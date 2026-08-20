"""EquivariantConv with per-relation message paths (v6.6 training).

Supports:
- thermo one-hots (generic / wrapped / dehydron) on contact graph, or
- role one-hots (packing / dehydron / spoke / ribbon) on role-typed graph.

Training-only — does not change GraphBuilder / Normalizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn import o3
from torch_geometric.nn import MessagePassing

from science.tokyo_eye.thermo_edge_features import GEO_DIM

# Default: thermo layout (onehots at GEO+2). Role layout uses offset=GEO_DIM.
_DEFAULT_ONEHOT_OFFSET = GEO_DIM + 2
_DEFAULT_NUM_RELATIONS = 3


def _make_radial_mlp(weight_numel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, weight_numel),
    )


class EquivariantConvMultiRel(MessagePassing):
    """SE(3) conv with one radial MLP per edge relation."""

    def __init__(
        self,
        hidden_dim: int,
        irreps_hidden: str = "32x0e + 8x1e",
        *,
        num_relations: int = _DEFAULT_NUM_RELATIONS,
        onehot_offset: int = _DEFAULT_ONEHOT_OFFSET,
        spoke_rho_col: int | None = None,
        spoke_relation_id: int = 2,
        spoke_edge_scale: float = 1.0,
        ribbon_relation_id: int = 3,
        ribbon_edge_scale: float = 1.0,
        coupling_strength_col: int | None = None,
        coupling_relation_id: int = 4,
        dehydron_barcode_col: int | None = None,
        dehydron_relation_id: int = 1,
        dehydron_angular_scale: float = 1.0,
        ha_strength_col: int | None = None,
        ha_packing_relation_id: int = 0,
        ha_dehydron_relation_id: int = 1,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.irreps_hidden = Irreps(irreps_hidden)
        self.irreps_sh = Irreps("1x0e + 1x1e")
        self.tp = FullyConnectedTensorProduct(
            self.irreps_hidden,
            self.irreps_sh,
            self.irreps_hidden,
            shared_weights=False,
        )
        self.num_relations = int(num_relations)
        self.onehot_offset = int(onehot_offset)
        self.spoke_rho_col = spoke_rho_col
        self.spoke_relation_id = int(spoke_relation_id)
        self.spoke_edge_scale = float(spoke_edge_scale)
        self.ribbon_relation_id = int(ribbon_relation_id)
        self.ribbon_edge_scale = float(ribbon_edge_scale)
        self.coupling_strength_col = coupling_strength_col
        self.coupling_relation_id = int(coupling_relation_id)
        self.dehydron_barcode_col = dehydron_barcode_col
        self.dehydron_relation_id = int(dehydron_relation_id)
        self.dehydron_angular_scale = float(dehydron_angular_scale)
        self.ha_strength_col = ha_strength_col
        self.ha_packing_relation_id = int(ha_packing_relation_id)
        self.ha_dehydron_relation_id = int(ha_dehydron_relation_id)
        self.radial_mlps = nn.ModuleList(
            [_make_radial_mlp(self.tp.weight_numel) for _ in range(self.num_relations)]
        )
        # Telemetry / legacy callers expect ``radial_mlp``.
        self.radial_mlp = self.radial_mlps[0]
        self.hidden_dim = hidden_dim
        num_scalars = sum(
            mul * ir.dim for mul, ir in self.irreps_hidden if ir.l == 0
        )
        self.num_scalars = num_scalars
        self._proj = nn.Linear(num_scalars, hidden_dim)

    def _min_attr_width(self) -> int:
        return self.onehot_offset + self.num_relations

    def _relation_masks(self, edge_attr: torch.Tensor) -> list[torch.Tensor]:
        """Boolean masks [E] for each relation; narrow attr → all edges as rel 0."""
        e = edge_attr.size(0)
        device = edge_attr.device
        if edge_attr.size(-1) < self._min_attr_width():
            masks = [
                torch.zeros(e, dtype=torch.bool, device=device)
                for _ in range(self.num_relations)
            ]
            masks[0] = torch.ones(e, dtype=torch.bool, device=device)
            return masks
        onehots = edge_attr[
            :, self.onehot_offset : self.onehot_offset + self.num_relations
        ]
        masks = [onehots[:, r] > 0.5 for r in range(self.num_relations)]
        assigned = torch.stack(masks, dim=-1).any(dim=-1)
        if not bool(assigned.all()):
            masks[0] = masks[0] | (~assigned)
        return masks

    def forward(self, x, edge_index, edge_attr):
        rel_pos = edge_attr[:, :3]
        dist = edge_attr[:, 3:4]
        sh = o3.spherical_harmonics(
            l=[0, 1], x=rel_pos, normalize=True, normalization="component"
        )
        irreps_dim = self.irreps_hidden.dim
        if x.size(-1) < irreps_dim:
            padding = torch.zeros(
                x.size(0),
                irreps_dim - x.size(-1),
                device=x.device,
                dtype=x.dtype,
            )
            x_irreps = torch.cat([x, padding], dim=-1)
        else:
            x_irreps = x[:, :irreps_dim]

        masks = self._relation_masks(edge_attr)
        out = torch.zeros(
            x_irreps.size(0),
            self.irreps_hidden.dim,
            device=x.device,
            dtype=x.dtype,
        )
        for r, mask in enumerate(masks):
            if not bool(mask.any()):
                continue
            ei = edge_index[:, mask]
            edge_weights = self.radial_mlps[r](dist[mask])
            if (
                self.spoke_rho_col is not None
                and r == self.spoke_relation_id
                and edge_attr.size(-1) > self.spoke_rho_col
            ):
                rho_w = edge_attr[
                    mask, self.spoke_rho_col : self.spoke_rho_col + 1
                ].clamp(0.0, 2.0)
                edge_weights = edge_weights * (1.0 + rho_w) * self.spoke_edge_scale
            if r == self.ribbon_relation_id and self.ribbon_edge_scale != 1.0:
                edge_weights = edge_weights * self.ribbon_edge_scale
            if (
                self.coupling_strength_col is not None
                and r == self.coupling_relation_id
                and edge_attr.size(-1) > self.coupling_strength_col
            ):
                c_w = edge_attr[
                    mask, self.coupling_strength_col : self.coupling_strength_col + 1
                ].clamp(0.0, 2.0)
                edge_weights = edge_weights * (1.0 + c_w)
            if (
                self.dehydron_barcode_col is not None
                and r == self.dehydron_relation_id
                and edge_attr.size(-1)
                > self.dehydron_barcode_col + 3
            ):
                bc = edge_attr[
                    mask,
                    self.dehydron_barcode_col : self.dehydron_barcode_col + 4,
                ]
                # wrap_deficit + normalized local H1 persistence → local leak strength
                strength = (bc[:, 0:1] + bc[:, 3:4].clamp(0.0, 1.0)).clamp(0.0, 2.0)
                edge_weights = edge_weights * (1.0 + strength)
            if (
                self.ha_strength_col is not None
                and r
                in (self.ha_packing_relation_id, self.ha_dehydron_relation_id)
                and edge_attr.size(-1) > self.ha_strength_col
            ):
                ha_w = edge_attr[
                    mask, self.ha_strength_col : self.ha_strength_col + 1
                ].clamp(0.0, 1.0)
                edge_weights = edge_weights * (1.0 + ha_w)
            sh_r = sh[mask]
            if (
                r == self.dehydron_relation_id
                and self.dehydron_angular_scale != 1.0
                and sh_r.size(-1) >= 4
            ):
                sh_r = sh_r.clone()
                sh_r[:, 1:4] = sh_r[:, 1:4] * self.dehydron_angular_scale
                # scale=0 → isotropic dehydron messages only (radial detach)
            out = out + self.propagate(
                ei, x=x_irreps, sh=sh_r, edge_weights=edge_weights
            )
        return self._proj(out[:, : self.num_scalars])

    def message(self, x_j, sh, edge_weights):
        return self.tp(x_j, sh, edge_weights)


__all__ = [
    "EquivariantConvMultiRel",
]
