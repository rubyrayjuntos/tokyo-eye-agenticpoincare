"""EquivariantConv with thermodynamic message modulation (v6.6 training).

Geometric channels ``edge_attr[:, :4]`` drive SH + distance radial MLP (warm-start
compatible). When thermo columns are present, an **additive** TP-weight head
(non-zero init) lets affinity reshape messages — the prior multiplicative
identity gate was too weak to move geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn import o3
from torch_geometric.nn import MessagePassing

from science.tokyo_eye.thermo_edge_features import EDGE_ATTR_THERMO_DIM, GEO_DIM, THERMO_DIM


class EquivariantConvThermo(MessagePassing):
    """SE(3) conv + additive thermo → TP weights when ``edge_attr`` is wide enough."""

    def __init__(
        self,
        hidden_dim: int,
        irreps_hidden: str = "32x0e + 8x1e",
        *,
        thermo_dim: int = THERMO_DIM,
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
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel),
        )
        self.thermo_dim = int(thermo_dim)
        # Kept for checkpoint key continuity with soft-gate A/B; unused in forward.
        self.thermo_gate = nn.Sequential(
            nn.Linear(self.thermo_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.thermo_gate[-1].weight)
        nn.init.zeros_(self.thermo_gate[-1].bias)

        self.thermo_to_weights = nn.Linear(self.thermo_dim, self.tp.weight_numel)
        nn.init.xavier_uniform_(self.thermo_to_weights.weight, gain=0.25)
        nn.init.zeros_(self.thermo_to_weights.bias)

        self.hidden_dim = hidden_dim
        num_scalars = sum(
            mul * ir.dim for mul, ir in self.irreps_hidden if ir.l == 0
        )
        self.num_scalars = num_scalars
        self._proj = nn.Linear(num_scalars, hidden_dim)

    def _edge_weights(self, edge_attr: torch.Tensor) -> torch.Tensor:
        dist = edge_attr[:, 3:4]
        base = self.radial_mlp(dist)
        if edge_attr.size(-1) < EDGE_ATTR_THERMO_DIM:
            return base
        thermo = edge_attr[:, GEO_DIM : GEO_DIM + self.thermo_dim]
        return base + self.thermo_to_weights(thermo)

    def forward(self, x, edge_index, edge_attr):
        rel_pos = edge_attr[:, :3]
        sh = o3.spherical_harmonics(
            l=[0, 1], x=rel_pos, normalize=True, normalization="component"
        )
        edge_weights = self._edge_weights(edge_attr)
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
        out = self.propagate(edge_index, x=x_irreps, sh=sh, edge_weights=edge_weights)
        return self._proj(out[:, : self.num_scalars])

    def message(self, x_j, sh, edge_weights):
        return self.tp(x_j, sh, edge_weights)
