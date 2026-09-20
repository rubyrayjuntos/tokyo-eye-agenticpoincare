"""EquiformerV3 → node (s, v) cut BEFORE energy_block (geoopt_restore contract).

SE(3)-lite is forbidden on the sealed path. This module wraps official
EquiformerV3_OC (optionally MPtrj-warmed) and exposes residue/atom node
features after ``_forward_blocks``, before graph energy pooling.

Addendum §2.3: ``cold_init=True`` (no ckpt / ignore ckpt) is the freeze-
recon default — random architecture weights, not MPtrj pretraining claim.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from science.tokyo_eye.v8.protein_eqf_batch import ca_coords_to_eqf_data

_SCIENCE_TP = Path(__file__).resolve().parents[2] / "third_party"

# Official MPtrj gradient finetune YAML dims (omat24/mptrj/...yml)
MPTRJ_GRADIENT_KWARGS: dict[str, Any] = dict(
    use_pbc=False,
    use_pbc_single=True,
    otf_graph=True,
    regress_forces=False,
    regress_stress=False,
    direct_prediction=False,
    max_neighbors=50,  # laptop 4G; graph construction only (not weight shape)
    max_radius=8.0,
    num_radial_basis=10,
    num_layers=7,
    num_channels=128,
    attn_hidden_channels=32,
    num_heads=8,
    attn_alpha_channels=64,
    attn_value_channels=16,
    ffn_hidden_channels=512,
    norm_type="merge_layer_norm",
    lmax=4,
    mmax=2,
    attn_grid_resolution_list=[14, 8],
    ffn_grid_resolution_list=[14, 14],
    edge_channels=128,
    use_grid_mlp=True,
)


def _ensure_vendor_path() -> None:
    tp = str(_SCIENCE_TP)
    if _SCIENCE_TP.is_dir() and tp not in sys.path:
        sys.path.insert(0, tp)


class EquiformerPoolFrontend(nn.Module):
    """Official EquiformerV3 forward cut at pre-energy node features → (s, v)."""

    def __init__(
        self,
        *,
        equiformer_ckpt: str | Path | None = None,
        scalar_dim: int = 128,
        vector_dim: int = 3,
        freeze: bool = True,
        max_neighbors: int | None = None,
        cold_init: bool = False,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.equiformer_ckpt = (
            None if equiformer_ckpt is None else Path(equiformer_ckpt)
        )
        kwargs = dict(MPTRJ_GRADIENT_KWARGS)
        if max_neighbors is not None:
            kwargs["max_neighbors"] = int(max_neighbors)
        # §2.3: cold random-init wins over warm MPtrj unless explicitly disabled.
        use_cold = bool(cold_init) or self.equiformer_ckpt is None
        self._backbone = self._build_backbone(
            self.equiformer_ckpt, kwargs, cold_init=use_cold
        )
        if freeze:
            for p in self._backbone.parameters():
                p.requires_grad_(False)
            self._backbone.eval()
        self._s_proj: nn.Module = nn.Identity()
        self._v_proj: nn.Module = nn.Identity()
        self.frontend_mode = "equiformer_v3_pool"
        self.live_backbone = False  # never SE(3)-lite

    def _build_backbone(
        self,
        ckpt: Path | None,
        kwargs: dict[str, Any],
        *,
        cold_init: bool,
    ) -> nn.Module:
        _ensure_vendor_path()
        try:
            from equiformer_v3_model.equiformer_v3 import EquiformerV3_OC
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "EquiformerV3_OC unavailable under science/third_party. "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        model = EquiformerV3_OC(**kwargs)
        if cold_init:
            self._load_report = {
                "missing": 0,
                "unexpected": 0,
                "ckpt": None if ckpt is None else str(ckpt),
                "mode": "equiformer_v3_pool_cold_init",
                "cold_init": True,
                "kwargs": {
                    k: kwargs[k]
                    for k in ("num_layers", "num_channels", "lmax", "max_neighbors")
                },
            }
            return model

        if ckpt is None or not ckpt.is_file():
            raise FileNotFoundError(f"Equiformer MPtrj ckpt missing: {ckpt}")

        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        if isinstance(state, dict) and any(str(k).startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        self._load_report = {
            "missing": len(missing),
            "unexpected": len(unexpected),
            "ckpt": str(ckpt),
            "mode": "equiformer_v3_pool_loaded",
            "cold_init": False,
            "kwargs": {
                k: kwargs[k]
                for k in ("num_layers", "num_channels", "lmax", "max_neighbors")
            },
        }
        if missing or unexpected:
            # Shape-matched MPtrj should be 0/0; surface for preflight.
            self._load_report["missing_sample"] = list(missing)[:8]
            self._load_report["unexpected_sample"] = list(unexpected)[:8]
        return model

    def load_info(self) -> dict[str, Any]:
        return dict(getattr(self, "_load_report", {}))

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self._backbone.eval()
        return self

    def set_live_backbone(self, live: bool) -> None:
        if live:
            raise RuntimeError(
                "EquiformerPoolFrontend forbids SE(3)-lite live_backbone under geoopt_restore"
            )

    def forward_blocks_features(self, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Equiformer through blocks; return (x_scalar, x_full) pre-energy."""
        bb = self._backbone
        bb.device = data.pos.device
        bb.dtype = data.pos.dtype
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            *_rest,
        ) = bb.generate_graph(
            data,
            enforce_max_neighbors_strictly=getattr(
                bb, "enforce_max_neighbors_strictly", True
            ),
            use_pbc_single=True,
        )
        atomic_numbers = data.atomic_numbers.long()
        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]
        edge_distance, edge_envelope_weight = bb._forward_edge(
            edge_distance, edge_distance_vec
        )
        x = bb._forward_embedding(
            atomic_numbers, edge_distance, edge_index, edge_envelope_weight
        )
        x_scalar, x = bb._forward_blocks(
            x,
            source_atomic_numbers,
            target_atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
            data.batch,
        )
        return x_scalar, x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        *,
        data: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(s, v)`` for the hyperbolic spine.

        If ``data`` is omitted, builds CA→Equiformer batch from ``x`` ([N,3] coords).
        """
        del edge_index, edge_type  # R0–R5 edges are spine-side; Equiformer builds its own graph
        if data is None:
            data = ca_coords_to_eqf_data(x)
        x_scalar, x_full = self.forward_blocks_features(data)
        s = x_scalar
        if x_full.ndim == 3 and x_full.shape[1] >= 4:
            v = x_full[:, 1:4, :].mean(dim=-1)
        elif x_full.ndim == 2 and x_full.shape[-1] >= 3:
            v = x_full[:, :3]
        else:
            v = torch.zeros(s.shape[0], 3, device=s.device, dtype=s.dtype)
        if s.shape[-1] != self.scalar_dim:
            if not isinstance(self._s_proj, nn.Linear):
                self._s_proj = nn.Linear(s.shape[-1], self.scalar_dim).to(s.device)
            s = self._s_proj(s)
        if v.shape[-1] != self.vector_dim:
            if not isinstance(self._v_proj, nn.Linear):
                self._v_proj = nn.Linear(v.shape[-1], self.vector_dim).to(v.device)
            v = self._v_proj(v)
        return s, v


__all__ = ["EquiformerPoolFrontend", "MPTRJ_GRADIENT_KWARGS"]
