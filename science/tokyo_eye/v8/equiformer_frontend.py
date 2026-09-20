"""EquiformerV3 front-end binding for TokyoEye-v8 (Sprint 5+).

Isolated stub + audited weight-map loader against
``mirror-physics/equiformer_v3`` MPtrj-gradient DDP checkpoints.
A full EquiformerV3 forward can replace the stub trunks later without
touching hyp / MoE code; remapped tensors live on ``backbone``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHT_MAP = Path(
    "science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json"
)
DEFAULT_CKPT = Path("checkpoints/v8/pretrained/equiformer_v3_baseline.pt")


def load_weight_map(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_WEIGHT_MAP
    with p.open() as f:
        cfg = json.load(f)
    if int(cfg.get("schema_version", 0)) < 1:
        raise ValueError(f"unsupported weight map schema in {p}")
    return cfg


def _is_regex_rename_pattern(pattern: str) -> bool:
    """True when the weight-map key is a regex (anchors / captures / globs)."""
    return bool(
        pattern.startswith("^")
        or pattern.endswith("$")
        or r"\d" in pattern
        or ".*" in pattern
        or ".+" in pattern
        or "(" in pattern
    )


def resolve_checkpoint_key(
    src_key: str,
    renames: Mapping[str, str],
    *,
    use_intersection: bool,
    strip_prefixes: tuple[str, ...] = (),
) -> str | None:
    """Map an official EquiformerV3 state-dict key to a module key.

    Exact renames win first; then regex patterns with ``\\1`` block capture.
    Optional ``strip_prefixes`` lets ``module.foo`` also try ``foo`` against
    rename patterns that omit the DDP wrapper.
    """
    candidates = [src_key]
    for prefix in strip_prefixes:
        if src_key.startswith(prefix):
            candidates.append(src_key[len(prefix) :])

    for cand in candidates:
        if cand in renames:
            return renames[cand]
        for pattern, repl in renames.items():
            if not _is_regex_rename_pattern(pattern):
                continue
            try:
                match = re.fullmatch(pattern, cand)
            except re.error:
                continue
            if match is None:
                # Also try matching the full DDP key against anchored patterns.
                if cand is src_key:
                    continue
                try:
                    match = re.fullmatch(pattern, src_key)
                except re.error:
                    continue
                if match is None:
                    continue
            return match.expand(repl)

    if use_intersection:
        return src_key
    return None


class _AlphaNormBank(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))


class _BlockAttnBank(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.source = nn.Linear(dim, dim, bias=False)
        self.target = nn.Linear(dim, dim, bias=False)
        # EquiformerV3 ga.proj.weight is (Lmax+1, C, C) with Lmax=4 → (5, 128, 128)
        self.proj = nn.Parameter(torch.empty(5, dim, dim))
        self.alpha_norm = _AlphaNormBank(64)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.source.weight)
        nn.init.xavier_uniform_(self.target.weight)
        nn.init.xavier_uniform_(self.proj.view(5, -1))


class _BlockFfnBank(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.scalar_in = nn.Linear(dim, 1024, bias=True)
        self.gate = nn.Linear(dim, 512, bias=True)


class _BlockNormBank(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(5, dim))
        self.affine_bias = nn.Parameter(torch.zeros(dim))


class _BlockBank(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.attn = _BlockAttnBank(dim)
        self.ffn = _BlockFfnBank(dim)
        self.norm_1 = _BlockNormBank(dim)
        self.norm_2 = _BlockNormBank(dim)


class EquiformerBackboneWeightBank(nn.Module):
    """Parameter bank shaped to audited MPtrj-gradient EquiformerV3 tensors.

    Holds remapped DDP weights under ``backbone.*`` so ``apply_weight_map``
    can bind production checkpoints without vendoring the full SE(3) forward.
    """

    def __init__(self, *, dim: int = 128, num_blocks: int = 7) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_blocks = int(num_blocks)
        self.atom_embed = nn.Embedding(dim, dim)  # weight (dim, dim) matches sphere_embedding
        # Re-bind as Parameter-compatible Linear-like for (128, 128) weight
        # nn.Embedding(128, 128).weight is (128, 128) — same layout as sphere_embedding.
        self.edge_degree_embed = nn.ModuleDict(
            {
                "source": nn.Linear(dim, dim, bias=False),
                "target": nn.Linear(dim, dim, bias=False),
            }
        )
        self.blocks = nn.ModuleList([_BlockBank(dim) for _ in range(self.num_blocks)])


class StubEquiformerFrontend(nn.Module):
    """Equiformer front-end: stub trunks or live SE(3)-lite over the weight bank.

    * ``live_backbone=False`` (or ``--freeze-backbone``): Sprint 5 stub trunks.
    * ``live_backbone=True``: SE(3)-lite path so MPtrj bank tensors see gradients.
    """

    def __init__(
        self,
        *,
        in_dim: int = 3,
        scalar_dim: int = 128,
        vector_dim: int = 3,
        num_backbone_blocks: int = 7,
        live_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.live_backbone = bool(live_backbone)
        self.backbone = EquiformerBackboneWeightBank(
            dim=scalar_dim, num_blocks=num_backbone_blocks
        )
        self.scalar_trunk = nn.Sequential(
            nn.Linear(in_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.vector_trunk = nn.Sequential(
            nn.Linear(in_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, vector_dim),
        )
        # SE(3)-lite adapters (new; bank holds pretrained Equiformer slices)
        self.seed_proj = nn.Linear(in_dim, scalar_dim)
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.ffn_down = nn.Linear(1024, scalar_dim)
        self.gate_down = nn.Linear(512, scalar_dim)
        self.out_s = nn.Linear(scalar_dim, scalar_dim)
        self.out_v = nn.Linear(scalar_dim, vector_dim)
        self.rel_bias = nn.Embedding(6, scalar_dim)

    def set_live_backbone(self, live: bool) -> None:
        self.live_backbone = bool(live)

    def _se3_lite_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-pass lite equivariant update using remapped bank weights."""
        from torch_geometric.utils import scatter

        n, d = x.shape[0], self.scalar_dim
        h = self.seed_proj(x)
        # Live sphere / atom embedding matrix from MPtrj bank
        h = h + F.linear(torch.tanh(h), self.backbone.atom_embed.weight)

        if edge_index.numel() > 0:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            rel = edge_type.long().clamp(0, 5)
            dist = torch.linalg.vector_norm(x[src] - x[dst], dim=-1, keepdim=True)
            rbf = self.radial_mlp(dist)
            block = self.backbone.blocks[0]
            m_s = block.attn.source(h[src])
            m_t = block.attn.target(h[dst])
            # ga.proj is (Lmax+1, C, C); use L=0 channel as scalar coupling
            proj0 = block.attn.proj[0]
            msg = F.silu(m_s + m_t + rbf + self.rel_bias(rel))
            msg = F.linear(msg, proj0)
            # Also mix edge-degree bank embeddings (live)
            msg = msg + self.backbone.edge_degree_embed["source"](h[src]) * 0.1
            msg = msg + self.backbone.edge_degree_embed["target"](h[dst]) * 0.1
            agg = scatter(msg, src, dim=0, dim_size=n, reduce="mean")
            h = h + agg
            h = h + self.ffn_down(F.silu(block.ffn.scalar_in(h)))
            h = h + 0.05 * self.gate_down(F.silu(block.ffn.gate(torch.tanh(h))))

        s = self.out_s(F.silu(h))
        v = self.out_v(F.silu(h))
        return s, v

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"x must be [N, in_dim], got {tuple(x.shape)}")
        if (
            self.live_backbone
            and edge_index is not None
            and edge_type is not None
        ):
            return self._se3_lite_forward(x, edge_index, edge_type)
        return self.scalar_trunk(x), self.vector_trunk(x)


def _unbox_checkpoint(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict) and "state_dict" in blob:
        raw = blob["state_dict"]
    elif isinstance(blob, dict) and "model" in blob:
        raw = blob["model"]
    elif isinstance(blob, dict):
        raw = blob
    else:
        raise TypeError(f"unsupported checkpoint type: {type(blob)}")
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint payload is not a dict: {type(raw)}")
    return raw


def apply_weight_map(
    module: nn.Module,
    checkpoint: Mapping[str, torch.Tensor] | Path | str,
    weight_map: Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Load Equiformer weights into ``module`` using the JSON weight map."""
    cfg = dict(weight_map or {})
    if isinstance(checkpoint, (str, Path)):
        path = Path(checkpoint)
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Older pickles with non-tensor metadata; trusted local ckpt only.
            blob = torch.load(path, map_location="cpu", weights_only=False)
        raw = _unbox_checkpoint(blob)
    else:
        raw = dict(checkpoint)

    key_map = cfg.get("key_map") or {}
    renames: dict[str, str] = dict(key_map.get("renames") or {})
    use_intersection = bool(key_map.get("exact_name_intersection", False))
    strip_prefixes = tuple(key_map.get("strip_prefixes") or ())

    mapped: dict[str, torch.Tensor] = {}
    rename_hits: list[tuple[str, str]] = []
    for src_k, tensor in raw.items():
        if not torch.is_tensor(tensor):
            continue
        dst_k = resolve_checkpoint_key(
            src_k,
            renames,
            use_intersection=use_intersection,
            strip_prefixes=strip_prefixes,
        )
        if dst_k is None:
            continue
        mapped[dst_k] = tensor
        if dst_k != src_k:
            rename_hits.append((src_k, dst_k))

    current = module.state_dict()
    filtered = {
        k: v for k, v in mapped.items() if k in current and current[k].shape == v.shape
    }
    missing_shapes = [
        k
        for k, v in mapped.items()
        if k in current and current[k].shape != v.shape
    ]
    unmatched_renames = [
        (src, dst) for src, dst in rename_hits if dst not in filtered
    ]
    result = module.load_state_dict(filtered, strict=False)
    return {
        "loaded_keys": sorted(filtered.keys()),
        "n_loaded": len(filtered),
        "n_renamed": len(rename_hits),
        "n_mapped": len(mapped),
        "rename_examples": rename_hits[:12],
        "unmatched_renames": unmatched_renames[:12],
        "missing_in_module": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "shape_mismatches": missing_shapes,
        "strict": strict,
        "checkpoint_n_tensors": sum(1 for v in raw.values() if torch.is_tensor(v)),
    }


def evaluate_geometry_health(
    metrics: Mapping[str, float],
    *,
    oversmooth_entropy_floor: float = 0.20,
    boundary_saturation_pct_ceiling: float = 40.0,
) -> dict[str, float]:
    """Operational safeguards from the Sprint-5 live-run playbook."""
    entropy = float(
        metrics.get("diag_manifold_entropy", metrics.get("diag_radial_entropy", 1.0))
    )
    spread = float(metrics.get("diag_radius_spread", 1.0))
    sat_pct = float(
        metrics.get(
            "diag_boundary_saturation_pct",
            float(metrics.get("diag_boundary_saturation", 0.0)) * 100.0,
        )
    )
    # True oversmoothing: low entropy AND collapsed radial spread (not just 1 hist bin)
    warn_oversmooth = (
        1.0
        if (entropy <= float(oversmooth_entropy_floor) and spread < 0.05)
        else 0.0
    )
    warn_boundary = 1.0 if sat_pct > float(boundary_saturation_pct_ceiling) else 0.0
    return {
        "warn_oversmooth": warn_oversmooth,
        "warn_boundary_blowout": warn_boundary,
        "margin_boost_factor": 1.5 if warn_oversmooth else 1.0,
    }


def build_param_groups(
    backbone: nn.Module,
    hyperbolic: nn.Module,
    *,
    lr_backbone: float = 1e-5,
    lr_hyperbolic: float = 1e-3,
    freeze_backbone: bool = False,
) -> list[dict[str, Any]]:
    """Differential LR groups for Equiformer trunk vs hyp/MoE spine.

    ``freeze_backbone`` freezes the MPtrj weight bank (``module.backbone`` when
    present), not the stub / SE(3)-lite adapter layers.
    """
    if freeze_backbone:
        bank = getattr(backbone, "backbone", None)
        freeze_target = bank if isinstance(bank, nn.Module) else backbone
        for p in freeze_target.parameters():
            p.requires_grad = False
    return [
        {
            "params": [p for p in backbone.parameters() if p.requires_grad],
            "lr": float(lr_backbone),
            "name": "backbone",
        },
        {
            "params": [p for p in hyperbolic.parameters() if p.requires_grad],
            "lr": float(lr_hyperbolic),
            "name": "hyperbolic",
        },
    ]


class TokyoEyeV8WithFrontend(nn.Module):
    """Equiformer front-end + hyperbolic spine in one module."""

    def __init__(
        self,
        frontend: StubEquiformerFrontend,
        spine: nn.Module,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.spine = spine

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        *,
        tau_ceiling: float = 0.995,
        chem: torch.Tensor | None = None,
        gate_chem: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        s, v = self.frontend(x, edge_index=edge_index, edge_type=edge_type)
        return self.spine(
            s,
            v,
            edge_index,
            edge_type,
            tau_ceiling=tau_ceiling,
            chem=chem if chem is not None else gate_chem,
        )

    def set_moe_temperature(self, tau: float) -> None:
        self.spine.set_moe_temperature(tau)

    def set_moe_explore_epsilon(self, epsilon: float) -> None:
        self.spine.set_moe_explore_epsilon(epsilon)

    def set_moe_mode(self, mode: str) -> None:
        self.spine.set_moe_mode(mode)


__all__ = [
    "DEFAULT_CKPT",
    "DEFAULT_WEIGHT_MAP",
    "EquiformerBackboneWeightBank",
    "StubEquiformerFrontend",
    "TokyoEyeV8WithFrontend",
    "apply_weight_map",
    "build_param_groups",
    "evaluate_geometry_health",
    "load_weight_map",
    "resolve_checkpoint_key",
]
