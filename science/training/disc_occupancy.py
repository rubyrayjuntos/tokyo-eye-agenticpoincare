"""Centered disc occupancy metrics — primary signal for 2D Poincaré spread."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

# Promotion / save gates (tune after first healthy checkpoint)
DISC_SIGMA2_SIGMA1_PROMOTE_MIN = 0.35
DISC_EFFECTIVE_RANK_PROMOTE_MIN = 1.6
DISC_R_STD_PROMOTE_MIN = 0.02
DISC_LINE_THICKNESS_PROMOTE_MIN = 0.02
DISC_MIN_R_MEAN_FOR_OCCUPANCY = 0.05


def _effective_rank_np(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or s[0] < 1e-15:
        return 0.0
    return float(np.sum(s * s) / (s[0] * s[0]))


def disc_occupancy_from_numpy(xy: np.ndarray) -> dict[str, float]:
    """Centered SVD occupancy on [N, 2] disc coordinates."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return {
            "disc_sigma2_sigma1": 0.0,
            "disc_effective_rank": 0.0,
            "disc_line_thickness_rms": 0.0,
        }
    X = pts - pts.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    ratio = float(s[1] / (s[0] + 1e-12)) if s.size >= 2 else 0.0
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]
    resid = X - (X @ pc1)[:, None] * pc1[None, :]
    thickness = np.linalg.norm(resid, axis=1)
    return {
        "disc_sigma2_sigma1": ratio,
        "disc_effective_rank": _effective_rank_np(s),
        "disc_line_thickness_rms": float(np.sqrt((thickness**2).mean())),
    }


def disc_occupancy_from_tensor(hyp_proj_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable centered σ₂/σ₁ and effective rank for [N, 2] disc coords."""
    xy = hyp_proj_2d
    if xy.dim() != 2 or xy.shape[-1] != 2:
        raise ValueError(f"expected [N, 2] hyp_proj_2d, got {tuple(xy.shape)}")
    X = xy - xy.mean(dim=0, keepdim=True)
    if X.shape[0] < 3:
        z = torch.zeros((), device=xy.device, dtype=xy.dtype)
        return z, z + 1.0
    s = torch.linalg.svdvals(X)
    ratio = s[1] / (s[0] + 1e-8) if s.numel() >= 2 else torch.zeros((), device=xy.device, dtype=xy.dtype)
    eff = (s * s).sum() / (s[0] * s[0] + 1e-8)
    return ratio, eff


def _disc_r_collapse_penalty(
    hyp_proj_2d: torch.Tensor,
    *,
    min_disc_r_mean: float,
    collapse_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize origin collapse; returns (collapse_loss, r_mean)."""
    r_mean = hyp_proj_2d.norm(dim=-1).mean()
    collapse = torch.relu(min_disc_r_mean - r_mean) * collapse_scale
    return collapse, r_mean


def _occupancy_weight(r_mean: torch.Tensor, min_disc_r_mean: float) -> torch.Tensor:
    """Zero occupancy pressure when |proj| mean is below floor (differentiable ramp)."""
    return torch.clamp(r_mean / (min_disc_r_mean + 1e-8), max=1.0)


def disc_line_thickness_from_tensor(hyp_proj_2d: torch.Tensor) -> torch.Tensor:
    """Differentiable RMS line thickness (perpendicular residual to PC1).

    SVD is always run in float32 — half/bfloat16 inputs are unsupported and
  under autocast would otherwise break thickness floors and their gradients.
    """
    input_dtype = hyp_proj_2d.dtype
    xy = hyp_proj_2d.float()
    z = torch.zeros((), device=xy.device, dtype=torch.float32)
    X = xy - xy.mean(dim=0, keepdim=True)
    if X.shape[0] < 3:
        return z.to(dtype=input_dtype)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    pc1 = Vh[0]
    along = (X @ pc1).unsqueeze(-1) * pc1.unsqueeze(0)
    resid = X - along
    thickness = torch.sqrt((resid.pow(2).sum(dim=-1)).mean() + 1e-12)
    return thickness.to(dtype=input_dtype)


def disc_origin_circular_spread(hyp_proj_2d: torch.Tensor) -> torch.Tensor:
    """1 - |mean(exp(iθ))| — higher when angular spread from origin is wider."""
    theta = torch.atan2(hyp_proj_2d[:, 1], hyp_proj_2d[:, 0])
    mean_cos = torch.cos(theta).mean()
    mean_sin = torch.sin(theta).mean()
    R = torch.sqrt(mean_cos * mean_cos + mean_sin * mean_sin + 1e-12)
    return 1.0 - R


def disc_line_thickness_floor_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_thickness: float = 0.025,
    scale: float = 8.0,
    min_disc_r_mean: float = DISC_MIN_R_MEAN_FOR_OCCUPANCY,
) -> dict[str, torch.Tensor]:
    """Penalize thin disc wedges (low line_thickness_rms)."""
    thickness = disc_line_thickness_from_tensor(hyp_proj_2d)
    r_mean = hyp_proj_2d.norm(dim=-1).mean()
    weight = _occupancy_weight(r_mean, min_disc_r_mean)
    loss = weight * torch.relu(min_thickness - thickness) * scale
    return {
        "disc_thickness_floor": loss,
        "disc_line_thickness_rms": thickness.detach(),
    }


def ball_line_thickness_floor_loss(
    ball_points: torch.Tensor,
    *,
    min_thickness: float = 0.18,
    scale: float = 8.0,
) -> dict[str, torch.Tensor]:
    """Penalize low centered line thickness on ball coordinates [N, D]."""
    thickness = disc_line_thickness_from_tensor(ball_points)
    loss = torch.relu(min_thickness - thickness.float()) * scale
    return {
        "x_hyp_thickness_floor": loss,
        "x_hyp_line_thickness_rms": thickness.detach().float(),
    }


def disc_origin_span_floor_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_circular_spread: float = 0.12,
    scale: float = 6.0,
    min_disc_r_mean: float = DISC_MIN_R_MEAN_FOR_OCCUPANCY,
) -> dict[str, torch.Tensor]:
    """Penalize narrow origin angular span (proxy: low circular spread)."""
    spread = disc_origin_circular_spread(hyp_proj_2d)
    r_mean = hyp_proj_2d.norm(dim=-1).mean()
    weight = _occupancy_weight(r_mean, min_disc_r_mean)
    loss = weight * torch.relu(min_circular_spread - spread) * scale
    return {
        "disc_origin_span_floor": loss,
        "disc_origin_circular_spread": spread.detach(),
    }


def disc_occupancy_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_sigma_ratio: float = DISC_SIGMA2_SIGMA1_PROMOTE_MIN,
    scale: float = 5.0,
    min_disc_r_mean: float = DISC_MIN_R_MEAN_FOR_OCCUPANCY,
    collapse_scale: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Penalize rank-1 disc streaks via centered σ₂/σ₁ floor (guarded against origin collapse)."""
    collapse, r_mean = _disc_r_collapse_penalty(
        hyp_proj_2d,
        min_disc_r_mean=min_disc_r_mean,
        collapse_scale=collapse_scale,
    )
    ratio, eff = disc_occupancy_from_tensor(hyp_proj_2d)
    occ_term = torch.relu(min_sigma_ratio - ratio) * scale
    occ_weight = _occupancy_weight(r_mean, min_disc_r_mean)
    loss = occ_weight * occ_term + collapse
    return {
        "disc_occupancy": loss,
        "disc_r_collapse": collapse.detach(),
        "disc_r_mean": r_mean.detach(),
        "disc_sigma2_sigma1": ratio.detach(),
        "disc_effective_rank": eff.detach(),
    }


def disc_pc_repulsion_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_pc2_std: float = 0.08,
    scale: float = 4.0,
    min_disc_r_mean: float = DISC_MIN_R_MEAN_FOR_OCCUPANCY,
) -> dict[str, torch.Tensor]:
    """Penalize collapse along PC2 — spreads mass off the dominant streak axis."""
    xy = hyp_proj_2d
    r_mean = xy.norm(dim=-1).mean()
    z = torch.zeros((), device=xy.device, dtype=xy.dtype)
    X = xy - xy.mean(dim=0, keepdim=True)
    if X.shape[0] < 3:
        return {"disc_pc_repulsion": z, "disc_pc2_std": z}
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    pc2 = Vh[1]
    pc2_coords = X @ pc2
    pc2_std = pc2_coords.std()
    repel = torch.relu(min_pc2_std - pc2_std) * scale
    repel_weight = _occupancy_weight(r_mean, min_disc_r_mean)
    loss = repel_weight * repel
    return {"disc_pc_repulsion": loss, "disc_pc2_std": pc2_std.detach()}


def disc_eff_rank_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_eff_rank: float = DISC_EFFECTIVE_RANK_PROMOTE_MIN,
    scale: float = 3.0,
    min_disc_r_mean: float = DISC_MIN_R_MEAN_FOR_OCCUPANCY,
) -> dict[str, torch.Tensor]:
    """Penalize low centered effective rank on hyp_projections_2d."""
    _, eff = disc_occupancy_from_tensor(hyp_proj_2d)
    r_mean = hyp_proj_2d.norm(dim=-1).mean()
    weight = _occupancy_weight(r_mean, min_disc_r_mean)
    loss = weight * torch.relu(min_eff_rank - eff) * scale
    return {"disc_eff_rank": loss, "disc_effective_rank": eff.detach()}


def batch_diversity_repulsion_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_pairwise_dist: float = 0.035,
    scale: float = 4.0,
    min_disc_r_mean: float = DISC_MIN_R_MEAN_FOR_OCCUPANCY,
    subsample_pairs: int = 2048,
) -> dict[str, torch.Tensor]:
    """PC2-residual pairwise repulsion — spread points off the dominant streak axis."""
    xy = hyp_proj_2d
    r_mean = xy.norm(dim=-1).mean()
    z = torch.zeros((), device=xy.device, dtype=xy.dtype)
    X = xy - xy.mean(dim=0, keepdim=True)
    n = X.shape[0]
    if n < 4:
        return {"disc_batch_diversity": z, "disc_batch_min_pairwise": z}
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    pc1 = Vh[0]
    # Residuals perpendicular to PC1 (dominant streak) — repel in the opening direction.
    along_pc1 = (X @ pc1).unsqueeze(-1) * pc1.unsqueeze(0)
    resid = X - along_pc1
    d = torch.cdist(resid, resid)
    mask = torch.triu(torch.ones(n, n, device=xy.device, dtype=torch.bool), diagonal=1)
    d_pairs = d[mask]
    if d_pairs.numel() > subsample_pairs:
        idx = torch.randperm(d_pairs.numel(), device=xy.device)[:subsample_pairs]
        d_pairs = d_pairs[idx]
    repel = torch.relu(min_pairwise_dist - d_pairs).pow(2).mean() * scale
    weight = _occupancy_weight(r_mean, min_disc_r_mean)
    loss = weight * repel
    return {
        "disc_batch_diversity": loss,
        "disc_batch_min_pairwise": d_pairs.min().detach(),
    }


def _finite_probe(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def disc_occupancy_ineligibility_reasons(
    health: dict[str, Any],
    *,
    min_sigma_ratio: float = DISC_SIGMA2_SIGMA1_PROMOTE_MIN,
    min_effective_rank: float | None = DISC_EFFECTIVE_RANK_PROMOTE_MIN,
    min_disc_r_std: float | None = DISC_R_STD_PROMOTE_MIN,
    min_line_thickness: float | None = None,
) -> list[str]:
    """Return eligibility failures from epoch health disc occupancy means."""
    ratio = _finite_probe(health.get("disc_sigma2_sigma1_mean"))
    eff = _finite_probe(health.get("disc_effective_rank_mean"))
    r_std = _finite_probe(health.get("disc_r_std_mean"))
    thickness = _finite_probe(health.get("disc_line_thickness_rms_mean"))
    if ratio is None and eff is None and r_std is None and thickness is None:
        return []

    reasons: list[str] = []
    if r_std is not None and min_disc_r_std is not None and r_std < min_disc_r_std:
        reasons.append(
            f"disc_r_std={r_std:.4f}<{min_disc_r_std} (origin-collapse / no radial spread)"
        )
    if (
        thickness is not None
        and min_line_thickness is not None
        and thickness < min_line_thickness
    ):
        reasons.append(
            f"disc_line_thickness_rms={thickness:.5f}<{min_line_thickness} "
            "(rank-1 streak / thin disc cloud)"
        )
    if ratio is not None and min_sigma_ratio is not None and ratio < min_sigma_ratio:
        reasons.append(
            f"disc_sigma2_sigma1={ratio:.3f}<{min_sigma_ratio} (rank-1 disc streak)"
        )
    if (
        min_effective_rank is not None
        and eff is not None
        and eff < min_effective_rank
    ):
        reasons.append(
            f"disc_effective_rank={eff:.3f}<{min_effective_rank} (collapsed disc)"
        )
    return reasons
