"""Centered disc occupancy metrics — primary signal for 2D Poincaré spread.

Convention (centered SVD on [N, 2] disc coordinates):
  disc_sigma2_sigma1 = σ₂ / σ₁ = s[1] / s[0]  with s[0] ≥ s[1]

  → 1.0  healthy 2D spread (both principal axes carry variance)
  → 0.0  rank-1 streak / collapse (PC2 ≪ PC1)

Promote floors penalize *low* σ₂/σ₁ (see DISC_SIGMA2_SIGMA1_PROMOTE_MIN).
"""

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

# Tier 2 biology gate — block angular emergence claims on rank-1 crescent streaks.
CRESCENT_TIER2_BLOCK_MIN_EFFECTIVE_RANK = 1.35
CRESCENT_TIER2_BLOCK_MIN_LINE_THICKNESS = 0.02
CRESCENT_TIER2_BLOCK_MIN_SIGMA2_SIGMA1 = 0.22
CRESCENT_TIER2_BLOCK_MAX_R_SPAN = 0.12  # narrow radial band + low thickness ⇒ 1D arc


def crescent_geometry_metrics(xy: np.ndarray) -> dict[str, float]:
    """Occupancy metrics on disc xy for crescent-collapse detection."""
    pts = np.asarray(xy, dtype=np.float64)
    occ = disc_occupancy_from_numpy(pts)
    r_span = 0.0
    if pts.ndim == 2 and pts.shape[0] >= 2:
        r = np.linalg.norm(pts, axis=1)
        r_span = float(r.max() - r.min())
    occ["disc_r_span"] = r_span
    return occ


def is_crescent_collapsed(
    xy: np.ndarray,
    *,
    min_effective_rank: float = CRESCENT_TIER2_BLOCK_MIN_EFFECTIVE_RANK,
    min_line_thickness: float = CRESCENT_TIER2_BLOCK_MIN_LINE_THICKNESS,
    min_sigma2_sigma1: float = CRESCENT_TIER2_BLOCK_MIN_SIGMA2_SIGMA1,
    max_r_span: float = CRESCENT_TIER2_BLOCK_MAX_R_SPAN,
) -> tuple[bool, dict[str, float]]:
    """
  Return True when disc occupancy is a thin 1D crescent (r–θ entangled).

  Blocked if effective rank is low AND (thin streak OR narrow radial band).
  """
    metrics = crescent_geometry_metrics(xy)
    eff = metrics["disc_effective_rank"]
    thick = metrics["disc_line_thickness_rms"]
    s21 = metrics["disc_sigma2_sigma1"]
    r_span = metrics["disc_r_span"]
    rank_collapsed = eff < min_effective_rank
    thin_streak = thick < min_line_thickness or s21 < min_sigma2_sigma1
    narrow_r = r_span < max_r_span
    blocked = rank_collapsed and (thin_streak or narrow_r)
    return blocked, metrics


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


def core_radial_floor_loss(
    hyp_proj_2d: torch.Tensor,
    target_rho: torch.Tensor,
    target_tau: torch.Tensor | None = None,
    *,
    min_r: float = 0.15,
    rho_scale: float = 30.0,
) -> dict[str, torch.Tensor]:
    """Soft min-radius for core / high-ρ low-τ residues (e1-style niche).

    Weight ``w ∝ (ρ/ρ_scale) · (1 − τ)`` so surface (τ≈1) and low-ρ residues
    are barely touched. ``min_r`` is a *near-origin* floor (~0.15), not a rim push —
    preserves core-at-center identity while lifting global ``disc_r_mean``.
    """
    device = hyp_proj_2d.device
    dtype = hyp_proj_2d.dtype
    z = torch.zeros((), device=device, dtype=dtype)
    if hyp_proj_2d.ndim != 2 or hyp_proj_2d.shape[0] < 1:
        return {
            "core_radial_floor": z,
            "core_radial_floor_weight_mean": z,
            "core_radial_floor_r_weighted": z,
        }
    r = hyp_proj_2d.norm(dim=-1)
    rho = target_rho.reshape(-1).to(device=device, dtype=dtype)
    if rho.numel() != r.numel():
        n = min(int(rho.numel()), int(r.numel()))
        rho = rho[:n]
        r = r[:n]
    rho_n = (rho / float(rho_scale)).clamp(0.0, 1.0)
    if target_tau is None:
        tau_n = torch.zeros_like(rho_n)
    else:
        tau = target_tau.reshape(-1).to(device=device, dtype=dtype)
        if tau.numel() != rho_n.numel():
            tau = tau[: rho_n.numel()]
        tau_n = tau.clamp(0.0, 1.0)
    w = rho_n * (1.0 - tau_n)
    w_sum = w.sum().clamp_min(1e-8)
    hinge = torch.relu(float(min_r) - r).pow(2)
    loss = (w * hinge).sum() / w_sum
    r_w = (w * r).sum() / w_sum
    return {
        "core_radial_floor": loss,
        "core_radial_floor_weight_mean": w.mean().detach(),
        "core_radial_floor_r_weighted": r_w.detach(),
    }


def disc_angular_coverage_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_r: float = 0.12,
    n_bins: int = 12,
    min_bin_frac: float = 0.40,
    temperature: float = 0.20,
    scale: float = 4.0,
    r_softness: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Penalize under-filled angular sectors on the Poincaré disc (empty-wedge pressure).

    Soft-assigns mid/rim residues (r ≳ ``min_r``) to equally spaced θ bins and
    floors each bin's mass at ``min_bin_frac / n_bins`` of the soft-weighted mass.
    Unlike rim repulsion (which only separates existing rays), this term creates
    gradient into *empty* sectors.
    """
    xy = hyp_proj_2d
    z = torch.zeros((), device=xy.device, dtype=xy.dtype)
    n = xy.shape[0]
    n_bins = int(n_bins)
    if n < 4 or n_bins < 4:
        return {
            "disc_angular_coverage": z,
            "disc_angular_coverage_empty_bins": z,
            "disc_angular_coverage_min_mass": z,
        }

    r = xy.norm(dim=-1).clamp(min=1e-8)
    dirs = xy / r.unsqueeze(-1)
    # Soft rim weight — avoid hard cutoff so near-gap mid-disc points can migrate.
    w = torch.sigmoid((r - float(min_r)) / max(float(r_softness), 1e-4))
    if float(w.detach().sum()) < 1e-4:
        return {
            "disc_angular_coverage": z,
            "disc_angular_coverage_empty_bins": z,
            "disc_angular_coverage_min_mass": z,
        }

    angles = torch.linspace(
        0.0, 2.0 * math.pi, n_bins + 1, device=xy.device, dtype=xy.dtype
    )[:-1]
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [B, 2]
    # Soft assignments via circular cosine similarity / temperature.
    logits = (dirs @ centers.T) / max(float(temperature), 1e-4)
    soft = torch.softmax(logits, dim=-1)  # [N, B]
    mass = (soft * w.unsqueeze(-1)).sum(dim=0)
    mass = mass / mass.sum().clamp(min=1e-8)

    floor = float(min_bin_frac) / float(n_bins)
    deficit = torch.relu(floor - mass)
    loss = deficit.pow(2).mean() * float(scale)

    hard = soft.argmax(dim=-1)
    # Detached diagnostics on soft-weighted mid/rim set.
    rim_mask = w > 0.5
    if bool(rim_mask.any()):
        hard_rim = hard[rim_mask]
        counts = torch.bincount(hard_rim, minlength=n_bins).float()
        empty = (counts < 0.5).float().sum()
        min_mass = (counts / counts.sum().clamp(min=1.0)).min()
    else:
        empty = z
        min_mass = z

    return {
        "disc_angular_coverage": loss,
        "disc_angular_coverage_empty_bins": empty.detach(),
        "disc_angular_coverage_min_mass": min_mass.detach(),
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
