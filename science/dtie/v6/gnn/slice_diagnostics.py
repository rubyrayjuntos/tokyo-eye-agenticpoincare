"""Rank / cone / disc spread diagnostics for radial×angular lift and projection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def effective_rank_np(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or s[0] < 1e-15:
        return 0.0
    return float(np.sum(s * s) / (s[0] * s[0]))


def _angular_span_deg(xy: np.ndarray, *, from_origin: bool) -> float | None:
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 3:
        return None
    if from_origin:
        ang = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    else:
        xc = xy - xy.mean(axis=0, keepdims=True)
        ang = np.degrees(np.arctan2(xc[:, 1], xc[:, 0]))
    return float(np.percentile(ang, 95) - np.percentile(ang, 5))


def disc_2d_stats(xy: np.ndarray) -> dict[str, Any]:
    """Full disc occupancy stats (thickness, origin span, SVD)."""
    pts = np.asarray(xy, dtype=np.float64)
    base = centered_svd_stats(pts)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        base.update(
            {
                "origin_angular_span_p5_p95_deg": None,
                "centroid_angular_span_p5_p95_deg": None,
                "disc_r_p50": None,
            }
        )
        return base
    r = np.linalg.norm(pts, axis=1)
    base.update(
        {
            "origin_angular_span_p5_p95_deg": _angular_span_deg(pts, from_origin=True),
            "centroid_angular_span_p5_p95_deg": _angular_span_deg(pts, from_origin=False),
            "disc_r_min": float(r.min()),
            "disc_r_p50": float(np.median(r)),
            "disc_r_max": float(r.max()),
        }
    )
    return base


def centered_svd_stats(pts: np.ndarray) -> dict[str, Any]:
    """Centered SVD occupancy on [N, D]."""
    x = np.asarray(pts, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        return {"n": int(x.shape[0]), "sigma": [], "sigma_ratio": [], "eff_rank": 0.0}
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    ratio = [float(s[i] / (s[0] + 1e-12)) for i in range(min(3, s.size))]
    energy = float((s * s).sum())
    pc1_frac = float((s[0] * s[0]) / (energy + 1e-12)) if energy > 0 else 0.0
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    pc1 = vt[0]
    resid = xc - (xc @ pc1)[:, None] * pc1[None, :]
    thickness = np.linalg.norm(resid, axis=1)
    return {
        "n": int(x.shape[0]),
        "sigma": [float(v) for v in s[: min(3, s.size)]],
        "sigma_ratio": ratio,
        "eff_rank": effective_rank_np(s),
        "line_thickness_rms": float(np.sqrt((thickness**2).mean())),
        "pc1_energy_frac": pc1_frac,
    }


def cone_slice_stats(
    radial_depth: torch.Tensor,
    angular_direction: torch.Tensor,
    *,
    hidden: int,
) -> dict[str, float]:
    """How collinear is radial×angular with angular directions (planar cone slice)."""
    r = radial_depth.detach().float().cpu().numpy().reshape(-1, 1)
    a = angular_direction.detach().float().cpu().numpy()
    if a.shape[0] < 3:
        return {"cone_pc1_frac": 0.0, "angular_pc1_frac": 0.0}
    tangent = r * a
    ang_stats = centered_svd_stats(a)
    tan_stats = centered_svd_stats(tangent)
    return {
        "cone_pc1_frac": float(tan_stats.get("pc1_energy_frac", 0.0)),
        "angular_pc1_frac": float(ang_stats.get("pc1_energy_frac", 0.0)),
        "tangent_eff_rank": float(tan_stats.get("eff_rank", 0.0)),
        "angular_eff_rank": float(ang_stats.get("eff_rank", 0.0)),
    }


def recompute_tangent(
    radial_depth: torch.Tensor,
    angular_direction: torch.Tensor,
    *,
    mode: str,
    fusion: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Counterfactual tangent vectors for offline ablation."""
    cone = radial_depth * angular_direction
    if mode == "multiply":
        return cone
    if mode == "mlp_fusion" and fusion is not None:
        fused_in = torch.cat([radial_depth, angular_direction], dim=-1)
        return cone + fusion(fused_in)
    if mode == "angular_only" or mode == "angular_lift":
        base = angular_direction
        if mode == "angular_lift" and fusion is not None:
            fused_in = torch.cat([radial_depth, angular_direction], dim=-1)
            return base + fusion(fused_in)
        return base
    raise ValueError(f"unknown recompute mode: {mode}")
