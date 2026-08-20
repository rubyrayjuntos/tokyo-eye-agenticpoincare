"""Within-rim dehydron angular spread metrics (v6.6 evaluation).

Tier-2 KS (dehydron vs non-dehydron θ) is underpowered when the outer rim is
almost purely dehydron. These metrics compare angular diversity among high-r
dehydrons only — the SSOT control target for feeler spike diagnosis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def circular_delta_deg(a: float, b: float) -> float:
    """Smallest absolute angular separation on [-180, 180] degrees."""
    d = abs(float(a) - float(b)) % 360.0
    return float(min(d, 360.0 - d))


def max_pairwise_circular_delta_deg(theta_deg: np.ndarray) -> float:
    """Maximum circular pairwise separation (degrees)."""
    t = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    n = int(t.size)
    if n < 2:
        return 0.0
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            max_d = max(max_d, circular_delta_deg(t[i], t[j]))
    return float(max_d)


def _effective_rank_2d(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return 0.0
    X = xy - xy.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0 or s[0] < 1e-15:
        return 0.0
    return float(np.sum(s * s) / (s[0] * s[0]))


def rim_dehydron_mask(
    disc_r: np.ndarray,
    dehydron: np.ndarray,
    *,
    rim_quantile: float = 0.5,
) -> np.ndarray:
    """High-r dehydrons: dehydron label and disc_r at/above rim_quantile."""
    r = np.asarray(disc_r, dtype=np.float64).reshape(-1)
    d = np.asarray(dehydron, dtype=bool).reshape(-1)
    if r.shape[0] != d.shape[0]:
        raise ValueError("disc_r and dehydron length mismatch")
    if not bool(d.any()):
        return np.zeros_like(d, dtype=bool)
    thresh = float(np.quantile(r[d], rim_quantile)) if d.sum() >= 2 else float(r[d].max())
    return d & (r >= thresh)


@dataclass
class RimDehydronAngularStats:
    structure_id: str
    n_dehydron_total: int
    n_rim_dehydron: int
    rim_disc_r_min: float
    rim_disc_r_max: float
    rim_disc_r_mean: float
    within_rim_angular_std_deg: float
    within_rim_angular_p5_p95_span_deg: float
    max_pairwise_delta_theta_deg: float
    rim_xy_effective_rank: float
    rim_disc_r_span: float
    corr_rho_disc_r: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_rim_dehydron_angular_stats(
    *,
    structure_id: str,
    disc_r: np.ndarray,
    disc_theta_deg: np.ndarray,
    disc_xy: np.ndarray,
    rho: np.ndarray,
    dehydron: np.ndarray,
    rim_quantile: float = 0.5,
) -> RimDehydronAngularStats:
    """Angular spread among high-r dehydrons (Q3–Q4 analog when quantile=0.5)."""
    r = np.asarray(disc_r, dtype=np.float64).reshape(-1)
    theta = np.asarray(disc_theta_deg, dtype=np.float64).reshape(-1)
    xy = np.asarray(disc_xy, dtype=np.float64)
    rho_a = np.asarray(rho, dtype=np.float64).reshape(-1)
    dehyd = np.asarray(dehydron, dtype=bool).reshape(-1)

    rim = rim_dehydron_mask(r, dehyd, rim_quantile=rim_quantile)
    n_total = int(dehyd.sum())
    n_rim = int(rim.sum())

    if n_rim < 2:
        corr = float(np.corrcoef(rho_a, r)[0, 1]) if len(rho_a) > 2 else float("nan")
        return RimDehydronAngularStats(
            structure_id=structure_id,
            n_dehydron_total=n_total,
            n_rim_dehydron=n_rim,
            rim_disc_r_min=float(r[rim].min()) if n_rim else float("nan"),
            rim_disc_r_max=float(r[rim].max()) if n_rim else float("nan"),
            rim_disc_r_mean=float(r[rim].mean()) if n_rim else float("nan"),
            within_rim_angular_std_deg=0.0,
            within_rim_angular_p5_p95_span_deg=0.0,
            max_pairwise_delta_theta_deg=0.0,
            rim_xy_effective_rank=0.0,
            rim_disc_r_span=0.0,
            corr_rho_disc_r=corr,
            note="insufficient rim dehydrons (need >= 2)",
        )

    t_rim = theta[rim]
    xy_rim = xy[rim]
    r_rim = r[rim]
    corr = float(np.corrcoef(rho_a, r)[0, 1]) if len(rho_a) > 2 else float("nan")

    return RimDehydronAngularStats(
        structure_id=structure_id,
        n_dehydron_total=n_total,
        n_rim_dehydron=n_rim,
        rim_disc_r_min=float(r_rim.min()),
        rim_disc_r_max=float(r_rim.max()),
        rim_disc_r_mean=float(r_rim.mean()),
        within_rim_angular_std_deg=float(np.std(t_rim)),
        within_rim_angular_p5_p95_span_deg=float(
            np.percentile(t_rim, 95) - np.percentile(t_rim, 5)
        ),
        max_pairwise_delta_theta_deg=max_pairwise_circular_delta_deg(t_rim),
        rim_xy_effective_rank=_effective_rank_2d(xy_rim),
        rim_disc_r_span=float(r_rim.max() - r_rim.min()),
        corr_rho_disc_r=corr,
        note="high-r dehydrons (disc_r >= median among dehydrons)",
    )


__all__ = [
    "RimDehydronAngularStats",
    "circular_delta_deg",
    "compute_rim_dehydron_angular_stats",
    "max_pairwise_circular_delta_deg",
    "rim_dehydron_mask",
]
