"""Poincaré ball radius conventions — geoopt vs model clamp.

geoopt stereographic ball (curvature c > 0): points satisfy ||x|| < r_ball with r_ball = 1/sqrt(c).
The legacy model clamp used a unit-ball ceiling (0.99) on Euclidean norm, ignoring r_ball.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Interior margin inside the model clamp ceiling (matches model code).
CLAMP_INTERIOR_FRACTION = 0.99


def ball_radius(c: float) -> float:
    if c <= 0:
        raise ValueError("curvature c must be positive")
    return 1.0 / math.sqrt(c)


def model_clamp_ceiling(c: float, interior: float = CLAMP_INTERIOR_FRACTION) -> float:
    """Maximum Euclidean norm after geoopt-correct model clamp."""
    return interior * ball_radius(c)


def legacy_unit_clamp_ceiling(interior: float = CLAMP_INTERIOR_FRACTION) -> float:
    """Legacy clamp cap (c=1 assumption) used before r_ball unification."""
    return interior


@dataclass
class ConventionProbe:
    """Empirical probe — run before trusting invariants 1, 2, 4."""

    curvature: float
    r_ball: float
    model_clamp_ceiling: float
    disc_max_raw_norm: float
    disc_max_over_r_ball: float
    disc_max_over_legacy_ceiling: float
    legacy_unit_clamp_detected: bool
    effective_norm_boundary: float
    note: str


def probe_disc_convention(disc: np.ndarray, c: float) -> ConventionProbe:
    """One-line convention check (paste-friendly)."""
    norms = np.linalg.norm(np.asarray(disc, dtype=np.float64), axis=1)
    r = ball_radius(c)
    ceiling = model_clamp_ceiling(c)
    legacy = legacy_unit_clamp_ceiling()
    max_raw = float(norms.max()) if len(norms) else 0.0
    max_over_r = max_raw / r if r > 0 else 0.0
    max_over_legacy = max_raw / legacy if legacy > 0 else 0.0

    # Persisted data from pre-fix model: max raw ~0.99 but only ~0.77 of r_ball.
    legacy_detected = max_over_r < 0.85 and max_raw > 0.95

    if legacy_detected:
        effective = legacy
        note = (
            "LEGACY: disc capped at unit-ball 0.99 while geoopt r_ball=1/sqrt(c). "
            "Use norm/0.99 for saturation; re-run pipeline after clamp fix for unified metrics."
        )
    else:
        effective = ceiling
        note = "Unified: disc clamp ceiling ≈ 0.99 * r_ball (geoopt-consistent)."

    return ConventionProbe(
        curvature=float(c),
        r_ball=r,
        model_clamp_ceiling=ceiling,
        disc_max_raw_norm=max_raw,
        disc_max_over_r_ball=max_over_r,
        disc_max_over_legacy_ceiling=max_over_legacy,
        legacy_unit_clamp_detected=legacy_detected,
        effective_norm_boundary=effective,
        note=note,
    )


def rescale_tangent_before_expmap(tangent_vector, c):  # type: ignore[no-untyped-def]
    """Scale tangent magnitude with tanh(‖v‖/r_ball) before expmap0 to avoid boundary saturation."""
    import torch

    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c, device=tangent_vector.device, dtype=tangent_vector.dtype)
    tv_norm = tangent_vector.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ball_r = torch.rsqrt(torch.clamp(c, min=1e-8))
    return tangent_vector * (torch.tanh(tv_norm / ball_r) / tv_norm)


def estimate_clamp_applied_fraction(
    norms: np.ndarray,
    c: float,
    *,
    legacy_unit_clamp: bool,
    tol: float = 1e-4,
) -> float:
    """Fraction of points sitting on the model clamp ceiling (non-isometric rescale)."""
    if len(norms) == 0:
        return 0.0
    ceiling = legacy_unit_clamp_ceiling() if legacy_unit_clamp else model_clamp_ceiling(c)
    return float(np.mean(norms >= ceiling - tol))


def mobius_add_2d(
    a: tuple[float, float],
    x: np.ndarray,
    c: float,
) -> np.ndarray:
    """Möbius addition in the Poincaré disc (matches visualizer math.ts mobiusAdd2D)."""
    ax, ay = a
    x0, x1 = x[:, 0], x[:, 1]
    norm_a_sq = ax * ax + ay * ay
    norm_x_sq = x0 * x0 + x1 * x1
    dot_ax = ax * x0 + ay * x1
    den = 1.0 + 2.0 * c * dot_ax + c * c * norm_a_sq * norm_x_sq
    num_x = (1.0 + 2.0 * c * dot_ax + c * norm_x_sq) * ax + (1.0 - c * norm_a_sq) * x0
    num_y = (1.0 + 2.0 * c * dot_ax + c * norm_x_sq) * ay + (1.0 - c * norm_a_sq) * x1
    return np.column_stack([num_x / den, num_y / den])


def mobius_recenter_2d(
    disc: np.ndarray,
    center: tuple[float, float],
    c: float,
) -> np.ndarray:
    """Transport so `center` maps to origin via Möbius addition with -center."""
    neg = (-center[0], -center[1])
    return mobius_add_2d(neg, disc, c)
