"""Per-residue routing entropy sparsity helpers (Fix-1 MoE sharpening).

See ``docs/specs/routing-entropy-sparsity/design.md``.
"""

from __future__ import annotations

from typing import Any

import torch


def mean_residue_routing_entropy(
    scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean Shannon entropy over residues: ``(1/N) Σ_i H(p_i)`` in nats.

    Parameters
    ----------
    scores:
        Soft expert weights ``[N, E]`` (rows should sum to ~1).
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be [N, E], got shape {tuple(scores.shape)}")
    p = scores.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    h = -(p * p.log()).sum(dim=-1)
    return h.mean()


def sparse_coeff_at_epoch(
    epoch_1indexed: int,
    *,
    peak: float,
    warmup: int,
) -> float:
    """Linear warmup of λ_sparse; full peak at ``warmup`` and thereafter."""
    if warmup <= 0:
        return float(peak)
    t = max(1, int(epoch_1indexed))
    return float(peak) * min(1.0, t / float(warmup))


def advance_sparse_lam(
    *,
    peak: float,
    warmup: int,
    last_lam: float,
    slope_scale: float = 1.0,
    hold_remaining: int = 0,
    hold_lam: float = 0.0,
) -> tuple[float, int]:
    """Compute λ for the next epoch, consuming one hold epoch if active.

    Returns ``(lam, hold_remaining_after)``. When not holding, advances by
    ``(peak / warmup) * slope_scale`` from ``last_lam`` (capped at peak).
    With ``slope_scale=1`` and ``last_lam=0``, matches ``sparse_coeff_at_epoch``.
    """
    if float(peak) <= 0:
        return 0.0, 0
    if int(hold_remaining) > 0:
        return float(hold_lam), int(hold_remaining) - 1
    if int(warmup) <= 0:
        return float(peak), 0
    step = float(peak) / float(warmup) * float(slope_scale)
    return min(float(peak), float(last_lam) + step), 0


def sparsity_governor_update(
    *,
    max_soft_share: float,
    newly_banned: list[int],
    prev_lam: float,
    slope_scale: float,
    half_slope_applied: bool,
    timeout_share: float = 0.45,
    warn_share: float = 0.40,
    hold_epochs: int = 2,
) -> dict[str, Any]:
    """End-of-epoch soft governor for sparsity λ ramp.

    - New timeout ban **and** ``max_soft_share >= timeout_share`` → hold λ at
      ``prev_lam`` for ``hold_epochs``.
    - ``warn_share <= max_soft_share < timeout_share`` (once) → half remaining
      ramp slope.
    """
    events: list[str] = []
    new_scale = float(slope_scale)
    new_half = bool(half_slope_applied)
    hold_remaining = 0
    hold_lam = float(prev_lam)
    mx = float(max_soft_share)
    if newly_banned and mx >= float(timeout_share):
        hold_remaining = int(hold_epochs)
        hold_lam = float(prev_lam)
        events.append("timeout_hold")
    if (
        float(warn_share) <= mx < float(timeout_share)
        and not half_slope_applied
    ):
        new_scale = 0.5 * float(slope_scale)
        new_half = True
        events.append("half_ramp")
    return {
        "slope_scale": new_scale,
        "half_slope_applied": new_half,
        "hold_remaining": hold_remaining,
        "hold_lam": hold_lam,
        "events": events,
    }


def sparse_vs_cap_ratio(
    *,
    lam_sparse: float,
    l_sparse: float,
    balance_coeff: float,
    capacity_loss: float,
    eps: float = 1e-12,
) -> float:
    """Weighted sparsity vs capacity contribution ratio."""
    return (float(lam_sparse) * float(l_sparse)) / (
        float(balance_coeff) * float(capacity_loss) + float(eps)
    )


def detect_sparse_capacity_collision(
    epoch_rows: list[dict[str, Any]],
    *,
    cap_floor: float = 1e-4,
    max_share_warn: float = 0.40,
) -> int | None:
    """First 1-indexed epoch where sparsity bites and capacity/monopoly reacts.

    Collision = mean residue-H decreased vs previous epoch **and** either
    ``capacity_loss >= max(cap_floor, 2 * ep1 capacity)`` or
    ``usage_max_soft_share >= max_share_warn``.
    """
    if not epoch_rows:
        return None
    cap0 = float(epoch_rows[0].get("capacity_loss") or 0.0)
    for i in range(1, len(epoch_rows)):
        prev, cur = epoch_rows[i - 1], epoch_rows[i]
        h_prev = float(prev["routing_entropy_mean_residue"])
        h_cur = float(cur["routing_entropy_mean_residue"])
        cap = float(cur.get("capacity_loss") or 0.0)
        mx = float(cur.get("usage_max_soft_share") or 0.0)
        if h_cur < h_prev and (
            cap >= max(cap_floor, 2.0 * cap0) or mx >= max_share_warn
        ):
            return i + 1
    return None
