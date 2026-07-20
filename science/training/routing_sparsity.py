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
