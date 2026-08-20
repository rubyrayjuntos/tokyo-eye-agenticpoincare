"""Core (dehydron=0) capacity quotas with frozen dehydron-dominant guard.

Pre-reg: CORE_QUOTA_* in docs/audit/GNNV7_SUCCESS_CRITERIA.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def freeze_dehydron_dominant_mask(
    soft_weights: torch.Tensor,
    dehydron: torch.Tensor,
    *,
    rate_threshold: float = 0.50,
    min_residues: int = 5,
    n_experts: int | None = None,
) -> torch.Tensor:
    """Ge0 freeze: bool mask [E] — True = dehydron-dominant (protected from core overflow).

    Soft-argmax dehydron rate per expert; dominant if rate > threshold and
    count >= min_residues. If zero non-dominant experts, mark the two highest-
    dehydron-rate experts as dominant.
    """
    w = soft_weights.detach().float()
    dh = dehydron.detach().float().reshape(-1)[: w.shape[0]]
    e = int(n_experts if n_experts is not None else w.shape[-1])
    hard = w.argmax(dim=-1)
    rates = torch.zeros(e, device=w.device, dtype=torch.float32)
    counts = torch.zeros(e, device=w.device, dtype=torch.float32)
    for i in range(e):
        m = hard == i
        c = int(m.sum().item())
        counts[i] = float(c)
        rates[i] = float(dh[m].mean().item()) if c > 0 else 0.0
    dominant = (rates > float(rate_threshold)) & (counts >= float(min_residues))
    if int((~dominant).sum().item()) == 0:
        # Always protect an axis pole: top-2 by dehydron rate.
        top2 = torch.topk(rates, k=min(2, e)).indices
        dominant = torch.zeros(e, device=w.device, dtype=torch.bool)
        dominant[top2] = True
    return dominant


def apply_core_capacity_quota(
    soft_weights: torch.Tensor,
    dehydron: torch.Tensor,
    dominant_mask: torch.Tensor,
    *,
    tau_cap: float = 0.40,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Greedy core quota assignment; unplaced keep soft-argmax (never force cross-axis).

    Returns hard one-hot assignment [N, E] and stats. Differentiable STE should
    be applied by the caller if needed.
    """
    w = soft_weights.float()
    n, e = w.shape
    dh = dehydron.float().reshape(-1)[:n]
    dominant = dominant_mask.to(device=w.device, dtype=torch.bool).reshape(-1)[:e]
    hard_out = F.one_hot(w.argmax(dim=-1), e).to(dtype=w.dtype)

    core = dh <= 0.0
    n_core = int(core.sum().item())
    if n_core == 0 or tau_cap <= 0:
        return hard_out, {
            "n_core": n_core,
            "quota_unplaced_core": 0,
            "quota_forced_cross_axis": 0,
            "cap_per_expert": 0,
            "tau_cap": float(tau_cap),
        }

    cap = max(int(float(tau_cap) * n_core), 1)
    remaining = torch.full((e,), cap, device=w.device, dtype=torch.long)
    # Dehydron-dominant experts: capacity for *core* is zero (cannot receive core).
    remaining = torch.where(dominant, torch.zeros_like(remaining), remaining)

    core_idx = core.nonzero(as_tuple=False).view(-1)
    # Place highest-confidence core residues first.
    core_conf = w[core_idx].max(dim=-1).values
    order = torch.argsort(core_conf, descending=True)
    core_idx = core_idx[order]

    unplaced = 0
    assigned = torch.zeros(n, dtype=torch.bool, device=w.device)
    assigned_counts = torch.zeros(e, device=w.device, dtype=torch.long)
    for idx in core_idx.tolist():
        scores_i = w[idx]
        # Rank experts by score; skip dominant and full.
        order_e = torch.argsort(scores_i, descending=True)
        placed = False
        for ei in order_e.tolist():
            if dominant[ei]:
                continue
            if int(remaining[ei].item()) <= 0:
                continue
            hard_out[idx] = 0
            hard_out[idx, ei] = 1
            remaining[ei] -= 1
            assigned_counts[ei] += 1
            placed = True
            assigned[idx] = True
            break
        if not placed:
            # Tie-break: unplaced — keep soft argmax (already in hard_out).
            unplaced += 1

    return hard_out, {
        "n_core": n_core,
        "quota_unplaced_core": int(unplaced),
        "quota_unplaced_frac": float(unplaced) / float(max(n_core, 1)),
        "quota_forced_cross_axis": 0,
        "cap_per_expert": int(cap),
        "tau_cap": float(tau_cap),
        "n_dominant_experts": int(dominant.sum().item()),
        "remaining_capacity": [int(x) for x in remaining.tolist()],
        "assigned_under_quota": [int(x) for x in assigned_counts.tolist()],
    }


def apply_core_capacity_quota_ste(
    soft_weights: torch.Tensor,
    dehydron: torch.Tensor,
    dominant_mask: torch.Tensor,
    *,
    tau_cap: float = 0.40,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """STE: core residues use quota hard assignment; dehydron residues keep soft.

    Forward: quota hard one-hot on core (unplaced = soft argmax one-hot).
    Backward: soft weights everywhere.
    """
    soft = soft_weights.float()
    hard, stats = apply_core_capacity_quota(
        soft.detach(), dehydron, dominant_mask, tau_cap=tau_cap
    )
    dh = dehydron.float().reshape(-1)[: soft.shape[0]]
    core = (dh <= 0.0).unsqueeze(-1)
    # STE on core; soft on dehydron=1.
    ste = hard + (soft - soft.detach())
    out = torch.where(core, ste, soft)
    return out, stats
