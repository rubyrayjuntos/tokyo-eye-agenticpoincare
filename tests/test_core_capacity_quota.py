"""Unit tests for core capacity quotas (pre-reg CORE_QUOTA_*)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from science.training.core_capacity_quota import (
    apply_core_capacity_quota,
    apply_core_capacity_quota_ste,
    freeze_dehydron_dominant_mask,
)


def test_freeze_dehydron_dominant_mask_threshold() -> None:
    # Expert 0: all dehydron=1; expert 1: all core; 2/3 empty→not dominant.
    n, e = 20, 4
    soft = torch.zeros(n, e)
    soft[:10, 0] = 1.0
    soft[10:, 1] = 1.0
    dh = torch.cat([torch.ones(10), torch.zeros(10)])
    mask = freeze_dehydron_dominant_mask(soft, dh, rate_threshold=0.50, min_residues=5)
    assert mask.tolist() == [True, False, False, False]


def test_freeze_marks_top2_when_all_dominant() -> None:
    n, e = 40, 4
    soft = torch.zeros(n, e)
    for i in range(e):
        soft[i * 10 : (i + 1) * 10, i] = 1.0
    dh = torch.ones(n)  # every expert dehydron-rich
    mask = freeze_dehydron_dominant_mask(soft, dh, rate_threshold=0.50, min_residues=5)
    assert int(mask.sum().item()) == 2
    assert bool(mask.all()) is False


def test_quota_respects_cap_and_never_force_cross_axis() -> None:
    # 10 core residues; τ=0.40 → cap=4. Expert 0/1 eligible; 2/3 dominant.
    n, e = 10, 4
    soft = torch.zeros(n, e)
    # All prefer expert 0 first, then 2 (dominant), then 1.
    soft[:, 0] = 0.70
    soft[:, 2] = 0.20
    soft[:, 1] = 0.09
    soft[:, 3] = 0.01
    soft = soft / soft.sum(dim=-1, keepdim=True)
    dh = torch.zeros(n)
    dominant = torch.tensor([False, False, True, True])
    _hard, stats = apply_core_capacity_quota(soft, dh, dominant, tau_cap=0.40)
    assert stats["assigned_under_quota"] == [4, 4, 0, 0]
    assert stats["quota_forced_cross_axis"] == 0
    assert stats["quota_unplaced_core"] == 2


def test_quota_tiebreak_unplaced_when_only_dominant_under_cap() -> None:
    # Only under-cap slots are dehydron-dominant → unplaced, never force.
    n, e = 8, 4
    soft = torch.zeros(n, e)
    soft[:, 0] = 0.9
    soft[:, 1] = 0.05
    soft[:, 2] = 0.03
    soft[:, 3] = 0.02
    dh = torch.zeros(n)
    dominant = torch.tensor([True, True, True, True])  # all protected
    hard, stats = apply_core_capacity_quota(soft, dh, dominant, tau_cap=0.40)
    assert stats["quota_unplaced_core"] == n
    assert stats["quota_forced_cross_axis"] == 0
    # Soft argmax retained (expert 0)
    assert (hard.argmax(dim=-1) == 0).all()


def test_quota_ste_leaves_dehydron_soft() -> None:
    logits = torch.randn(12, 4, requires_grad=True)
    soft = F.softmax(logits, dim=-1)
    dh = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
    dominant = torch.tensor([False, False, True, True])
    out, stats = apply_core_capacity_quota_ste(soft, dh, dominant, tau_cap=0.40)
    assert torch.allclose(out[dh > 0], soft[dh > 0])
    assert stats["quota_forced_cross_axis"] == 0
    out.sum().backward()
    assert logits.grad is not None


def test_frozen_mask_not_recomputed_semantics() -> None:
    """Snapshot is a stored tensor — recomputing later must not mutate it."""
    soft = torch.eye(4).repeat(4, 1)  # 16×4
    dh = torch.cat([torch.ones(8), torch.zeros(8)])
    frozen = freeze_dehydron_dominant_mask(soft, dh)
    frozen_copy = frozen.clone()
    soft2 = soft.clone()
    soft2[:, :] = 0
    soft2[:, 0] = 1.0
    _live = freeze_dehydron_dominant_mask(soft2, dh)
    assert torch.equal(frozen, frozen_copy)
