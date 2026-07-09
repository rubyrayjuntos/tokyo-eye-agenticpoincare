"""Tests for ExpertTimeoutController and train-only gate masking."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from science.training.expert_timeout import ExpertTimeoutController


class _FakeGate(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.expert_timeout_banned: list[int] = []
        self.logits = nn.Parameter(torch.zeros(1, num_experts))

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        adjusted = self.logits.expand(8, -1).clone()
        if self.training and self.expert_timeout_banned:
            for ban_idx in self.expert_timeout_banned:
                adjusted[:, int(ban_idx)] = -1e9
        return torch.softmax(adjusted, dim=-1)


class _FakeModel(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.gate = _FakeGate(num_experts)


def test_observe_bans_dominant_expert() -> None:
    ctl = ExpertTimeoutController(num_experts=4, max_share=0.50, ban_epochs=1)
    newly = ctl.observe_loads([0.20, 0.15, 0.55, 0.10], epoch=3)
    assert newly == [2]
    assert ctl.active_bans() == [2]
    assert ctl.ban_remaining[2] == 1


def test_default_is_fifty_percent_one_epoch() -> None:
    ctl = ExpertTimeoutController(num_experts=4)
    assert ctl.max_share == pytest.approx(0.50)
    assert ctl.ban_epochs == 1
    # Mild lead under 50% — no ban
    assert ctl.observe_loads([0.20, 0.20, 0.45, 0.15], epoch=1) == []
    newly = ctl.observe_loads([0.15, 0.15, 0.55, 0.15], epoch=2)
    assert newly == [2]


def test_one_ban_per_epoch_and_keep_two_experts() -> None:
    ctl = ExpertTimeoutController(num_experts=4, max_share=0.50, ban_epochs=1)
    # Two experts over threshold — only the worst is banned.
    newly = ctl.observe_loads([0.55, 0.51, 0.0, 0.0], epoch=1)
    assert newly == [0]
    assert ctl.active_bans() == [0]


def test_ban_lasts_one_full_epoch_with_end_tick() -> None:
    ctl = ExpertTimeoutController(num_experts=4, max_share=0.50, ban_epochs=1)
    ctl.observe_loads([0.10, 0.10, 0.70, 0.10], epoch=5)
    assert ctl.active_bans() == [2]
    # One train epoch under ban, then tick → cleared + cooldown
    still = ctl.tick_end_of_epoch()
    assert still == []
    assert ctl.cooldown_remaining.get(2, 0) == 1
    # Same expert over threshold during cooldown → skip
    newly = ctl.observe_loads([0.10, 0.10, 0.70, 0.10], epoch=6)
    assert newly == []
    # Cooldown tick
    ctl.tick_end_of_epoch()
    assert ctl.cooldown_remaining.get(2, 0) == 0
    newly = ctl.observe_loads([0.10, 0.10, 0.70, 0.10], epoch=7)
    assert newly == [2]


def test_skip_ban_when_would_leave_fewer_than_two() -> None:
    ctl = ExpertTimeoutController(num_experts=4, max_share=0.50, ban_epochs=1)
    ctl.ban_remaining = {0: 1, 1: 1}  # already two banned
    newly = ctl.observe_loads([0.05, 0.05, 0.80, 0.10], epoch=2)
    assert newly == []  # would leave only e3


def test_apply_and_clear_gate_mask() -> None:
    model = _FakeModel(4)
    ctl = ExpertTimeoutController(num_experts=4, max_share=0.50, ban_epochs=1)
    ctl.observe_loads([0.55, 0.15, 0.15, 0.15], epoch=1)
    ctl.apply_to_gate(model)
    assert model.gate.expert_timeout_banned == [0]
    model.train()
    scores = model.gate(torch.zeros(1))
    assert float(scores[:, 0].detach().max()) < 1e-6
    assert abs(float(scores.sum(dim=-1).detach().mean()) - 1.0) < 1e-5
    ctl.clear_gate(model)
    assert model.gate.expert_timeout_banned == []
    model.eval()
    scores_eval = model.gate(torch.zeros(1))
    assert float(scores_eval[:, 0].detach().mean()) > 0.2


def test_eligible_experts_restricts_bans() -> None:
    """Only listed experts may be banned (e.g. generalist-only timeout)."""
    ctl = ExpertTimeoutController(
        num_experts=4,
        max_share=0.42,
        ban_epochs=1,
        eligible_experts=[2],
    )
    # e0 over threshold but not eligible — no ban
    assert ctl.observe_loads([0.50, 0.20, 0.20, 0.10], epoch=1) == []
    # e2 over 0.42 — banned
    newly = ctl.observe_loads([0.20, 0.20, 0.45, 0.15], epoch=2)
    assert newly == [2]
    assert ctl.active_bans() == [2]


def test_capacity_aware_gate_timeout_mask_train_only() -> None:
    """Real MoE gate: ban masks train softmax, not eval."""
    from science.dtie.v6.gnn.model import TopologicalMoEGateV6

    gate = TopologicalMoEGateV6(hidden_dim=8, num_experts=4, expert_dropout_p=0.0)
    gate.expert_timeout_banned = [1]
    n = 16
    x = torch.randn(n, 8)
    clustering = torch.rand(n)
    cone_depth = torch.rand(n, 1)
    degree = torch.ones(n)
    rho = torch.rand(n)
    ss_onehot = torch.zeros(n, 3)
    ss_onehot[:, 0] = 1.0
    gate.train()
    scores, _, _ = gate(x, clustering, cone_depth, degree, rho, ss_onehot)
    assert float(scores[:, 1].detach().max()) < 1e-5
    gate.eval()
    scores_e, _, _ = gate(x, clustering, cone_depth, degree, rho, ss_onehot)
    # Eval ignores timeout bans — expert 1 can receive mass again.
    assert float(scores_e[:, 1].detach().sum()) > 0.0
