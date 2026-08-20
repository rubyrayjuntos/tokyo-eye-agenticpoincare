"""Sprint 9: MoE min-load quota + exponential Gumbel schedule."""

from __future__ import annotations

import math

import pytest
import torch

from science.tokyo_eye.v8.engine import GumbelTemperatureSchedule
from science.tokyo_eye.v8.moe import (
    TopologyAwareHardMoE,
    cv_load_balance_loss,
    min_load_quota_loss,
)


def test_quota_hinge_fires_below_floor() -> None:
    monopoly = torch.zeros(20, 4)
    monopoly[:, 0] = 1.0
    q = float(min_load_quota_loss(monopoly, floor=0.05))
    # Three experts at 0 → 3 * (0.05)**2
    assert q == pytest.approx(3 * (0.05**2), rel=1e-5)


def test_quota_zero_when_all_above_floor() -> None:
    uniform = torch.full((16, 4), 0.25)
    assert float(min_load_quota_loss(uniform, floor=0.05)) == pytest.approx(0.0)


def test_cv_unscaled_and_aux_has_quota() -> None:
    torch.manual_seed(0)
    moe = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=1.0, cv_coeff=99.0)
    moe.train()
    z = torch.randn(8, 6) * 0.04
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    _, aux = moe(z, edge_index)
    # Internal cv_coeff must NOT scale aux["cv_loss"] (harness SSOT)
    raw = float(cv_load_balance_loss(aux["routing"]))
    assert float(aux["cv_loss"]) == pytest.approx(raw, rel=1e-5)
    assert "quota_loss" in aux
    assert "load" in aux
    assert aux["load"].shape == (4,)


def test_exponential_gumbel_hits_end_near_half_epochs() -> None:
    sched = GumbelTemperatureSchedule(
        1.0, 0.3, total_epochs=24, schedule="exponential", half_epochs=12
    )
    assert sched.alpha == pytest.approx(math.log(1.0 / 0.3) / 12.0, rel=1e-4)
    assert sched.temperature(0) == pytest.approx(1.0)
    assert sched.temperature(12) == pytest.approx(0.3, abs=1e-3)
    assert sched.temperature(20) == pytest.approx(0.3)
    # Monotonic nonincreasing
    prev = sched.temperature(0)
    for e in range(1, 16):
        cur = sched.temperature(e)
        assert cur <= prev + 1e-9
        prev = cur


def test_linear_gumbel_matches_sprint8() -> None:
    sched = GumbelTemperatureSchedule(
        1.0, 0.3, total_epochs=10, schedule="linear"
    )
    assert sched.temperature(0) == pytest.approx(1.0)
    assert sched.temperature(10) == pytest.approx(0.3)
    assert sched.temperature(5) == pytest.approx(0.65)
