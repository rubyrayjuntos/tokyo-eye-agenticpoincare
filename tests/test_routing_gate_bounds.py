"""Property tests for N-aware routing gate bounds."""

from __future__ import annotations

import math

import pytest

from science.training.checkpoint_score import ROUTING_ENTROPY_SAVE_MAX
from science.training.mlflow_governance import stage_a_gate_passed
from science.training.routing_gate_bounds import (
    effective_experts_fraction_at_ceiling,
    model_num_experts,
    routing_entropy_save_ceiling,
    scale_routing_entropy_ceiling,
    stage_a_effective_experts_bounds,
    uniform_routing_entropy,
)


def test_stage_a_bounds_scale_linearly_with_num_experts() -> None:
    n4 = stage_a_effective_experts_bounds(4)
    n6 = stage_a_effective_experts_bounds(6)
    assert n4 == pytest.approx((3.0, 4.5, 2.5))
    assert n6 == pytest.approx((4.5, 6.75, 3.75))


def test_entropy_save_ceiling_scales_with_n_preserving_effective_fraction() -> None:
    assert routing_entropy_save_ceiling(num_experts=4) == ROUTING_ENTROPY_SAVE_MAX
    assert routing_entropy_save_ceiling(num_experts=6) == pytest.approx(
        ROUTING_ENTROPY_SAVE_MAX + math.log(6 / 4)
    )
    frac4 = effective_experts_fraction_at_ceiling(4)
    frac6 = effective_experts_fraction_at_ceiling(6)
    assert frac4 == pytest.approx(frac6, rel=1e-9)


def test_scale_routing_entropy_ceiling_shifts_n4_ramp_endpoints() -> None:
    assert scale_routing_entropy_ceiling(1.40, 6) == pytest.approx(1.40 + math.log(6 / 4))
    assert scale_routing_entropy_ceiling(1.21, 6) == pytest.approx(
        routing_entropy_save_ceiling(num_experts=6)
    )


def test_uniform_entropy_scales_with_n() -> None:
    assert uniform_routing_entropy(4) == pytest.approx(math.log(4))
    assert uniform_routing_entropy(6) == pytest.approx(math.log(6))


def test_stage_gate_n6_uses_scaled_band_not_n4_bar() -> None:
    """N=6 healthy routing must not fail solely because N=4 band was [3, 4.5]."""
    health = {
        "disc_sigma2_sigma1_mean": 0.665,
        "probe_r_depth_sasa": 0.73,
    }
    routing = {
        "effective_experts": 5.2,
        "effective_experts_min": 4.0,
        "min_routing_fraction": 0.18,
    }
    losses = {
        **routing,
        "per_fold_loss.3_40_50_300": 1.0,
        "per_fold_loss.3_80_20_20": 1.1,
    }
    assert stage_a_gate_passed(health, losses, num_experts=4) == 0
    assert stage_a_gate_passed(health, losses, num_experts=6) == 1


def test_model_num_experts_from_module_list() -> None:
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(1, 1) for _ in range(6)])

    assert model_num_experts(_M()) == 6
