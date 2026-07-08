"""Crescent collapse detection and Tier 2 biology gate."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.crescent_biology_projection import (
    QuartileKs,
    _tier2_verdict,
)
from science.training.disc_occupancy import is_crescent_collapsed


def test_is_crescent_collapsed_detects_thin_arc() -> None:
    """A tight high-r arc has low thickness and low effective rank."""
    n = 120
    t = np.linspace(-0.4, 0.2, n)
    r = 0.84 + 0.01 * np.sin(t)
    xy = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    blocked, metrics = is_crescent_collapsed(xy)
    assert blocked is True
    assert metrics["disc_effective_rank"] < 1.35
    assert metrics["disc_r_span"] < 0.12


def test_is_crescent_collapsed_allows_2d_wedge() -> None:
    """A filled wedge should not be blocked."""
    rng = np.random.default_rng(0)
    n = 200
    r = rng.uniform(0.2, 0.7, size=n)
    theta = rng.uniform(-2.0, 2.0, size=n)
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    blocked, metrics = is_crescent_collapsed(xy)
    assert blocked is False
    assert metrics["disc_line_thickness_rms"] > 0.02


def test_tier2_verdict_crescent_blocked_before_residual() -> None:
    status, interp = _tier2_verdict(
        0.001,
        [],
        within_quartile_max_perm_p=0.0001,
        crescent_geometry_blocked=True,
    )
    assert status == "crescent_blocked"
    assert "1D crescent" in interp


def test_tier2_verdict_residual_fail_when_not_blocked() -> None:
    status, _ = _tier2_verdict(
        0.28,
        [QuartileKs(1, 0.1, 0.2, 10, 10, 0.5, 0.01, 0.02)],
        within_quartile_max_perm_p=0.0002,
        crescent_geometry_blocked=False,
    )
    assert status == "residual_fail"


def test_topology_crescent_recovery_phase_config() -> None:
    from science.training.config import topology_crescent_recovery_phase_config

    phase = topology_crescent_recovery_phase_config()
    assert phase.topology_crescent_recovery_train is True
    assert phase.coeffs.disc_thickness_floor_coeff >= 6.0
    assert phase.coeffs.disc_eff_rank_coeff > 0
    assert phase.coeffs.angular_coeff > 0
    assert phase.min_disc_line_thickness_save == 0.03


def test_topology_crescent_recovery_freeze() -> None:
    from experiments.training.v6.train_loop import set_topology_crescent_recovery_freeze
    from science.dtie.v6.gnn.model import GOSPConeMapperV6

    model = GOSPConeMapperV6(expert_depth_decouple=True)
    set_topology_crescent_recovery_freeze(model)
    assert any(p.requires_grad for p in model.angular_head.parameters())
    assert any(p.requires_grad for p in model.hyp_proj_head_2d.parameters())
    assert not any(p.requires_grad for p in model.radial_head.parameters())
    assert model.expert_depth_bias is not None
    assert not model.expert_depth_bias.requires_grad
