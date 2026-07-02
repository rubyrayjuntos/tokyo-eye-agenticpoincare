"""Tests for Poincaré ball-radius convention probe and property suite guards."""

from __future__ import annotations

import numpy as np
import pytest

from science.dtie.common.curvature_loader import CANONICAL_V6_CURVATURE
from science.dtie.common.poincare_conventions import (
    estimate_clamp_applied_fraction,
    model_clamp_ceiling,
    probe_disc_convention,
)
from science.dtie.common.poincare_properties import run_property_suite


def _legacy_clamped_disc(n: int, c: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate pre-fix model: project then clamp to Euclidean norm 0.99."""
    angles = rng.uniform(0, 2 * np.pi, size=n)
    radii = rng.uniform(0.2, 1.5, size=n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * np.minimum(1.0, 0.99 / (norms + 1e-8))


def test_convention_probe_detects_legacy_unit_clamp():
    c = CANONICAL_V6_CURVATURE
    disc = _legacy_clamped_disc(200, c, np.random.default_rng(0))
    probe = probe_disc_convention(disc, c)
    assert probe.disc_max_raw_norm == pytest.approx(0.99, abs=0.01)
    assert probe.r_ball == pytest.approx(1 / np.sqrt(c), rel=1e-3)
    assert probe.disc_max_over_r_ball == pytest.approx(0.99 * np.sqrt(c), rel=0.02)
    assert probe.legacy_unit_clamp_detected is True


def test_convention_probe_unified_clamp_ceiling():
    c = CANONICAL_V6_CURVATURE
    ceiling = model_clamp_ceiling(c)
    rng = np.random.default_rng(1)
    angles = rng.uniform(0, 2 * np.pi, size=100)
    radii = rng.uniform(0.1, ceiling, size=100)
    disc = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    probe = probe_disc_convention(disc, c)
    assert probe.disc_max_over_r_ball == pytest.approx(probe.disc_max_raw_norm / probe.r_ball, rel=1e-3)
    assert probe.legacy_unit_clamp_detected is False


def test_saturation_unreachable_under_legacy_misnormalization():
    """frac>0.99 vs r_ball is structurally ~0 when legacy clamp caps at 0.99."""
    c = CANONICAL_V6_CURVATURE
    disc = _legacy_clamped_disc(500, c, np.random.default_rng(2))
    result = run_property_suite(
        structure_id="synthetic",
        curvature=c,
        source="test",
        disc_2d=disc,
        cone_depth=np.linspace(0, 1, len(disc)),
    )
    assert result.norm_disc.fraction_above_0_99_of_ceiling > 0.0
    assert result.norm_disc.ratio_vs_r_ball_p99 < 0.85


def test_invariant3_skipped_from_db_x_hyp():
    c = CANONICAL_V6_CURVATURE
    disc = _legacy_clamped_disc(50, c, np.random.default_rng(3))
    x_hyp = np.random.default_rng(4).normal(size=(50, 32))
    result = run_property_suite(
        structure_id="4uj1",
        curvature=c,
        source="db",
        disc_2d=disc,
        cone_depth=np.linspace(0, 1, 50),
        x_hyp_highd=x_hyp,
    )
    assert result.ball_disc is not None
    assert result.ball_disc.skipped is True
    assert result.norm_routed_highd is None


def test_physics_skips_flat_cone_depth():
    c = 1.0
    disc = np.random.default_rng(5).uniform(-0.5, 0.5, size=(30, 2))
    result = run_property_suite(
        structure_id="flat",
        curvature=c,
        source="test",
        disc_2d=disc,
        cone_depth=np.full(30, 7.4675),
    )
    assert result.physics.skipped_depth_correlation is True
    assert np.isnan(result.physics.radial_vs_cone_depth_r)


def test_clamp_applied_fraction_on_ceiling():
    c = CANONICAL_V6_CURVATURE
    norms = np.array([0.5, 0.9, 0.99, 0.99, 0.88])
    frac = estimate_clamp_applied_fraction(norms, c, legacy_unit_clamp=True)
    assert frac == pytest.approx(0.4, abs=0.01)
