"""Unit tests for SHP2 2SHP→6MCF OOD migration gate scoring."""

from __future__ import annotations

import numpy as np
import pytest

from science.dtie.common.shp2_state_migration import (
    DEFAULT_BARS,
    grade_shp2_migration,
    hub_set_jaccard,
    score_calibration,
    score_perturbation_invariance,
    score_state_transition,
)


def test_state_transition_pass_on_correlated_axes() -> None:
    rng = np.random.default_rng(0)
    inactive = rng.random(80)
    active = inactive + 0.05 * rng.random(80)
    maps = {
        "2SHP": {i + 1: i for i in range(80)},
        "6CRF": {i + 1: i for i in range(80)},
    }
    s = score_state_transition(
        {"2SHP": inactive, "6CRF": active},
        maps,
        spearman_bar=0.50,
    )
    assert s["spearman"] > 0.9
    assert s["pass"] is True
    assert s["n_shared"] == 80


def test_state_transition_fail_on_anticorrelated() -> None:
    x = np.arange(50, dtype=np.float64)
    y = -x
    maps = {
        "2SHP": {i: i for i in range(50)},
        "6CRF": {i: i for i in range(50)},
    }
    s = score_state_transition(
        {"2SHP": x, "6CRF": y},
        maps,
        spearman_bar=0.50,
    )
    assert s["pass"] is False


def test_hub_set_jaccard_identity() -> None:
    oe = np.linspace(0.01, 1.0, 100)
    assert hub_set_jaccard(oe, oe, k_frac=0.1) == pytest.approx(1.0)


def test_perturbation_invariance_requires_all_arms() -> None:
    base = np.linspace(0.2, 1.0, 40)
    good = base + 0.01 * np.arange(40)[::-1] / 40.0
    bad = base[::-1]
    s = score_perturbation_invariance(
        base,
        {
            "cutoff_7p5": good,
            "cutoff_8p5": good,
            "edge_truncate_0p8": bad,
        },
        jaccard_bar=0.50,
    )
    assert s["pass"] is False
    assert s["arms"]["cutoff_7p5"]["pass"] is True
    assert s["arms"]["edge_truncate_0p8"]["pass"] is False


def test_calibration_band_and_no_blowup() -> None:
    # Construct mild inequality with G ≈ 0.16 (champion-like band)
    n = 100
    oe_inact = np.linspace(0.05, 0.25, n)
    oe_act = np.linspace(0.06, 0.22, n)
    s = score_calibration(oe_inact, oe_act, bars=DEFAULT_BARS)
    assert 0.12 <= s["gini_active"] <= 0.25
    assert s["pass"] is True


def test_calibration_fails_on_monopoly_collapse() -> None:
    oe_inact = np.full(100, 0.1)
    oe_act = np.zeros(100)
    oe_act[0] = 1.0
    s = score_calibration(oe_inact, oe_act, bars=DEFAULT_BARS)
    assert s["pass"] is False


def test_aggregate_grade_requires_all_three() -> None:
    g = grade_shp2_migration(
        state_transition={"pass": True, "spearman": 0.7},
        perturbation={"pass": True},
        calibration={"pass": False},
    )
    assert g["pass"] is False
    assert g["verdict"] == "FAIL"
    g2 = grade_shp2_migration(
        state_transition={"pass": True},
        perturbation={"pass": True},
        calibration={"pass": True},
    )
    assert g2["pass"] is True
    assert g2["verdict"] == "PASS"
