"""Unit tests for hub-knockout classical grading."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.hub_knockout_classical import (
    grade_arm_compare,
    score_structure_knockout_vs_classical,
)


def test_score_structure_positive_spearman_holds() -> None:
    rng = np.random.default_rng(0)
    btw = rng.random(40)
    out = btw + 0.05 * rng.random(40)
    s = score_structure_knockout_vs_classical(out, btw, seed=1)
    assert s["spearman_betweenness"] is not None
    assert s["spearman_betweenness"] > 0.8
    assert s["holds"] is True


def test_grade_arm_compare_pass_on_clear_delta() -> None:
    baseline = [
        {"spearman_betweenness": 0.20, "holds": False},
        {"spearman_betweenness": 0.25, "holds": False},
        {"spearman_betweenness": 0.22, "holds": False},
    ]
    swap = [
        {"spearman_betweenness": 0.45, "holds": True},
        {"spearman_betweenness": 0.50, "holds": True},
        {"spearman_betweenness": 0.40, "holds": True},
    ]
    g = grade_arm_compare(baseline, swap)
    assert g["grade"] == "Pass"
    assert g["delta_median"] >= 0.10


def test_grade_arm_compare_partial_when_flat() -> None:
    baseline = [
        {"spearman_betweenness": 0.40, "holds": True},
        {"spearman_betweenness": 0.42, "holds": True},
        {"spearman_betweenness": 0.41, "holds": True},
    ]
    swap = [
        {"spearman_betweenness": 0.43, "holds": True},
        {"spearman_betweenness": 0.41, "holds": True},
        {"spearman_betweenness": 0.42, "holds": True},
    ]
    g = grade_arm_compare(baseline, swap)
    assert g["grade"] == "Partial"
    assert g["flat_by_prereg"] is True
