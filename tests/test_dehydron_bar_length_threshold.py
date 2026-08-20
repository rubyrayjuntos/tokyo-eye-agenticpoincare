"""Unit tests for dehydron bar-length diagnostic helpers (no GUDHI / PDB)."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.dehydron_bar_length_threshold import (
    DOMINANCE_FRAC,
    build_dim_report,
    dominance_by_structure,
    largest_gap_threshold,
    persistences_for_dim,
    recommend_threshold,
    summarize_lengths,
)


def test_persistences_for_dim_filters_and_floor():
    bars = [
        {"dim": 0, "persistence": 0.05, "birth": 0.0, "death": 0.05},
        {"dim": 1, "persistence": 0.2, "birth": 0.0, "death": 0.2},
        {"dim": 1, "persistence": 1.5, "birth": 0.0, "death": 1.5},
    ]
    h1_all = persistences_for_dim(bars, 1, min_persistence=0.0)
    h1_floor = persistences_for_dim(bars, 1, min_persistence=0.1)
    h0_floor = persistences_for_dim(bars, 0, min_persistence=0.1)
    np.testing.assert_allclose(h1_all, [0.2, 1.5])
    np.testing.assert_allclose(h1_floor, [0.2, 1.5])
    assert h0_floor.size == 0  # 0.05 < 0.1


def test_summarize_lengths_empty_and_populated():
    empty = summarize_lengths(np.asarray([], dtype=np.float64))
    assert empty["n"] == 0
    assert empty["percentiles"]["75"] is None

    vals = summarize_lengths(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
    assert vals["n"] == 4
    assert vals["min"] == 1.0
    assert vals["max"] == 4.0
    assert vals["percentiles"]["50"] == 2.5


def test_largest_gap_finds_upper_cluster_break():
    # Dense short cluster + sparse long tail with a clear gap.
    values = np.asarray(
        [0.2, 0.25, 0.3, 0.35, 0.4, 2.0, 2.1, 2.2],
        dtype=np.float64,
    )
    gap = largest_gap_threshold(values, search_from_percentile=50.0, min_tail_count=3)
    assert gap is not None
    assert gap["lo"] < 1.0 < gap["hi"]
    assert gap["n_bars_strictly_above_lo"] >= 3


def test_dominance_flags_majority_owner():
    # Stage A-shaped equal split (~8% each) must not flag.
    equal = {f"P{i}:A": 10 for i in range(12)}
    mild = dominance_by_structure(equal)
    assert mild["dominated"] is False
    assert mild["max_frac"] < DOMINANCE_FRAC

    heavy = dominance_by_structure({"A:A": 80, "B:A": 10, "C:A": 10})
    assert heavy["dominated"] is True
    assert heavy["max_structure"] == "A:A"
    assert heavy["max_frac"] >= DOMINANCE_FRAC


def test_build_dim_report_and_recommend_with_dominance():
    per = {
        "BIG:A": [
            {"dim": 1, "persistence": 0.5, "birth": 0.0, "death": 0.5},
            {"dim": 1, "persistence": 0.6, "birth": 0.0, "death": 0.6},
            {"dim": 1, "persistence": 0.7, "birth": 0.0, "death": 0.7},
            {"dim": 1, "persistence": 3.0, "birth": 0.0, "death": 3.0},
        ]
        * 5,
        "SMALL:A": [
            {"dim": 1, "persistence": 0.4, "birth": 0.0, "death": 0.4},
        ],
    }
    pooled = [b for bars in per.values() for b in bars]
    report = build_dim_report(
        pooled_bars=pooled,
        per_structure_bars=per,
        dim=1,
        noise_floor=0.1,
    )
    assert report["pooled"]["above_noise_floor"]["n"] == 21
    assert report["dominance_above_noise"]["dominated"] is True
    assert "BIG:A" in report["per_structure"]

    rec = recommend_threshold(report)
    assert rec["recommended_angstrom"] is not None
    assert rec["needs_manual_review"] is True
    assert "MANUAL REVIEW" in rec["rationale"]
