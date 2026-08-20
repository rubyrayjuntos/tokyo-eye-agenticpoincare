from __future__ import annotations

import numpy as np
import pytest

from experiments.diagnostics.chem_mvp_disulf_pair_probe import (
    aggregate_verdict,
    layer_verdict,
    summarize_pair_distances_disc,
    summarize_pair_distances_trunk,
)


def test_summarize_pair_distances_trunk_normalizes_by_all_pair_median() -> None:
    trunk = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    summary = summarize_pair_distances_trunk(trunk, [(0, 1), (0, 2)])
    assert summary["chem_mean_euclidean"] == pytest.approx(1.5)
    assert summary["chem_mean_euclidean_normalized"] == pytest.approx(
        summary["chem_mean_euclidean"] / summary["all_pair_median_euclidean"]
    )


def test_summarize_pair_distances_disc_reports_poincare_pairs() -> None:
    disc = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [-0.2, 0.0]],
        dtype=np.float64,
    )
    summary = summarize_pair_distances_disc(disc, [(0, 1), (0, 2)], curvature=1.0)
    assert summary["n_pairs"] == 2
    assert summary["chem_mean_poincare"] > 0.0
    assert "0-1" in summary["pairs"]


def test_layer_verdict_flags_projection_only() -> None:
    assert layer_verdict(-0.005, -0.08) == "projection_only"
    assert layer_verdict(-0.06, -0.07) == "trunk_and_disc_same_sign"


def test_aggregate_verdict_requires_all_three_trunk() -> None:
    def row(pid: str, trunk_d: float, disc_d: float, n_pairs: int) -> dict:
        return {
            "pdb_id": pid,
            "pair_counts": {"mapped_pairs_type": n_pairs},
            "delta_chem_minus_baseline": {
                "chem_mean_euclidean_trunk_normalized": trunk_d,
                "chem_mean_poincare_normalized": disc_d,
                "layer_verdict": layer_verdict(trunk_d, disc_d),
                "trunk_compacts_nontrivial": trunk_d <= -0.02,
                "disc_compacts_nontrivial": disc_d <= -0.02,
                "trunk_token_compaction_only": -0.02 < trunk_d < 0.0,
            },
        }

    # 1IVO-heavy pool with only 1IVO trunk win → partial
    partial = aggregate_verdict(
        [
            row("1LYZ", -0.005, -0.01, 4),
            row("1F88", -0.004, -0.01, 1),
            row("1IVO", -0.10, -0.12, 18),
        ]
    )
    assert partial["outcome"] == "partial"
    assert partial["dominance_guard_passes"] is False

    # Balanced nontrivial trunk on all three → win only if dominance ok
    # With equal shares of 4 each, dominance passes.
    win = aggregate_verdict(
        [
            row("1LYZ", -0.05, -0.06, 4),
            row("1F88", -0.05, -0.06, 4),
            row("1IVO", -0.05, -0.06, 4),
        ]
    )
    assert win["outcome"] == "win"
    assert win["n_structures_trunk_ok"] == 3
