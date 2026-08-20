"""Pure helper tests for the barcode scalar orthogonality diagnostic."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest


MODULE = "experiments.diagnostics.dehydron_scalar_orthogonality"


def _module():
    assert importlib.util.find_spec(MODULE) is not None, (
        "orthogonality diagnostic module must exist"
    )
    return importlib.import_module(MODULE)


def test_candidate_matrix_uses_h1_only_and_broadcasts_to_touching_residues():
    mod = _module()
    bars = [
        {"dim": 0, "persistence": 20.0},
        {"dim": 1, "persistence": 1.0},
        {"dim": 1, "persistence": 4.0},
    ]
    touching_counts = np.asarray([0, 2, 1], dtype=np.float64)

    matrix = mod.build_candidate_matrix(
        bars,
        touching_counts,
        long_lived_threshold=3.11,
    )

    assert list(matrix) == [
        "total_persistence_h1",
        "max_persistence_h1",
        "num_h1_bars",
        "fraction_long_lived_h1",
        "n_dehydrons_touching",
    ]
    np.testing.assert_allclose(matrix["total_persistence_h1"], [0.0, 5.0, 5.0])
    np.testing.assert_allclose(matrix["max_persistence_h1"], [0.0, 4.0, 4.0])
    np.testing.assert_allclose(matrix["num_h1_bars"], [0.0, 2.0, 2.0])
    np.testing.assert_allclose(matrix["fraction_long_lived_h1"], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(matrix["n_dehydrons_touching"], [0.0, 2.0, 1.0])


def test_correlation_report_separates_marginal_and_within_ss_classes():
    mod = _module()
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.asarray([0.0, 2.0, 4.0, 3.0, 2.0, 1.0])
    ss = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    report = mod.correlation_report(x, y, ss)

    assert report["marginal"]["n"] == 6
    assert set(report["within_ss"]) == {"0", "1"}
    assert report["within_ss"]["0"]["pearson"] == pytest.approx(1.0)
    assert report["within_ss"]["1"]["pearson"] == pytest.approx(-1.0)


def test_redundancy_verdict_fails_on_any_valid_absolute_correlation_at_cut():
    mod = _module()
    reports = {
        "rho": {
            "marginal": {"pearson": 0.2, "spearman": 0.3},
            "within_ss": {
                "0": {"pearson": 0.71, "spearman": 0.6},
                "1": {"pearson": None, "spearman": None},
            },
        }
    }

    verdict = mod.redundancy_verdict(reports, threshold=0.70)

    assert verdict["passes"] is False
    assert verdict["max_abs_correlation"] == 0.71
    assert verdict["worst_path"] == "rho.within_ss.0.pearson"


def test_constant_vectors_are_reported_as_unscorable_not_zero_correlation():
    mod = _module()
    ss = np.zeros(4)
    report = mod.correlation_report(np.ones(4), np.arange(4.0), ss)

    assert report["marginal"]["pearson"] is None
    assert report["marginal"]["spearman"] is None
    assert report["marginal"]["reason"] == "constant"


def test_touching_counts_align_by_residue_identity_when_feature_rows_are_filtered():
    mod = _module()
    from science.dtie.common.dehydron_barcode_features import DehydronMidpoint

    residue_index_map = {("A", 10): 0, ("A", 11): 1, ("A", 802): 2}
    midpoints = [
        DehydronMidpoint(np.zeros(3), donor_idx=0, acceptor_idx=2, wrapping_count=2),
        DehydronMidpoint(np.ones(3), donor_idx=1, acceptor_idx=0, wrapping_count=3),
    ]

    counts = mod.align_touching_counts(
        midpoints,
        residue_index_map,
        feature_residue_indices=[10, 11],  # 802 was filtered from SSOT nodes
    )

    np.testing.assert_allclose(counts, [2.0, 1.0])


def test_payload_gate_excludes_expected_all_residue_support_mask_alignment():
    mod = _module()
    reports = {
        "rho": {
            "all_residues": {
                "marginal": {"pearson": 0.9, "spearman": 0.9},
                "within_ss": {},
            },
            "touching_only": {
                "marginal": {"pearson": 0.2, "spearman": 0.3},
                "within_ss": {},
            },
        },
        "structure_level": {
            "mean_rho": {"pearson": 0.1, "spearman": 0.2},
        },
    }

    gate_reports = mod.payload_baseline_gate_reports(reports)
    verdict = mod.redundancy_verdict(gate_reports, threshold=0.70)

    assert "all_residues" not in str(gate_reports)
    assert verdict["passes"] is True
