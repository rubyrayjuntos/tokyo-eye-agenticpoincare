"""Unit tests for angular shell geometry statistics."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.angular_shell_geometry_test import (
    _assign_radial_shells,
    _bh_adjust,
    _circular_mi_discrete,
    _cohort_verdict,
    _perm_pvalue,
    _permute_within_seq_blocks,
    StructureAngularReport,
)


def test_circular_mi_independent_near_zero() -> None:
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=200)
    labels = rng.integers(0, 3, size=200)
    mi = _circular_mi_discrete(theta, labels)
    assert mi < 0.15


def test_circular_mi_structured_positive() -> None:
    theta = np.linspace(-np.pi, np.pi, 120, endpoint=False)
    labels = np.where(theta < 0, 0, np.where(theta < np.pi / 2, 1, 2))
    mi = _circular_mi_discrete(theta, labels)
    assert mi > 0.2


def test_perm_pvalue_extreme() -> None:
    nulls = np.linspace(0, 0.1, 50)
    assert _perm_pvalue(0.5, nulls) < 0.05
    assert _perm_pvalue(0.0, nulls) > 0.9


def test_bh_adjust_controls_grid() -> None:
    pvals = [0.01, 0.02, 0.03, 0.5, 0.6]
    adj = _bh_adjust(pvals)
    assert adj[0] <= 0.05
    assert max(adj) <= 1.0


def test_radial_shells_cover_all_points() -> None:
    r = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    shells = _assign_radial_shells(r, 5)
    assert shells.min() >= 0
    assert shells.max() <= 4
    assert shells.size == r.size


def test_seq_block_permutation_preserves_block_sizes() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    blocks = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    rng = np.random.default_rng(1)
    out = _permute_within_seq_blocks(labels, blocks, rng)
    for b in np.unique(blocks):
        assert np.sort(labels[blocks == b]).tolist() == np.sort(out[blocks == b]).tolist()


def test_cohort_verdict_requires_replication() -> None:
    reports = [
        StructureAngularReport("A", "A", 100, 0.3, 0.03, verdict="pass"),
        StructureAngularReport("B", "A", 100, 0.3, 0.03, verdict="fail"),
        StructureAngularReport("C", "A", 100, 0.3, 0.03, verdict="fail"),
    ]
    assert _cohort_verdict(reports) == "fail"

    reports[1].verdict = "pass"
    assert _cohort_verdict(reports) == "pass"
