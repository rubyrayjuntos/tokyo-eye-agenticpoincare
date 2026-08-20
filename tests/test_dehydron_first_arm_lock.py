"""Tests for locked first-arm dehydron scalars, corpus z-norm, and min-dehydron guard."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.training.v6.precompute_dehydron_barcodes import (
    stats_missing_for_training_rows,
)
from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    LONG_LIVED_PERSISTENCE_ANGSTROM,
    MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS,
    SCALAR_DIM,
    SCALAR_NAMES,
    DehydronMidpoint,
    PersistenceBar,
    aggregate_residue_barcode_features,
    apply_corpus_zscore,
    compute_corpus_zscore_stats,
    compute_witness_persistence,
)


def test_locked_first_arm_scalar_contract():
    assert BARCODE_FEATURE_VERSION == "dehydron_barcode_v1_2"
    assert SCALAR_DIM == 3
    assert SCALAR_NAMES == [
        "total_persistence_h1",
        "fraction_long_lived_h1",
        "n_dehydrons_touching",
    ]
    assert LONG_LIVED_PERSISTENCE_ANGSTROM == 3.11
    assert MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS == 5


def test_aggregate_uses_h1_only_for_total_and_fraction():
    midpoints = [
        DehydronMidpoint(np.zeros(3), donor_idx=0, acceptor_idx=1, wrapping_count=2.0),
        DehydronMidpoint(np.ones(3), donor_idx=1, acceptor_idx=0, wrapping_count=3.0),
    ]
    bars = [
        PersistenceBar(dim=0, birth=0.0, death=20.0, persistence=20.0),
        PersistenceBar(dim=1, birth=0.0, death=1.0, persistence=1.0),
        PersistenceBar(dim=1, birth=0.0, death=4.0, persistence=4.0),
    ]
    out = aggregate_residue_barcode_features(
        2,
        midpoints,
        bars,
        long_lived_persistence_angstrom=3.11,
    )
    # both residues touch; H1 total=5 → log1p(5); frac long-lived = 1/2
    assert out["scalars"].shape == (2, 3)
    np.testing.assert_allclose(out["scalars"][0, 0], np.log1p(5.0), rtol=1e-6)
    np.testing.assert_allclose(out["scalars"][0, 1], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out["scalars"][0, 2], np.log1p(2.0), rtol=1e-6)
    # H0 persistence must not inflate total
    assert out["scalars"][0, 0] < np.log1p(25.0)


def test_min_dehydron_guard_skips_witness_below_threshold():
    midpoints = [
        DehydronMidpoint(np.array([float(i), 0.0, 0.0]), donor_idx=0, acceptor_idx=1, wrapping_count=1.0)
        for i in range(MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS - 1)
    ]
    assert compute_witness_persistence(midpoints) == []


def test_corpus_zscore_stats_ignore_missing_and_apply_round_trips():
    scalars = [
        np.asarray([[1.0, 0.2, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[3.0, 0.8, 1.5]], dtype=np.float32),
    ]
    missing = [
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([[0.0]], dtype=np.float32),
    ]
    stats = compute_corpus_zscore_stats(scalars, missing)
    assert stats["n_valid_rows"] == 2
    np.testing.assert_allclose(stats["mean"], [2.0, 0.5, 1.0], rtol=1e-6)

    z = apply_corpus_zscore(scalars[0], missing[0], stats)
    # valid row becomes z-scored; missing row stays zero
    assert z[1].tolist() == [0.0, 0.0, 0.0]
    assert abs(z[0, 0]) > 0.0


def test_zscore_stats_mask_excludes_pdb_rows_filtered_from_training_graph():
    payload = {
        "residue_indices": np.asarray([1, 2, 802], dtype=np.int32),
        "missing": np.zeros((3, 1), dtype=np.float32),
    }

    stats_missing = stats_missing_for_training_rows(payload, [1, 2])

    np.testing.assert_array_equal(
        stats_missing,
        np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32),
    )
