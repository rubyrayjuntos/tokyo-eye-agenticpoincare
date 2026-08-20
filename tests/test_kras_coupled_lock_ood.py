from __future__ import annotations

import numpy as np
import pytest

from experiments.diagnostics.kras_coupled_lock_ood import (
    TARGET_RESIDUES,
    resolve_target_indices,
    summarize_euclidean_motif_distances,
    summarize_motif_distances,
)


def test_resolve_target_indices_checks_pdb_identity_and_graph_numbering() -> None:
    residue_ids = ["A:11:", "A:12:", "A:32:", "A:60:", "A:61:", "A:62:"]
    pdb_resnames = {12: "ASP", 32: "TYR", 60: "GLY", 61: "GLN"}

    resolved = resolve_target_indices(residue_ids, pdb_resnames)

    assert resolved == {12: 1, 32: 2, 60: 3, 61: 4}


def test_resolve_target_indices_rejects_wt_gly12() -> None:
    residue_ids = ["A:12:", "A:32:", "A:60:", "A:61:"]
    pdb_resnames = {12: "GLY", 32: "TYR", 60: "GLY", 61: "GLN"}

    with pytest.raises(ValueError, match="12.*ASP.*GLY"):
        resolve_target_indices(residue_ids, pdb_resnames)


def test_summarize_motif_distances_reports_lock_pairs_and_normalized_compactness() -> None:
    disc = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [0.1, 0.2], [-0.2, 0.0]],
        dtype=np.float64,
    )
    cone = np.asarray([1.0, 1.2, 1.4, 1.8, 0.5], dtype=np.float64)
    indices = {resnum: idx for idx, resnum in enumerate(TARGET_RESIDUES)}

    summary = summarize_motif_distances(
        disc,
        cone,
        indices,
        curvature=1.0,
    )

    assert set(summary["lock_pairs"]) == {"12-32", "12-60", "12-61"}
    assert summary["lock_mean_poincare"] > 0.0
    assert summary["all_pair_median_poincare"] > 0.0
    assert summary["lock_mean_poincare_normalized"] == pytest.approx(
        summary["lock_mean_poincare"] / summary["all_pair_median_poincare"]
    )
    assert summary["lock_pairs"]["12-32"]["cone_depth_delta"] == pytest.approx(0.2)


def test_summarize_euclidean_motif_distances_uses_trunk_l2() -> None:
    trunk = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    indices = {resnum: idx for idx, resnum in enumerate(TARGET_RESIDUES)}

    summary = summarize_euclidean_motif_distances(trunk, indices)

    assert summary["lock_pairs"]["12-32"]["euclidean"] == pytest.approx(1.0)
    assert summary["lock_pairs"]["12-60"]["euclidean"] == pytest.approx(2.0)
    assert summary["lock_pairs"]["12-61"]["euclidean"] == pytest.approx(3.0)
    assert summary["lock_mean_euclidean"] == pytest.approx(2.0)
    assert summary["lock_mean_euclidean_normalized"] == pytest.approx(
        summary["lock_mean_euclidean"] / summary["all_pair_median_euclidean"]
    )
