"""Unit tests for KRAS hub-migration scoring helpers."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.diagnostics.kras_hub_migration_ood import (
    resolve_hub_indices,
    score_in_centrality,
)


def test_score_logs_raw_and_normalized_r() -> None:
    in_c = np.array([1.0, 2.0, 10.0, 4.0, 3.0], dtype=np.float64)
    # indices: 151→1, 163→2
    indices = {12: 0, 151: 1, 163: 2}
    s = score_in_centrality(in_c, indices)
    assert s["in_163_raw"] == 10.0
    assert s["in_151_raw"] == 2.0
    assert s["median_in"] == 3.0
    assert abs(s["R_163"] - 10.0 / 3.0) < 1e-12
    assert abs(s["H_163_over_151"] - 5.0) < 1e-12
    assert s["rank_163"] == 1.0


def test_resolve_hub_refuses_identity_mismatch() -> None:
    residue_ids = [f"A:{i}:" for i in (12, 151, 163)]
    pdb_names = {12: "GLY", 151: "GLY", 163: "ILE"}
    ok = resolve_hub_indices(residue_ids, pdb_names, {12: "GLY", 151: "GLY", 163: "ILE"})
    assert ok[163] == 2
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_hub_indices(
            residue_ids, {12: "ASP", 151: "GLY", 163: "ILE"}, {12: "GLY", 151: "GLY", 163: "ILE"}
        )
