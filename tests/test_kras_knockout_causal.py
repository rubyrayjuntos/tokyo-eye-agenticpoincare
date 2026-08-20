"""Unit tests for forward-pass knockout scoring."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.kras_knockout_causal import score_knockout


def test_score_knockout_logs_raw_and_r_out() -> None:
    # 5 residues; knockout-163 out-effect is largest.
    out_effect = np.array([1.0, 2.0, 8.0, 3.0, 2.5], dtype=np.float64)
    # delta_rows[i][j] = effect at j when knocking i
    delta_rows = [
        np.array([0.0, 0.5, 1.0, 0.4, 0.3]),  # knock 0 → effect at 163(idx2)=1.0
        np.array([0.2, 0.0, 0.8, 0.3, 0.2]),
        np.array([2.0, 2.0, 0.0, 2.0, 2.0]),  # knock 163
        np.array([0.1, 0.1, 0.5, 0.0, 0.1]),
        np.array([0.2, 0.2, 0.6, 0.2, 0.0]),
    ]
    indices = {12: 0, 151: 1, 163: 2}
    s = score_knockout(out_effect, delta_rows, indices)
    assert s["out_163_raw"] == 8.0
    assert s["median_out"] == 2.5
    assert abs(s["R_out_163"] - 8.0 / 2.5) < 1e-12
    assert s["in_163_from_12_raw"] == 1.0
    assert s["rank_out_163"] == 1.0
