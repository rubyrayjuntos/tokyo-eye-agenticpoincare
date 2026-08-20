"""Unit tests for Ledger B sensitivity helpers."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.ledger_b_sensitivity_check import (
    JITTER_SPEARMAN_BAR,
    _ranks_desc,
)


def test_jitter_bar_locked() -> None:
    assert JITTER_SPEARMAN_BAR == 0.85


def test_ranks_desc_ordering() -> None:
    scores = np.array([0.1, 0.9, 0.5])
    ranks = _ranks_desc(scores)
    assert ranks.tolist() == [3, 1, 2]
