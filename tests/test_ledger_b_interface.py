"""Unit tests for Ledger B interface scoring helpers."""

from __future__ import annotations

import numpy as np

from science.dtie.common.ledger_b_interface import (
    interface_alignment_score,
    resolve_interface_set,
)


def test_resolve_drops_missing_and_adds_conditional() -> None:
    spec = {
        "primary_resseqs": [10, 20, 999],
        "conditional_resseqs_if_present": {"SH2": {"range": [148, 150]}},
    }
    present = {10, 20, 148, 149, 150}
    r = resolve_interface_set(spec, present)
    assert 999 in r["missing_from_deposit"]
    assert set(r["interface_resseqs"]) == {10, 20, 148, 149, 150}


def test_resolve_applies_literature_to_auth_offset() -> None:
    # 3PP0: lit 273 → auth 706 (offset 433)
    spec = {
        "literature_to_auth_offset": 433,
        "primary_resseqs": [273, 295],
        "conditional_resseqs_if_present": {"SH3": {"range": [81, 82]}},
    }
    present = {706, 728, 900}
    r = resolve_interface_set(spec, present)
    assert set(r["interface_resseqs"]) == {706, 728}
    assert r["missing_from_deposit"] == []
    assert r["conditional_included"] == []  # 81+433=514 not in deposit


def test_recall_pass() -> None:
    # 10 residues; interface {0,1,2,3}; top 10% = 1 hub → need more hubs
    # n=20, top 10% = 2 hubs. Put high scores on interface sites 0 and 1.
    oe = np.zeros(20)
    oe[0] = 10.0
    oe[1] = 9.0
    oe[5] = 1.0
    idx = {i: i for i in range(20)}
    iface = [0, 1, 2, 3]
    s = interface_alignment_score(oe, idx, iface, k_frac=0.10, recall_threshold=0.25)
    # hub set size 2 → overlap {0,1} → recall 2/4 = 0.5
    assert s["recall_at_top_k"] == 0.5
    assert s["pass"] is True
    assert set(s["hub_overlap_resseqs"]) == {0, 1}
    assert s["hub_resseqs"] == [0, 1]
    assert s["hub_outside_interface"] == []
