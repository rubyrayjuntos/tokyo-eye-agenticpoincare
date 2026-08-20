"""Unit tests for full-chain knockout concentration / Gini metrics."""

from __future__ import annotations

import numpy as np
import pytest

from science.dtie.common.gini_flow_concentration import (
    compare_concentration,
    concentration_metrics,
    gini_coefficient,
)


def test_gini_perfect_equality_is_zero() -> None:
    x = np.full(40, 0.12)
    assert gini_coefficient(x) == pytest.approx(0.0, abs=1e-12)


def test_gini_single_node_monopoly_near_one() -> None:
    x = np.zeros(100)
    x[0] = 1.0
    assert gini_coefficient(x) == pytest.approx(1.0 - 1.0 / 100.0, abs=1e-12)


def test_concentration_metrics_mass_shares_and_max_median() -> None:
    # 10 residues: one large, nine equal small
    oe = np.array([0.50] + [0.05] * 9, dtype=np.float64)
    m = concentration_metrics(oe)
    assert m["n"] == 10
    assert m["gini"] == pytest.approx(gini_coefficient(oe))
    assert m["max"] == pytest.approx(0.50)
    assert m["median"] == pytest.approx(0.05)
    assert m["max_over_median"] == pytest.approx(10.0)
    assert m["top1pct_mass_share"] == pytest.approx(0.50 / oe.sum())
    # top 10% of 10 → 1 residue
    assert m["top10pct_mass_share"] == pytest.approx(0.50 / oe.sum())
    assert "out_effect" not in m


def test_compare_concentration_positive_delta_g_means_reduction() -> None:
    base = np.array([0.9] + [0.01] * 99, dtype=np.float64)
    champ = np.full(100, 0.1, dtype=np.float64)
    cmp_ = compare_concentration(base, champ)
    assert cmp_["delta_gini"] == pytest.approx(
        cmp_["baseline"]["gini"] - cmp_["champion"]["gini"]
    )
    assert cmp_["delta_gini"] > 0.0
    assert cmp_["monopoly_reduced"] is True


def test_compare_concentration_archives_full_arrays() -> None:
    base = np.arange(5, dtype=np.float64) + 1.0
    champ = np.full(5, 2.0)
    cmp_ = compare_concentration(base, champ, archive_arrays=True)
    assert cmp_["baseline"]["out_effect"] == base.tolist()
    assert cmp_["champion"]["out_effect"] == champ.tolist()


def test_gini_rejects_empty_and_all_zero() -> None:
    assert np.isnan(gini_coefficient(np.array([])))
    assert np.isnan(gini_coefficient(np.zeros(8)))
