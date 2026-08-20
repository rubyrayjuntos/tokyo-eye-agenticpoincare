"""Tests for generic flow–centrality concordance helpers."""

from __future__ import annotations

import numpy as np

from science.dtie.common.topo_flow_concordance import (
    concordance_report,
    panel_verdict,
    stability_ok,
    top_k_hub_mask,
)


def test_top_k_hub_mask() -> None:
    btw = np.arange(10, dtype=np.float64)  # 9 is top
    mask = top_k_hub_mask(btw, k_frac=0.1)
    assert mask.sum() == 1
    assert mask[-1]


def test_concordance_pass_and_fail() -> None:
    btw = np.linspace(0, 1, 20)
    good = concordance_report(btw, btw, pass_rho=0.50)
    assert good["pass"] is True
    assert good["spearman_full"] > 0.99
    bad = concordance_report(-btw, btw, pass_rho=0.50)
    assert bad["pass"] is False


def test_panel_and_stability() -> None:
    assert stability_ok(0.60, 0.58) is True
    assert stability_ok(0.60, 0.50) is False
    rows = [
        {"spearman_full": 0.7, "pass": True, "stability_pass": True},
        {"spearman_full": 0.6, "pass": True, "stability_pass": False},
    ]
    v = panel_verdict(rows)
    assert v["pass"] is False
    assert v["n_pass"] == 1
