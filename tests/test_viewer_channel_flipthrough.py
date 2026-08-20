"""Smoke tests for viewer_channel_flipthrough diagnostic."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.viewer_channel_flipthrough import (
    analyze,
    color_bin_fractions,
)


def test_color_bin_fractions_binary_tau() -> None:
    # 60% τ=1 → ~60% red under min-max continuous scale (same path as depth).
    v = np.array([0.0] * 40 + [1.0] * 60, dtype=np.float64)
    bins = color_bin_fractions(v)
    assert abs(bins["red_gt_0.66"] - 0.60) < 1e-9
    assert abs(bins["blue_lt_0.33"] - 0.40) < 1e-9


def test_analyze_focus_pairs_include_depth_tau() -> None:
    rng = np.random.default_rng(0)
    n = 40
    rho = rng.uniform(0, 30, size=n)
    tau = (rho < 13).astype(np.float64)
    depth = 0.5 * tau + 0.1 * rng.normal(size=n)
    points = [
        {
            "r": float(0.1 * depth[i]),
            "cone_depth": float(depth[i]),
            "tau": float(tau[i]),
            "rho": float(rho[i]),
            "physics_investigation": float(tau[i] * (1.0 - rho[i] / 30.0)),
            "investigation": float(rng.random()),
            "aleatoric": float(1.0 + 0.01 * rng.random()),
            "epistemic": float(1.0 + 0.01 * rng.random()),
        }
        for i in range(n)
    ]
    report = analyze(points)
    assert report["n_residues"] == n
    pairs = {(r["a"], r["b"]) for r in report["focus_pairs"]}
    assert ("cone_depth", "tau") in pairs
    assert ("rho", "tau") in pairs
    assert report["tau_vs_depth_visual"]["tau1_frac"] == float(tau.mean())
