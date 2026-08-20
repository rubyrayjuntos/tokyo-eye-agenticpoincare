"""Unit checks for |ρ−TAU| Steps 1–3 ceiling diagnostic."""

from __future__ import annotations

import numpy as np

from experiments.diagnostics.rho_tau_abs_dist_ceiling import build_report
from science.dtie.common.residue_features import TAU


def test_build_report_swap_lifts_when_tau_redundant():
    rng = np.random.default_rng(0)
    n = 2000
    # Skewed dehydron-like mix (near-TAU mass + high-ρ tail) — uniform ρ
    # understates the redundancy lift seen on Stage A-12.
    rho = np.concatenate(
        [
            rng.normal(8.0, 3.0, n // 2).clip(0.0, 40.0),
            rng.normal(20.0, 8.0, n // 2).clip(0.0, 40.0),
        ]
    )
    tau = (rho < TAU).astype(np.float64)
    ss = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.35, 0.10, 0.55])
    X = np.column_stack([rho, tau, ss])
    report = build_report(
        X,
        min_lift=0.10,
        redundancy_threshold=0.70,
        chem_mvp_zscore_enabled=False,
    )
    assert report["gate"]["verdict"] == "CLEAR_FOR_MATCHED_COLD_ARMS"
    assert report["step2_ceiling"]["lift_zscore"] >= 0.10
    assert report["step3_resolution"]["ss_redundancy_verdict"]["passes"] is True
    assert report["step3_resolution"]["resolution_beyond_binary_tau"][
        "adds_within_class_resolution"
    ]


def test_build_report_stop_when_abs_dist_is_constant_rewrite():
    # Degenerate: abs_dist constant ⇒ no resolution / no lift expected.
    n = 400
    rho = np.full(n, TAU + 5.0)
    tau = np.zeros(n)
    ss = np.zeros(n)
    # Force a second orthogonal channel so baseline isn't rank-0.
    ss[:200] = 1.0
    X = np.column_stack([rho, tau, ss])
    report = build_report(
        X,
        min_lift=0.10,
        redundancy_threshold=0.70,
        chem_mvp_zscore_enabled=False,
    )
    assert report["gate"]["verdict"] == "STOP_NO_PREDICTED_CEILING_LIFT"
    assert report["step2_ceiling"]["meaningful_predicted_lift"] is False
