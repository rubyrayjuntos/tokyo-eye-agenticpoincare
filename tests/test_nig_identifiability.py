"""Tests for NIG loss identifiability analysis (no GPU)."""

from __future__ import annotations

import numpy as np

from science.training.nig_identifiability import (
    g4a_p8_magnitude_gate,
    aleatoric_shaping_holdout_design,
    analyze_nig_loss_coupling,
    analyze_v3_vs_v6_uncertainty_training,
    decoupled_head_changes,
    g5_epistemic_provenance_checklist,
    head_output_quantity,
    loss_philosophy_options,
    v3_aleatoric_shaping_terms,
)


def test_regularizer_couples_nu_and_alpha() -> None:
    info = analyze_nig_loss_coupling()
    assert info["nll_entangles_all_evidence"] is True
    assert set(info["regularizer_couples"]) == {"nu", "alpha"}


def test_decoupled_head_still_uses_shared_nig_loss() -> None:
    changes = decoupled_head_changes()
    assert changes["architecture"]["shared_nig_loss"] is True
    assert "epi_ale_decorrelation_loss" in str(changes["phase4_training_signals_that_can_help"])


def test_floors_target_head_output_not_canonical_der() -> None:
    qty = head_output_quantity()
    assert qty["reported_epistemic"] == "(1/ν) · epistemic_temp_scaling"
    assert qty["canonical_der_epistemic"] != qty["reported_epistemic"]


def test_v3_shaping_documents_g4_circularity() -> None:
    v3 = v3_aleatoric_shaping_terms()
    assert "var_penalty" in v3["shaping_terms"]
    assert "g4_circularity_risk" in v3


def test_g4_holdout_design_requires_holdout_p8() -> None:
    g4 = aleatoric_shaping_holdout_design()
    assert g4["gate"] == "G4"
    assert "holdout" in g4["p8_eval"].lower()
    assert g4["pass_frozen"]["transfer_ratio_min"] == 0.70


def test_loss_philosophy_lists_three_modes() -> None:
    opts = loss_philosophy_options()
    assert set(opts["options"]) == {"PRIMARY_NIG", "PRIMARY_GAUSSIAN", "BLENDED"}


def test_g4a_p8_magnitude_gate_documents_relative_lift() -> None:
    g4a = g4a_p8_magnitude_gate()
    assert g4a["gate"] == "G4a"
    assert "relative_lift" in g4a["fix"]


def test_g5_provenance_flags_distilled_proxy() -> None:
    from science.training.evidential_validation import g5_epistemic_provenance_report

    n = 50
    rng = np.random.default_rng(0)
    sasa = rng.uniform(0.1, 0.9, n)
    teacher = 2.0 * sasa + rng.normal(0, 0.05, n)
    student = teacher + rng.normal(0, 0.03, n)
    rows = [
        {
            "epistemic": float(student[i]),
            "aleatoric": 0.1,
            "teacher_epistemic": float(teacher[i]),
            "rho": float(10 + i * 0.5),
            "sasa": float(sasa[i]),
        }
        for i in range(n)
    ]
    report = g5_epistemic_provenance_report(rows)
    assert report["distilled_proxy"] is True


def test_g5b_flags_rho_proxy_when_ood_collapses() -> None:
    from science.training.evidential_validation import (
        g5b_rho_feature_proxy_report,
        ood_epistemic_contrast_rho_residualized,
    )

    n = 80
    rho_in = np.linspace(5.0, 25.0, n)
    epi_in = 0.5 * rho_in + 1.0
    in_rows = [
        {"epistemic": float(epi_in[i]), "aleatoric": 0.1, "rho": float(rho_in[i])}
        for i in range(n)
    ]
    rho_ood = np.linspace(30.0, 50.0, 40)
    epi_ood = 0.5 * rho_ood + 1.0
    ood_rows = [
        {"epistemic": float(epi_ood[i]), "aleatoric": 0.1, "rho": float(rho_ood[i])}
        for i in range(40)
    ]
    residual = ood_epistemic_contrast_rho_residualized(in_rows, ood_rows)
    assert residual["ood_separation_collapsed_after_rho"] is True
    g5b = g5b_rho_feature_proxy_report(in_rows, ood_rows)
    assert g5b["rho_feature_proxy"] is True


def test_bootstrap_teacher_student_ci_straddles_threshold() -> None:
    from science.training.evidential_validation import bootstrap_structure_pooled_correlation

    rows_by_structure = {
        "A": [
            {"epistemic": 1.0 + i * 0.01, "teacher_epistemic": 1.0 + i * 0.01}
            for i in range(30)
        ],
        "B": [
            {"epistemic": 2.0 + i * 0.02, "teacher_epistemic": 1.6 + i * 0.015}
            for i in range(30)
        ],
        "C": [
            {"epistemic": 0.5 + i * 0.01, "teacher_epistemic": 0.4 + i * 0.008}
            for i in range(30)
        ],
    }
    boot = bootstrap_structure_pooled_correlation(
        rows_by_structure, n_bootstrap=200, seed=1
    )
    assert boot["ok"] is True
    assert boot["ci_low"] <= boot["ci_high"]


def test_sasa_sign_negative_interpretation() -> None:
    from science.training.evidential_validation import sasa_epistemic_sign_interpretation

    info = sasa_epistemic_sign_interpretation(-0.51)
    assert info["direction"] == "negative"
    assert info["proxy_flag_uses_abs_r"] is True


def test_v3_vs_v6_summary_includes_g5() -> None:
    summary = analyze_v3_vs_v6_uncertainty_training()
    assert summary["gates"]["G5"]["gate"] == "G5"
    assert g5_epistemic_provenance_checklist()["distillation_path"]
