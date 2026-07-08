"""Tests for NIG loss identifiability analysis (no GPU)."""

from __future__ import annotations

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


def test_loss_philosophy_lists_three_modes() -> None:
    opts = loss_philosophy_options()
    assert set(opts["options"]) == {"PRIMARY_NIG", "PRIMARY_GAUSSIAN", "BLENDED"}


def test_g4a_p8_magnitude_gate_documents_relative_lift() -> None:
    g4a = g4a_p8_magnitude_gate()
    assert g4a["gate"] == "G4a"
    assert "relative_lift" in g4a["fix"]


def test_v3_vs_v6_summary_includes_g5() -> None:
    summary = analyze_v3_vs_v6_uncertainty_training()
    assert summary["gates"]["G5"]["gate"] == "G5"
    assert g5_epistemic_provenance_checklist()["distillation_path"]
