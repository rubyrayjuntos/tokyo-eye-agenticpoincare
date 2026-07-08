"""Tests for NIG loss identifiability analysis (no GPU)."""

from __future__ import annotations

from science.training.nig_identifiability import (
    analyze_nig_loss_coupling,
    decoupled_head_changes,
    head_output_quantity,
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
