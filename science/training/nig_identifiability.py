"""NIG evidential loss identifiability — analytical checks before Phase 4 retrain.

Vanilla Deep Evidential Regression can couple epistemic and aleatoric through the
loss surface (Meinert & Lavin; Bengs et al.), not only through shared weights.
Architecture splits alone may not decouple reported uncertainties.

See ``docs/audit/EVIDENTIAL_UNCERTAINTY.md`` § loss identifiability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NigLossStructure:
    """Structural terms in ``evidential_regression_loss`` (v5/v6)."""

    nll_terms: tuple[str, ...]
    regularizer: str
    regularizer_couples: tuple[str, ...]
    all_evidence_in_nll: tuple[str, ...]


NIG_LOSS_STRUCTURE = NigLossStructure(
    nll_terms=(
        "log(π/ν)",
        "−α·log(ω)",
        "(α+½)·log((y−μ)²ν + ω)",
        "lgamma(α)",
        "ω = 2β(1+ν)",
    ),
    regularizer="|y−μ| · (2ν + α)",
    regularizer_couples=("nu", "alpha"),
    all_evidence_in_nll=("mu", "nu", "alpha", "beta"),
)


def analyze_nig_loss_coupling() -> dict[str, Any]:
    """Return a structured audit of where NIG loss entangles evidence parameters."""
    s = NIG_LOSS_STRUCTURE
    return {
        "framework": "Deep Evidential Regression (Amini et al.)",
        "loss": "evidential_regression_loss in science/dtie/v5/gnn/model.py",
        "nll_entangles_all_evidence": True,
        "nll_evidence_params": list(s.all_evidence_in_nll),
        "regularizer": s.regularizer,
        "regularizer_couples": list(s.regularizer_couples),
        "identifiability_risk": (
            "The evidence regularizer multiplies |y−μ| by (2ν+α). Gradients on ν and α "
            "are tied to the same residual magnitude — coupling can persist even when "
            "DecoupledEvidentialHead uses separate MLP trunks, because both trunks are "
            "fit against the same NIG objective."
        ),
        "critique_refs": [
            "Meinert & Lavin — DER identifiability limits",
            "Bengs et al. — epistemic/aleatoric separation under NIG",
        ],
    }


def decoupled_head_changes() -> dict[str, Any]:
    """What DecoupledEvidentialHead changes vs coupled EvidentialHead."""
    return {
        "architecture": {
            "epi_trunk": "ν (exposure) → reported epistemic = (1/ν)·temp",
            "ale_trunk": "μ, α, β → reported aleatoric = β/(α−1) (log-clamped)",
            "shared_nig_loss": True,
        },
        "does_not_change": [
            "evidential_regression_loss (same NIG NLL + regularizer)",
            "Regularizer (2ν+α) coupling in loss surface",
        ],
        "phase4_training_signals_that_can_help": [
            "epi_ale_decorrelation_loss — penalizes r²(epi, ale) on head *outputs*",
            "epistemic_decoupling_loss — B-factor residual + SASA partial on epistemic",
            "epistemic_anticollapse — min std on epistemic",
            "max_probe_r_epi_ale_save gate (<0.70 in p4_head_decouple)",
        ],
        "phase4_presets_with_decorrelation": [
            "p4_head_decouple_phase_config (epi_ale_decorrelation_coeff=0.75)",
            "p4_head_decouple_decorr_only_phase_config (G3 ablation A — no B-factor/SASA)",
        ],
        "g3_circularity_risk": (
            "epistemic_decoupling_loss supervises epistemic toward B-factor/SASA. "
            "P8/P11 are not independent validation if they pass only with that loss. "
            "Run G3 A/B before trusting τ/OOD claims. See GNNV7_SUCCESS_CRITERIA.md G3."
        ),
        "open_question": (
            "Head split + epi_ale_decorrelation may reduce r(epi,ale) without fixing "
            "NIG identifiability. Confirm on holdout before trusting aleatoric semantics."
        ),
        "cheap_pre_retrain_check": (
            "Read evidential_regression_loss regularizer; verify Phase 4 enables "
            "epi_ale_decorrelation_coeff > 0 and/or epistemic_decoupling_coeff > 0 — "
            "not decoupled_uncertainty_heads flag alone."
        ),
    }


def head_output_quantity() -> dict[str, Any]:
    """Units for validation floors — head-reported vs canonical DER."""
    return {
        "reported_epistemic": "(1/ν) · epistemic_temp_scaling",
        "reported_aleatoric": "exp(clamp(log(β/(α−1))))",
        "canonical_der_epistemic": "β / (ν · (α−1))",
        "default_temp_scaling": 1.0,
        "floor_calibration_note": (
            "Corpus floors (1e-4 alive, 1e-3 non-degenerate) apply to *head-reported* "
            "epistemic and aleatoric tensors, not canonical DER. temp is a constructor "
            "hyperparameter (default 1.0, not learned). Passing epistemic std >> floor "
            "does not prove ν is informative if temp were raised — use nu_cv_floor in "
            "parallel (std(ν)/mean(ν) on evidence ν)."
        ),
        "scale_invariant_guard": "evidence_nu_cv = std(ν)/mean(ν) from forward evidence",
        "p9_ranking": "Sparsification ranks by reported epistemic — scale-invariant.",
    }
