"""NIG evidential loss identifiability — analytical checks before Phase 4 retrain.

Vanilla Deep Evidential Regression can couple epistemic and aleatoric through the
loss surface (Meinert & Lavin; Bengs et al.), not only through shared weights.
Architecture splits alone may not decouple reported uncertainties.

See ``docs/audit/EVIDENTIAL_UNCERTAINTY.md`` § loss identifiability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from science.training.evidential_validation import (
    TAU_ALE_RELATIVE_LIFT_MIN,
)
from science.training.uncertainty_diagnostics import NODE_ALE_INFORMATIVE_FLOOR


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


def v3_aleatoric_shaping_terms() -> dict[str, Any]:
    """v3 gosp_loss terms that v6 dropped — separate mechanism from G1 NIG coupling."""
    return {
        "source": "science/dtie/v3/gnn/model.py gosp_loss",
        "tokyo_loss_mode_shipped": "gaussian_likelihood (v2_bridge_epoch_014.pt)",
        "shaping_terms": {
            "var_penalty": "(1 - target_dehydron) * aleatoric — mean over regular residues",
            "aleatoric_hinge": "(1 - target_dehydron) * relu(aleatoric - target)^2",
            "regularity_mask": "1 - target_dehydron (ρ/TAU label — same signal P8 uses)",
        },
        "shipped_weights": {
            "w_var_penalty": 2.8,
            "w_aleatoric_hinge": 4.8,
            "aleatoric_hinge_target": 1.0,
        },
        "aleatoric_clamp_log": "[-5.0, 0.3] → exp(0.3) ≈ 1.35 ceiling on reported aleatoric",
        "epistemic_temp_scaling_shipped": 2.8,
        "g4_circularity_risk": (
            "Training aleatoric with dehydron-mask shaping then validating P8 on the same "
            "residues proves hinge convergence, not discovered τ-boundary ambiguity. "
            "Require held-out residues excluded from var_penalty/hinge before P8 eval."
        ),
        "not_a_substitute_for_g1": (
            "Recovering v3 shaping fixes missing supervision; it does not remove NIG "
            "regularizer coupling (G1). Retrain must address both."
        ),
    }


def g4a_p8_magnitude_gate() -> dict[str, Any]:
    """G4a — P8 must be magnitude-relative + informative aleatoric (prerequisite for G4)."""
    return {
        "gate": "G4a",
        "problem": (
            "P8 with min_lift=0 passes on flat aleatoric when τ-strata show tiny "
            "directional noise — same failure shape as flat-std at sign level."
        ),
        "fix": {
            "relative_lift": "(ale_tau - ale_non) / std(ale) >= TAU_ALE_RELATIVE_LIFT_MIN",
            "informative_std": "std(ale) >= NODE_ALE_INFORMATIVE_FLOOR before P8 ok",
            "strata": "continuous |ρ - TAU| <= band (not tau_flag)",
        },
        "constants": {
            "TAU_ALE_RELATIVE_LIFT_MIN": TAU_ALE_RELATIVE_LIFT_MIN,
            "NODE_ALE_INFORMATIVE_FLOOR": NODE_ALE_INFORMATIVE_FLOOR,
        },
        "blocks": "G4 holdout contrast until P8 is meaningful",
        "re_evaluate": "Re-run G3 A/B P8 interpretation through G4a-hardened P8",
    }


def aleatoric_shaping_holdout_design(
    *,
    holdout_fraction: float = 0.20,
    stratify_by_dehydron: bool = True,
) -> dict[str, Any]:
    """G4 — design for dehydron-mask shaping without circular P8 validation."""
    return {
        "gate": "G4",
        "holdout_fraction": holdout_fraction,
        "stratify_by_dehydron": stratify_by_dehydron,
        "train_mask": "residues receiving var_penalty and aleatoric_hinge",
        "holdout_mask": "residues excluded from shaping loss; still in NIG/gaussian primary loss",
        "p8_eval": "tau_boundary_aleatoric_elevation(rows[holdout_mask]) only",
        "pass": "holdout P8 lift > 0 and holdout aleatoric_std ≥ informative floor",
        "fail": "full-corpus P8 pass but holdout P8 fail → mask memorization",
        "ablation": {
            "A": "no var_penalty/hinge (or baseline P4)",
            "B": "shaping on train mask only",
            "compare": "holdout P8 B vs A",
        },
    }


def g5_epistemic_provenance_checklist() -> dict[str, Any]:
    """G5 — is v6 P7 creditable to native learning vs v3 teacher proxy?"""
    return {
        "gate": "G5",
        "distillation_path": "experiments/training/v6/v2_teacher.py — depth + epistemic only",
        "teacher_checkpoint": "science/dtie/v3/checkpoints/v2_bridge_epoch_014.pt",
        "teacher_epistemic_risk": "v3 epistemic ≈ SASA proxy (SASA in x[:,3], coupled head, temp=2.8)",
        "metrics_to_compute": [
            "r(epi, SASA) marginal on Stage A corpus",
            "r(epi, SASA | ρ) partial (or residualized)",
            "r(student_epi, teacher_epi) per structure",
            "r(student_epi, teacher_epi) on proteins outside teacher precompute cache",
        ],
        "flag_distilled_proxy_if": (
            "marginal r(epi,SASA) ≥ 0.85 AND teacher-student r ≥ 0.80 on same graphs"
        ),
        "pass_interpretation": (
            "partial r(epi,SASA|ρ) materially below marginal, OR holdout structures "
            "show student epistemic spread not explained by teacher alignment"
        ),
        "does_not_block_retrain": True,
        "blocks_claim": "emergent DER epistemic without provenance caveat",
        "s6_credit_rule_if_distilled_proxy": (
            "S1 (routing), S4 (hyperbolic MP), S5 (edge telemetry) count at full weight. "
            "S6 and any epistemic-based production claim carry asterisk: "
            "'informative but largely SASA-proxy inherited via v3 teacher distillation — "
            "not v6-native DER exposure discovery.' Do not count S6 epistemic leg toward "
            "production uncertainty gates without native-learning corroboration."
        ),
    }


def loss_philosophy_options() -> dict[str, Any]:
    """Explicit decision required before p4_v3_aleatoric_recovery."""
    return {
        "decision_required_before": "p4_v3_aleatoric_recovery phase config",
        "options": {
            "PRIMARY_NIG": {
                "primary": "evidential_regression_loss",
                "aleatoric_semantics": "NIG-derived β/(α−1) (clamped)",
                "p9_p11_valid_under": "DER interpretation (subject to G1/G3/G4)",
                "v6_default": True,
            },
            "PRIMARY_GAUSSIAN": {
                "primary": "0.5*log(ale) + 0.5*(y-μ)²/ale on ρ (v3 tokyo_loss_mode)",
                "aleatoric_semantics": "heteroscedastic variance — not NIG-evidential",
                "p9_p11_valid_under": "re-scope tests; aleatoric is supervised variance",
                "v3_teacher": True,
            },
            "BLENDED": {
                "primary": "evidential_regression_loss",
                "auxiliary": ["gaussian_likelihood on ρ", "var_penalty + hinge with G4 holdout"],
                "aleatoric_semantics": "hybrid — document canonical reporting quantity",
                "requires": ["G4 holdout pass", "explicit MLflow tag loss_philosophy=blended"],
            },
        },
        "recommendation": (
            "Do not port v3 shaping under PRIMARY_NIG without G4 holdout. If "
            "gaussian_likelihood becomes co-primary, update EVIDENTIAL_UNCERTAINTY.md "
            "and P9/P11 interpretation — different epistemic story than pure DER."
        ),
    }


def analyze_v3_vs_v6_uncertainty_training() -> dict[str, Any]:
    """Summary for audit docs — two mechanisms + philosophy fork."""
    return {
        "g1_nig_coupling": analyze_nig_loss_coupling(),
        "v3_shaping": v3_aleatoric_shaping_terms(),
        "v6_dropped": [
            "gaussian_likelihood primary mode",
            "var_penalty on regularity_mask",
            "aleatoric_hinge on regularity_mask",
            "epistemic_temp_scaling (constructor hyperparam in v3, default 1.0 in v6)",
        ],
        "v6_added": [
            "DecoupledEvidentialHead (optional)",
            "epi_ale_decorrelation_loss",
            "epistemic_decoupling_loss (B-factor/SASA)",
            "S6 joint save gate with P8",
        ],
        "gates": {
            "G3": "epistemic B-factor/SASA circularity (A/B decorr vs full)",
            "G4a": g4a_p8_magnitude_gate(),
            "G4": aleatoric_shaping_holdout_design(),
            "G5": g5_epistemic_provenance_checklist(),
        },
        "loss_philosophy": loss_philosophy_options(),
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
