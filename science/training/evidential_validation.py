"""Validation checks for Deep Evidential Regression decomposition (read-only).

Guards against the flat-std failure mode (decomposition that looks principled
but carries no per-input signal). See ``docs/audit/EVIDENTIAL_UNCERTAINTY.md``.

Property tests: ``tests/test_evidential_validation.py`` (P7–P11).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from science.dtie.common.residue_features import TAU
from science.training.uncertainty_diagnostics import (
    NODE_ALE_INFORMATIVE_FLOOR,
    NODE_ALE_STD_FLOOR,
    NODE_EPI_STD_FLOOR,
)

# Minimum corpus std for epistemic/aleatoric — applied to *head-reported* outputs
# (epistemic = (1/ν)·temp, aleatoric = clamped β/(α−1)), NOT canonical DER.
# Chosen order-of-magnitude vs flat-std≈0.0041 incident; see head_output_quantity().
EPISTEMIC_CORPUS_STD_FLOOR = 1e-3
ALEATORIC_CORPUS_STD_FLOOR = 1e-3
# Scale-invariant exposure guard: std(ν)/mean(ν) on evidence ν (temp-independent).
NU_CV_FLOOR = 0.02
# Aleatoric must exceed non-τ mean by this margin for τ-boundary elevation check.
TAU_ALE_LIFT_MIN = 0.0
# Corpus expansion: epistemic should shrink more than aleatoric on shared residues.
EXPANSION_EPI_SHRINK_RATIO_MIN = 1.5
# OOD policy — see docs/audit/EVIDENTIAL_UNCERTAINTY.md § P11
OOD_MANIFEST_PATH = "manifests/v6_corpus_stage_a_small_v1.json"
# Explicit holdout: dropped from Stage A v1 (enabled=false), CATH 3.10.20.10 β-grasp.
OOD_PINNED_STRUCTURES: tuple[tuple[str, str], ...] = (("1PGB", "A"),)
OOD_DISTANCE_BASIS = "manifest_enabled=false + CATH fold not in training fold set"
OOD_EPI_RATIO_MIN = 1.1

# S6 joint uncertainty save gate (Phase 4) — decoupling AND biophysical aleatoric.
S6_MAX_R_EPI_ALE = 0.70
S6_MIN_TAU_ALE_LIFT = 0.0  # strict: > 0

# G3 ablation variant names (pre-retrain circularity gate).
G3_VARIANT_DECORR_ONLY = "p4_head_decouple_decorr_only"
G3_VARIANT_FULL_P4 = "p4_head_decouple_plus_epistemic_decoupling"


def uncertainty_s6_joint_pass(
    rows: Sequence[dict[str, Any]],
    health: dict[str, float] | None = None,
    *,
    max_r_epi_ale: float = S6_MAX_R_EPI_ALE,
    min_tau_ale_lift: float = S6_MIN_TAU_ALE_LIFT,
) -> dict[str, Any]:
    """S6 — joint gate: low r(epi,ale) AND P8 τ-aleatoric elevation (not decorrelation alone)."""
    var = uncertainty_corpus_variance(rows)
    r_epi_ale = float("nan")
    if var["epistemic_std"] > 1e-12 and var["aleatoric_std"] > 1e-12:
        r_epi_ale = float(np.corrcoef(_col(rows, "epistemic"), _col(rows, "aleatoric"))[0, 1])
    tau = tau_boundary_aleatoric_elevation(rows, min_lift=min_tau_ale_lift)
    decouple_ok = math.isfinite(r_epi_ale) and abs(r_epi_ale) <= max_r_epi_ale
    p8_ok = bool(tau["ok"])
    ok = decouple_ok and p8_ok
    reasons: list[str] = []
    if not decouple_ok:
        reasons.append(f"r_epi_ale={r_epi_ale:.3f}>{max_r_epi_ale}")
    if not p8_ok:
        reasons.append(tau.get("reason", "p8_fail"))
    if health is not None:
        # Optional cross-check with epoch health scalars
        h_lift = health.get("node_aleatoric_tau_lift")
        if h_lift is not None and math.isfinite(float(h_lift)) and float(h_lift) <= min_tau_ale_lift:
            reasons.append(f"health_tau_lift={float(h_lift):.4f}")
    return {
        "ok": ok,
        "reason": "ok" if ok else ";".join(reasons),
        "r_epi_ale": r_epi_ale,
        "tau_boundary": tau,
        "decouple_ok": decouple_ok,
        "p8_ok": p8_ok,
    }


def g3_supervision_circularity_report(
    decorr_only_audit: dict[str, Any],
    full_supervision_audit: dict[str, Any],
) -> dict[str, Any]:
    """G3 — compare P8/P11 pass with vs without B-factor/SASA epistemic supervision.

    If P8/P11 pass only under ``epistemic_decoupling_coeff > 0``, uncertainty
    semantics are downstream of direct supervision (Goodhart risk), not emergent.
    """
    def _flags(audit: dict[str, Any]) -> dict[str, bool]:
        dec = audit.get("decomposition") or audit
        checks = dec.get("checks") or {}
        return {
            "p8": bool((dec.get("tau_boundary") or {}).get("ok", checks.get("tau_aleatoric_elevated"))),
            "p11": bool(audit.get("p11_ok", audit.get("ood_ok"))),
            "p9": bool(checks.get("sparsification_monotone")),
            "s6_joint": bool(audit.get("s6_joint", {}).get("ok")),
        }

    a = _flags(decorr_only_audit)
    b = _flags(full_supervision_audit)
    emergent = {
        k: a.get(k) and not b.get(k) for k in ("p8", "p11", "p9")
    }
    supervised_only = {
        k: (not a.get(k)) and b.get(k) for k in ("p8", "p11", "p9")
    }
    return {
        "decorrelation_only": a,
        "with_epistemic_decoupling": b,
        "emergent_under_decorr_only": emergent,
        "supervision_required_for_pass": supervised_only,
        "g3_pass": not any(supervised_only.values()),
        "interpretation": (
            "g3_fail: P8/P11 pass only with B-factor/SASA supervision — not independent validation"
            if any(supervised_only.values())
            else "g3_ok: no supervised-only pass detected (compare still required)"
        ),
    }


def _evidence_nu_col(rows: Sequence[dict[str, Any]]) -> np.ndarray | None:
    if not rows or "evidence_nu" not in rows[0]:
        return None
    vals = [r["evidence_nu"] for r in rows if r.get("evidence_nu") is not None]
    if len(vals) < 3:
        return None
    return np.array(vals, dtype=np.float64)


def exposure_nu_coefficient_of_variation(rows: Sequence[dict[str, Any]]) -> float:
    """std(ν)/mean(ν) — scale-invariant check on evidence ν (not temp-scaled epistemic)."""
    nu = _evidence_nu_col(rows)
    if nu is None or nu.size < 3:
        return float("nan")
    mean = float(np.mean(nu))
    if mean <= 1e-12:
        return float("nan")
    return float(np.std(nu) / mean)


def exposure_non_degenerate(
    rows: Sequence[dict[str, Any]],
    *,
    nu_cv_floor: float = NU_CV_FLOOR,
) -> tuple[bool, str]:
    """ν spread independent of epistemic_temp_scaling."""
    cv = exposure_nu_coefficient_of_variation(rows)
    if not math.isfinite(cv):
        return False, "evidence_nu_missing_or_flat"
    if cv < nu_cv_floor:
        return False, f"nu_cv={cv:.2e}<{nu_cv_floor}"
    return True, "ok"


def _col(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def uncertainty_corpus_variance(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Variance of uncertainty outputs across residues — primary flat-std guard."""
    if not rows:
        return {
            "epistemic_std": float("nan"),
            "aleatoric_std": float("nan"),
            "epistemic_cv": float("nan"),
            "aleatoric_cv": float("nan"),
        }
    epi = _col(rows, "epistemic")
    ale = _col(rows, "aleatoric")
    epi_std = float(np.std(epi))
    ale_std = float(np.std(ale))
    epi_mean = float(np.mean(epi))
    ale_mean = float(np.mean(ale))
    return {
        "epistemic_std": epi_std,
        "aleatoric_std": ale_std,
        "epistemic_mean": epi_mean,
        "aleatoric_mean": ale_mean,
        "epistemic_cv": epi_std / max(epi_mean, 1e-12),
        "aleatoric_cv": ale_std / max(ale_mean, 1e-12),
    }


def epistemic_var_non_degenerate(
    rows: Sequence[dict[str, Any]],
    *,
    std_floor: float = EPISTEMIC_CORPUS_STD_FLOOR,
) -> tuple[bool, str]:
    """P7 — epistemic spread across inputs exceeds floor (not flat everywhere)."""
    if not rows:
        return False, "empty"
    std = uncertainty_corpus_variance(rows)["epistemic_std"]
    if not math.isfinite(std):
        return False, "epistemic_non_finite"
    if std < std_floor:
        return False, f"epistemic_flat_std={std:.2e}"
    return True, "ok"


def aleatoric_var_non_degenerate(
    rows: Sequence[dict[str, Any]],
    *,
    std_floor: float = ALEATORIC_CORPUS_STD_FLOOR,
) -> tuple[bool, str]:
    """Aleatoric spread across inputs exceeds floor."""
    if not rows:
        return False, "empty"
    std = uncertainty_corpus_variance(rows)["aleatoric_std"]
    if not math.isfinite(std):
        return False, "aleatoric_non_finite"
    if std < std_floor:
        return False, f"aleatoric_flat_std={std:.2e}"
    return True, "ok"


def tau_boundary_aleatoric_elevation(
    rows: Sequence[dict[str, Any]],
    *,
    rho_band: float = 1.0,
    min_lift: float = TAU_ALE_LIFT_MIN,
) -> dict[str, Any]:
    """P8 domain check — aleatoric higher near ρ≈TAU (biophysical ambiguity)."""
    if not rows:
        return {"ok": False, "reason": "empty", "aleatoric_tau_lift": float("nan")}
    rho = _col(rows, "rho")
    ale = _col(rows, "aleatoric")
    tau_mask = np.abs(rho - TAU) <= rho_band
    if not tau_mask.any() or not (~tau_mask).any():
        return {
            "ok": False,
            "reason": "insufficient_tau_strata",
            "tau_boundary_n": int(tau_mask.sum()),
            "aleatoric_tau_lift": float("nan"),
        }
    ale_tau = float(np.mean(ale[tau_mask]))
    ale_non = float(np.mean(ale[~tau_mask]))
    lift = ale_tau - ale_non
    ok = lift > min_lift
    return {
        "ok": ok,
        "reason": "ok" if ok else f"aleatoric_tau_lift={lift:.4f}<={min_lift}",
        "tau_boundary_n": int(tau_mask.sum()),
        "non_tau_n": int((~tau_mask).sum()),
        "aleatoric_mean_tau_boundary": ale_tau,
        "aleatoric_mean_non_tau_boundary": ale_non,
        "aleatoric_tau_lift": lift,
    }


def sparsification_curve(
    rows: Sequence[dict[str, Any]],
    *,
    sort_key: str = "epistemic",
    error_key: str = "prediction_error",
    n_points: int = 11,
) -> list[dict[str, float]]:
    """Rank by ``sort_key``, progressively drop highest-uncertainty fraction."""
    if not rows:
        return []
    errors = _col(rows, error_key)
    sort_vals = _col(rows, sort_key)
    if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(sort_vals)):
        return []
    order = np.argsort(sort_vals)[::-1]  # highest uncertainty first
    n = len(order)
    fractions = np.linspace(0.0, 1.0, n_points)
    curve: list[dict[str, float]] = []
    for frac in fractions:
        k_remove = int(round(frac * n))
        keep = order[k_remove:]
        if keep.size == 0:
            mean_err = float("nan")
        else:
            mean_err = float(np.mean(errors[keep]))
        curve.append(
            {
                "removed_fraction": float(frac),
                "kept_fraction": float(1.0 - frac),
                "n_kept": int(keep.size),
                "mean_error": mean_err,
            }
        )
    return curve


def sparsification_error_monotone(
    curve: Sequence[dict[str, float]],
    *,
    min_drop: float = 0.0,
) -> tuple[bool, str]:
    """P9 — removing high-epistemic residues should not increase mean error."""
    if len(curve) < 2:
        return False, "curve_too_short"
    errors = [c["mean_error"] for c in curve if math.isfinite(c["mean_error"])]
    if len(errors) < 2:
        return False, "non_finite_errors"
    # As we remove more high-epistemic points, mean error on remainder should fall.
    for i in range(1, len(errors)):
        if errors[i] > errors[i - 1] + min_drop:
            return False, f"error_increased_at_step_{i}"
    if errors[-1] >= errors[0] - min_drop:
        return False, "no_error_improvement"
    return True, "ok"


def corpus_expansion_sensitivity(
    baseline_rows: Sequence[dict[str, Any]],
    expanded_rows: Sequence[dict[str, Any]],
    *,
    residue_key: str = "residue_id",
    epi_shrink_ratio_min: float = EXPANSION_EPI_SHRINK_RATIO_MIN,
) -> dict[str, Any]:
    """P10 — epistemic shrinks on shared residues after corpus expansion; aleatoric stable."""
    base_map = {str(r[residue_key]): r for r in baseline_rows}
    exp_map = {str(r[residue_key]): r for r in expanded_rows}
    shared = sorted(set(base_map) & set(exp_map))
    if len(shared) < 3:
        return {
            "ok": False,
            "reason": "insufficient_shared_residues",
            "n_shared": len(shared),
        }
    epi_base = np.array([float(base_map[k]["epistemic"]) for k in shared])
    epi_exp = np.array([float(exp_map[k]["epistemic"]) for k in shared])
    ale_base = np.array([float(base_map[k]["aleatoric"]) for k in shared])
    ale_exp = np.array([float(exp_map[k]["aleatoric"]) for k in shared])
    epi_delta = float(np.mean(epi_exp - epi_base))
    ale_delta = float(np.mean(ale_exp - ale_base))
    epi_shrink = float(np.mean(epi_base - epi_exp))
    ale_move = float(np.mean(np.abs(ale_exp - ale_base)))
    # Epistemic should drop (positive shrink); aleatoric movement should be smaller.
    ratio = epi_shrink / max(ale_move, 1e-12)
    ok = epi_shrink > 0 and ratio >= epi_shrink_ratio_min
    return {
        "ok": ok,
        "reason": "ok" if ok else f"epi_shrink_ratio={ratio:.2f}<{epi_shrink_ratio_min}",
        "n_shared": len(shared),
        "mean_epistemic_delta": epi_delta,
        "mean_aleatoric_delta": ale_delta,
        "epistemic_shrink": epi_shrink,
        "aleatoric_abs_move": ale_move,
        "epi_shrink_to_ale_move_ratio": ratio,
    }


def out_of_corpus_epistemic_contrast(
    in_corpus_rows: Sequence[dict[str, Any]],
    ood_rows: Sequence[dict[str, Any]],
    *,
    ratio_min: float = OOD_EPI_RATIO_MIN,
) -> dict[str, Any]:
    """P11 — epistemic spikes on structurally distant proteins, not uniformly."""
    if not in_corpus_rows or not ood_rows:
        return {"ok": False, "reason": "empty"}
    epi_in = float(np.mean(_col(in_corpus_rows, "epistemic")))
    epi_ood = float(np.mean(_col(ood_rows, "epistemic")))
    ale_in = float(np.mean(_col(in_corpus_rows, "aleatoric")))
    ale_ood = float(np.mean(_col(ood_rows, "aleatoric")))
    epi_ratio = epi_ood / max(epi_in, 1e-12)
    ale_ratio = ale_ood / max(ale_in, 1e-12)
    # OOD epistemic elevation should exceed aleatoric elevation (coverage gap, not physics).
    ok = epi_ratio >= ratio_min and epi_ratio > ale_ratio
    return {
        "ok": ok,
        "reason": "ok" if ok else f"epi_ratio={epi_ratio:.2f}",
        "epistemic_mean_in_corpus": epi_in,
        "epistemic_mean_ood": epi_ood,
        "aleatoric_mean_in_corpus": ale_in,
        "aleatoric_mean_ood": ale_ood,
        "epistemic_ood_ratio": epi_ratio,
        "aleatoric_ood_ratio": ale_ratio,
    }


def assess_evidential_decomposition(
    rows: Sequence[dict[str, Any]],
    *,
    decoupled_head: bool = False,
    require_tau_lift: bool = True,
) -> dict[str, Any]:
    """Composite validation report for a single structure or merged corpus rows."""
    var = uncertainty_corpus_variance(rows)
    epi_ok, epi_reason = epistemic_var_non_degenerate(rows)
    ale_ok, ale_reason = aleatoric_var_non_degenerate(rows)
    nu_ok, nu_reason = exposure_non_degenerate(rows)
    nu_cv = exposure_nu_coefficient_of_variation(rows)
    tau = tau_boundary_aleatoric_elevation(rows)
    r_epi_ale = float("nan")
    if var["epistemic_std"] > 1e-12 and var["aleatoric_std"] > 1e-12:
        r_epi_ale = float(np.corrcoef(_col(rows, "epistemic"), _col(rows, "aleatoric"))[0, 1])

    sp_curve = sparsification_curve(rows)
    sp_ok, sp_reason = sparsification_error_monotone(sp_curve) if sp_curve else (False, "no_curve")

    decoupled_ok = (not decoupled_head) or (math.isfinite(r_epi_ale) and abs(r_epi_ale) < 0.85)
    informative_ale = var["aleatoric_std"] >= NODE_ALE_INFORMATIVE_FLOOR

    checks = {
        "epistemic_non_degenerate": epi_ok,
        "exposure_nu_non_degenerate": nu_ok,
        "aleatoric_non_degenerate": ale_ok,
        "aleatoric_informative": informative_ale,
        "epi_ale_low_correlation": decoupled_ok,
        "tau_aleatoric_elevated": tau["ok"] if require_tau_lift else True,
        "sparsification_monotone": sp_ok,
    }
    ok = all(checks.values())
    return {
        "ok": ok,
        "checks": checks,
        "variance": var,
        "evidence_nu_cv": nu_cv,
        "head_epistemic_quantity": "(1/nu)*epistemic_temp_scaling",
        "r_epi_ale": r_epi_ale,
        "epistemic_reason": epi_reason,
        "aleatoric_reason": ale_reason,
        "exposure_nu_reason": nu_reason,
        "tau_boundary": tau,
        "sparsification_reason": sp_reason,
        "sparsification_curve": sp_curve,
        "alive_floors": {
            "epistemic_std_floor": EPISTEMIC_CORPUS_STD_FLOOR,
            "aleatoric_std_floor": ALEATORIC_CORPUS_STD_FLOOR,
            "nu_cv_floor": NU_CV_FLOOR,
            "epistemic_alive_floor": NODE_EPI_STD_FLOOR,
            "aleatoric_alive_floor": NODE_ALE_STD_FLOOR,
            "aleatoric_informative_floor": NODE_ALE_INFORMATIVE_FLOOR,
        },
    }
