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
# Aleatoric must exceed non-τ mean by this absolute margin (legacy; insufficient alone).
TAU_ALE_LIFT_MIN = 0.0
# G4a — magnitude-relative τ lift: (ale_τ − ale_baseline) / std(ale). Analogous to nu_cv for epi.
TAU_ALE_RELATIVE_LIFT_MIN = 0.20
# P8 requires informative aleatoric spread before τ-boundary claims count.
TAU_ALE_REQUIRE_INFORMATIVE_STD = True
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
S6_MIN_TAU_ALE_LIFT = TAU_ALE_LIFT_MIN
S6_MIN_TAU_ALE_RELATIVE_LIFT = TAU_ALE_RELATIVE_LIFT_MIN

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


# G5 epistemic provenance — v3 teacher proxy vs v6-native (see GNNV7_SUCCESS_CRITERIA.md G5).
G5_EPI_SASA_MARGINAL_PROXY = 0.85
G5_EPI_RHO_MARGINAL_PROXY = 0.85  # G5b — ρ is a direct input feature
G5_TEACHER_STUDENT_R_PROXY = 0.80
G5_PARTIAL_DROP_MIN = 0.15  # marginal − partial r(epi,SASA) to suggest non-proxy signal
G5_BOOTSTRAP_N_DEFAULT = 1000
G5_BOOTSTRAP_CI = (0.025, 0.975)

# G4 — holdout P8 pass (frozen before v3 shaping training runs).
G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN = 0.70  # holdout_rel / full_rel
G4_TRANSFER_RATIO_FULL_REL_LIFT_MIN = TAU_ALE_RELATIVE_LIFT_MIN  # full corpus rel lift floor
G4_ALE_RHO_MARGINAL_PROXY = 0.85

# G4 w_var_penalty sweep — population separation (frozen before coefficient probes).
G4_POPULATION_GAP_STD_MULT_MIN = 2.0  # |μ_dehyd − μ_regular| / σ_pooled
G4_POPULATION_SEPARATION_WVP_WEIGHTS = (2.8, 1.0, 0.3)  # probe grid (frozen)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """OLS residual y − E[y|z] (with intercept)."""
    z1 = np.column_stack([np.ones(len(z)), z])
    coef, _, _, _ = np.linalg.lstsq(z1, y, rcond=None)
    return y - z1 @ coef


def partial_correlation_epi_sasa_given_rho(rows: Sequence[dict[str, Any]]) -> float:
    """r(epi, SASA | ρ) via residualization on governed features."""
    epi = _col(rows, "epistemic")
    rho = _col(rows, "rho")
    sasa_vals = [r.get("sasa") for r in rows]
    if any(v is None or not math.isfinite(float(v)) for v in sasa_vals):
        return float("nan")
    sasa = np.array([float(v) for v in sasa_vals], dtype=np.float64)
    epi_r = _residualize(epi, rho)
    sasa_r = _residualize(sasa, rho)
    return _pearson_r(epi_r, sasa_r)


def _fit_affine_residualize(
    y: np.ndarray, z: np.ndarray, *, fit_y: np.ndarray, fit_z: np.ndarray
) -> np.ndarray:
    """OLS residual y − E[y|z] using coefficients fit on (fit_y, fit_z)."""
    z1_fit = np.column_stack([np.ones(len(fit_z)), fit_z])
    coef, _, _, _ = np.linalg.lstsq(z1_fit, fit_y, rcond=None)
    z1 = np.column_stack([np.ones(len(z)), z])
    return y - z1 @ coef


def partial_correlation_epi_teacher_given_rho(rows: Sequence[dict[str, Any]]) -> float:
    """r(epi_student, epi_teacher | ρ) — distillation alignment beyond ρ."""
    teacher_col = [r.get("teacher_epistemic") for r in rows]
    if any(v is None or not math.isfinite(float(v)) for v in teacher_col):
        return float("nan")
    epi = _col(rows, "epistemic")
    rho = _col(rows, "rho")
    teacher = np.array([float(v) for v in teacher_col], dtype=np.float64)
    epi_r = _residualize(epi, rho)
    teacher_r = _residualize(teacher, rho)
    return _pearson_r(epi_r, teacher_r)


def sasa_epistemic_sign_interpretation(r_epi_sasa: float) -> dict[str, Any]:
    """Mechanistic read on marginal r(epi, SASA) — symmetric |r| used for proxy flag."""
    if not math.isfinite(r_epi_sasa):
        return {"direction": "missing", "note": "SASA unavailable on rows"}
    direction = "positive" if r_epi_sasa > 0.05 else ("negative" if r_epi_sasa < -0.05 else "near_zero")
    notes = {
        "positive": (
            "Higher SASA associates with higher epistemic — consistent with v3 teacher "
            "SASA-proxy risk when |r| is large."
        ),
        "negative": (
            "Higher SASA associates with lower epistemic — exposed/corpus-familiar "
            "residues carry lower model uncertainty; opposite of v3 SASA pass-through "
            "(which would be strongly positive). Strong |r| still implies SASA coupling "
            "even when sign is negative — proxy flag uses |r|, not sign alone."
        ),
        "near_zero": "Marginal SASA coupling weak at corpus level.",
    }
    return {
        "r_epi_sasa_marginal": r_epi_sasa,
        "direction": direction,
        "note": notes[direction],
        "proxy_flag_uses_abs_r": True,
    }


def ood_epistemic_contrast_rho_residualized(
    in_corpus_rows: Sequence[dict[str, Any]],
    ood_rows: Sequence[dict[str, Any]],
    *,
    ratio_min: float = OOD_EPI_RATIO_MIN,
) -> dict[str, Any]:
    """P11-style contrast on epistemic residualized w.r.t. ρ (fit on in-corpus only)."""
    if not in_corpus_rows or not ood_rows:
        return {"ok": False, "reason": "empty"}
    rho_in = _col(in_corpus_rows, "rho")
    epi_in = _col(in_corpus_rows, "epistemic")
    rho_ood = _col(ood_rows, "rho")
    epi_ood = _col(ood_rows, "epistemic")
    epi_res_in = _fit_affine_residualize(
        epi_in, rho_in, fit_y=epi_in, fit_z=rho_in
    )
    epi_res_ood = _fit_affine_residualize(
        epi_ood, rho_ood, fit_y=epi_in, fit_z=rho_in
    )
    mean_in = float(np.mean(epi_in))
    mean_ood = float(np.mean(epi_ood))
    mean_res_in = float(np.mean(epi_res_in))
    mean_res_ood = float(np.mean(epi_res_ood))
    mag_res_in = float(np.mean(np.abs(epi_res_in)))
    mag_res_ood = float(np.mean(np.abs(epi_res_ood)))
    raw_ratio = mean_ood / max(mean_in, 1e-12)
    residual_ratio = mag_res_ood / max(mag_res_in, 1e-12)
    raw_elevated = raw_ratio >= ratio_min
    residual_elevated = residual_ratio >= ratio_min
    collapsed = bool(raw_elevated and not residual_elevated)
    residual_ok = residual_elevated and (not raw_elevated or residual_ratio > raw_ratio * 0.85)
    return {
        "ok": residual_ok,
        "reason": "ok" if residual_ok else f"residual_epi_mag_ratio={residual_ratio:.2f}",
        "epistemic_mean_in_corpus": mean_in,
        "epistemic_mean_ood": mean_ood,
        "epistemic_mean_in_corpus_rho_residual": mean_res_in,
        "epistemic_mean_ood_rho_residual": mean_res_ood,
        "epistemic_mag_in_corpus_rho_residual": mag_res_in,
        "epistemic_mag_ood_rho_residual": mag_res_ood,
        "epistemic_ood_ratio_raw": raw_ratio,
        "epistemic_ood_mag_ratio_rho_residual": residual_ratio,
        "epistemic_ood_ratio_rho_residual": residual_ratio,
        "raw_ood_elevated": raw_elevated,
        "ood_separation_collapsed_after_rho": collapsed,
        "ratio_min": ratio_min,
    }


def bootstrap_structure_pooled_correlation(
    rows_by_structure: dict[str, Sequence[dict[str, Any]]],
    *,
    x_key: str = "epistemic",
    y_key: str = "teacher_epistemic",
    n_bootstrap: int = G5_BOOTSTRAP_N_DEFAULT,
    seed: int = 0,
    ci_quantiles: tuple[float, float] = G5_BOOTSTRAP_CI,
    threshold: float = G5_TEACHER_STUDENT_R_PROXY,
) -> dict[str, Any]:
    """Bootstrap r(x,y) by resampling proteins with replacement (pooled residues)."""
    structure_ids = sorted(rows_by_structure.keys())
    if len(structure_ids) < 2:
        return {
            "ok": False,
            "reason": "insufficient_structures",
            "n_structures": len(structure_ids),
        }
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(structure_ids, size=len(structure_ids), replace=True)
        xs: list[float] = []
        ys: list[float] = []
        for sid in drawn:
            for row in rows_by_structure[sid]:
                xv, yv = row.get(x_key), row.get(y_key)
                if xv is None or yv is None:
                    continue
                if not math.isfinite(float(xv)) or not math.isfinite(float(yv)):
                    continue
                xs.append(float(xv))
                ys.append(float(yv))
        if len(xs) >= 3:
            r = _pearson_r(np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64))
            if math.isfinite(r):
                samples.append(r)
    if len(samples) < 10:
        return {"ok": False, "reason": "bootstrap_too_few_finite_samples", "n_samples": len(samples)}
    arr = np.array(samples, dtype=np.float64)
    ci_low, ci_high = float(np.quantile(arr, ci_quantiles[0])), float(
        np.quantile(arr, ci_quantiles[1])
    )
    point = float(np.mean(arr))
    straddles = ci_low < threshold < ci_high
    if straddles:
        interpretation = (
            f"borderline: bootstrap CI [{ci_low:.3f}, {ci_high:.3f}] straddles "
            f"threshold {threshold:.2f} — point estimate not a robust binary"
        )
    elif ci_high < threshold:
        interpretation = f"robust_non_trigger: CI upper {ci_high:.3f} < {threshold:.2f}"
    elif ci_low >= threshold:
        interpretation = f"robust_trigger: CI lower {ci_low:.3f} ≥ {threshold:.2f}"
    else:
        interpretation = f"below_threshold: CI [{ci_low:.3f}, {ci_high:.3f}]"
    return {
        "ok": True,
        "point_estimate_bootstrap_mean": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_quantiles": list(ci_quantiles),
        "n_bootstrap": n_bootstrap,
        "n_finite_samples": len(samples),
        "n_structures": len(structure_ids),
        "threshold": threshold,
        "ci_straddles_threshold": straddles,
        "interpretation": interpretation,
    }


def g5b_rho_feature_proxy_report(
    corpus_rows: Sequence[dict[str, Any]],
    ood_rows: Sequence[dict[str, Any]] | None = None,
    *,
    rho_marginal_threshold: float = G5_EPI_RHO_MARGINAL_PROXY,
    ood_ratio_min: float = OOD_EPI_RATIO_MIN,
) -> dict[str, Any]:
    """G5b — is epistemic a monotonic ρ reparameterization vs independent novelty (OOD)?"""
    if not corpus_rows:
        return {"gate": "G5b", "ok": False, "reason": "empty", "rho_feature_proxy": False}
    epi = _col(corpus_rows, "epistemic")
    rho = _col(corpus_rows, "rho")
    r_epi_rho = _pearson_r(epi, rho)
    r_epi_teacher_partial = partial_correlation_epi_teacher_given_rho(corpus_rows)

    ood_raw: dict[str, Any] | None = None
    ood_residual: dict[str, Any] | None = None
    if ood_rows:
        ood_raw = out_of_corpus_epistemic_contrast(corpus_rows, ood_rows, ratio_min=ood_ratio_min)
        ood_residual = ood_epistemic_contrast_rho_residualized(
            corpus_rows, ood_rows, ratio_min=ood_ratio_min
        )

    collapsed = bool(
        ood_residual is not None and ood_residual.get("ood_separation_collapsed_after_rho")
    )
    rho_dominant_marginal = (
        math.isfinite(r_epi_rho) and abs(r_epi_rho) >= rho_marginal_threshold
    )
    rho_feature_proxy = rho_dominant_marginal and collapsed
    native_novelty = bool(
        ood_residual is not None
        and ood_residual.get("ok")
    ) or bool(
        ood_residual is not None
        and not collapsed
        and ood_raw is not None
        and ood_raw.get("ok")
    )

    if rho_feature_proxy:
        interpretation = (
            "rho_feature_proxy: |r(epi,ρ)| high and raw OOD epistemic elevation "
            "collapses after ρ residualization — epistemic likely ρ monotonic reparameterization"
        )
    elif rho_dominant_marginal:
        interpretation = (
            f"|r(epi,ρ)|={abs(r_epi_rho):.2f}≥{rho_marginal_threshold} — ρ-dominant marginal "
            "coupling; inspect OOD residual contrast before crediting exposure-gap semantics"
        )
    else:
        interpretation = "no rho_dominant marginal coupling or OOD absent"

    return {
        "gate": "G5b",
        "ok": not rho_feature_proxy,
        "rho_dominant_marginal": rho_dominant_marginal,
        "rho_feature_proxy": rho_feature_proxy,
        "rho_coupling_advisory": rho_dominant_marginal and not rho_feature_proxy,
        "native_novelty_after_rho_control": native_novelty,
        "r_epi_rho_marginal": r_epi_rho,
        "r_student_teacher_partial_given_rho": r_epi_teacher_partial,
        "ood_contrast_raw": ood_raw,
        "ood_contrast_rho_residual": ood_residual,
        "thresholds": {
            "epi_rho_marginal_proxy": rho_marginal_threshold,
            "ood_ratio_min": ood_ratio_min,
        },
        "interpretation": interpretation,
    }


def g5_epistemic_provenance_report(
    rows: Sequence[dict[str, Any]],
    *,
    per_structure: dict[str, dict[str, Any]] | None = None,
    ood_rows: Sequence[dict[str, Any]] | None = None,
    rows_by_structure: dict[str, Sequence[dict[str, Any]]] | None = None,
    epi_sasa_marginal_threshold: float = G5_EPI_SASA_MARGINAL_PROXY,
    teacher_student_threshold: float = G5_TEACHER_STUDENT_R_PROXY,
    partial_drop_min: float = G5_PARTIAL_DROP_MIN,
    bootstrap_n: int = G5_BOOTSTRAP_N_DEFAULT,
) -> dict[str, Any]:
    """G5 — student epistemic vs SASA/ρ and v3 teacher alignment on Stage A corpus."""
    from science.training.nig_identifiability import g5_epistemic_provenance_checklist

    if not rows:
        return {"gate": "G5", "ok": False, "reason": "empty", "distilled_proxy": False}

    epi = _col(rows, "epistemic")
    rho = _col(rows, "rho")
    sasa_vals = [r.get("sasa") for r in rows]
    has_sasa = not any(v is None or not math.isfinite(float(v)) for v in sasa_vals)
    sasa = (
        np.array([float(v) for v in sasa_vals], dtype=np.float64) if has_sasa else None
    )

    r_epi_sasa = _pearson_r(epi, sasa) if has_sasa and sasa is not None else float("nan")
    r_epi_rho = _pearson_r(epi, rho)
    r_epi_sasa_partial = (
        partial_correlation_epi_sasa_given_rho(rows) if has_sasa else float("nan")
    )
    partial_drop = (
        abs(r_epi_sasa) - abs(r_epi_sasa_partial)
        if math.isfinite(r_epi_sasa) and math.isfinite(r_epi_sasa_partial)
        else float("nan")
    )

    teacher_col = [r.get("teacher_epistemic") for r in rows]
    has_teacher = all(
        v is not None and math.isfinite(float(v)) for v in teacher_col
    )
    teacher_epi = (
        np.array([float(v) for v in teacher_col], dtype=np.float64) if has_teacher else None
    )
    r_student_teacher = (
        _pearson_r(epi, teacher_epi) if has_teacher and teacher_epi is not None else float("nan")
    )
    r_student_teacher_partial_rho = (
        partial_correlation_epi_teacher_given_rho(rows) if has_teacher else float("nan")
    )

    var = uncertainty_corpus_variance(rows)
    distilled_sasa_proxy = (
        math.isfinite(r_epi_sasa)
        and math.isfinite(r_student_teacher)
        and abs(r_epi_sasa) >= epi_sasa_marginal_threshold
        and abs(r_student_teacher) >= teacher_student_threshold
    )

    native_signal = (
        math.isfinite(r_epi_sasa_partial)
        and math.isfinite(r_epi_sasa)
        and partial_drop >= partial_drop_min
    ) or (
        per_structure is not None
        and any(
            math.isfinite(float(s.get("r_student_teacher", float("nan"))))
            and abs(float(s["r_student_teacher"])) < teacher_student_threshold
            for s in per_structure.values()
        )
    )

    g5b = g5b_rho_feature_proxy_report(rows, ood_rows)

    bootstrap: dict[str, Any] | None = None
    if rows_by_structure and has_teacher:
        bootstrap = bootstrap_structure_pooled_correlation(
            rows_by_structure,
            n_bootstrap=bootstrap_n,
            threshold=teacher_student_threshold,
        )

    teacher_student_borderline = bool(
        bootstrap
        and bootstrap.get("ok")
        and bootstrap.get("ci_straddles_threshold")
    )
    distilled_proxy = distilled_sasa_proxy
    if (
        bootstrap
        and bootstrap.get("ok")
        and bootstrap.get("ci_low") is not None
        and has_sasa
        and math.isfinite(r_epi_sasa)
        and abs(r_epi_sasa) >= epi_sasa_marginal_threshold
        and float(bootstrap["ci_low"]) >= teacher_student_threshold
    ):
        distilled_proxy = True

    if bootstrap:
        bootstrap = dict(bootstrap)
        n_struct = bootstrap.get("n_structures")
        bootstrap["ci_precision_caveat"] = (
            f"CI computed over n={n_struct} protein resampling units; true uncertainty "
            "may be wider given per-structure r(stu,tea) spread."
        )

    ood_raw_contrast = g5b.get("ood_contrast_raw") or {}
    ood_ratio = ood_raw_contrast.get("epistemic_ood_ratio")
    epistemic_ood_inverted = (
        ood_ratio is not None
        and math.isfinite(float(ood_ratio))
        and float(ood_ratio) < 1.0
    )
    epistemic_novelty_claim_blocked = bool(
        epistemic_ood_inverted or g5b.get("rho_dominant_marginal")
    )
    novelty_block_text = (
        "epistemic is ρ-dominant and shows no OOD elevation (raw or ρ-residualized) "
        "on the pinned OOD test (1PGB) — do not cite epistemic uncertainty for "
        "novelty/review-flagging claims until this is independently resolved."
    )
    checklist = g5_epistemic_provenance_checklist()

    s6_asterisk = distilled_proxy or g5b.get("rho_feature_proxy")
    s6_caveat = bool(teacher_student_borderline)
    g5_ok = (not distilled_proxy or native_signal) and not g5b.get("rho_feature_proxy", False)
    reasons: list[str] = []
    if distilled_proxy:
        reasons.append("distilled_sasa_proxy_flag")
    if g5b.get("rho_feature_proxy"):
        reasons.append("rho_feature_proxy_flag")
    if epistemic_ood_inverted:
        reasons.append("epistemic_ood_inverted_on_1pgb")
    if epistemic_novelty_claim_blocked:
        reasons.append("epistemic_novelty_claim_blocked")
    if teacher_student_borderline:
        reasons.append("teacher_student_borderline_bootstrap")
    if native_signal:
        reasons.append("native_signal_partial_or_structure_divergence")
    if g5b.get("native_novelty_after_rho_control"):
        reasons.append("ood_novelty_survives_rho_control")

    if s6_asterisk:
        interpretation = checklist["s6_credit_rule_if_distilled_proxy"]
    elif epistemic_novelty_claim_blocked:
        interpretation = novelty_block_text
    else:
        interpretation = (
            "no distilled_proxy / rho_feature_proxy — epistemic may be creditable to v6-native spread"
        )

    return {
        "gate": "G5",
        "ok": g5_ok,
        "distilled_proxy": distilled_proxy,
        "distilled_sasa_proxy": distilled_sasa_proxy,
        "rho_feature_proxy": bool(g5b.get("rho_feature_proxy")),
        "epistemic_ood_inverted": epistemic_ood_inverted,
        "epistemic_novelty_claim_blocked": epistemic_novelty_claim_blocked,
        "s6_epistemic_asterisk": s6_asterisk,
        "s6_epistemic_caveat": s6_caveat,
        "s6_epistemic_novelty_blocked": epistemic_novelty_claim_blocked,
        "s6_epistemic_novelty_block_text": novelty_block_text if epistemic_novelty_claim_blocked else None,
        "native_signal_corroboration": native_signal,
        "teacher_student_borderline": teacher_student_borderline,
        "reason": "ok" if g5_ok and not s6_asterisk and not epistemic_novelty_claim_blocked else ";".join(reasons) or "review",
        "interpretation": interpretation,
        "n_residues": len(rows),
        "epistemic_std": var["epistemic_std"],
        "r_epi_sasa_marginal": r_epi_sasa,
        "r_epi_rho_marginal": r_epi_rho,
        "r_epi_sasa_partial_given_rho": r_epi_sasa_partial,
        "partial_drop_abs_marginal_minus_partial": partial_drop,
        "r_student_teacher_corpus": r_student_teacher,
        "r_student_teacher_partial_given_rho": r_student_teacher_partial_rho,
        "sasa_sign": sasa_epistemic_sign_interpretation(r_epi_sasa),
        "teacher_student_bootstrap": bootstrap,
        "g5b": g5b,
        "thresholds": {
            "epi_sasa_marginal_proxy": epi_sasa_marginal_threshold,
            "epi_rho_marginal_proxy": G5_EPI_RHO_MARGINAL_PROXY,
            "teacher_student_proxy": teacher_student_threshold,
            "partial_drop_min": partial_drop_min,
        },
        "per_structure": per_structure or {},
        "checklist": checklist,
    }


def aleatoric_dehydron_stratification_report(
    rows: Sequence[dict[str, Any]],
    *,
    dehydron_key: str = "tau_flag",
    dehydron_threshold: float = 0.5,
    informative_std_floor: float = NODE_ALE_INFORMATIVE_FLOOR,
) -> dict[str, Any]:
    """Split aleatoric spread by dehydron vs regular residues (mask wiring diagnostic)."""
    if not rows:
        return {"ok": False, "reason": "empty"}
    ale = _col(rows, "aleatoric")
    dehyd = np.array(
        [float(r.get(dehydron_key, 0.0)) >= dehydron_threshold for r in rows],
        dtype=bool,
    )
    regular = ~dehyd
    global_std = float(np.std(ale))

    def _slice(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"n": 0, "ale_std": float("nan"), "ale_mean": float("nan")}
        vals = ale[mask]
        std = float(np.std(vals))
        return {
            "n": int(mask.sum()),
            "ale_std": std,
            "ale_mean": float(np.mean(vals)),
            "informative_aleatoric": std >= informative_std_floor,
        }

    dehyd_stats = _slice(dehyd)
    regular_stats = _slice(regular)
    global_collapsed = global_std < informative_std_floor
    dehyd_collapsed = (
        dehyd_stats["n"] > 0
        and not dehyd_stats.get("informative_aleatoric", False)
    )
    regular_collapsed = (
        regular_stats["n"] > 0
        and not regular_stats.get("informative_aleatoric", False)
    )

    if global_collapsed and dehyd_collapsed:
        interpretation = (
            "global aleatoric collapsed and dehydron-flagged residues also near-floor — "
            "shaping hinge is not selectively suppressing regular residues; suspect global "
            "penalty or mask not gating (not underpowered signal alone)"
        )
    elif global_collapsed and not dehyd_collapsed:
        interpretation = (
            "global aleatoric collapsed but dehydron residues retain spread — consistent "
            "with selective regular-residue suppression (hinge may be wired correctly)"
        )
    elif not global_collapsed:
        interpretation = "corpus aleatoric informative globally — use holdout P8 for G4 verdict"

    return {
        "global_ale_std": global_std,
        "global_informative_aleatoric": not global_collapsed,
        "dehydron_residues": dehyd_stats,
        "regular_residues": regular_stats,
        "dehydron_also_collapsed": dehyd_collapsed,
        "regular_also_collapsed": regular_collapsed,
        "informative_std_floor": informative_std_floor,
        "interpretation": interpretation,
    }


def aleatoric_population_separation_report(
    rows: Sequence[dict[str, Any]],
    *,
    dehydron_key: str = "tau_flag",
    dehydron_threshold: float = 0.5,
    gap_std_mult_min: float = G4_POPULATION_GAP_STD_MULT_MIN,
    informative_std_floor: float = NODE_ALE_INFORMATIVE_FLOOR,
) -> dict[str, Any]:
    """Frozen G4 var_penalty sweep criterion — dehydron vs regular aleatoric separation.

    **Not a site-certification gate.** For druggability / conformational triage use
    ``flag_investigation_sites()`` in ``aleatoric_residue_diagnostics.py``.
    Corpus ``global_ale_std`` here is a shaping-sweep / collapse monitor only.
    """
    if not rows:
        return {"ok": False, "reason": "empty"}
    ale = _col(rows, "aleatoric")
    dehyd = np.array(
        [float(r.get(dehydron_key, 0.0)) >= dehydron_threshold for r in rows],
        dtype=bool,
    )
    regular = ~dehyd
    if not dehyd.any() or not regular.any():
        return {"ok": False, "reason": "insufficient_strata"}

    mean_dehyd = float(np.mean(ale[dehyd]))
    mean_regular = float(np.mean(ale[regular]))
    std_dehyd = float(np.std(ale[dehyd]))
    std_regular = float(np.std(ale[regular]))
    global_std = float(np.std(ale))
    gap = abs(mean_dehyd - mean_regular)
    gap_over_pooled_std = (
        gap / global_std if global_std > 1e-12 else float("nan")
    )
    global_informative = global_std >= informative_std_floor
    gap_ok = math.isfinite(gap_over_pooled_std) and gap_over_pooled_std >= gap_std_mult_min
    ok = global_informative and gap_ok

    if ok:
        interpretation = (
            f"populations separated: gap/σ_pooled={gap_over_pooled_std:.2f}≥{gap_std_mult_min}, "
            f"global σ={global_std:.4f}≥{informative_std_floor}"
        )
    elif not global_informative and gap_ok:
        interpretation = (
            "gap opens vs pooled std but global aleatoric still below informativeness floor "
            f"(σ={global_std:.4f}<{informative_std_floor}) — partial differentiation only"
        )
    elif global_informative and not gap_ok:
        interpretation = (
            f"global σ informative but populations not separated "
            f"(gap/σ_pooled={gap_over_pooled_std:.2f}<{gap_std_mult_min})"
        )
    else:
        interpretation = (
            f"no separation: gap/σ_pooled={gap_over_pooled_std:.2f}, global σ={global_std:.4f}"
        )

    return {
        "ok": ok,
        "global_ale_std": global_std,
        "global_informative_aleatoric": global_informative,
        "mean_ale_dehydron": mean_dehyd,
        "mean_ale_regular": mean_regular,
        "std_ale_dehydron": std_dehyd,
        "std_ale_regular": std_regular,
        "mean_gap_abs": gap,
        "gap_over_pooled_std": gap_over_pooled_std,
        "gap_std_mult_min": gap_std_mult_min,
        "informative_std_floor": informative_std_floor,
        "n_dehydron": int(dehyd.sum()),
        "n_regular": int(regular.sum()),
        "interpretation": interpretation,
    }


def g4_aleatoric_shaping_holdout_report(
    rows: Sequence[dict[str, Any]],
    *,
    holdout_key: str = "ale_shaping_holdout",
    relative_lift_transfer_min: float = G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN,
    transfer_full_rel_lift_min: float = G4_TRANSFER_RATIO_FULL_REL_LIFT_MIN,
    ale_rho_proxy_threshold: float = G4_ALE_RHO_MARGINAL_PROXY,
    holdout_metadata: dict[str, Any] | None = None,
    checkpoint_eligible: bool | None = None,
) -> dict[str, Any]:
    """G4 — holdout P8 vs full corpus + transfer ratio + aleatoric ρ coupling."""
    if not rows:
        return {"gate": "G4", "ok": False, "reason": "empty"}
    holdout_rows = [r for r in rows if r.get(holdout_key)]
    if len(holdout_rows) < 10:
        return {
            "gate": "G4",
            "ok": False,
            "reason": f"insufficient_holdout_rows={len(holdout_rows)}",
            "n_holdout": len(holdout_rows),
        }

    p8_full = tau_boundary_aleatoric_elevation(rows)
    p8_holdout = tau_boundary_aleatoric_elevation(holdout_rows)
    full_rel = float(p8_full.get("aleatoric_tau_lift_relative", float("nan")))
    hold_rel = float(p8_holdout.get("aleatoric_tau_lift_relative", float("nan")))
    transfer_ratio = (
        hold_rel / full_rel
        if math.isfinite(full_rel) and math.isfinite(hold_rel) and abs(full_rel) > 1e-12
        else float("nan")
    )
    transfer_evaluable = bool(p8_full.get("informative_aleatoric")) and math.isfinite(
        full_rel
    ) and full_rel >= transfer_full_rel_lift_min
    if transfer_evaluable:
        transfer_ok = (
            math.isfinite(transfer_ratio)
            and transfer_ratio >= relative_lift_transfer_min
        )
        transfer_status = "evaluated"
    else:
        transfer_ok = False
        transfer_status = "not_evaluable_sub_threshold"

    ale = _col(rows, "aleatoric")
    rho = _col(rows, "rho")
    ale_std = float(np.std(ale))
    r_ale_rho = _pearson_r(ale, rho)
    r_ale_rho_holdout = _pearson_r(
        _col(holdout_rows, "aleatoric"), _col(holdout_rows, "rho")
    )
    epi_vals = [r.get("epistemic") for r in rows if r.get("epistemic") is not None]
    if len(epi_vals) >= 3:
        r_epi_ale = _pearson_r(
            np.array([float(r["epistemic"]) for r in rows if r.get("epistemic") is not None]),
            np.array([float(r["aleatoric"]) for r in rows if r.get("epistemic") is not None]),
        )
    else:
        r_epi_ale = float("nan")
    r_epi_ale_interpretable = ale_std >= NODE_ALE_INFORMATIVE_FLOOR
    rho_dominant_marginal = (
        math.isfinite(r_ale_rho) and abs(r_ale_rho) >= ale_rho_proxy_threshold
    )
    rho_dominant_holdout = (
        math.isfinite(r_ale_rho_holdout)
        and abs(r_ale_rho_holdout) >= ale_rho_proxy_threshold
    )
    aleatoric_rho_reparameterization_risk = rho_dominant_marginal or rho_dominant_holdout

    memorization_evaluable = bool(p8_full.get("ok"))
    if memorization_evaluable:
        memorization_detected = not bool(p8_holdout.get("ok"))
        memorization_status = (
            "detected" if memorization_detected else "not_detected"
        )
    else:
        memorization_detected = False
        memorization_status = "vacuous_full_p8_fail"

    p8_holdout_ok = bool(p8_holdout.get("ok"))
    g4_pass = (
        p8_holdout_ok
        and not memorization_detected
        and transfer_ok
        and transfer_evaluable
    )

    reasons: list[str] = []
    if not p8_holdout_ok:
        reasons.append(f"holdout_p8_fail:{p8_holdout.get('reason', '?')}")
    if memorization_detected:
        reasons.append("mask_memorization")
    if transfer_status == "not_evaluable_sub_threshold":
        reasons.append("transfer_ratio_not_evaluable_sub_threshold")
    elif not transfer_ok:
        reasons.append(
            f"relative_lift_transfer={transfer_ratio:.3f}<{relative_lift_transfer_min}"
            if math.isfinite(transfer_ratio)
            else "relative_lift_transfer=nan"
        )
    if aleatoric_rho_reparameterization_risk:
        reasons.append("aleatoric_rho_dominant_marginal")
    if checkpoint_eligible is False:
        reasons.append("checkpoint_ineligible_do_not_certify")

    if g4_pass and aleatoric_rho_reparameterization_risk:
        interpretation = (
            f"g4_p8_pass but |r(ale,ρ)|≥{ale_rho_proxy_threshold} — aleatoric may be ρ "
            "reparameterization (shaping target uses dehydron mask); do not cite aleatoric "
            "as independent biophysical ambiguity until ρ coupling is resolved"
        )
    elif g4_pass:
        interpretation = (
            "g4_ok: holdout P8 (G4a) passes, transfer ratio meets floor, no memorization"
        )
    elif memorization_detected:
        interpretation = "g4_fail: full-corpus P8 pass but holdout P8 fail — shaping memorized mask"
    elif transfer_status == "not_evaluable_sub_threshold":
        interpretation = (
            "g4_fail: transfer ratio not evaluable — full-corpus P8 did not clear "
            f"informative ale_std or relative_lift≥{transfer_full_rel_lift_min:.2f}; "
            "ratio would be noise÷noise"
        )
    elif not transfer_ok:
        interpretation = (
            f"g4_fail: holdout relative lift < {relative_lift_transfer_min:.0%} of "
            "full-corpus relative lift — partial memorization / weak transfer"
        )
    else:
        interpretation = f"g4_fail: holdout P8 does not meet G4a floor — {p8_holdout.get('reason')}"

    rule_summary = {
        "holdout_p8": "pass" if p8_holdout_ok else "fail",
        "memorization": (
            memorization_status
            if memorization_status != "not_detected"
            else "pass"
        ),
        "transfer_ratio": (
            "pass"
            if transfer_evaluable and transfer_ok
            else ("not_evaluable" if transfer_status == "not_evaluable_sub_threshold" else "fail")
        ),
    }

    return {
        "gate": "G4",
        "ok": g4_pass,
        "reason": "ok" if g4_pass else ";".join(reasons),
        "p8_full_corpus": p8_full,
        "p8_holdout_only": p8_holdout,
        "relative_lift_transfer_ratio": transfer_ratio,
        "relative_lift_transfer_min": relative_lift_transfer_min,
        "relative_lift_transfer_ok": transfer_ok,
        "relative_lift_transfer_evaluable": transfer_evaluable,
        "relative_lift_transfer_status": transfer_status,
        "mask_memorization_detected": memorization_detected,
        "mask_memorization_evaluable": memorization_evaluable,
        "mask_memorization_status": memorization_status,
        "r_ale_rho_marginal": r_ale_rho,
        "r_ale_rho_holdout": r_ale_rho_holdout,
        "r_epi_ale_marginal": r_epi_ale,
        "r_epi_ale_interpretable": r_epi_ale_interpretable,
        "rho_dominant_marginal": rho_dominant_marginal,
        "rho_dominant_holdout": rho_dominant_holdout,
        "aleatoric_rho_reparameterization_risk": aleatoric_rho_reparameterization_risk,
        "aleatoric_dehydron_stratification": aleatoric_dehydron_stratification_report(rows),
        "rule_summary": rule_summary,
        "n_corpus": len(rows),
        "n_holdout": len(holdout_rows),
        "holdout_metadata": holdout_metadata or {},
        "checkpoint_eligible": checkpoint_eligible,
        "thresholds_frozen": {
            "holdout_p8_g4a": "relative_lift≥0.20 AND ale_std≥0.05 on holdout",
            "transfer_ratio_min": relative_lift_transfer_min,
            "transfer_ratio_requires": (
                f"full informative ale_std AND full relative_lift≥{transfer_full_rel_lift_min:.2f}"
            ),
            "memorization_fail": "full_p8_ok AND NOT holdout_p8_ok (vacuous when full P8 fails)",
            "ale_rho_proxy": ale_rho_proxy_threshold,
        },
        "interpretation": interpretation,
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
    min_relative_lift: float = TAU_ALE_RELATIVE_LIFT_MIN,
    require_informative_std: bool = TAU_ALE_REQUIRE_INFORMATIVE_STD,
    informative_std_floor: float = NODE_ALE_INFORMATIVE_FLOOR,
) -> dict[str, Any]:
    """P8 — τ-boundary aleatoric elevation (G4a-hardened).

  Strata use **continuous ρ** (``|ρ − TAU| ≤ band``), not ``tau_flag``.

  Pass requires ALL of:
    - absolute lift > ``min_lift`` (default 0 — legacy sign check),
    - **relative lift** ``(ale_τ − ale_non) / std(ale)`` ≥ ``min_relative_lift``,
    - corpus ``std(ale)`` ≥ informative floor when ``require_informative_std``.
  """
    if not rows:
        return {"ok": False, "reason": "empty", "aleatoric_tau_lift": float("nan")}
    rho = _col(rows, "rho")
    ale = _col(rows, "aleatoric")
    ale_std = float(np.std(ale))
    tau_mask = np.abs(rho - TAU) <= rho_band
    if not tau_mask.any() or not (~tau_mask).any():
        return {
            "ok": False,
            "reason": "insufficient_tau_strata",
            "strata_basis": "continuous_rho",
            "tau_boundary_n": int(tau_mask.sum()),
            "aleatoric_tau_lift": float("nan"),
            "aleatoric_tau_lift_relative": float("nan"),
        }
    ale_tau = float(np.mean(ale[tau_mask]))
    ale_non = float(np.mean(ale[~tau_mask]))
    lift = ale_tau - ale_non
    relative_lift = lift / ale_std if ale_std > 1e-12 else float("nan")

    informative_ok = (not require_informative_std) or ale_std >= informative_std_floor
    relative_ok = math.isfinite(relative_lift) and relative_lift >= min_relative_lift
    absolute_ok = lift > min_lift
    ok = informative_ok and relative_ok and absolute_ok

    reasons: list[str] = []
    if not informative_ok:
        reasons.append(
            f"aleatoric_std={ale_std:.4f}<{informative_std_floor} (not informative; P8 blocked)"
        )
    if not relative_ok:
        reasons.append(
            f"aleatoric_tau_lift_relative={relative_lift:.4f}<{min_relative_lift}"
            if math.isfinite(relative_lift)
            else "aleatoric_tau_lift_relative=nan"
        )
    if not absolute_ok:
        reasons.append(f"aleatoric_tau_lift={lift:.4f}<={min_lift}")

    return {
        "ok": ok,
        "reason": "ok" if ok else ";".join(reasons),
        "strata_basis": "continuous_rho",
        "tau_boundary_n": int(tau_mask.sum()),
        "non_tau_n": int((~tau_mask).sum()),
        "aleatoric_std": ale_std,
        "aleatoric_mean_tau_boundary": ale_tau,
        "aleatoric_mean_non_tau_boundary": ale_non,
        "aleatoric_tau_lift": lift,
        "aleatoric_tau_lift_relative": relative_lift,
        "informative_aleatoric": informative_ok,
        "g4a_relative_lift_min": min_relative_lift,
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
