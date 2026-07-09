"""Minimal probe: does aleatoric vary independently of ρ and expert after controls?

Read-only diagnostics — not a training gate. Answers whether residue-level
aleatoric carries signal beyond dehydron density (ρ) and MoE routing assignment.

See ``docs/audit/GNNV7_SUCCESS_CRITERIA.md`` § Aleatoric independence probe.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# Marginal / partial |r| above this → ρ or expert proxy (mirrors G5 / G4 families).
ALE_RHO_PROXY_ABS_R = 0.85
ALE_EXPERT_ETA2_PROXY = 0.85
# Fraction of raw std(ale) that must remain after ρ + expert OLS to claim independence.
ALE_RESIDUAL_STD_RATIO_MIN = 0.50
# Full-model R² above this with low residual ratio → variance fully explained by proxies.
ALE_FULL_MODEL_R2_EXPLAINED = 0.85
# Coefficient of variation std/mean — below this the probe is not decision-grade.
ALE_CV_MIN = 0.005


def _col(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual y − E[y|x] with intercept (x may be [n] or [n, k])."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        design = np.column_stack([np.ones(len(x_arr)), x_arr])
    else:
        design = np.column_stack([np.ones(len(x_arr)), x_arr])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _eta_squared_categorical(y: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA η² = SS_between / SS_total."""
    if y.size < 3:
        return float("nan")
    total_var = float(np.var(y))
    if total_var < 1e-18:
        return 0.0
    uniq = np.unique(groups)
    if uniq.size < 2:
        return 0.0
    grand = float(np.mean(y))
    ss_between = sum(
        int(np.sum(groups == g)) * (float(np.mean(y[groups == g])) - grand) ** 2
        for g in uniq
    )
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total < 1e-18:
        return 0.0
    return float(ss_between / ss_total)


def _expert_onehot(experts: np.ndarray) -> np.ndarray | None:
    """Expert indicators (drop-first encoding). Returns None if <2 valid experts."""
    valid_mask = experts >= 0
    if int(valid_mask.sum()) < 3:
        return None
    uniq = sorted(int(e) for e in np.unique(experts[valid_mask]))
    if len(uniq) < 2:
        return None
    # Drop first category to avoid collinearity with intercept.
    cols = []
    for e in uniq[1:]:
        cols.append((experts == e).astype(np.float64))
    return np.column_stack(cols)


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    resid = _residualize(y, x)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot < 1e-18:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def aleatoric_independence_probe(
    rows: Sequence[dict[str, Any]],
    *,
    include_geometry: bool = False,
    rho_proxy_abs_r: float = ALE_RHO_PROXY_ABS_R,
    expert_eta2_proxy: float = ALE_EXPERT_ETA2_PROXY,
    residual_std_ratio_min: float = ALE_RESIDUAL_STD_RATIO_MIN,
    full_model_r2_explained: float = ALE_FULL_MODEL_R2_EXPLAINED,
    cv_min: float = ALE_CV_MIN,
) -> dict[str, Any]:
    """Probe whether ν_ale varies beyond ρ and expert assignment.

    Returns marginal and partial associations, OLS decomposition, and a verdict.
    """
    if not rows:
        return {"ok": False, "reason": "empty", "verdict": "empty"}

    ale = _col(rows, "aleatoric")
    rho = _col(rows, "rho")
    experts = np.array([int(r.get("expert", -1)) for r in rows], dtype=np.int64)

    ale_std = float(np.std(ale))
    ale_mean = float(np.mean(ale))
    ale_cv = ale_std / max(abs(ale_mean), 1e-12)

    r_ale_rho_marginal = _pearson_r(ale, rho)
    eta2_expert_marginal = _eta_squared_categorical(ale, experts)

    expert_oh = _expert_onehot(experts)
    partial_r_ale_rho_given_expert = float("nan")
    eta2_expert_given_rho = float("nan")
    r2_rho_only = float("nan")
    r2_rho_expert = float("nan")
    r2_full = float("nan")
    residual_std = float("nan")
    residual_std_ratio = float("nan")

    if expert_oh is not None:
        ale_r_expert = _residualize(ale, expert_oh)
        rho_r_expert = _residualize(rho, expert_oh)
        partial_r_ale_rho_given_expert = _pearson_r(ale_r_expert, rho_r_expert)

        ale_r_rho = _residualize(ale, rho)
        eta2_expert_given_rho = _eta_squared_categorical(ale_r_rho, experts)

        r2_rho_only = _ols_r2(ale, rho)
        r2_rho_expert = _ols_r2(ale, np.column_stack([rho, expert_oh]))
        design_parts: list[np.ndarray] = [rho, expert_oh]
        if include_geometry:
            geom = _geometry_matrix(rows)
            if geom is not None:
                design_parts.append(geom)
        full_x = np.column_stack(design_parts)
        r2_full = _ols_r2(ale, full_x)
        resid = _residualize(ale, full_x)
        residual_std = float(np.std(resid))
        residual_std_ratio = residual_std / max(ale_std, 1e-12)

    within_expert: list[dict[str, Any]] = []
    for e in sorted(int(x) for x in np.unique(experts) if int(x) >= 0):
        mask = experts == e
        if int(mask.sum()) < 5:
            continue
        sub_ale = ale[mask]
        sub_rho = rho[mask]
        within_expert.append(
            {
                "expert": e,
                "n": int(mask.sum()),
                "ale_std": float(np.std(sub_ale)),
                "r_ale_rho": _pearson_r(sub_ale, sub_rho),
            }
        )

    verdict, interpretation = _interpret_probe(
        ale_std=ale_std,
        ale_cv=ale_cv,
        cv_min=cv_min,
        r_ale_rho_marginal=r_ale_rho_marginal,
        partial_r_ale_rho_given_expert=partial_r_ale_rho_given_expert,
        eta2_expert_marginal=eta2_expert_marginal,
        eta2_expert_given_rho=eta2_expert_given_rho,
        r2_full=r2_full,
        residual_std_ratio=residual_std_ratio,
        rho_proxy_abs_r=rho_proxy_abs_r,
        expert_eta2_proxy=expert_eta2_proxy,
        residual_std_ratio_min=residual_std_ratio_min,
        full_model_r2_explained=full_model_r2_explained,
    )

    return {
        "ok": True,
        "n_residues": len(rows),
        "aleatoric_std": ale_std,
        "aleatoric_mean": ale_mean,
        "aleatoric_cv": ale_cv,
        "cv_min": cv_min,
        "dynamic_range_ok": ale_cv >= cv_min,
        "marginal": {
            "r_ale_rho": r_ale_rho_marginal,
            "eta2_expert": eta2_expert_marginal,
        },
        "partial_after_controls": {
            "r_ale_rho_given_expert": partial_r_ale_rho_given_expert,
            "eta2_expert_given_rho": eta2_expert_given_rho,
        },
        "ols": {
            "r2_rho_only": r2_rho_only,
            "r2_rho_plus_expert": r2_rho_expert,
            "r2_full_model": r2_full,
            "residual_std": residual_std,
            "residual_std_ratio": residual_std_ratio,
            "include_geometry": include_geometry,
        },
        "within_expert": within_expert,
        "thresholds": {
            "rho_proxy_abs_r": rho_proxy_abs_r,
            "expert_eta2_proxy": expert_eta2_proxy,
            "residual_std_ratio_min": residual_std_ratio_min,
            "full_model_r2_explained": full_model_r2_explained,
        },
        "verdict": verdict,
        "interpretation": interpretation,
        "independent_signal_candidate": verdict == "independent_signal_candidate",
    }


def _geometry_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray | None:
    keys = ("disc_r", "clustering", "cone_depth")
    cols: list[np.ndarray] = []
    for key in keys:
        vals = [r.get(key) for r in rows]
        if any(v is None or not math.isfinite(float(v)) for v in vals):
            return None
        cols.append(np.array([float(v) for v in vals], dtype=np.float64))
    return np.column_stack(cols)


def _interpret_probe(
    *,
    ale_std: float,
    ale_cv: float,
    cv_min: float,
    r_ale_rho_marginal: float,
    partial_r_ale_rho_given_expert: float,
    eta2_expert_marginal: float,
    eta2_expert_given_rho: float,
    r2_full: float,
    residual_std_ratio: float,
    rho_proxy_abs_r: float,
    expert_eta2_proxy: float,
    residual_std_ratio_min: float,
    full_model_r2_explained: float,
) -> tuple[str, str]:
    if ale_cv < cv_min or ale_std < 1e-4:
        return (
            "not_yet_meaningful",
            "Aleatoric dynamic range too narrow (low CV/std) — probe cannot distinguish "
            "proxy structure from independent signal; head not decision-grade.",
        )

    rho_m = abs(r_ale_rho_marginal) if math.isfinite(r_ale_rho_marginal) else 0.0
    rho_p = (
        abs(partial_r_ale_rho_given_expert)
        if math.isfinite(partial_r_ale_rho_given_expert)
        else 0.0
    )
    eta_m = eta2_expert_marginal if math.isfinite(eta2_expert_marginal) else 0.0
    eta_p = eta2_expert_given_rho if math.isfinite(eta2_expert_given_rho) else 0.0
    r2 = r2_full if math.isfinite(r2_full) else 0.0
    res_ratio = residual_std_ratio if math.isfinite(residual_std_ratio) else 0.0

    if (
        r2 >= full_model_r2_explained
        and res_ratio < residual_std_ratio_min
    ):
        return (
            "proxy_fully_explained",
            f"ρ + expert (+ geometry) explain R²={r2:.3f} of aleatoric; "
            f"residual_std_ratio={res_ratio:.3f} — no independent variance left.",
        )

    if rho_p >= rho_proxy_abs_r or (rho_m >= rho_proxy_abs_r and rho_p >= 0.70):
        return (
            "rho_proxy",
            f"|r(ale,ρ)| marginal={rho_m:.3f}, partial|expert={rho_p:.3f} — "
            "aleatoric tracks dehydron density, not independent ensemble ambiguity.",
        )

    if eta_p >= expert_eta2_proxy or (eta_m >= expert_eta2_proxy and eta_p >= 0.70):
        return (
            "expert_proxy",
            f"η²(expert) marginal={eta_m:.3f}, given ρ={eta_p:.3f} — "
            "aleatoric is largely routing-assignment structured.",
        )

    if (
        res_ratio >= residual_std_ratio_min
        and rho_p < rho_proxy_abs_r
        and eta_p < expert_eta2_proxy
    ):
        return (
            "independent_signal_candidate",
            f"After ρ + expert controls: residual_std_ratio={res_ratio:.3f}, "
            f"|partial r(ale,ρ)|={rho_p:.3f}, η²(expert|ρ)={eta_p:.3f} — "
            "variance remains that is not fully explained by proxies; warrants site-level follow-up.",
        )

    return (
        "ambiguous",
        "Spread exists but proxy partials are borderline — extend corpus or add geometry controls.",
    )
