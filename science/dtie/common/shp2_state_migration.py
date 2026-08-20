"""SHP2 inactive→active OOD migration scoring (2SHP → 6CRF).

Mechanistic axes = N-SH2 / PTP coupling (not Src C-lobe language).
Bars are locked in ``DEFAULT_BARS`` / prereg JSON before first grade run.
Active open deposit is ``6CRF`` (E76K); ``6MCF`` is not SHP2.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from science.dtie.common.gini_flow_concentration import concentration_metrics
from science.dtie.common.kras_topo_matrix import align_resseq_vectors, spearman_rho

DEFAULT_BARS: dict[str, float] = {
    # Arm 1 — state-transition consistency (aligned auth_resseq)
    "state_spearman_min": 0.50,
    # Arm 2 — highway (top-decile) invariance under graph perturbations
    "perturb_jaccard_min": 0.50,
    "perturb_k_frac": 0.10,
    # Arm 3 — calibration / no open-state monopoly collapse
    "gini_active_min": 0.12,
    "gini_active_max": 0.25,
    "max_over_median_max": 3.0,
    "gini_blowup_slack": 0.05,  # G_active ≤ G_inactive + slack
}


def hub_set_jaccard(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    k_frac: float = 0.10,
) -> float:
    """Jaccard of top-k% index sets (descending)."""
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    k = max(1, int(np.ceil(k_frac * aa.size)))
    Ha = set(np.argsort(-np.nan_to_num(aa, nan=-np.inf))[:k].tolist())
    Hb = set(np.argsort(-np.nan_to_num(bb, nan=-np.inf))[:k].tolist())
    union = Ha | Hb
    if not union:
        return float("nan")
    return float(len(Ha & Hb) / len(union))


def score_state_transition(
    out_effect_by_pdb: Mapping[str, Sequence[float]],
    resseq_maps: Mapping[str, Mapping[int, int]],
    *,
    inactive_id: str = "2SHP",
    active_id: str = "6CRF",
    spearman_bar: float = DEFAULT_BARS["state_spearman_min"],
    k_frac: float = DEFAULT_BARS["perturb_k_frac"],
) -> dict[str, Any]:
    """Spearman of knockout flow on shared residue numbers across states."""
    shared, aligned = align_resseq_vectors(
        out_effect_by_pdb,
        resseq_maps,
        structures=[inactive_id, active_id],
    )
    x = aligned[inactive_id]
    y = aligned[active_id]
    rho = float(spearman_rho(x, y))
    jacc = hub_set_jaccard(x, y, k_frac=k_frac)
    passed = bool(np.isfinite(rho) and rho >= float(spearman_bar))
    return {
        "inactive_id": inactive_id,
        "active_id": active_id,
        "n_shared": len(shared),
        "spearman": rho,
        "spearman_bar": float(spearman_bar),
        "hub_jaccard_top10pct": jacc,
        "hub_jaccard_report_only": True,
        "pass": passed,
        "pass_rule": f"Spearman(out_effect_{inactive_id}, out_effect_{active_id}) "
        f"on shared auth_resseq ≥ {spearman_bar}",
        "mechanistic_axes_note": "N-SH2/PTP coupling axes (SHP2); residue IDs may shift",
    }


def score_perturbation_invariance(
    baseline_out_effect: Sequence[float] | np.ndarray,
    perturbed: Mapping[str, Sequence[float]],
    *,
    jaccard_bar: float = DEFAULT_BARS["perturb_jaccard_min"],
    k_frac: float = DEFAULT_BARS["perturb_k_frac"],
) -> dict[str, Any]:
    """Top-decile highway Jaccard vs baseline for each perturbation arm."""
    arms: dict[str, Any] = {}
    all_pass = True
    for name, oe in perturbed.items():
        j = hub_set_jaccard(baseline_out_effect, oe, k_frac=k_frac)
        arm_pass = bool(np.isfinite(j) and j >= float(jaccard_bar))
        all_pass = all_pass and arm_pass
        arms[name] = {
            "jaccard_top10pct": j,
            "jaccard_bar": float(jaccard_bar),
            "pass": arm_pass,
        }
    return {
        "k_frac": float(k_frac),
        "jaccard_bar": float(jaccard_bar),
        "arms": arms,
        "pass": bool(all_pass and len(arms) > 0),
        "pass_rule": (
            f"all arms Jaccard(H_top{int(100 * k_frac)}%) ≥ {jaccard_bar} "
            "vs unperturbed 6CRF"
        ),
    }


def score_calibration(
    inactive_out_effect: Sequence[float] | np.ndarray,
    active_out_effect: Sequence[float] | np.ndarray,
    *,
    bars: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Gini / max-median regime on active state; no monopoly blow-up vs inactive."""
    b = dict(DEFAULT_BARS if bars is None else bars)
    m_in = concentration_metrics(inactive_out_effect)
    m_act = concentration_metrics(active_out_effect)
    g_in = float(m_in["gini"])
    g_act = float(m_act["gini"])
    mm = float(m_act["max_over_median"])

    in_band = bool(
        np.isfinite(g_act)
        and float(b["gini_active_min"]) <= g_act <= float(b["gini_active_max"])
    )
    no_mm_blowup = bool(np.isfinite(mm) and mm <= float(b["max_over_median_max"]))
    no_gini_blowup = bool(
        np.isfinite(g_in)
        and np.isfinite(g_act)
        and g_act <= g_in + float(b["gini_blowup_slack"])
    )
    passed = bool(in_band and no_mm_blowup and no_gini_blowup)
    return {
        "inactive": m_in,
        "active": m_act,
        "gini_inactive": g_in,
        "gini_active": g_act,
        "max_over_median_active": mm,
        "bars": {
            "gini_active_min": float(b["gini_active_min"]),
            "gini_active_max": float(b["gini_active_max"]),
            "max_over_median_max": float(b["max_over_median_max"]),
            "gini_blowup_slack": float(b["gini_blowup_slack"]),
        },
        "checks": {
            "gini_in_band": in_band,
            "max_over_median_ok": no_mm_blowup,
            "no_gini_blowup_vs_inactive": no_gini_blowup,
        },
        "pass": passed,
        "pass_rule": (
            f"G(6CRF)∈[{b['gini_active_min']},{b['gini_active_max']}] "
            f"and max/median≤{b['max_over_median_max']} "
            f"and G(6CRF)≤G(2SHP)+{b['gini_blowup_slack']}"
        ),
    }


def grade_shp2_migration(
    *,
    state_transition: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate three-arm Pass/Fail."""
    p1 = bool(state_transition.get("pass"))
    p2 = bool(perturbation.get("pass"))
    p3 = bool(calibration.get("pass"))
    ok = p1 and p2 and p3
    return {
        "pass": ok,
        "verdict": "PASS" if ok else "FAIL",
        "arms": {
            "state_transition": p1,
            "perturbation_invariance": p2,
            "cross_structure_calibration": p3,
        },
    }
