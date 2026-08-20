#!/usr/bin/env python3
"""Steps 1–3 prerequisite for |ρ−TAU| swap on chem-MVP Stage A-12.

Forward-pass / corpus stats only — no training.

  Step 1: r(ρ,τ), r(ρ,|ρ−TAU|)
  Step 2: T1a-style input effective-rank ceiling (τ vs |ρ−TAU| substitution)
  Step 3: resolution / orthogonality bar vs everything except τ

Usage:
  GNN_INPUT_MODE=topology_three_vector python -m experiments.diagnostics.rho_tau_abs_dist_ceiling \\
    --corpus manifests/v6_corpus_stage_a_small_v1.json \\
    --max-proteins 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from experiments.diagnostics.embedding_occupancy_audit import _effective_rank
from experiments.diagnostics.trunk_hidden_occupancy import _svd_pack
from experiments.training.v6.corpus import load_training_proteins
from science.dtie.common.input_feature_norm import rewrite_tau_channel
from science.dtie.common.residue_features import TAU

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = (
    _REPO_ROOT
    / "checkpoints"
    / "v66"
    / "diagnostics"
    / "rho_tau_abs_dist_ceiling"
)

# Absolute ER lift below this is "no meaningful predicted ceiling lift".
DEFAULT_MIN_LIFT = 0.10
DEFAULT_REDUNDANCY_THRESHOLD = 0.70


def _corr_pair(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    n = int(x_arr.size)
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None, "reason": "n_lt_3"}
    if np.ptp(x_arr) == 0.0 or np.ptp(y_arr) == 0.0:
        return {"n": n, "pearson": None, "spearman": None, "reason": "constant"}
    return {
        "n": n,
        "pearson": float(pearsonr(x_arr, y_arr).statistic),
        "spearman": float(spearmanr(x_arr, y_arr).statistic),
        "reason": None,
    }


def _ss_key(value: float) -> str:
    return f"{float(value):g}"


def correlation_report(x: np.ndarray, y: np.ndarray, ss: np.ndarray) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    ss_arr = np.asarray(ss, dtype=np.float64).reshape(-1)
    within = {}
    for value in sorted(float(v) for v in np.unique(ss_arr[np.isfinite(ss_arr)])):
        mask = ss_arr == value
        within[_ss_key(value)] = _corr_pair(x_arr[mask], y_arr[mask])
    return {"marginal": _corr_pair(x_arr, y_arr), "within_ss": within}


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = np.maximum(X.std(axis=0, keepdims=True), 1e-8)
    return (X - mu) / sd


def _board_pack(X: np.ndarray, *, name: str, zscore: bool) -> dict[str, Any]:
    Xc = np.asarray(X, dtype=np.float64)
    if zscore:
        Xc = _zscore(Xc)
    pack = _svd_pack(Xc, name=name)
    return {
        "name": name,
        "zscore": zscore,
        "n": int(Xc.shape[0]),
        "dim": int(Xc.shape[1]),
        "effective_rank": pack.get("effective_rank"),
        "sigma2_sigma1": pack.get("sigma2_sigma1"),
        "top1_explained_var": (pack.get("explained_var_topk") or {}).get("k1"),
        "participation_ratio": pack.get("participation_ratio"),
        "sigma": pack.get("sigma"),
    }


def _walk_correlations(node: Any, prefix: str = ""):
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key in ("pearson", "spearman") and isinstance(value, (float, int)):
            if math.isfinite(float(value)):
                yield path, float(value)
        elif isinstance(value, dict):
            yield from _walk_correlations(value, path)


def redundancy_verdict(
    reports: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    values = list(_walk_correlations(reports))
    if not values:
        return {
            "passes": False,
            "threshold": threshold,
            "max_abs_correlation": None,
            "worst_path": None,
            "reason": "no_scorable_correlations",
        }
    worst_path, worst_value = max(values, key=lambda item: abs(item[1]))
    max_abs = abs(worst_value)
    return {
        "passes": max_abs < threshold,
        "threshold": threshold,
        "max_abs_correlation": max_abs,
        "signed_correlation": worst_value,
        "worst_path": worst_path,
        "reason": None,
    }


def _resolution_beyond_flag(rho: np.ndarray, tau: np.ndarray, abs_dist: np.ndarray) -> dict[str, Any]:
    """Does |ρ−TAU| vary within each binary-τ class?"""
    out: dict[str, Any] = {}
    for flag, label in ((0.0, "tau_0"), (1.0, "tau_1")):
        mask = np.isclose(tau, flag)
        d = abs_dist[mask]
        out[label] = {
            "n": int(mask.sum()),
            "abs_dist_std": float(d.std()) if d.size else None,
            "abs_dist_mean": float(d.mean()) if d.size else None,
            "abs_dist_p10": float(np.percentile(d, 10)) if d.size else None,
            "abs_dist_p90": float(np.percentile(d, 90)) if d.size else None,
            "rho_std": float(rho[mask].std()) if mask.any() else None,
        }
    # Within-flag continuous resolution: std > 0 on both sides ⇒ not a flat rewrite of τ.
    both = (
        out["tau_0"]["n"] > 10
        and out["tau_1"]["n"] > 10
        and (out["tau_0"]["abs_dist_std"] or 0.0) > 1e-6
        and (out["tau_1"]["abs_dist_std"] or 0.0) > 1e-6
    )
    out["adds_within_class_resolution"] = bool(both)
    return out


def build_report(
    X: np.ndarray,
    *,
    min_lift: float,
    redundancy_threshold: float,
    chem_mvp_zscore_enabled: bool,
) -> dict[str, Any]:
    if X.ndim != 2 or X.shape[1] < 3:
        raise ValueError(f"expected (N,≥3) topology features, got {X.shape}")
    X3 = np.asarray(X[:, :3], dtype=np.float64)
    rho = X3[:, 0]
    tau = X3[:, 1]
    ss = X3[:, 2]
    abs_dist = np.abs(rho - float(TAU))
    X_swap = rewrite_tau_channel(X3, mode="abs_dist")
    assert isinstance(X_swap, np.ndarray)

    step1 = {
        "TAU": float(TAU),
        "n_residues": int(X3.shape[0]),
        "frac_tau_exact_rho_lt_TAU": float(
            np.mean(np.isclose(tau, (rho < TAU).astype(np.float64), atol=1e-5))
        ),
        "corr_rho_tau": _corr_pair(rho, tau),
        "corr_rho_abs_dist": _corr_pair(rho, abs_dist),
    }

    boards = {
        "baseline_tau_raw": _board_pack(X3, name="baseline_tau_raw", zscore=False),
        "swap_abs_dist_raw": _board_pack(X_swap, name="swap_abs_dist_raw", zscore=False),
        "baseline_tau_zscore": _board_pack(X3, name="baseline_tau_zscore", zscore=True),
        "swap_abs_dist_zscore": _board_pack(
            X_swap, name="swap_abs_dist_zscore", zscore=True
        ),
        "drop_tau_zscore": _board_pack(
            X3[:, [0, 2]], name="drop_tau_zscore", zscore=True
        ),
    }

    # Primary ceiling: z-scored 3-vector (T1a methodology). Also report raw
    # because chem-MVP currently trains with input_feature_zscore=False.
    base_z = float(boards["baseline_tau_zscore"]["effective_rank"])
    swap_z = float(boards["swap_abs_dist_zscore"]["effective_rank"])
    base_r = float(boards["baseline_tau_raw"]["effective_rank"])
    swap_r = float(boards["swap_abs_dist_raw"]["effective_rank"])
    lift_z = swap_z - base_z
    lift_r = swap_r - base_r

    primary_board = "zscore"  # analytic capacity always judged on z-scored board
    primary_lift = lift_z
    predicted_ceiling = swap_z
    baseline_ceiling = base_z
    meaningful = primary_lift >= float(min_lift)

    step2 = {
        "methodology": (
            "Centered SVD effective_rank on full input feature matrix "
            "(T1a-style); |ρ−TAU| replaces τ channel; boards fit separately."
        ),
        "legacy_t1a_note": "Do not assume legacy ~2.33; this is re-derived on chem-MVP Stage A-12.",
        "chem_mvp_input_feature_zscore_enabled": bool(chem_mvp_zscore_enabled),
        "boards": boards,
        "baseline_er_zscore": base_z,
        "predicted_ceiling_zscore": predicted_ceiling,
        "lift_zscore": lift_z,
        "baseline_er_raw": base_r,
        "predicted_ceiling_raw": swap_r,
        "lift_raw": lift_r,
        "drop_tau_er_zscore": float(boards["drop_tau_zscore"]["effective_rank"]),
        "primary_board": primary_board,
        "min_lift": float(min_lift),
        "meaningful_predicted_lift": meaningful,
    }

    # Step 3: vs everything except τ. ρ correlation is expected (telemetry).
    vs_rho = correlation_report(abs_dist, rho, ss)
    vs_ss = correlation_report(abs_dist, ss, ss)
    ss_gate = redundancy_verdict({"vs_ss": vs_ss}, threshold=redundancy_threshold)
    resolution = _resolution_beyond_flag(rho, tau, abs_dist)

    step3 = {
        "note": (
            "Substitution, not a new independent feature. Correlation with ρ is "
            "expected; gate is resolution beyond binary τ + non-redundancy with ss."
        ),
        "vs_rho_telemetry": vs_rho,
        "vs_ss": vs_ss,
        "ss_redundancy_verdict": ss_gate,
        "resolution_beyond_binary_tau": resolution,
        "redundancy_threshold": float(redundancy_threshold),
    }

    if not meaningful:
        gate = "STOP_NO_PREDICTED_CEILING_LIFT"
        next_action = (
            "Do not burn a cold run. Revisit which feature is redundant, "
            "or whether a genuinely new orthogonal input is required."
        )
    else:
        gate = "CLEAR_FOR_MATCHED_COLD_ARMS"
        next_action = (
            "Steps 1–3 clear. Proceed only after user go-ahead: matched chem-MVP "
            "baseline vs replace_tau_with_abs_dist cold arms; Part A/B grading."
        )

    return {
        "lineage": "chem_mvp_stage_a12_topology_three_vector",
        "step1_correlations": step1,
        "step2_ceiling": step2,
        "step3_resolution": step3,
        "gate": {
            "verdict": gate,
            "predicted_ceiling": predicted_ceiling,
            "baseline_er": baseline_ceiling,
            "lift": primary_lift,
            "min_lift": float(min_lift),
            "next_action": next_action,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="|ρ−TAU| Steps 1–3 ceiling re-derivation (chem-MVP Stage A-12)"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_REPO_ROOT / "manifests" / "v6_corpus_stage_a_small_v1.json",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path(os.environ.get("PDB_DIR", "/tmp/dtie_pdb_cache")),
    )
    parser.add_argument("--max-proteins", type=int, default=12)
    parser.add_argument("--max-residues", type=int, default=None)
    parser.add_argument("--min-lift", type=float, default=DEFAULT_MIN_LIFT)
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=DEFAULT_REDUNDANCY_THRESHOLD,
    )
    parser.add_argument(
        "--chem-mvp-zscore-enabled",
        action="store_true",
        help="Set if the live chem-MVP recipe enables input_feature_zscore",
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args(argv)

    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")

    if args.max_residues is None:
        try:
            from science.training.corpus_governance import STAGE_A_MAX_RESIDUES

            max_residues = int(STAGE_A_MAX_RESIDUES)
        except Exception:
            max_residues = 650
    else:
        max_residues = int(args.max_residues)

    proteins, failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=args.max_proteins,
        max_residues=max_residues,
    )
    if not proteins:
        raise SystemExit(f"no proteins loaded from {args.corpus} (failed={failed})")

    widths = {int(p["data"].x.size(-1)) for p in proteins}
    if widths != {3}:
        raise SystemExit(
            f"expected topology_three_vector width 3, got widths={sorted(widths)}; "
            "set GNN_INPUT_MODE=topology_three_vector and clear mismatched caches"
        )

    X = np.concatenate(
        [p["data"].x.detach().cpu().numpy()[:, :3] for p in proteins], axis=0
    )
    report = build_report(
        X,
        min_lift=args.min_lift,
        redundancy_threshold=args.redundancy_threshold,
        chem_mvp_zscore_enabled=bool(args.chem_mvp_zscore_enabled),
    )
    report["corpus"] = str(args.corpus)
    report["n_proteins"] = len(proteins)
    report["failed_loads"] = int(failed)
    report["pdb_ids"] = [str(p.get("pdb_id", p.get("id", "?"))) for p in proteins]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    g = report["gate"]
    s1 = report["step1_correlations"]
    s2 = report["step2_ceiling"]
    print(
        json.dumps(
            {
                "gate": g["verdict"],
                "r_rho_tau": s1["corr_rho_tau"]["pearson"],
                "r_rho_abs_dist": s1["corr_rho_abs_dist"]["pearson"],
                "baseline_er_zscore": s2["baseline_er_zscore"],
                "predicted_ceiling_zscore": s2["predicted_ceiling_zscore"],
                "lift_zscore": s2["lift_zscore"],
                "lift_raw": s2["lift_raw"],
                "ss_redundancy_passes": report["step3_resolution"][
                    "ss_redundancy_verdict"
                ]["passes"],
                "within_class_resolution": report["step3_resolution"][
                    "resolution_beyond_binary_tau"
                ]["adds_within_class_resolution"],
            },
            indent=2,
        )
    )
    print(
        f"{g['verdict']}: ER {g['baseline_er']:.3f}→{g['predicted_ceiling']:.3f} "
        f"(Δ={g['lift']:+.3f}, min={g['min_lift']}) | wrote {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
