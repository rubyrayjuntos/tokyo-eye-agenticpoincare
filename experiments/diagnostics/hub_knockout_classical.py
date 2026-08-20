#!/usr/bin/env python3
"""Forward-knockout hub scaffolding vs classical betweenness (Stage A-12).

Causal Part B substitute under z-norm (Jacobian forbidden): Spearman of
knockout ``out_effect`` vs classical betweenness — hub scaffolding, not a
163-directionality hunt.

Pre-registration companion to ``docs/specs/rho-tau-abs-dist-swap/ablation.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.jacobian_flow_influence import (
    HOLD_SPEARMAN_MIN,
    bootstrap_spearman,
    load_proteins_from_cache,
    spearman_corr,
)
from experiments.diagnostics.kras_knockout_causal import knockout_scan
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.classical_network_metrics import classical_network_metrics
from science.training.gnn_lineage import load_model_from_checkpoint

STAGE_A12 = (
    "1MBN",
    "1LYZ",
    "1BG1",
    "1F88",
    "2Z6H",
    "1HHP",
    "1TEN",
    "1UBQ",
    "1TIM",
    "4OBE",
    "1IVO",
    "2SHP",
)

# Pre-registered before interpretation (Move 2).
PREREG = {
    "primary_metric": (
        "Stage A-12 median Spearman(knockout out_effect, classical betweenness); "
        "out_effect[i] = mean_j≠i ||Δencoder_h[j]|| after zeroing input x[i]"
    ),
    "method": "forward_pass_input_knockout_no_gradients",
    "not_tested": "KRAS 163 / 4OBE↔4DSO directionality (separate probe)",
    "hold": {
        "spearman_min": HOLD_SPEARMAN_MIN,
        "ci_lo_min": 0.0,
        "note": "hub-scaffolding hold only (no Jacobian asymmetry term)",
    },
    "arm_compare": {
        "clear_delta_median": 0.10,
        "flat_abs_delta_median": 0.05,
        "clear_delta_holds": 2,
        "pass": (
            "Δ(median_swap − median_baseline) ≥ +0.10 and swap_median ≥ 0.30, "
            "OR Δ_holds ≥ +2 with swap_median ≥ baseline_median and swap_median ≥ 0.30"
        ),
        "fail": (
            "Δ_median ≤ −0.10, OR Δ_holds ≤ −2 with swap_median < baseline_median"
        ),
        "partial": "otherwise (includes flat |Δ_median| < 0.05 and |Δ_holds| ≤ 1)",
    },
}


def score_structure_knockout_vs_classical(
    out_effect: np.ndarray,
    betweenness: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Spearman + hold for one structure."""
    btw = bootstrap_spearman(out_effect, betweenness, seed=seed)
    rho = btw["spearman"]
    ci_lo = btw["ci_lo"]
    spearman_ok = (
        np.isfinite(rho)
        and np.isfinite(ci_lo)
        and float(rho) >= HOLD_SPEARMAN_MIN
        and float(ci_lo) > 0.0
    )
    return {
        "spearman_betweenness": float(rho) if np.isfinite(rho) else None,
        "ci_lo": float(ci_lo) if np.isfinite(ci_lo) else None,
        "ci_hi": float(btw["ci_hi"]) if np.isfinite(btw["ci_hi"]) else None,
        "n": int(btw["n"]),
        "spearman_ok": bool(spearman_ok),
        "holds": bool(spearman_ok),
        "median_out_effect": float(np.median(out_effect[np.isfinite(out_effect)])),
        "out_effect_cv": float(
            np.std(out_effect[np.isfinite(out_effect)])
            / max(np.mean(out_effect[np.isfinite(out_effect)]), 1e-12)
        ),
    }


def grade_arm_compare(
    baseline_rows: Sequence[Mapping[str, Any]],
    swap_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply pre-registered Pass / Fail / Partial on matched arms."""
    def _medians(rows: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
        rhos = [
            float(r["spearman_betweenness"])
            for r in rows
            if r.get("spearman_betweenness") is not None
            and np.isfinite(float(r["spearman_betweenness"]))
        ]
        holds = sum(1 for r in rows if r.get("holds"))
        return float(np.median(rhos)) if rhos else float("nan"), holds

    base_med, base_holds = _medians(baseline_rows)
    swap_med, swap_holds = _medians(swap_rows)
    d_med = swap_med - base_med
    d_holds = swap_holds - base_holds
    clear = float(PREREG["arm_compare"]["clear_delta_median"])  # type: ignore[index]
    flat = float(PREREG["arm_compare"]["flat_abs_delta_median"])  # type: ignore[index]
    hold_jump = int(PREREG["arm_compare"]["clear_delta_holds"])  # type: ignore[index]

    if (d_med >= clear and swap_med >= HOLD_SPEARMAN_MIN) or (
        d_holds >= hold_jump
        and swap_med >= base_med
        and swap_med >= HOLD_SPEARMAN_MIN
    ):
        grade = "Pass"
    elif (d_med <= -clear) or (d_holds <= -hold_jump and swap_med < base_med):
        grade = "Fail"
    else:
        grade = "Partial"

    flat_note = abs(d_med) < flat and abs(d_holds) <= 1
    return {
        "grade": grade,
        "baseline_median_spearman": base_med,
        "swap_median_spearman": swap_med,
        "delta_median": d_med,
        "baseline_holds": base_holds,
        "swap_holds": swap_holds,
        "delta_holds": d_holds,
        "flat_by_prereg": bool(flat_note),
        "preregistration": PREREG,
    }


def run_arm(
    checkpoint: Path,
    proteins: Mapping[str, dict[str, Any]],
    pdb_ids: Sequence[str],
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid in pdb_ids:
        prot = proteins[pid]
        print(f"knockout-scan {pid} on {checkpoint} ...", flush=True)
        model = load_model_from_checkpoint(checkpoint, device)
        model.eval()
        ca = prot.get("ca_coords")
        if ca is None:
            raise ValueError(f"{pid}: missing ca_coords")
        data0 = prepare_training_batch(model, prot, device)
        n = residue_node_count(data0, prot)
        ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
        ca_np = ca_np[:n]
        classical = classical_network_metrics(ca_np)
        scan = knockout_scan(model, prot, device)
        # Drop heavy delta_rows before scoring
        out_effect = np.asarray(scan["out_effect"], dtype=np.float64)
        scored = score_structure_knockout_vs_classical(
            out_effect,
            np.asarray(classical["betweenness"], dtype=np.float64),
            seed=hash(pid) % 10_000,
        )
        # Secondary telemetry (not primary grade)
        anm = spearman_corr(
            out_effect, np.asarray(classical["anm_msf"], dtype=np.float64)
        )
        cf = spearman_corr(
            out_effect,
            np.asarray(classical["current_flow_betweenness"], dtype=np.float64),
        )
        rows.append(
            {
                "pdb_id": pid,
                "n_residues": int(n),
                **scored,
                "spearman_anm_msf": float(anm) if np.isfinite(anm) else None,
                "spearman_current_flow": float(cf) if np.isfinite(cf) else None,
            }
        )
        print(
            f"  {pid}: ρ(btw)={scored['spearman_betweenness']:.3f} "
            f"holds={scored['holds']}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/chem_mvp_znorm_stage_a12_cold_v1/v66_best.pt"
        ),
    )
    parser.add_argument(
        "--swap-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/chem_mvp_tau_abs_dist_stage_a12_cold_v1/v66_best.pt"
        ),
    )
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
    )
    parser.add_argument("--stage-a12", action="store_true", default=True)
    parser.add_argument(
        "--pdb-id",
        action="append",
        dest="pdb_ids",
        help="Optional subset; default Stage A-12.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/rho_tau_abs_dist_swap/"
            "part_b_causal_knockout_stage_a12.json"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--prereg-only",
        action="store_true",
        help="Write pre-registration JSON and exit (no model runs).",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prereg_only:
        prereg_path = args.output.with_name("part_b_causal_knockout_prereg.json")
        prereg_path.write_text(json.dumps({"preregistration": PREREG}, indent=2) + "\n")
        print(f"wrote {prereg_path}")
        return 0

    pdb_ids = [p.upper() for p in (args.pdb_ids or list(STAGE_A12))]
    proteins = load_proteins_from_cache(args.corpus_cache, pdb_ids)

    print("=== baseline (matched z-norm) ===", flush=True)
    baseline_rows = run_arm(args.baseline_checkpoint, proteins, pdb_ids, args.device)
    print("=== swap (|ρ−TAU|) ===", flush=True)
    swap_rows = run_arm(args.swap_checkpoint, proteins, pdb_ids, args.device)

    compare = grade_arm_compare(baseline_rows, swap_rows)
    out = {
        "schema_version": 1,
        "probe": "hub_knockout_classical",
        "preregistration": PREREG,
        "GNN_INPUT_MODE": os.environ.get("GNN_INPUT_MODE"),
        "corpus_cache": str(args.corpus_cache),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "swap_checkpoint": str(args.swap_checkpoint),
        "baseline": {"structures": baseline_rows},
        "swap": {"structures": swap_rows},
        "compare": compare,
        "move_3_hint": (
            "CLEANUP 2026-07-18: soft Pass via Δ_holds is WITHDRAWN for SSOT. "
            "Clear Pass requires Δ_median ≥ +0.10. Path 2 agent pilot ORPHANED — "
            "do not authorize training from this grade. Resume from locked "
            "|ρ−TAU| / matched z-norm trunk only after a user-registered bet."
        ),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(compare, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
