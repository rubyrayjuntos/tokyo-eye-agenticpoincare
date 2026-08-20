"""Monopole-as-composition audit: re-score absolute share vs corpus majority fraction.

Compares per-structure committed_hard_share_max against natural floor
max(frac_core, frac_dh) under clean bipartition framing (SSOT § composition).
Optionally reloads ge30 checkpoints for prototype-proximity correlations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.diagnostics.monopole_and_seed_dilution_audit import audit_monopole_proto
from experiments.training.v6.corpus import load_training_proteins

MONOPOLE_ABS_THRESHOLD = 0.56
EXCESS_RESOLVED_THRESHOLD = 0.05
EXCESS_WATCH_THRESHOLD = 0.10

_CORPUS_DH_FRAC: dict[str, float] | None = None


def _corpus_dehydron_fractions() -> dict[str, float]:
    global _CORPUS_DH_FRAC
    if _CORPUS_DH_FRAC is not None:
        return _CORPUS_DH_FRAC
    proteins, _ = load_training_proteins(
        Path("/tmp/dtie_pdb_cache"),
        Path("/app/manifests/v6_corpus_stage_a_small_v1.json"),
        max_proteins=12,
    )
    out: dict[str, float] = {}
    for prot in proteins:
        pdb = str(prot.get("pdb_id") or "?").upper()
        dh = prot.get("target_dehydron")
        if dh is None:
            continue
        arr = np.asarray(dh, dtype=float).reshape(-1)
        out[pdb] = float(np.mean(arr > 0.5))
    _CORPUS_DH_FRAC = out
    return out


def _enrich_per_structure(per_structure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus = _corpus_dehydron_fractions()
    enriched: list[dict[str, Any]] = []
    for row in per_structure:
        row = dict(row)
        if row.get("corpus_dehydron_frac") is None:
            pdb = str(row.get("pdb_id") or "?").upper()
            if pdb in corpus:
                row["corpus_dehydron_frac"] = corpus[pdb]
        enriched.append(row)
    return enriched


def _load_ge30_per_structure(run_dir: Path) -> list[dict[str, Any]]:
    jsonl = run_dir / "prototype_repulsion_per_epoch.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    by_ge = {int(r["global_epoch"]): r for r in rows}
    if 30 not in by_ge:
        raise KeyError(f"ge30 missing in {jsonl}")
    return _enrich_per_structure(list(by_ge[30].get("per_structure") or []))


def score_composition(
    per_structure: list[dict[str, Any]],
    *,
    run_label: str,
    node_emb_width: int | None = None,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in per_structure:
        pdb = str(row.get("pdb_id") or "?").upper()
        frac_dh = row.get("corpus_dehydron_frac")
        share = row.get("committed_hard_share_max")
        if frac_dh is None or share is None:
            continue
        frac_dh_f = float(frac_dh)
        share_f = float(share)
        frac_core = 1.0 - frac_dh_f
        natural_floor = max(frac_core, frac_dh_f)
        excess = share_f - natural_floor
        abs_mono = share_f >= MONOPOLE_ABS_THRESHOLD
        maj_class = "dehydron" if frac_dh_f >= 0.5 else "core"
        scored.append(
            {
                "pdb_id": pdb,
                "frac_core": frac_core,
                "frac_dh": frac_dh_f,
                "corpus_majority_class": maj_class,
                "committed_hard_share_max": share_f,
                "natural_floor": natural_floor,
                "excess": excess,
                "abs_monopole_ge_0_56": abs_mono,
                "n_committed": int(row.get("n_committed") or 0),
                "n_minority": int(row.get("n_minority") or 0),
                "majority_dehydron_rate": row.get("majority_dehydron_rate"),
                "minority_dehydron_rate": row.get("minority_dehydron_rate"),
                "classification": _classify_row(excess, abs_mono),
            }
        )

    scored.sort(key=lambda r: r["pdb_id"])
    abs_monos = [r for r in scored if r["abs_monopole_ge_0_56"]]
    resolved = [r for r in abs_monos if r["excess"] <= EXCESS_RESOLVED_THRESHOLD]
    watch = [
        r
        for r in scored
        if r["excess"] > EXCESS_WATCH_THRESHOLD and r["abs_monopole_ge_0_56"]
    ]
    genuine_residual = [
        r
        for r in abs_monos
        if r["excess"] > EXCESS_RESOLVED_THRESHOLD
    ]

    return {
        "run": run_label,
        "node_emb_width": node_emb_width,
        "n_structures": len(scored),
        "n_abs_monopole": len(abs_monos),
        "n_resolved_by_composition": len(resolved),
        "n_genuine_residual_abs_monopole": len(genuine_residual),
        "n_watch_excess_gt_0_10": len(watch),
        "mean_excess": float(np.mean([r["excess"] for r in scored])) if scored else None,
        "mean_excess_abs_monopoles": (
            float(np.mean([r["excess"] for r in abs_monos])) if abs_monos else None
        ),
        "worst_excess": max(scored, key=lambda r: r["excess"]) if scored else None,
        "per_structure": scored,
    }


def _classify_row(excess: float, abs_mono: bool) -> str:
    if not abs_mono:
        return "below_abs_threshold"
    if excess <= EXCESS_RESOLVED_THRESHOLD:
        return "resolved_by_composition"
    if excess <= EXCESS_WATCH_THRESHOLD:
        return "small_residual"
    return "genuine_residual"


def compare_runs(
    three_d: dict[str, Any], legacy: dict[str, Any] | None = None
) -> dict[str, Any]:
    deltas: list[dict[str, Any]] = []
    legacy_by = {}
    if legacy:
        legacy_by = {r["pdb_id"]: r for r in legacy.get("per_structure", [])}
    for row in three_d.get("per_structure", []):
        pdb = row["pdb_id"]
        leg = legacy_by.get(pdb)
        deltas.append(
            {
                "pdb_id": pdb,
                "share_3d": row["committed_hard_share_max"],
                "share_legacy": leg["committed_hard_share_max"] if leg else None,
                "delta_share": (
                    row["committed_hard_share_max"] - leg["committed_hard_share_max"]
                    if leg
                    else None
                ),
                "excess_3d": row["excess"],
                "excess_legacy": leg["excess"] if leg else None,
                "delta_excess": row["excess"] - leg["excess"] if leg else None,
                "abs_mono_3d": row["abs_monopole_ge_0_56"],
                "abs_mono_legacy": leg["abs_monopole_ge_0_56"] if leg else None,
            }
        )
    return {"per_structure_deltas": deltas}


def verdict_summary(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """High-level read for Outcome A vs B."""
    notes: list[str] = []
    outcomes: dict[str, str] = {}

    for key, comp in runs.items():
        if not key.startswith("three_vector"):
            continue
        n_abs = comp["n_abs_monopole"]
        n_res = comp["n_resolved_by_composition"]
        n_gen = comp["n_genuine_residual_abs_monopole"]
        frac_explained = (n_res / n_abs) if n_abs else 1.0
        if n_abs == 0:
            outcome = "NO_ABS_MONOPOLES_UNDER_WEAKER_COMMIT"
        elif frac_explained >= 0.75:
            outcome = "OUTCOME_A_COMPOSITION_STILL_DOMINANT"
        elif frac_explained <= 0.5:
            outcome = "OUTCOME_B_MORE_GENUINE_MONOPOLE"
        else:
            outcome = "MIXED_COMPOSITION_AND_RESIDUAL"
        outcomes[key] = outcome
        notes.append(
            f"{key}: {n_abs} abs≥0.56, {n_res} resolved (excess≤{EXCESS_RESOLVED_THRESHOLD}), "
            f"{n_gen} genuine residual; explained={frac_explained:.0%}"
        )

    return {"outcomes": outcomes, "notes": notes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/app/checkpoints/v66/diagnostics/three_vector_stack_battery_reverify/"
            "monopole_composition_audit.json"
        ),
    )
    parser.add_argument(
        "--reload-proto",
        action="store_true",
        help="Reload ge30 checkpoints for prototype-proximity audit (GPU).",
    )
    args = parser.parse_args()

    run_specs = [
        (
            "three_vector_seed1",
            Path("/app/checkpoints/v66/runs/fix1_s4_stack_three_vector_cold_seed1_v1"),
            3,
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_stack_three_vector_cold_seed1_v1/"
                "epochs/epoch_030.pt"
            ),
        ),
        (
            "three_vector_seed2",
            Path("/app/checkpoints/v66/runs/fix1_s4_stack_three_vector_cold_seed2_v1"),
            3,
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_stack_three_vector_cold_seed2_v1/"
                "epochs/epoch_030.pt"
            ),
        ),
        (
            "legacy4d_seed1",
            Path("/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1"),
            4,
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1/"
                "epochs/epoch_030.pt"
            ),
        ),
        (
            "legacy4d_seed2",
            Path("/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1"),
            4,
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1/"
                "epochs/epoch_030.pt"
            ),
        ),
    ]

    composition: dict[str, Any] = {}
    for label, run_dir, width, _ckpt in run_specs:
        if not run_dir.exists():
            composition[label] = {"error": f"missing run dir {run_dir}"}
            continue
        per_struct = _load_ge30_per_structure(run_dir)
        composition[label] = score_composition(
            per_struct, run_label=label, node_emb_width=width
        )

    comparisons = {
        "seed1": compare_runs(
            composition.get("three_vector_seed1", {}),
            composition.get("legacy4d_seed1"),
        ),
        "seed2": compare_runs(
            composition.get("three_vector_seed2", {}),
            composition.get("legacy4d_seed2"),
        ),
    }

    proto: dict[str, Any] | None = None
    if args.reload_proto:
        proto = {}
        for label, _run_dir, _width, ckpt in run_specs[:2]:
            if ckpt.exists():
                proto[label] = audit_monopole_proto(ckpt, label)

    report: dict[str, Any] = {
        "tag": "MONOPOLE_COMPOSITION_REVERIFY_THREE_VECTOR",
        "thresholds": {
            "abs_monopole": MONOPOLE_ABS_THRESHOLD,
            "excess_resolved": EXCESS_RESOLVED_THRESHOLD,
            "excess_watch": EXCESS_WATCH_THRESHOLD,
        },
        "composition": composition,
        "legacy_vs_three_vector": comparisons,
        "verdict": verdict_summary(composition),
    }
    if proto is not None:
        report["prototype_proximity_three_vector"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "per_structure"}
            for k, v in proto.items()
        }
        report["prototype_proximity_three_vector_per_structure"] = {
            k: v.get("per_structure") for k, v in proto.items()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", args.out)
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
