"""Per-structure routing audit across epoch snapshots (eval mode, no grad)."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features
from science.training.corpus_governance import STAGE_A_MAX_RESIDUES
from science.training.routing_metrics import effective_experts, min_routing_fraction

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
STARVE_THRESHOLD = 0.05


def eval_checkpoint_routing(
    checkpoint: Path,
    proteins: list[dict],
    device: str,
) -> dict[str, dict[str, float | list[float]]]:
    """Evaluate per-structure routing loads for a single checkpoint."""
    return _eval_epoch(checkpoint, proteins, device)


def summarize_routing_floor(
    rows: dict[str, dict[str, float | list[float]]],
    *,
    threshold: float = STARVE_THRESHOLD,
) -> dict[str, Any]:
    """Summarize batch routing floor across structures."""
    if not rows:
        return {
            "per_structure_min_routing": None,
            "structures_below_floor": [],
            "worst_pdb": None,
            "worst_min_r": None,
            "threshold": threshold,
        }
    worst_pdb, worst_row = min(
        rows.items(),
        key=lambda item: float(item[1]["min_routing_fraction"]),
    )
    batch_min = float(worst_row["min_routing_fraction"])
    below = sorted(
        pdb
        for pdb, row in rows.items()
        if float(row["min_routing_fraction"]) < threshold
    )
    return {
        "per_structure_min_routing": batch_min,
        "structures_below_floor": below,
        "worst_pdb": worst_pdb,
        "worst_min_r": batch_min,
        "threshold": threshold,
        "structure_count": len(rows),
    }


def _eval_epoch(
    checkpoint: Path,
    proteins: list[dict],
    device: str,
) -> dict[str, dict[str, float | list[float]]]:
    model = load_v6_model(checkpoint, device)
    rows: dict[str, dict[str, float | list[float]]] = {}
    with torch.no_grad():
        for prot in proteins:
            pdb_id = str(prot["pdb_id"]).upper()
            data = attach_v6_features(prot["data"].to(device))
            out = model(data)
            load = out["expert_load"].detach().cpu()
            rows[pdb_id] = {
                "min_routing_fraction": float(min_routing_fraction(load)),
                "effective_experts": float(effective_experts(load)),
                "expert_load": [float(x) for x in load.tolist()],
                "fold_id": str(prot.get("fold_id", "")),
                "gene": str(prot.get("gene", "")),
                "n_residues": int(prot.get("n_residues", 0)),
            }
    del model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-structure routing audit on epoch snapshots")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=_REPO / "manifests" / "v6_corpus_stage_a.json")
    parser.add_argument("--pdb-dir", type=Path, default=_REPO / "pdb_cache")
    parser.add_argument("--phase-min", type=int, default=2)
    parser.add_argument(
        "--p2-global-start",
        type=int,
        default=None,
        help="Global epoch when P2 began (default: infer from metrics.json or 137)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    proteins, failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=25,
        max_residues=STAGE_A_MAX_RESIDUES,
        use_cache=True,
    )
    if failed or len(proteins) != 25:
        raise SystemExit(f"Expected 25 proteins, loaded {len(proteins)} ({failed} failed)")

    epoch_dir = args.run_dir / "epochs"
    snapshots = sorted(epoch_dir.glob("epoch_*.pt"))
    if not snapshots:
        raise SystemExit(f"No epoch snapshots in {epoch_dir}")

    p2_global_start = args.p2_global_start
    if p2_global_start is None:
        metrics_path = args.run_dir / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            p2_starts = [
                int(m["global_epoch"])
                for m in metrics
                if m.get("phase") == 2
            ]
            if p2_starts:
                p2_global_start = min(p2_starts)
        if p2_global_start is None:
            p2_global_start = 137

    per_epoch: dict[str, dict[str, dict]] = {}
    starvation_counts: dict[str, int] = defaultdict(int)
    p2_epochs = 0

    for ckpt in snapshots:
        ge = int(ckpt.stem.split("_")[-1])
        phase = 2 if ge >= p2_global_start else 1
        if phase < args.phase_min:
            continue
        if phase == 2:
            p2_epochs += 1
        logger.info("Evaluating %s (global epoch %d)", ckpt.name, ge)
        rows = _eval_epoch(ckpt, proteins, args.device)
        per_epoch[str(ge)] = rows
        for pdb_id, row in rows.items():
            if row["min_routing_fraction"] < STARVE_THRESHOLD:
                starvation_counts[pdb_id] += 1

    stable = sorted(
        [(pdb, cnt) for pdb, cnt in starvation_counts.items()],
        key=lambda x: (-x[1], x[0]),
    )
    never = [p["pdb_id"].upper() for p in proteins if p["pdb_id"].upper() not in starvation_counts]

    meta = {
        str(p["pdb_id"]).upper(): {
            "fold_id": str(p.get("fold_id", "")),
            "gene": str(p.get("gene", "")),
            "n_residues": int(p.get("n_residues", 0)),
        }
        for p in proteins
    }

    report = {
        "run_dir": str(args.run_dir),
        "phase_min": args.phase_min,
        "p2_global_start": p2_global_start,
        "p2_epochs_evaluated": p2_epochs,
        "starvation_threshold": STARVE_THRESHOLD,
        "starvation_epoch_counts": dict(starvation_counts),
        "never_starved": never,
        "stable_ranked": [
            {
                "pdb_id": pdb,
                "starvation_epochs": cnt,
                "pct_of_p2": round(100.0 * cnt / max(p2_epochs, 1), 1),
                **meta.get(pdb, {}),
            }
            for pdb, cnt in stable
        ],
        "per_epoch_worst": [],
    }

    for ge_str, rows in sorted(per_epoch.items(), key=lambda x: int(x[0])):
        worst = min(rows.items(), key=lambda x: x[1]["min_routing_fraction"])
        report["per_epoch_worst"].append(
            {
                "global_epoch": int(ge_str),
                "worst_pdb": worst[0],
                "worst_min_r": worst[1]["min_routing_fraction"],
                "batch_min_r": min(r["min_routing_fraction"] for r in rows.values()),
            }
        )

    out_path = args.output or (args.run_dir / "per_structure_routing_p2.json")
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out_path)

    print("\n=== P2 per-structure starvation (min_r < 0.05) ===")
    print(f"P2 epochs evaluated: {p2_epochs}")
    for row in report["stable_ranked"][:15]:
        print(
            f"  {row['pdb_id']:5} {row.get('gene',''):6} fold={row.get('fold_id',''):16} "
            f"n={row.get('n_residues',0):4}  starved {row['starvation_epochs']}/{p2_epochs} "
            f"({row['pct_of_p2']}%)"
        )
    if never:
        print(f"\nNever starved ({len(never)}): {', '.join(sorted(never))}")


if __name__ == "__main__":
    main()
