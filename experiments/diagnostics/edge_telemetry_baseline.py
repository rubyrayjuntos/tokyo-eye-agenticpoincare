#!/usr/bin/env python3
"""Baseline edge telemetry on Stage A corpus — telemetry-alive check first.

Writes JSON summaries tagged for track-only MLflow (not P_ENTRY_GATE).
Includes per-edge arrays and flow-stratified same_expert_excess (spec §3).

Usage:
  TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/edge_telemetry_baseline.py \\
    --corpus manifests/v6_corpus_stage_a_small_v1.json \\
    --compare-checkpoints \\
      checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/v6_best.pt \\
      checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \\
    --json-out checkpoints/v6/runs/edge_telemetry_mvp_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.edge_telemetry import (
    MLFLOW_TAG_NOT_GATE,
    MLFLOW_TAG_TELEMETRY,
    aggregate_corpus_records,
    collect_edge_telemetry,
    record_to_export_dict,
)

logger = logging.getLogger("edge_telemetry_baseline")


def _run_checkpoint(
    ckpt: Path,
    proteins: list[dict],
    device: str,
    *,
    structural_disc_frozen: bool,
    include_per_edge: bool,
) -> dict:
    model = load_v6_model(ckpt, device)
    model.eval()
    records = []
    with torch.no_grad():
        for prot in proteins:
            pdb_id = str(prot.get("pdb_id", "?"))
            chain = str(prot.get("chain", "A"))
            data = prepare_training_batch(
                model, prot, device, structural_disc_frozen=structural_disc_frozen
            )
            out = model(data)
            ca = prot.get("ca_coords")
            ca_np = ca.detach().cpu().numpy() if ca is not None else None
            rec = collect_edge_telemetry(
                model,
                data,
                out,
                structure_id=pdb_id,
                chain=chain,
                ca_coords=ca_np,
                include_per_edge=include_per_edge,
            )
            records.append(rec)
            fs = rec.flow_stratification or {}
            delta = fs.get("same_expert_excess_high_minus_low", float("nan"))
            status = "ALIVE" if rec.telemetry_alive else f"DEAD({rec.telemetry_alive_reason})"
            logger.info(
                "%s:%s n_edges=%d corr=%.4f same=%.3f null=%.3f "
                "epi_std=%.3f ale_std=%.3f flow_Δ=%.3f [%s]",
                pdb_id,
                chain,
                rec.n_edges,
                rec.edge_embed_resistance_corr,
                rec.same_expert_rate,
                rec.same_expert_null_rate,
                rec.edge_epistemic_var_std,
                rec.edge_aleatoric_var_std,
                delta if delta == delta else float("nan"),
                status,
            )

    agg = aggregate_corpus_records(records)
    learned_c = float(model.curvature.detach().cpu().item())
    return {
        "checkpoint": str(ckpt.resolve()),
        "learned_curvature": learned_c,
        "structural_disc_frozen": structural_disc_frozen,
        "include_per_edge": include_per_edge,
        "mlflow_tags": [MLFLOW_TAG_TELEMETRY, MLFLOW_TAG_NOT_GATE],
        "structures": [record_to_export_dict(r) for r in records],
        "corpus_aggregate": agg,
        "signatures": {
            "telemetry_alive": agg.get("corpus_telemetry_alive_fraction", 0.0) >= 1.0,
            "collapsed_routing_null": {
                "note": "same_expert_rate vs same_expert_null_rate (Σp²)",
                "corpus_same_expert_rate_mean": agg.get("corpus_same_expert_rate_mean"),
                "corpus_same_expert_null_rate_mean": agg.get("corpus_same_expert_null_rate_mean"),
                "corpus_dominant_expert_share_mean": agg.get("corpus_dominant_expert_share_mean"),
            },
            "flow_stratification": {
                "note": "healthy when same_expert_excess_high > same_expert_excess_low",
                "corpus_excess_high_minus_low_mean": agg.get(
                    "corpus_flow_excess_high_minus_low_mean"
                ),
                "corpus_healthy_flow_alignment_fraction": agg.get(
                    "corpus_healthy_flow_alignment_fraction"
                ),
            },
            "resistance_corr_near_zero": abs(
                agg.get("corpus_edge_embed_resistance_corr_mean", 0.0)
            )
            < 0.15,
            "same_expert_excess_vs_null": {
                "corpus_same_expert_excess_mean": agg.get("corpus_same_expert_excess_mean"),
            },
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Edge telemetry MVP baseline")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--compare-checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--structural-disc-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--per-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export per-edge flow / uncertainty / same_expert arrays in JSON",
    )
    parser.add_argument("--pdb-local", action="store_true")
    args = parser.parse_args()

    if args.pdb_local:
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"

    checkpoints: list[Path] = []
    if args.compare_checkpoints:
        checkpoints = list(args.compare_checkpoints)
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        parser.error("Provide --checkpoint or --compare-checkpoints")

    proteins, failed = load_training_proteins(args.pdb_dir, args.corpus)
    if failed:
        logger.warning("%d corpus entries failed to load", failed)
    if not proteins:
        raise SystemExit("No proteins loaded")

    results = []
    for ckpt in checkpoints:
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        results.append(
            _run_checkpoint(
                ckpt,
                proteins,
                args.device,
                structural_disc_frozen=args.structural_disc_frozen,
                include_per_edge=args.per_edge,
            )
        )

    payload = {"version": "edge_telemetry_mvp_v2", "checkpoints": results}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.json_out}")

    for r in results:
        name = Path(r["checkpoint"]).parent.name
        agg = r["corpus_aggregate"]
        sig = r["signatures"]
        print(f"\n=== {name} ===")
        print(f"  telemetry_alive: {sig['telemetry_alive']}")
        print(
            f"  same_expert: rate={agg.get('corpus_same_expert_rate_mean', 0):.3f} "
            f"null={agg.get('corpus_same_expert_null_rate_mean', 0):.3f}"
        )
        print(
            f"  flow excess (high−low): "
            f"{agg.get('corpus_flow_excess_high_minus_low_mean', 0):.4f}  "
            f"healthy_frac={agg.get('corpus_healthy_flow_alignment_fraction', 0):.2f}"
        )
        print(
            f"  edge epi_std / ale_std (corpus mean of per-structure std): "
            f"{agg.get('corpus_edge_epistemic_var_std_mean', 0):.3f} / "
            f"{agg.get('corpus_edge_aleatoric_var_std_mean', 0):.3f}"
        )
        print(
            f"  edge_embed_resistance_corr_mean: "
            f"{agg.get('corpus_edge_embed_resistance_corr_mean', 0):.4f}"
        )


if __name__ == "__main__":
    main()
