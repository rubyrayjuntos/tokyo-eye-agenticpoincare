#!/usr/bin/env python3
"""G4 — holdout P8 eval for v3 aleatoric shaping (var_penalty / hinge).

Tags Stage A residues with stable G4 holdout masks (default: whole-protein holdout)
and evaluates frozen G4 pass rule (holdout P8 + transfer ratio + ρ coupling).

Usage:
  TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/g4_aleatoric_holdout_eval.py \\
    --checkpoint checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \\
    --holdout-seeds 42,7 \\
    --json-out checkpoints/v6/diagnostics/g4_holdout_report.json \\
    --pdb-local
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
from science.training.aleatoric_shaping_holdout import (
    G4_DEFAULT_HOLDOUT_FRACTION,
    G4_DEFAULT_HOLDOUT_MODE,
    G4_DEFAULT_MASK_SEED,
    corpus_protein_holdout_ids,
    holdout_corpus_contrast,
    holdout_mask_metadata,
    tag_residue_rows_with_shaping_holdout,
)
from science.training.evidential_validation import g4_aleatoric_shaping_holdout_report
from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows

logger = logging.getLogger("g4_aleatoric_holdout_eval")
STAGE_A_MANIFEST = ROOT / "manifests/v6_corpus_stage_a_small_v1.json"


def _parse_seeds(spec: str) -> list[int]:
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def eval_one_seed(
    checkpoint: Path,
    proteins: list[dict],
    model: torch.nn.Module,
    *,
    device: str,
    holdout_fraction: float,
    holdout_mode: str,
    holdout_seed: int,
    structural_disc_frozen: bool,
    checkpoint_eligible: bool | None = None,
) -> dict:
    pdb_ids = [str(p.get("pdb_id", "")).upper() for p in proteins]
    holdout_proteins = (
        corpus_protein_holdout_ids(
            pdb_ids, holdout_fraction=holdout_fraction, seed=holdout_seed
        )
        if holdout_mode == "protein"
        else None
    )
    rows: list[dict] = []
    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(
                model, prot, device, structural_disc_frozen=structural_disc_frozen
            )
            out = model(data)
            prot_rows = extract_residue_uncertainty_rows(out, prot)
            tag_residue_rows_with_shaping_holdout(
                prot_rows,
                prot,
                holdout_fraction=holdout_fraction,
                seed=holdout_seed,
                holdout_mode=holdout_mode,  # type: ignore[arg-type]
                corpus_holdout_proteins=holdout_proteins,
            )
            rows.extend(prot_rows)

    holdout_meta = holdout_mask_metadata(rows)
    if holdout_proteins is not None:
        holdout_meta["corpus_holdout_proteins"] = sorted(holdout_proteins)
        holdout_meta["holdout_corpus_contrast"] = holdout_corpus_contrast(
            proteins, holdout_proteins, holdout_seed=holdout_seed
        )

    g4 = g4_aleatoric_shaping_holdout_report(
        rows,
        holdout_metadata=holdout_meta,
        checkpoint_eligible=checkpoint_eligible,
    )
    return {
        "holdout_seed": holdout_seed,
        "holdout_metadata": holdout_meta,
        "g4_report": g4,
        "n_residues": len(rows),
        "n_holdout": g4.get("n_holdout"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="G4 aleatoric shaping holdout P8 eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--holdout-fraction", type=float, default=G4_DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument(
        "--holdout-mode",
        choices=("protein", "residue_stratified"),
        default=G4_DEFAULT_HOLDOUT_MODE,
    )
    parser.add_argument(
        "--holdout-seeds",
        default=str(G4_DEFAULT_MASK_SEED),
        help="Comma-separated seeds for eval-only holdout rotation (e.g. 42,7)",
    )
    parser.add_argument(
        "--structural-disc-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-eligible",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether training gates certified this checkpoint (default: unknown)",
    )
    parser.add_argument("--pdb-local", action="store_true")
    args = parser.parse_args()

    if args.pdb_local:
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"
    if not os.environ.get("TRAINING_LOAD_FROM_PDB"):
        logger.error("Set TRAINING_LOAD_FROM_PDB=1 or pass --pdb-local")
        sys.exit(2)
    if not args.checkpoint.is_file():
        logger.error("Missing checkpoint: %s", args.checkpoint)
        sys.exit(1)

    seeds = _parse_seeds(args.holdout_seeds)
    proteins, failed = load_training_proteins(args.pdb_dir, STAGE_A_MANIFEST)
    if failed:
        logger.warning("Corpus load: %d proteins failed", failed)

    model = load_v6_model(args.checkpoint, args.device)
    model.eval()

    by_seed: dict[str, dict] = {}
    for seed in seeds:
        logger.info("=== G4 eval holdout_seed=%d ===", seed)
        result = eval_one_seed(
            args.checkpoint,
            proteins,
            model,
            device=args.device,
            holdout_fraction=args.holdout_fraction,
            holdout_mode=args.holdout_mode,
            holdout_seed=seed,
            structural_disc_frozen=args.structural_disc_frozen,
            checkpoint_eligible=args.checkpoint_eligible,
        )
        by_seed[str(seed)] = result
        g4 = result["g4_report"]
        meta = result["holdout_metadata"]
        contrast = meta.get("holdout_corpus_contrast") or {}
        holdout_detail = (contrast.get("holdout_details") or [{}])[0]
        strat = g4.get("aleatoric_dehydron_stratification") or {}
        dehyd = strat.get("dehydron_residues") or {}
        regular = strat.get("regular_residues") or {}
        logger.info(
            "seed=%d pass=%s holdout=%s transfer=%s (%.3f) r(ale,ρ)=%.3f "
            "ale_std dehyd=%.4f regular=%.4f dehyd_z=%.2f rules=%s",
            seed,
            g4.get("ok"),
            meta.get("holdout_proteins"),
            g4.get("relative_lift_transfer_status"),
            g4.get("relative_lift_transfer_ratio", float("nan")),
            g4.get("r_ale_rho_marginal", float("nan")),
            dehyd.get("ale_std", float("nan")),
            regular.get("ale_std", float("nan")),
            (holdout_detail.get("z_vs_corpus") or {}).get(
                "dehydron_fraction_z_vs_corpus", float("nan")
            ),
            g4.get("rule_summary"),
        )

    g4_passes = sum(1 for r in by_seed.values() if r["g4_report"].get("ok"))
    transfer_passes = sum(
        1
        for r in by_seed.values()
        if (r["g4_report"].get("relative_lift_transfer_ok"))
    )
    primary = by_seed[str(seeds[0])]
    payload = {
        "version": "g4_aleatoric_holdout_eval_v3",
        "checkpoint": str(args.checkpoint.resolve()),
        "holdout_mode": args.holdout_mode,
        "holdout_seeds": seeds,
        "primary_seed": seeds[0],
        "eval_by_seed": by_seed,
        "multi_seed_consensus": {
            "n_seeds": len(seeds),
            "g4_pass_count": g4_passes,
            "transfer_pass_count": transfer_passes,
            "interpretation": (
                f"g4_pass on {g4_passes}/{len(seeds)} holdout seeds — "
                "single-seed pass alone is insufficient when n_holdout_proteins=1"
                if len(seeds) > 1
                else "single seed; add --holdout-seeds 42,7 for rotation check"
            ),
        },
        "g4_report": primary["g4_report"],
        "holdout_metadata": primary["holdout_metadata"],
        "n_residues": primary["n_residues"],
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", args.json_out)

    print(json.dumps(payload["g4_report"], indent=2))
    sys.exit(0 if primary["g4_report"].get("ok") else 1)


if __name__ == "__main__":
    main()
