#!/usr/bin/env python3
"""Run aleatoric independence probe on a checkpoint + corpus.

Tests: does ν_ale vary beyond ρ and MoE expert assignment after OLS controls?

Usage:
  python -m experiments.diagnostics.aleatoric_independence_probe \\
    --checkpoint checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \\
    --manifest manifests/v6_corpus_stage_a_small_v1.json \\
    --json-out checkpoints/v6/diagnostics/aleatoric_independence_route_v1.json \\
    --pdb-local
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diagnostics.aleatoric_corpus_diagnostics import (
    _load_manifest_structures,
    _parse_structures,
    collect_corpus_rows,
)
from experiments.training.v6.assess_checkpoint import load_v6_model
from science.training.aleatoric_independence_probe import aleatoric_independence_probe

logger = logging.getLogger("aleatoric_independence_probe")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Aleatoric independence from ρ and expert (minimal probe)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--include-geometry",
        action="store_true",
        help="Add disc_r, clustering, cone_depth to full OLS model",
    )
    parser.add_argument("--pdb-local", action="store_true")
    args = parser.parse_args()

    if args.pdb_local:
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"

    if args.structures:
        structures = _parse_structures(args.structures)
    elif args.manifest:
        structures = _load_manifest_structures(args.manifest)
    else:
        parser.error("Provide --manifest or --structures")

    model = load_v6_model(args.checkpoint, args.device)
    model.eval()

    rows = collect_corpus_rows(
        model,
        structures,
        args.pdb_dir,
        args.device,
        structural_disc_frozen=True,
    )
    report = aleatoric_independence_probe(rows, include_geometry=args.include_geometry)
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["manifest"] = str(args.manifest.resolve()) if args.manifest else None
    report["n_structures"] = len(structures)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote JSON → {args.json_out}")
    print(f"Verdict: {report.get('verdict')}")
    print(report.get("interpretation", ""))
    m = report.get("marginal") or {}
    p = report.get("partial_after_controls") or {}
    o = report.get("ols") or {}
    print(
        f"  r(ale,ρ)={m.get('r_ale_rho', float('nan')):.3f} "
        f"partial|expert={p.get('r_ale_rho_given_expert', float('nan')):.3f} "
        f"η²(expert)={m.get('eta2_expert', float('nan')):.3f} "
        f"η²(expert|ρ)={p.get('eta2_expert_given_rho', float('nan')):.3f} "
        f"R²_full={o.get('r2_full_model', float('nan')):.3f} "
        f"residual_std_ratio={o.get('residual_std_ratio', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
