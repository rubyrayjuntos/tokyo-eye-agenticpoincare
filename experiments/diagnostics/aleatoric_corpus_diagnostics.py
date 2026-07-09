#!/usr/bin/env python3
"""Corpus-wide residue-first aleatoric diagnostics.

Exports histogram, per-protein dehydron burden, active-learning ranking, and
local investigation sites. Global aleatoric std is reported as a **health monitor
only** — not a site certification gate.

Usage:
  python -m experiments.diagnostics.aleatoric_corpus_diagnostics \\
    --checkpoint checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \\
    --manifest manifests/v6_corpus_stage_a_small_v1.json \\
    --json-out checkpoints/v6/diagnostics/aleatoric_corpus_route_v1.json \\
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

from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.v5.gnn.model import precompute_clustering
from science.training.aleatoric_residue_diagnostics import (
    DEFAULT_T_ALE_PERCENTILE,
    aleatoric_dataset_health_report,
    global_aleatoric_health_monitor,
)
from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows

logger = logging.getLogger("aleatoric_corpus_diagnostics")


def _load_manifest_structures(manifest_path: Path) -> list[dict[str, str]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for entry in raw.get("proteins", []):
        if not entry.get("enabled", True):
            continue
        out.append(
            {
                "pdb_id": str(entry["pdb_id"]).upper(),
                "chain": str(entry.get("chain", "A")),
                "gene": str(entry.get("gene", "")),
                "fold_id": str(entry.get("fold_id", "")),
            }
        )
    return out


def _parse_structures(raw: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            pdb, chain = item.split(":", 1)
        else:
            pdb, chain = item, "A"
        out.append({"pdb_id": pdb.upper(), "chain": chain, "gene": "", "fold_id": ""})
    return out


def collect_corpus_rows(
    model: torch.nn.Module,
    structures: list[dict[str, str]],
    pdb_dir: Path,
    device: str,
    *,
    structural_disc_frozen: bool,
) -> list[dict]:
    all_rows: list[dict] = []
    for spec in structures:
        pdb_id = spec["pdb_id"]
        chain = spec["chain"]
        prot = load_protein_graph(pdb_id, chain, pdb_dir)
        if prot is None:
            logger.warning("Skip missing graph %s:%s", pdb_id, chain)
            continue
        with torch.no_grad():
            data = prepare_training_batch(
                model,
                prot,
                device,
                structural_disc_frozen=structural_disc_frozen,
            )
            data = precompute_clustering(data)
            out = model(data)
        rows = extract_residue_uncertainty_rows(
            out,
            prot,
            graph_data=data,
            structure_id=pdb_id.lower(),
            chain=chain,
        )
        all_rows.extend(rows)
        logger.info(
            "  %s:%s n=%d ale_mean=%.4f ale_max=%.4f",
            pdb_id,
            chain,
            len(rows),
            sum(r["aleatoric"] for r in rows) / max(len(rows), 1),
            max(r["aleatoric"] for r in rows) if rows else 0.0,
        )
    return all_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Residue-first aleatoric corpus diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Corpus manifest JSON (enabled proteins only)",
    )
    parser.add_argument(
        "--structures",
        default=None,
        help="Comma PDB:chain list (overrides manifest when set)",
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--t-ale",
        type=float,
        default=None,
        help="Absolute high-aleatoric threshold (overrides percentile default)",
    )
    parser.add_argument(
        "--t-ale-percentile",
        type=float,
        default=DEFAULT_T_ALE_PERCENTILE,
        help="Corpus percentile for t_ale when --t-ale omitted (default P90)",
    )
    parser.add_argument(
        "--structural-disc-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pdb-local", action="store_true")
    parser.add_argument("--print-top", type=int, default=10)
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
        structural_disc_frozen=args.structural_disc_frozen,
    )
    gene_by = {s["pdb_id"].lower(): s.get("gene", "") for s in structures}
    fold_by = {s["pdb_id"].lower(): s.get("fold_id", "") for s in structures}

    report = aleatoric_dataset_health_report(
        rows,
        t_ale=args.t_ale,
        t_ale_percentile=args.t_ale_percentile,
        gene_by_structure=gene_by,
        fold_by_structure=fold_by,
    )
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["manifest"] = str(args.manifest.resolve()) if args.manifest else None
    report["global_health"] = global_aleatoric_health_monitor(rows)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote JSON → {args.json_out}")

    hist = report["corpus_histogram"]
    t_meta = report.get("t_ale_resolution") or {}
    pct = t_meta.get("percentile")
    pct_label = f" P{int(pct)}" if pct is not None else ""
    print(
        f"Corpus n={hist.get('n_residues')} ale_std={hist.get('std'):.4f} "
        f"skew={hist.get('skewness'):.3f} right_skew_ok={hist.get('right_skew_ok')} "
        f"t_ale={report.get('t_ale'):.4f} ({t_meta.get('mode', '?')}{pct_label}) "
        f"frac>t_ale={hist.get('fraction_above_t_ale'):.3f}"
    )
    print(
        f"Investigation sites={report['n_investigation_sites']} "
        f"(local triage, not global gate)"
    )

    if args.print_top > 0:
        print(f"\n--- Active learning top {args.print_top} (max aleatoric per protein) ---")
        for item in report["active_learning_ranking"][: args.print_top]:
            print(
                f"  #{item['active_learning_rank']} {item['structure_id']}:{item['chain']} "
                f"max_ale={item['aleatoric_max']:.4f} "
                f"high_frac={item['high_aleatoric_fraction']:.3f} "
                f"var={item['aleatoric_variance']:.5f}"
            )


if __name__ == "__main__":
    main()
