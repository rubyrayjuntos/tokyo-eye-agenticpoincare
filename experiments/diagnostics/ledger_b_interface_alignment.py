#!/usr/bin/env python3
"""Ledger B grade — pre-registered interface recall@top-10% knockout flow.

Loads ONLY ``data/gates/ledger_b_interface_prereg_src_shp2.json``.
Never edits residue sets from smoke-test hubs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.diagnostics.kras_knockout_causal import knockout_scan
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.dtie.common.ledger_b_interface import (
    interface_alignment_score,
    resolve_interface_set,
)
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_PREREG = Path("data/gates/ledger_b_interface_prereg_src_shp2.json")


def _grade_structure(
    *,
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    structure_spec: dict[str, Any],
    pass_form: dict[str, Any],
    pdb_dir: Path,
    device: str,
) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")

    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    residue_ids = list(prot.get("residue_ids") or [])[:n]
    idx_map = residue_index_map(residue_ids)
    present = set(idx_map.keys())

    resolved = resolve_interface_set(structure_spec, present)
    scan = knockout_scan(model, prot, device)
    out_effect = np.asarray(scan["out_effect"], dtype=np.float64)[:n]

    scored = interface_alignment_score(
        out_effect,
        idx_map,
        resolved["interface_resseqs"],
        k_frac=0.10,
        recall_threshold=float(pass_form.get("primary_threshold", 0.25)),
        enrichment_threshold=float(pass_form.get("secondary_threshold", 1.25)),
    )

    # Secondary catalytic set (report only)
    secondary = resolved.get("secondary_resseqs") or []
    secondary_report = None
    if secondary:
        secondary_report = interface_alignment_score(
            out_effect,
            idx_map,
            secondary,
            k_frac=0.10,
            recall_threshold=float(pass_form.get("primary_threshold", 0.25)),
            enrichment_threshold=float(pass_form.get("secondary_threshold", 1.25)),
        )

    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "gene": structure_spec.get("gene"),
        "n_residues": int(n),
        "resolved": resolved,
        "primary": scored,
        "secondary_report_only": secondary_report,
        "pass": bool(scored.get("pass")),
        "refs": structure_spec.get("refs"),
    }


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        HEALTHY_FIX1_CKPT,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            FIX1_SPARSITY_CHAMPION_CKPT
            if FIX1_SPARSITY_CHAMPION_CKPT.is_file()
            else HEALTHY_FIX1_CKPT
        ),
    )
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/routing_sparsity/"
            "ledger_b_interface_alignment.json"
        ),
    )
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if not args.prereg.is_file():
        raise SystemExit(f"missing Ledger B pre-reg stamp: {args.prereg}")
    prereg = json.loads(args.prereg.read_text())
    pass_form = prereg["pass_form"]
    structures = prereg["structures"]

    rows: list[dict[str, Any]] = []
    for pdb_id, spec in structures.items():
        chain = str(spec.get("chain") or "A")
        print(f"Ledger B {pdb_id}:{chain} ...", flush=True)
        row = _grade_structure(
            checkpoint=args.checkpoint,
            pdb_id=pdb_id,
            chain=chain,
            structure_spec=spec,
            pass_form=pass_form,
            pdb_dir=args.pdb_dir,
            device=args.device,
        )
        print(
            f"  recall={row['primary']['recall_at_top_k']:.3f} "
            f"enrich={row['primary']['enrichment']:.3f} pass={row['pass']} "
            f"|I|={row['resolved']['n_interface']} missing={len(row['resolved']['missing_from_deposit'])}",
            flush=True,
        )
        rows.append(row)

    all_pass = all(bool(r["pass"]) for r in rows) and len(rows) >= 2
    verdict = {
        "pass": all_pass,
        "outcome": "Pass" if all_pass else "Fail",
        "n": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "rule": pass_form.get("panel_pass"),
    }

    out = {
        "schema_version": 1,
        "probe": "ledger_b_interface_alignment",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "prereg": str(args.prereg),
        "prereg_locked_at": prereg.get("locked_at"),
        "pass_form": pass_form,
        "panel": rows,
        "verdict": verdict,
        "notes": [
            "Residue sets loaded solely from Ledger B pre-reg JSON.",
            "Smoke-test hub lists were not used to select residues.",
            "Ledger A CB bar remains 0.50 (unchanged).",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.output}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
