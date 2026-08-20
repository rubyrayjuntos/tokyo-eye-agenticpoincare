#!/usr/bin/env python3
"""Full-chain Gini reduction analysis — structural monopoly closeout.

Compares knockout ``out_effect`` concentration on:

* baseline: ``HEALTHY_FIX1_CKPT`` (``v66_healthy_sealed.pt``)
* champion: ``FIX1_SPARSITY_CHAMPION_CKPT``

Panel: 2SHP, 3PP0, 4OBE, 4DSO, 5VQ2.

Metrics (full chain only — never hub-slice *H*):
  G_full, top-1% / top-10% mass share, max/median.
  ΔG = G_base − G_champion (positive ⇒ monopoly reduced).

Artifact: ``gini_reduction_analysis.json`` with archived arrays.
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
from science.dtie.common.gini_flow_concentration import compare_concentration
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_PANEL: list[tuple[str, str]] = [
    ("2SHP", "A"),
    ("3PP0", "A"),
    ("4OBE", "A"),
    ("4DSO", "A"),
    ("5VQ2", "A"),
]


def _scan_out_effect(
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
) -> tuple[np.ndarray, int]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain} from {pdb_dir}")

    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    scan = knockout_scan(model, prot, device)
    oe = np.asarray(scan["out_effect"], dtype=np.float64)[:n]
    return oe, int(n)


def grade_structure(
    *,
    baseline_ckpt: Path,
    champion_ckpt: Path,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
    archive_arrays: bool = True,
) -> dict[str, Any]:
    oe_base, n_b = _scan_out_effect(baseline_ckpt, pdb_id, chain, pdb_dir, device)
    oe_champ, n_c = _scan_out_effect(champion_ckpt, pdb_id, chain, pdb_dir, device)
    if n_b != n_c:
        raise RuntimeError(
            f"{pdb_id}:{chain} residue count mismatch baseline={n_b} champion={n_c}"
        )
    if oe_base.shape != oe_champ.shape:
        raise RuntimeError(
            f"{pdb_id}:{chain} out_effect shape mismatch "
            f"{oe_base.shape} vs {oe_champ.shape}"
        )

    cmp_ = compare_concentration(oe_base, oe_champ, archive_arrays=archive_arrays)
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "n_residues": int(n_b),
        **cmp_,
    }


def grade(
    *,
    baseline_ckpt: Path,
    champion_ckpt: Path,
    pdb_dir: Path,
    device: str,
    panel: list[tuple[str, str]] | None = None,
    archive_arrays: bool = True,
) -> dict[str, Any]:
    panel = list(panel or DEFAULT_PANEL)
    structures: list[dict[str, Any]] = []
    for pdb_id, chain in panel:
        structures.append(
            grade_structure(
                baseline_ckpt=baseline_ckpt,
                champion_ckpt=champion_ckpt,
                pdb_id=pdb_id,
                chain=chain,
                pdb_dir=pdb_dir,
                device=device,
                archive_arrays=archive_arrays,
            )
        )

    deltas = [float(s["delta_gini"]) for s in structures if np.isfinite(s["delta_gini"])]
    n_reduced = sum(1 for s in structures if s.get("monopoly_reduced"))
    mean_delta = float(np.mean(deltas)) if deltas else float("nan")

    return {
        "schema_version": 1,
        "probe": "gini_reduction_analysis",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "python_constant": "experiments.training.v66.healthy_fix1.HEALTHY_FIX1_CKPT",
            "checkpoint": str(baseline_ckpt),
            "role": "G_base — healthy sealed pre-sparsity continue",
        },
        "champion": {
            "python_constant": (
                "experiments.training.v66.healthy_fix1.FIX1_SPARSITY_CHAMPION_CKPT"
            ),
            "checkpoint": str(champion_ckpt),
            "role": "G_champion — sparsity confirm banked epoch",
        },
        "protocol": {
            "metric_scope": "full_chain_out_effect",
            "hub_slice_forbidden": True,
            "delta_gini": "G_base - G_champion",
            "positive_delta_means": "structural_monopoly_reduced",
            "suite": [
                "gini",
                "top1pct_mass_share",
                "top10pct_mass_share",
                "max_over_median",
            ],
            "panel": [{"pdb_id": p, "chain": c} for p, c in panel],
            "archive_arrays": bool(archive_arrays),
        },
        "structures": structures,
        "summary": {
            "n_structures": len(structures),
            "n_monopoly_reduced": int(n_reduced),
            "mean_delta_gini": mean_delta,
            "all_reduced": bool(n_reduced == len(structures) and len(structures) > 0),
            "per_structure_delta_gini": {
                s["pdb_id"]: s["delta_gini"] for s in structures
            },
        },
        "notes": [
            "Do not interpret hub-slice top/|H| ratios as monopoly evidence.",
            "Expert max_share is a separate MoE load metric; not graded here.",
            "Observational stamp — no hard Pass bar on ΔG magnitude in v1.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        HEALTHY_FIX1_CKPT,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=HEALTHY_FIX1_CKPT,
        help="G_base checkpoint (default: HEALTHY_FIX1_CKPT / v66_healthy_sealed.pt)",
    )
    p.add_argument(
        "--champion",
        type=Path,
        default=FIX1_SPARSITY_CHAMPION_CKPT,
        help="G_champion checkpoint (default: FIX1_SPARSITY_CHAMPION_CKPT)",
    )
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--no-archive-arrays",
        action="store_true",
        help="Omit full out_effect vectors from JSON (metrics only).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/routing_sparsity/gini_reduction_analysis.json"
        ),
    )
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if not args.baseline.is_file():
        raise SystemExit(f"missing baseline checkpoint: {args.baseline}")
    if not args.champion.is_file():
        raise SystemExit(f"missing champion checkpoint: {args.champion}")

    report = grade(
        baseline_ckpt=args.baseline,
        champion_ckpt=args.champion,
        pdb_dir=args.pdb_dir,
        device=args.device,
        archive_arrays=not args.no_archive_arrays,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
