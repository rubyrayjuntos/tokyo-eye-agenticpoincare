#!/usr/bin/env python3
"""Forward knockout vs classical betweenness on Fix-1 champion (z-norm safe).

Companion to hub_knockout_classical — single-checkpoint sweep for RAF1 mix targets.
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

from experiments.diagnostics.hub_knockout_classical import (
    HOLD_SPEARMAN_MIN,
    score_structure_knockout_vs_classical,
)
from experiments.diagnostics.kras_knockout_causal import knockout_scan
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.classical_network_metrics import classical_network_metrics
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_TARGETS: list[tuple[str, str]] = [
    ("4OBE", "A"),
    ("9AXM", "B"),
    ("3OMV", "A"),
]


def _align_prot_features(model: torch.nn.Module, prot: dict[str, Any]) -> dict[str, Any]:
    """Match training batch: topology 3-vector + optional SASA in gate."""
    data = prot["data"]
    in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
    if data.x.size(-1) > in_f:
        if getattr(data, "sasa", None) is None and data.x.size(-1) > 3:
            data.sasa = data.x[:, 3].detach().clone()
        data.x = data.x[:, :in_f].contiguous()
        prot = {**prot, "data": data}
    return prot


def run_one(
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain} from {pdb_dir}")

    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)

    ca = prot.get("ca_coords")
    if ca is None:
        raise ValueError(f"{pdb_id}:{chain} missing ca_coords")

    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
    ca_np = ca_np[:n]

    classical = classical_network_metrics(ca_np)
    scan = knockout_scan(model, prot, device)
    out_effect = np.asarray(scan["out_effect"], dtype=np.float64)
    btw = np.asarray(classical["betweenness"], dtype=np.float64)

    scored = score_structure_knockout_vs_classical(
        out_effect, btw, seed=hash(pdb_id + chain) % 10_000
    )

    # Top hub residues by classical betweenness (for attribution)
    top_idx = np.argsort(-btw)[:5]
    residue_ids = prot.get("residue_ids") or []

    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "checkpoint": str(checkpoint),
        "n_residues": int(n),
        **scored,
        "hold_threshold_spearman": HOLD_SPEARMAN_MIN,
        "top_betweenness": [
            {
                "graph_index": int(i),
                "residue_id": str(residue_ids[i]) if i < len(residue_ids) else None,
                "betweenness": float(btw[i]),
                "knockout_out_effect": float(out_effect[i]),
            }
            for i in top_idx
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/fix1_s4_raf1_mix_continue_v1/epochs/epoch_066.pt"
        ),
    )
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/fix1_expand_biology/"
            "hub_knockout_epoch066_sweep.json"
        ),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    rows: list[dict[str, Any]] = []
    for pdb_id, chain in DEFAULT_TARGETS:
        print(f"knockout {pdb_id}:{chain} ...", flush=True)
        row = run_one(args.checkpoint, pdb_id, chain, args.pdb_dir, args.device)
        rows.append(row)
        print(
            f"  ρ={row['spearman_betweenness']:.3f} holds={row['holds']} "
            f"median_out={row['median_out_effect']:.4f}",
            flush=True,
        )

    rhos = [r["spearman_betweenness"] for r in rows if r["spearman_betweenness"] is not None]
    report = {
        "schema_version": 1,
        "probe": "hub_knockout_classical",
        "method": "forward_pass_input_knockout_no_gradients",
        "note": "Jacobian forbidden on input_feature_zscore=True (Fix-1).",
        "checkpoint": str(args.checkpoint),
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "structures": rows,
        "summary": {
            "median_spearman_betweenness": float(np.median(rhos)) if rhos else None,
            "holds_count": sum(1 for r in rows if r["holds"]),
            "holds_fraction": sum(1 for r in rows if r["holds"]) / len(rows) if rows else 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
