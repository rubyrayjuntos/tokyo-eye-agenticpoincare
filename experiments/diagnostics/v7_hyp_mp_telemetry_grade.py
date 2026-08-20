#!/usr/bin/env python3
"""Grade cheap Hyp-MP telemetry on sealed (or Phase-A) Θ — one forward per structure.

Baseline concordance vs knockout hubs is report-only when --knockout-compare is set.
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
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.tokyo_eye.hyp_mp_telemetry import attach_hyp_mp_telemetry
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_TARGETS = ("4OBE:A", "4LPK:A", "6GOD:A")
DEFAULT_OUT = Path(
    "checkpoints/v7/diagnostics/hyp_mp_telemetry/hyp_mp_telemetry_baseline.json"
)


def _parse_targets(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for tok in raw.split():
        pdb, _, chain = tok.partition(":")
        out.append((pdb.upper(), chain or "A"))
    return out


def _run_one(
    *,
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
    k_frac: float,
) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    with torch.no_grad():
        out = model(data)
    attach_hyp_mp_telemetry(out, data, k_frac=k_frac)
    tel = out["hyp_mp_telemetry"]
    ids = list(prot.get("residue_ids") or [])[:n]
    idx_map = residue_index_map(ids)
    rev = {i: rs for rs, i in idx_map.items()}
    hubs = []
    for i, s in zip(tel.get("hub_indices") or [], tel.get("hub_strengths") or []):
        if i >= n:
            continue
        hubs.append(
            {
                "graph_index": int(i),
                "auth_resseq": rev.get(int(i)),
                "strength": float(s),
            }
        )
    audit = out.get("audit_trail") or {}
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "n_residues": n,
        "hyp_mp_primary": bool(audit.get("hyp_mp_primary")),
        "curvature_c": float(
            audit["curvature_value"].detach().cpu()
            if torch.is_tensor(audit.get("curvature_value"))
            else audit.get("curvature_value") or float("nan")
        ),
        "n_edges": tel.get("n_edges"),
        "n_hubs": tel.get("n_hubs"),
        "strength_cv": tel.get("strength_cv"),
        "hubs_top10pct": hubs,
        "hub_auth_resseqs": [h["auth_resseq"] for h in hubs],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--targets", default=" ".join(DEFAULT_TARGETS))
    p.add_argument("--k-frac", type=float, default=0.10)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    panel = []
    for pdb_id, chain in _parse_targets(args.targets):
        print(f"Hyp-MP telemetry {pdb_id}:{chain} ...", flush=True)
        row = _run_one(
            checkpoint=args.checkpoint,
            pdb_id=pdb_id,
            chain=chain,
            pdb_dir=args.pdb_dir,
            device=args.device,
            k_frac=float(args.k_frac),
        )
        print(
            f"  |H|={row['n_hubs']} edges={row['n_edges']} "
            f"cv={row['strength_cv']:.3f} hubs={row['hub_auth_resseqs'][:8]}...",
            flush=True,
        )
        panel.append(row)

    out = {
        "schema_version": 1,
        "probe": "tokyo_eye_v7_hyp_mp_telemetry",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "k_frac": float(args.k_frac),
        "panel": panel,
        "notes": [
            "One forward per structure; no knockout.",
            "Hub proxy = top-k% by Hyp-MP edge geodesic strength.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
