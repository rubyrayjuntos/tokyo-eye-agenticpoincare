#!/usr/bin/env python3
"""KRAS G12 neighborhood / conduit graft (matched-state OFF or ON).

OFF pre-reg: ``docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-prereg.md``
ON  pre-reg: ``docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-on-prereg.md``
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
from science.dtie.common.kras_g12_graft import (
    GRAFT_RESSEQ,
    align_out_effects,
    delta_field_spearman,
    graft_neighborhood_matched,
    graft_neighborhood_scramble,
    graft_resseq_from_donor,
    neighborhood_n12,
    neighborhood_verdict,
    residue_index_map,
    spearman_rho,
)
from science.training.gnn_lineage import load_model_from_checkpoint

ARMS = {
    "off": {
        "wt": "4LPK",
        "mut": "5US4",
        "chain": "A",
        "spec": "docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-prereg.md",
        "prereg": "data/gates/kras_g12_neighborhood_graft_prereg.json",
        "probe": "kras_g12_neighborhood_graft",
        "default_output": (
            "checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft.json"
        ),
    },
    "on": {
        "wt": "6GOD",
        "mut": "6GOF",
        "chain": "A",
        "spec": "docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-on-prereg.md",
        "prereg": "data/gates/kras_g12_neighborhood_graft_on_prereg.json",
        "probe": "kras_g12_neighborhood_graft_on",
        "default_output": (
            "checkpoints/v66/diagnostics/routing_sparsity/"
            "kras_g12_neighborhood_graft_on.json"
        ),
    },
}


def _load(pdb_id: str, chain: str, pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")
    return prot


def _scan_oe(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, dict[int, int], int]:
    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    ids = list(prot.get("residue_ids") or [])[:n]
    idx_map = residue_index_map(ids)
    oe = np.asarray(knockout_scan(model, prot, device)["out_effect"], dtype=np.float64)[:n]
    return oe, idx_map, n


def grade(
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str,
    arm: str = "off",
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    cfg = ARMS[arm]
    wt_id, mut_id, chain = cfg["wt"], cfg["mut"], cfg["chain"]

    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()

    print(f"=== neighborhood {arm.upper()}: {wt_id} ← N12 — {mut_id} ===", flush=True)
    wt = _align_prot_features(model, _load(wt_id, chain, pdb_dir))
    mut = _align_prot_features(model, _load(mut_id, chain, pdb_dir))

    n12 = neighborhood_n12(mut, wt, seed_resseq=GRAFT_RESSEQ)
    print(f"  |N12|={len(n12)}: {n12}", flush=True)

    single = graft_resseq_from_donor(
        wt, mut, target_resseq=GRAFT_RESSEQ, donor_resseq=GRAFT_RESSEQ
    )
    neigh = graft_neighborhood_matched(wt, mut, n12)
    scramble = graft_neighborhood_scramble(wt, mut, n12)

    print(f"  knockout WT {wt_id} ...", flush=True)
    oe_wt, map_wt, n_wt = _scan_oe(model, wt, device)
    print(f"  knockout mut {mut_id} ...", flush=True)
    oe_mut, map_mut, n_mut = _scan_oe(model, mut, device)
    print("  knockout single-site ...", flush=True)
    oe_single, map_single, n_single = _scan_oe(model, single, device)
    print("  knockout neighborhood ...", flush=True)
    oe_neigh, map_neigh, n_neigh = _scan_oe(model, neigh, device)
    print("  knockout scramble-neighborhood ...", flush=True)
    oe_scramble, map_scramble, n_scramble = _scan_oe(model, scramble, device)

    labels = ["wt", "mut", "single", "neighborhood", "scramble"]
    shared, aligned = align_out_effects(
        {
            "wt": oe_wt,
            "mut": oe_mut,
            "single": oe_single,
            "neighborhood": oe_neigh,
            "scramble": oe_scramble,
        },
        {
            "wt": map_wt,
            "mut": map_mut,
            "single": map_single,
            "neighborhood": map_neigh,
            "scramble": map_scramble,
        },
        labels,
    )

    rho_wt = spearman_rho(aligned["wt"], aligned["mut"])
    rho_single = spearman_rho(aligned["single"], aligned["mut"])
    rho_neigh = spearman_rho(aligned["neighborhood"], aligned["mut"])
    rho_scramble = spearman_rho(aligned["scramble"], aligned["mut"])
    verdict = neighborhood_verdict(
        rho_wt_mut=rho_wt,
        rho_single_mut=rho_single,
        rho_neighborhood_mut=rho_neigh,
        rho_scramble_mut=rho_scramble,
    )

    delta_field = delta_field_spearman(
        aligned["wt"], aligned["neighborhood"], aligned["wt"], aligned["mut"]
    )
    d = aligned["neighborhood"] - aligned["wt"]
    order = np.argsort(-np.abs(d))
    top_delta = [
        {
            "resseq": int(shared[i]),
            "delta_out_effect": float(d[i]),
            "wt": float(aligned["wt"][i]),
            "neighborhood": float(aligned["neighborhood"][i]),
            "mut": float(aligned["mut"][i]),
        }
        for i in order[:20]
    ]

    return {
        "schema_version": 1,
        "probe": cfg["probe"],
        "spec": cfg["spec"],
        "prereg": cfg["prereg"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "workstream": "gnn_perturbation_boundary",
        "arm": {"name": arm, "wt": wt_id, "mut": mut_id, "chain": chain},
        "n12": {
            "resseqs": n12,
            "size": len(n12),
            "includes_switch_lock_partners": True,
            "policy": "recomputed_on_mut_not_copied_from_other_arm",
        },
        "n_residues": {
            "wt": n_wt,
            "mut": n_mut,
            "single": n_single,
            "neighborhood": n_neigh,
            "scramble": n_scramble,
        },
        "n_shared": len(shared),
        "graft_meta": {
            "neighborhood": neigh.get("graft_meta"),
            "scramble": scramble.get("graft_meta"),
        },
        "spearman": {
            "wt_mut": float(rho_wt),
            "single_mut": float(rho_single),
            "neighborhood_mut": float(rho_neigh),
            "scramble_mut": float(rho_scramble),
        },
        "report_only": {
            "delta_field_spearman_neighborhood_vs_mut_minus_wt": float(delta_field),
            "top_abs_delta_neighborhood_minus_wt": top_delta,
            "delta_rho_ratio_vs_single": (
                float(verdict["delta_rho_neighborhood"] / verdict["delta_rho_single"])
                if abs(verdict["delta_rho_single"]) > 1e-12
                else None
            ),
        },
        "verdict": verdict,
        "notes": [
            "Pass requires neighborhood Δρ > single-site Δρ and beats neighborhood scramble.",
            "N12 recomputed per arm (not hard-coded across OFF/ON).",
            "Does not reopen CB concordance or Ledger B interface.",
            "GNN inputs are ρ/τ/ss + Cα — not AA identity.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        HEALTHY_FIX1_CKPT,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(ARMS), default="off")
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
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.output is None:
        args.output = Path(ARMS[args.arm]["default_output"])

    report = grade(
        checkpoint=args.checkpoint,
        pdb_dir=args.pdb_dir,
        device=args.device,
        arm=args.arm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"|N12|={report['n12']['size']} {report['n12']['resseqs']}")
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
