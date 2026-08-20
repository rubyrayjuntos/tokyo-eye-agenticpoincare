#!/usr/bin/env python3
"""KRAS G12 matched-state residue-12 graft validation (four-quadrant).

Pre-reg: ``docs/specs/kras-topo-structural-inference/kras-g12-residue-graft-prereg.md``
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
    DEFAULT_SCRAMBLE_RESSEQ,
    GRAFT_RESSEQ,
    align_out_effects,
    delta_field_spearman,
    graft_direction_verdict,
    graft_resseq_from_donor,
    residue_index_map,
    spearman_rho,
)
from science.training.gnn_lineage import load_model_from_checkpoint

ARMS: list[tuple[str, str, str, str]] = [
    ("off", "4LPK", "5US4", "A"),
    ("on", "6GOD", "6GOF", "A"),
]


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


def grade_arm(
    *,
    model: torch.nn.Module,
    arm: str,
    wt_id: str,
    mut_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
    scramble_donor: int,
) -> dict[str, Any]:
    print(f"=== arm {arm}: {wt_id} ←12— {mut_id} ===", flush=True)
    wt = _load(wt_id, chain, pdb_dir)
    mut = _load(mut_id, chain, pdb_dir)
    # Align feature widths before graft
    wt = _align_prot_features(model, wt)
    mut = _align_prot_features(model, mut)

    graft = graft_resseq_from_donor(wt, mut, target_resseq=GRAFT_RESSEQ, donor_resseq=GRAFT_RESSEQ)
    scramble = graft_resseq_from_donor(
        wt, mut, target_resseq=GRAFT_RESSEQ, donor_resseq=scramble_donor
    )

    print(f"  knockout WT {wt_id} ...", flush=True)
    oe_wt, map_wt, n_wt = _scan_oe(model, wt, device)
    print(f"  knockout mut {mut_id} ...", flush=True)
    oe_mut, map_mut, n_mut = _scan_oe(model, mut, device)
    print("  knockout graft ...", flush=True)
    oe_graft, map_graft, n_graft = _scan_oe(model, graft, device)
    print("  knockout scramble ...", flush=True)
    oe_scramble, map_scramble, n_scramble = _scan_oe(model, scramble, device)

    labels = ["wt", "mut", "graft", "scramble"]
    shared, aligned = align_out_effects(
        {
            "wt": oe_wt,
            "mut": oe_mut,
            "graft": oe_graft,
            "scramble": oe_scramble,
        },
        {
            "wt": map_wt,
            "mut": map_mut,
            "graft": map_graft,
            "scramble": map_scramble,
        },
        labels,
    )
    rho_wt = spearman_rho(aligned["wt"], aligned["mut"])
    rho_graft = spearman_rho(aligned["graft"], aligned["mut"])
    rho_scramble = spearman_rho(aligned["scramble"], aligned["mut"])
    verdict = graft_direction_verdict(
        rho_wt_mut=rho_wt,
        rho_graft_mut=rho_graft,
        rho_scramble_mut=rho_scramble,
    )

    delta_rho = delta_field_spearman(
        aligned["wt"], aligned["graft"], aligned["wt"], aligned["mut"]
    )
    # Largest |Δ| sites (graft - wt) on shared numbering
    d = aligned["graft"] - aligned["wt"]
    order = np.argsort(-np.abs(d))
    top_delta = [
        {
            "resseq": int(shared[i]),
            "delta_out_effect": float(d[i]),
            "wt": float(aligned["wt"][i]),
            "graft": float(aligned["graft"][i]),
            "mut": float(aligned["mut"][i]),
        }
        for i in order[:15]
    ]

    return {
        "arm": arm,
        "wt": wt_id,
        "mut": mut_id,
        "chain": chain,
        "n_residues": {"wt": n_wt, "mut": n_mut, "graft": n_graft, "scramble": n_scramble},
        "n_shared": len(shared),
        "graft_meta": graft.get("graft_meta"),
        "scramble_meta": scramble.get("graft_meta"),
        "scramble_donor_resseq": scramble_donor,
        "spearman": {
            "wt_mut": float(rho_wt),
            "graft_mut": float(rho_graft),
            "scramble_mut": float(rho_scramble),
        },
        "report_only": {
            "delta_field_spearman_graft_vs_mut_minus_wt": float(delta_rho),
            "top_abs_delta_graft_minus_wt": top_delta,
        },
        "verdict": verdict,
    }


def grade(
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str,
    scramble_donor: int = DEFAULT_SCRAMBLE_RESSEQ,
) -> dict[str, Any]:
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    arms = []
    for arm, wt_id, mut_id, chain in ARMS:
        arms.append(
            grade_arm(
                model=model,
                arm=arm,
                wt_id=wt_id,
                mut_id=mut_id,
                chain=chain,
                pdb_dir=pdb_dir,
                device=device,
                scramble_donor=scramble_donor,
            )
        )
    panel_pass = all(bool(a["verdict"]["pass"]) for a in arms)
    return {
        "schema_version": 1,
        "probe": "kras_g12_residue_graft",
        "spec": "docs/specs/kras-topo-structural-inference/kras-g12-residue-graft-prereg.md",
        "prereg": "data/gates/kras_g12_residue_graft_prereg.json",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "honesty": (
            "Node features are ρ/τ/ss — not AA one-hots. Graft copies mut local "
            "physics/geometry at the donor site onto the WT scaffold."
        ),
        "arms": arms,
        "verdict": {
            "pass": panel_pass,
            "outcome": "Pass" if panel_pass else "Fail",
            "panel_rule": "both OFF and ON arms Pass",
            "arms": {a["arm"]: a["verdict"]["pass"] for a in arms},
        },
        "notes": [
            "Does not reopen CB concordance or Ledger B interface Fail.",
            "Expansion to other G12 deposits is out of scope for this stamp.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        HEALTHY_FIX1_CKPT,
    )

    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--scramble-donor", type=int, default=DEFAULT_SCRAMBLE_RESSEQ)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/routing_sparsity/kras_g12_residue_graft.json"
        ),
    )
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    report = grade(
        checkpoint=args.checkpoint,
        pdb_dir=args.pdb_dir,
        device=args.device,
        scramble_donor=int(args.scramble_donor),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    for arm in report["arms"]:
        print(arm["arm"], json.dumps(arm["verdict"], indent=2))
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
