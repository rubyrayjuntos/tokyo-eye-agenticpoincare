#!/usr/bin/env python3
"""KRAS G12 hyperbolic latent graft (post-lift ``x_hyp``), OFF arm.

Pre-reg: ``docs/specs/kras-topo-structural-inference/kras-g12-hyperbolic-latent-graft-prereg.md``

Intervention is **after** ``expmap0`` only — Euclidean MP / ``data.x`` stay WT.
Primary metric: Spearman of hyperbolic depth ``dist0(x_hyp)`` vs mut.
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
from science.dtie.common.kras_g12_graft import (
    GRAFT_RESSEQ,
    align_out_effects,
    graft_neighborhood_matched,
    hyp_latent_neighborhood_matched,
    hyp_latent_neighborhood_scramble,
    hyp_latent_verdict,
    neighborhood_n12,
    residue_index_map,
    spearman_rho,
)
from science.training.gnn_lineage import load_model_from_checkpoint

SPEC = "docs/specs/kras-topo-structural-inference/kras-g12-hyperbolic-latent-graft-prereg.md"
PREREG = "data/gates/kras_g12_hyperbolic_latent_graft_prereg.json"
PROBE = "kras_g12_hyperbolic_latent_graft"
DEFAULT_OUTPUT = (
    "checkpoints/v66/diagnostics/routing_sparsity/kras_g12_hyperbolic_latent_graft.json"
)
WT_ID = "4LPK"
MUT_ID = "5US4"
CHAIN = "A"


def _load(pdb_id: str, chain: str, pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")
    return prot


def _forward_x_hyp(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, np.ndarray, dict[int, int], float, int]:
    """Full forward; return ``x_hyp``, ``dist0`` depth, resseq map, curvature, n."""
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    ids = list(prot.get("residue_ids") or [])[:n]
    idx_map = residue_index_map(ids)
    with torch.no_grad():
        out = model(data)
    x_hyp = out["x_hyp"].detach().cpu()[:n]
    depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)[:n].astype(np.float64)
    c_t = out["audit_trail"].get("curvature_value")
    if c_t is None:
        c = float(model.curvature.detach().cpu())
    else:
        c = float(c_t.detach().cpu()) if torch.is_tensor(c_t) else float(c_t)
    return x_hyp, depth, idx_map, c, n


def grade(
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str,
) -> dict[str, Any]:
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()

    print(f"=== hyp latent graft OFF: {WT_ID} ← N12 x_hyp — {MUT_ID} ===", flush=True)
    wt = _align_prot_features(model, _load(WT_ID, CHAIN, pdb_dir))
    mut = _align_prot_features(model, _load(MUT_ID, CHAIN, pdb_dir))

    n12 = neighborhood_n12(mut, wt, seed_resseq=GRAFT_RESSEQ)
    print(f"  |N12|={len(n12)}: {n12}", flush=True)

    print(f"  forward WT {WT_ID} (capture x_hyp) ...", flush=True)
    x_wt, depth_wt, map_wt, c_wt, n_wt = _forward_x_hyp(model, wt, device)
    print(f"  forward mut {MUT_ID} (capture x_hyp) ...", flush=True)
    x_mut, depth_mut, map_mut, c_mut, n_mut = _forward_x_hyp(model, mut, device)
    # Host curvature governs splice project/dist0 (WT scaffold).
    curvature = c_wt
    if abs(c_wt - c_mut) > 1e-6:
        print(
            f"  warn: curvature WT={c_wt:.6f} mut={c_mut:.6f}; using WT for splice",
            flush=True,
        )

    print("  hyp matched splice N12 ...", flush=True)
    _, depth_hyp, meta_hyp = hyp_latent_neighborhood_matched(
        x_wt, map_wt, x_mut, map_mut, n12, curvature=curvature
    )
    print("  hyp scramble splice N12 ...", flush=True)
    _, depth_scr, meta_scr = hyp_latent_neighborhood_scramble(
        x_wt,
        map_wt,
        x_mut,
        map_mut,
        n12,
        curvature=curvature,
        mut_prot_for_donors=mut,
    )

    print("  Euclidean neighborhood graft → full forward (hyp-depth baseline) ...", flush=True)
    euc_neigh = graft_neighborhood_matched(wt, mut, n12)
    _, depth_euc, map_euc, _, n_euc = _forward_x_hyp(model, euc_neigh, device)

    labels = ["wt", "mut", "hyp_graft", "euc_neigh", "hyp_scramble"]
    shared, aligned = align_out_effects(
        {
            "wt": depth_wt,
            "mut": depth_mut,
            "hyp_graft": depth_hyp,
            "euc_neigh": depth_euc,
            "hyp_scramble": depth_scr,
        },
        {
            "wt": map_wt,
            "mut": map_mut,
            "hyp_graft": map_wt,  # spliced on WT scaffold
            "euc_neigh": map_euc,
            "hyp_scramble": map_wt,
        },
        labels,
    )

    rho_wt = spearman_rho(aligned["wt"], aligned["mut"])
    rho_hyp = spearman_rho(aligned["hyp_graft"], aligned["mut"])
    rho_euc = spearman_rho(aligned["euc_neigh"], aligned["mut"])
    rho_scr = spearman_rho(aligned["hyp_scramble"], aligned["mut"])
    verdict = hyp_latent_verdict(
        rho_wt_mut=rho_wt,
        rho_hyp_graft_mut=rho_hyp,
        rho_euc_neigh_mut=rho_euc,
        rho_hyp_scramble_mut=rho_scr,
    )

    d = aligned["hyp_graft"] - aligned["wt"]
    order = np.argsort(-np.abs(d))
    top_delta = [
        {
            "resseq": int(shared[i]),
            "delta_depth": float(d[i]),
            "wt": float(aligned["wt"][i]),
            "hyp_graft": float(aligned["hyp_graft"][i]),
            "mut": float(aligned["mut"][i]),
            "in_n12": bool(int(shared[i]) in set(n12)),
        }
        for i in order[:20]
    ]

    ratio = None
    if abs(verdict["delta_rho_euc_neigh"]) > 1e-12:
        ratio = float(verdict["delta_rho_hyp"] / verdict["delta_rho_euc_neigh"])

    return {
        "schema_version": 1,
        "probe": PROBE,
        "spec": SPEC,
        "prereg": PREREG,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "workstream": "gnn_perturbation_boundary",
        "architecture": {
            "intervention_layer": "post_lift_x_hyp",
            "not": "hyperbolic_message_passing",
            "metric": "spearman_hyperbolic_depth_dist0",
        },
        "arm": {"name": "off", "wt": WT_ID, "mut": MUT_ID, "chain": CHAIN},
        "n12": {
            "resseqs": n12,
            "size": len(n12),
            "includes_switch_lock_partners": True,
            "policy": "same_as_child_2",
        },
        "n_residues": {
            "wt": n_wt,
            "mut": n_mut,
            "euc_neigh": n_euc,
            "hyp_graft_scaffold": n_wt,
        },
        "n_shared": len(shared),
        "curvature": {"wt": c_wt, "mut": c_mut, "splice_used": curvature},
        "graft_meta": {"hyp_matched": meta_hyp, "hyp_scramble": meta_scr},
        "spearman_hyp_depth": {
            "wt_mut": float(rho_wt),
            "hyp_graft_mut": float(rho_hyp),
            "euc_neigh_mut": float(rho_euc),
            "hyp_scramble_mut": float(rho_scr),
        },
        "report_only": {
            "delta_rho_hyp_over_euc": ratio,
            "top_abs_delta_hyp_graft_minus_wt": top_delta,
        },
        "verdict": verdict,
        "notes": [
            "Pass: Δρ_hyp_depth > Δρ_euc_neigh_on_hyp_depth AND hyp graft beats hyp scramble.",
            "Euclidean data.x / edges unchanged for hyp splice; only x_hyp at N12 replaced.",
            "Disc / Klein are views only — not Pass.",
            "Does not relocate MP trunk into hyperbolic space.",
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
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    report = grade(
        checkpoint=args.checkpoint,
        pdb_dir=args.pdb_dir,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"|N12|={report['n12']['size']} {report['n12']['resseqs']}")
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
