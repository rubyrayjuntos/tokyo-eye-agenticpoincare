#!/usr/bin/env python3
"""KRAS topo-structural Phase A triangulation grade (4OBE / 4DSO / 5VQ2).

Spec: ``docs/specs/kras-topo-structural-inference/design.md``
Checkpoint SSOT: ``FIX1_SPARSITY_CHAMPION_CKPT`` (forward knockout; no Jacobian ranks).
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
from science.dtie.common.kras_topo_matrix import (
    CRITICAL_RESSEQ,
    PRIMARY_ACTIVE,
    PRIMARY_MUT_INACTIVE,
    PRIMARY_WT_INACTIVE,
    SWITCH_LOCK_PAIRS,
    align_resseq_vectors,
    betweenness_proximity_verdict,
    ca_contact_edges_by_resseq,
    ca_distance,
    dehydron_wrapper_edges_by_resseq,
    edge_delta_verdict,
    edge_symdiff_size,
    primary_matrix_verdict,
    residue_index_map,
    spearman_rho,
    switch_lock_verdict,
)
from science.training.gnn_lineage import load_model_from_checkpoint

ROSTER: list[tuple[str, str, str]] = [
    (PRIMARY_WT_INACTIVE, "A", "wt_inactive"),
    (PRIMARY_MUT_INACTIVE, "A", "mut_inactive"),
    (PRIMARY_ACTIVE, "A", "active_reference"),
]


def _load_prot(pdb_id: str, chain: str, pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain} from {pdb_dir}")
    return prot


def _structure_bundle(
    *,
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    role: str,
    pdb_dir: Path,
    device: str,
    run_knockout: bool,
) -> dict[str, Any]:
    from experiments.training.v66._data import _download_pdb

    prot = _load_prot(pdb_id, chain, pdb_dir)
    ca = prot.get("ca_coords")
    if ca is None:
        raise ValueError(f"{pdb_id}: missing ca_coords")
    residue_ids = list(prot.get("residue_ids") or [])
    ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)

    edges_ca = ca_contact_edges_by_resseq(ca_np, residue_ids)
    pdb_path = Path(_download_pdb(pdb_id, pdb_dir))
    wrap_graphs = dehydron_wrapper_edges_by_resseq(pdb_path, chain)
    distances = {
        pair: ca_distance(ca_np, residue_ids, pair[0], pair[1]) for pair in SWITCH_LOCK_PAIRS
    }
    idx_map = residue_index_map(residue_ids)

    out_effect: list[float] | None = None
    n_res = len(residue_ids)
    if run_knockout:
        model = load_model_from_checkpoint(checkpoint, device)
        model.eval()
        prot = _align_prot_features(model, prot)
        data0 = prepare_training_batch(model, prot, device)
        n_res = residue_node_count(data0, prot)
        scan = knockout_scan(model, prot, device)
        out_effect = np.asarray(scan["out_effect"], dtype=np.float64)[:n_res].tolist()
        residue_ids = residue_ids[:n_res]
        idx_map = residue_index_map(residue_ids)

    critical: dict[str, Any] = {}
    for rs in CRITICAL_RESSEQ:
        if rs not in idx_map:
            critical[str(rs)] = None
            continue
        i = idx_map[rs]
        entry: dict[str, Any] = {"graph_index": i}
        if out_effect is not None:
            entry["knockout_out_effect"] = float(out_effect[i])
        critical[str(rs)] = entry

    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "role": role,
        "n_residues": int(n_res),
        "n_contact_edges_ca": len(edges_ca),
        "n_dehydron_edges": len(wrap_graphs["dehydron"]),
        "n_wrapped_hbond_edges": len(wrap_graphs["wrapped_hbond"]),
        "n_complementarity_edges": len(wrap_graphs["complementarity"]),
        "switch_lock_ca_distances": {
            f"{a}-{b}": distances[(a, b)] for a, b in SWITCH_LOCK_PAIRS
        },
        "critical_residues": critical,
        "knockout_out_effect": out_effect,
        "residue_ids": [str(r) for r in residue_ids],
        "_edges_ca": edges_ca,
        "_edges_dehydron": wrap_graphs["dehydron"],
        "_edges_complementarity": wrap_graphs["complementarity"],
        "_idx_map": idx_map,
    }


def grade(
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str,
    skip_knockout: bool = False,
) -> dict[str, Any]:
    run_ko = not skip_knockout
    bundles: dict[str, dict[str, Any]] = {}
    for pdb_id, chain, role in ROSTER:
        print(f"loading {role} {pdb_id}:{chain} (knockout={run_ko}) ...", flush=True)
        bundles[pdb_id] = _structure_bundle(
            checkpoint=checkpoint,
            pdb_id=pdb_id,
            chain=chain,
            role=role,
            pdb_dir=pdb_dir,
            device=device,
            run_knockout=run_ko,
        )

    wt, mut, active = PRIMARY_WT_INACTIVE, PRIMARY_MUT_INACTIVE, PRIMARY_ACTIVE
    shared_res = sorted(
        set(bundles[wt]["_idx_map"])
        & set(bundles[mut]["_idx_map"])
        & set(bundles[active]["_idx_map"])
    )

    d_mut = edge_symdiff_size(
        bundles[mut]["_edges_complementarity"],
        bundles[active]["_edges_complementarity"],
        shared_resseqs=shared_res,
    )
    d_wt = edge_symdiff_size(
        bundles[wt]["_edges_complementarity"],
        bundles[active]["_edges_complementarity"],
        shared_resseqs=shared_res,
    )
    edge_v = edge_delta_verdict(d_mut, d_wt)
    edge_v["graph"] = "dehydron_wrapper_complementarity"
    edge_v["observational_ca"] = {
        "delta_e_mut_active": edge_symdiff_size(
            bundles[mut]["_edges_ca"],
            bundles[active]["_edges_ca"],
            shared_resseqs=shared_res,
        ),
        "delta_e_wt_active": edge_symdiff_size(
            bundles[wt]["_edges_ca"],
            bundles[active]["_edges_ca"],
            shared_resseqs=shared_res,
        ),
        "note": "Phase A Cα contact ΔE — observational only (not a Pass bar)",
    }
    edge_v["dehydron_only"] = {
        "delta_e_mut_active": edge_symdiff_size(
            bundles[mut]["_edges_dehydron"],
            bundles[active]["_edges_dehydron"],
            shared_resseqs=shared_res,
        ),
        "delta_e_wt_active": edge_symdiff_size(
            bundles[wt]["_edges_dehydron"],
            bundles[active]["_edges_dehydron"],
            shared_resseqs=shared_res,
        ),
    }

    mut_dist = {
        pair: bundles[mut]["switch_lock_ca_distances"][f"{pair[0]}-{pair[1]}"]
        for pair in SWITCH_LOCK_PAIRS
    }
    wt_dist = {
        pair: bundles[wt]["switch_lock_ca_distances"][f"{pair[0]}-{pair[1]}"]
        for pair in SWITCH_LOCK_PAIRS
    }
    switch_v = switch_lock_verdict(mut_dist, wt_dist)

    if run_ko:
        values = {
            s: bundles[s]["knockout_out_effect"]
            for s in (wt, mut, active)
        }
        maps = {s: bundles[s]["_idx_map"] for s in (wt, mut, active)}
        shared_sorted, aligned = align_resseq_vectors(
            values, maps, structures=[wt, mut, active]
        )
        rho_mut = spearman_rho(aligned[mut], aligned[active])
        rho_wt = spearman_rho(aligned[wt], aligned[active])
        btw_v = betweenness_proximity_verdict(rho_mut, rho_wt)
        btw_v["n_aligned_residues"] = len(shared_sorted)
    else:
        btw_v = {
            "pass": False,
            "skipped": True,
            "reason": "knockout skipped (--skip-knockout)",
            "rule": "rho(4DSO,5VQ2) > rho(4OBE,5VQ2)",
        }

    aggregate = primary_matrix_verdict(
        betweenness=btw_v, edge_delta=edge_v, switch_lock=switch_v
    )

    # Strip private keys for JSON
    public_structs = {}
    for pdb_id, b in bundles.items():
        public_structs[pdb_id] = {
            k: v for k, v in b.items() if not k.startswith("_")
        }

    return {
        "schema_version": 1,
        "probe": "kras_topo_structural_matrix",
        "spec": "docs/specs/kras-topo-structural-inference/design.md",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "roster": {
            "wt_inactive": wt,
            "mut_inactive": mut,
            "active_reference": active,
            "active_observational": None,
            "section_13": "docs/specs/kras-topo-structural-inference/section-13-amendment.md",
        },
        "structures": public_structs,
        "gates": {
            "betweenness_proximity": btw_v,
            "edge_symmetric_difference": edge_v,
            "switch_lock_tethers": switch_v,
        },
        "verdict": aggregate,
        "completed_related": {
            "hub_migration_R": (
                "checkpoints/v66/diagnostics/routing_sparsity/"
                "kras_hub_migration_4obe_4dso.json"
            ),
            "status": "PASS (separate probe)",
        },
        "notes": [
            "Blocking Pass = betweenness ∧ switch_lock; historical triad edge ΔE is report-only.",
            "Rewiring edge-ΔE closeout: make grade-v66-kras-topo-edge-four-quadrant (4LPK/6GOD/5US4/6GOF).",
            "Historical Pass ΔE used dehydron-wrapper complementarity; Cα ΔE remains observational on triad.",
            "Switch-I 12–32 Cα cutoff relaxed to 11.0 Å for Switch-I breathing.",
            "Uncertainty telemetry is monitor-only and not graded here.",
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
    p.add_argument(
        "--skip-knockout",
        action="store_true",
        help="Classical gates only (CI / smoke); betweenness proximity will Fail/skip.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/routing_sparsity/"
            "kras_topo_matrix_4obe_4dso_5vq2.json"
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
        skip_knockout=bool(args.skip_knockout),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps(report["gates"], indent=2))
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
