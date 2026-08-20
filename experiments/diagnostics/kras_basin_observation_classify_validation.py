#!/usr/bin/env python3
"""KRAS basin observation classify — four-quadrant static smoke.

Pre-reg: ``docs/specs/kras-topo-structural-inference/kras-basin-observation-classify-prereg.md``
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
from science.dtie.common.kras_basin_observation import (
    ROSTER,
    allele_delta_site_specificity,
    basin_observation_verdict,
    build_arm_n12,
    check_structure_guards,
    pairwise_spearman_on_rstar,
    r_star_for_structure,
    residue_index_map,
)
from science.training.gnn_lineage import load_model_from_checkpoint

SPEC = "docs/specs/kras-topo-structural-inference/kras-basin-observation-classify-prereg.md"
PREREG = "data/gates/kras_basin_observation_classify_prereg.json"
PROBE = "kras_basin_observation_classify"
DEFAULT_OUTPUT = (
    "checkpoints/v66/diagnostics/routing_sparsity/kras_basin_observation_classify.json"
)

ORDER = ("4LPK", "5US4", "6GOD", "6GOF")
CHAIN = "A"


def _load(pdb_id: str, pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, CHAIN, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{CHAIN}")
    return prot


def _forward_depth(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, dict[int, int], int]:
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    ids = list(prot.get("residue_ids") or [])[:n]
    idx_map = residue_index_map(ids)
    with torch.no_grad():
        out = model(data)
    depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)[:n].astype(np.float64)
    return depth, idx_map, n


def grade(
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str,
) -> dict[str, Any]:
    print("=== basin observation classify (4LPK/5US4/6GOD/6GOF) ===", flush=True)

    # ── Guards first (before GNN) ─────────────────────────────────────────
    guard_rows: dict[str, Any] = {}
    prots_raw: dict[str, dict[str, Any]] = {}
    for pdb_id in ORDER:
        prot = _load(pdb_id, pdb_dir)
        prots_raw[pdb_id] = prot
        pdb_path = prot.get("pdb_path") or str(pdb_dir / f"{pdb_id}.pdb")
        row = check_structure_guards(pdb_id, pdb_path, chain=CHAIN)
        guard_rows[pdb_id] = row
        print(
            f"  guard {pdb_id}: nuc={row['parsed']['nucleotide']} "
            f"allele={row['parsed']['allele']} ok={row['pass']}",
            flush=True,
        )
    guards_ok = all(guard_rows[p]["pass"] for p in ORDER)

    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()

    # Align + N12 per arm
    prots = {p: _align_prot_features(model, prots_raw[p]) for p in ORDER}
    n12_off = build_arm_n12(prots["5US4"], prots["4LPK"])
    n12_on = build_arm_n12(prots["6GOF"], prots["6GOD"])
    print(f"  N12_OFF={n12_off}", flush=True)
    print(f"  N12_ON={n12_on}", flush=True)

    depths: dict[str, np.ndarray] = {}
    maps: dict[str, dict[int, int]] = {}
    ns: dict[str, int] = {}
    rstars: dict[str, list[int]] = {}
    for pdb_id in ORDER:
        print(f"  forward {pdb_id} ...", flush=True)
        depth, idx_map, n = _forward_depth(model, prots[pdb_id], device)
        depths[pdb_id] = depth
        maps[pdb_id] = idx_map
        ns[pdb_id] = n
        n12 = n12_off if pdb_id in ("4LPK", "5US4") else n12_on
        rstars[pdb_id] = r_star_for_structure(set(idx_map), n12)
        print(f"    |R★|={len(rstars[pdb_id])}", flush=True)

    # Pairwise Spearmans on R★
    pairs = {
        "4LPK_5US4": ("4LPK", "5US4"),
        "6GOD_6GOF": ("6GOD", "6GOF"),
        "4LPK_6GOD": ("4LPK", "6GOD"),
        "4LPK_6GOF": ("4LPK", "6GOF"),
        "5US4_6GOD": ("5US4", "6GOD"),
        "5US4_6GOF": ("5US4", "6GOF"),
    }
    pair_stats: dict[str, Any] = {}
    for key, (a, b) in pairs.items():
        pair_stats[key] = pairwise_spearman_on_rstar(
            depths[a], maps[a], rstars[a], depths[b], maps[b], rstars[b]
        )
        print(f"  ρ({a},{b})={pair_stats[key]['rho']:.4f} n={pair_stats[key]['n_shared']}", flush=True)

    same_nuc = float(
        np.mean([pair_stats["4LPK_5US4"]["rho"], pair_stats["6GOD_6GOF"]["rho"]])
    )
    cross_nuc = float(
        np.mean(
            [
                pair_stats["4LPK_6GOD"]["rho"],
                pair_stats["4LPK_6GOF"]["rho"],
                pair_stats["5US4_6GOD"]["rho"],
                pair_stats["5US4_6GOF"]["rho"],
            ]
        )
    )

    # Allele site-specificity both arms
    # R★ for allele: use union of arm R★ for wt/mut pair intersection via helper
    allele_off = allele_delta_site_specificity(
        depths["4LPK"],
        maps["4LPK"],
        depths["5US4"],
        maps["5US4"],
        rstar=sorted(set(rstars["4LPK"]) | set(rstars["5US4"])),
        n12=n12_off,
        mut_prot=prots["5US4"],
    )
    allele_on = allele_delta_site_specificity(
        depths["6GOD"],
        maps["6GOD"],
        depths["6GOF"],
        maps["6GOF"],
        rstar=sorted(set(rstars["6GOD"]) | set(rstars["6GOF"])),
        n12=n12_on,
        mut_prot=prots["6GOF"],
    )
    print(
        f"  allele OFF: N12={allele_off['mean_abs_delta_n12']:.6f} "
        f"scr={allele_off['mean_abs_delta_scramble']:.6f} ok={allele_off['pass_arm']}",
        flush=True,
    )
    print(
        f"  allele ON:  N12={allele_on['mean_abs_delta_n12']:.6f} "
        f"scr={allele_on['mean_abs_delta_scramble']:.6f} ok={allele_on['pass_arm']}",
        flush=True,
    )

    verdict = basin_observation_verdict(
        guards_ok=guards_ok,
        same_nuc=same_nuc,
        cross_nuc=cross_nuc,
        allele_off_pass=allele_off["pass_arm"],
        allele_on_pass=allele_on["pass_arm"],
    )

    return {
        "schema_version": 1,
        "probe": PROBE,
        "spec": SPEC,
        "prereg": PREREG,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "workstream": "kras_basin_routing_automaton",
        "architecture": {
            "sensor": "hyp_depth_on_R_star",
            "referee": "deposit_nucleotide_and_g12_allele",
            "memory": "discrete_basin_labels",
            "not": ["md_integration", "hyperbolic_message_passing", "disc_alone_pass"],
        },
        "roster": {k: ROSTER[k] for k in ORDER},
        "guards": guard_rows,
        "n12": {"off": n12_off, "on": n12_on},
        "r_star_sizes": {k: len(rstars[k]) for k in ORDER},
        "n_residues": ns,
        "pairwise_spearman_rstar": {
            k: {"rho": v["rho"], "n_shared": v["n_shared"]} for k, v in pair_stats.items()
        },
        "nucleotide_axis": {
            "same_nuc": same_nuc,
            "cross_nuc": cross_nuc,
            "same_pairs": ["4LPK_5US4", "6GOD_6GOF"],
            "cross_pairs": ["4LPK_6GOD", "4LPK_6GOF", "5US4_6GOD", "5US4_6GOF"],
        },
        "allele_axis": {"off": allele_off, "on": allele_on},
        "verdict": verdict,
        "notes": [
            "Pass: guards + same_nuc > cross_nuc + N12 |Δ| > scramble on OFF and ON.",
            "Observation = cone_depth/dist0(x_hyp) on Switch I∪II ∪ N12.",
            "Does not claim dynamic transitions or hyperbolic MP.",
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
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
