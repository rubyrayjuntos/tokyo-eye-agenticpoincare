#!/usr/bin/env python3
"""SHP2 inactive→active OOD migration — 2SHP → 6CRF on sparsity champion.

Three locked arms (see ``data/gates/shp2_2shp_6mcf_ood_prereg.json``):

1. State-transition consistency — Spearman of knockout ``out_effect`` on shared
   auth_resseq (N-SH2/PTP mechanistic axes).
2. Perturbation invariance — top-decile highway Jaccard under contact-cutoff
   ±0.5 Å and edge-sparsity truncation on **6CRF** (open SHP2).
3. Cross-structure calibration — Gini(6CRF) in champion band; no monopoly blow-up.

**PDB identity:** Active state is ``6CRF`` (E76K open). Historical label ``6MCF``
was a Tier-2 typo (6MCF is 7SK/Tat RNA, not SHP2).

Make: ``make grade-v66-fix1-shp2-2shp-6mcf-ood`` (alias) or
``make grade-v66-fix1-shp2-2shp-6crf-ood``
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.diagnostics.kras_knockout_causal import knockout_scan
from experiments.training.v66._data import EDGE_CUTOFF, load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.dtie.common.shp2_state_migration import (
    DEFAULT_BARS,
    grade_shp2_migration,
    score_calibration,
    score_perturbation_invariance,
    score_state_transition,
)
from science.dtie.v5.gnn.model import precompute_clustering
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_PREREG = Path("data/gates/shp2_2shp_6mcf_ood_prereg.json")
DEFAULT_OUT = Path(
    "checkpoints/v66/diagnostics/routing_sparsity/shp2_2shp_6crf_ood.json"
)
INACTIVE = ("2SHP", "A")
ACTIVE = ("6CRF", "A")  # open SHP2 E76K (NOT 6MCF — that deposit is 7SK/Tat RNA)


def _rebuild_ca_graph(
    prot: dict[str, Any],
    ca_np: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, Any]:
    out = copy.deepcopy(prot)
    ca_np = np.asarray(ca_np, dtype=np.float64)
    n = ca_np.shape[0]
    data0 = out["data"]
    x = data0.x.detach().cpu().clone()
    if x.shape[0] != n:
        raise ValueError(f"ca/x length mismatch: {n} vs {x.shape[0]}")
    dists = cdist(ca_np, ca_np)
    src, dst = np.where((dists < float(cutoff)) & (dists > 0.1))
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    rel_pos = ca_np[dst] - ca_np[src]
    edge_dist = dists[src, dst]
    edge_attr = np.column_stack([rel_pos, edge_dist]).astype(np.float32)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    if hasattr(data0, "sasa") and data0.sasa is not None:
        data.sasa = data0.sasa.detach().cpu().clone()
    data = precompute_clustering(data)
    out["data"] = data
    out["ca_coords"] = torch.tensor(ca_np, dtype=torch.float32)
    return out


def _truncate_farthest_edges(
    prot: dict[str, Any],
    *,
    keep_frac: float,
) -> dict[str, Any]:
    """Keep the closest ``keep_frac`` of directed contact edges (by distance)."""
    out = copy.deepcopy(prot)
    data0 = out["data"]
    ei = data0.edge_index.detach().cpu()
    ea = data0.edge_attr.detach().cpu()
    if ea.ndim != 2 or ea.size(-1) < 4:
        raise ValueError("expected edge_attr [..., 4] with distance in last column")
    dist = ea[:, -1].numpy()
    n_keep = max(1, int(np.floor(keep_frac * dist.size)))
    order = np.argsort(dist)[:n_keep]
    data = Data(
        x=data0.x.detach().cpu().clone(),
        edge_index=ei[:, order].contiguous(),
        edge_attr=ea[order].contiguous(),
    )
    if hasattr(data0, "sasa") and data0.sasa is not None:
        data.sasa = data0.sasa.detach().cpu().clone()
    data = precompute_clustering(data)
    out["data"] = data
    return out


def _scan(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, dict[int, int], int]:
    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    idx_map = residue_index_map(list(prot.get("residue_ids") or [])[:n])
    oe = np.asarray(knockout_scan(model, prot, device)["out_effect"], dtype=np.float64)[:n]
    return oe, idx_map, n


def _load_prot(pdb_id: str, chain: str, pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain} from {pdb_dir}")
    return prot


def grade(
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str,
    bars: dict[str, float] | None = None,
    archive_arrays: bool = True,
    skip_perturbation: bool = False,
) -> dict[str, Any]:
    bars = dict(DEFAULT_BARS if bars is None else {**DEFAULT_BARS, **bars})
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()

    print("scan 2SHP (inactive) ...", flush=True)
    prot_in = _load_prot(*INACTIVE, pdb_dir)
    oe_in, map_in, n_in = _scan(model, prot_in, device)

    print("scan 6CRF (active/open SHP2 E76K) ...", flush=True)
    prot_act = _load_prot(*ACTIVE, pdb_dir)
    oe_act, map_act, n_act = _scan(model, prot_act, device)

    state = score_state_transition(
        {"2SHP": oe_in, "6CRF": oe_act},
        {"2SHP": map_in, "6CRF": map_act},
        inactive_id="2SHP",
        active_id="6CRF",
        spearman_bar=float(bars["state_spearman_min"]),
        k_frac=float(bars["perturb_k_frac"]),
    )

    calib = score_calibration(oe_in, oe_act, bars=bars)

    if skip_perturbation:
        perturb = {
            "skipped": True,
            "pass": False,
            "pass_rule": "skipped (--skip-perturbation)",
            "arms": {},
        }
    else:
        ca = prot_act["ca_coords"]
        ca = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
        ca = ca[:n_act]
        base_cut = float(EDGE_CUTOFF)
        perturbed: dict[str, np.ndarray] = {}

        print(f"  perturb cutoff {base_cut - 0.5:.1f} Å ...", flush=True)
        prot_lo = _rebuild_ca_graph(prot_act, ca, cutoff=base_cut - 0.5)
        oe_lo, _, _ = _scan(model, prot_lo, device)
        perturbed["cutoff_7p5"] = oe_lo

        print(f"  perturb cutoff {base_cut + 0.5:.1f} Å ...", flush=True)
        prot_hi = _rebuild_ca_graph(prot_act, ca, cutoff=base_cut + 0.5)
        oe_hi, _, _ = _scan(model, prot_hi, device)
        perturbed["cutoff_8p5"] = oe_hi

        print("  perturb edge truncate keep=0.8 ...", flush=True)
        prot_tr = _truncate_farthest_edges(prot_act, keep_frac=0.8)
        oe_tr, _, _ = _scan(model, prot_tr, device)
        perturbed["edge_truncate_0p8"] = oe_tr

        perturb = score_perturbation_invariance(
            oe_act,
            perturbed,
            jaccard_bar=float(bars["perturb_jaccard_min"]),
            k_frac=float(bars["perturb_k_frac"]),
        )
        if archive_arrays:
            perturb["perturbed_out_effect"] = {
                k: v.tolist() for k, v in perturbed.items()
            }

    aggregate = grade_shp2_migration(
        state_transition=state,
        perturbation=perturb,
        calibration=calib,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "shp2_2shp_6crf_ood",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_ssot": "experiments.training.v66.healthy_fix1.FIX1_SPARSITY_CHAMPION_CKPT",
        "prereg": str(DEFAULT_PREREG),
        "bars": bars,
        "structures": {
            "2SHP": {"chain": "A", "n_residues": int(n_in), "role": "inactive_autoinhibited"},
            "6CRF": {
                "chain": "A",
                "n_residues": int(n_act),
                "role": "active_open_E76K",
                "note": "Replaces erroneous 6MCF (7SK/Tat RNA) from historical Tier-2 label",
            },
        },
        "state_transition": state,
        "perturbation_invariance": perturb,
        "cross_structure_calibration": {
            k: v
            for k, v in calib.items()
            if k not in ("inactive", "active")
        },
        "calibration_metrics": {
            "inactive": calib["inactive"],
            "active": calib["active"],
        },
        "verdict": aggregate,
        "notes": [
            "Mechanistic axes = N-SH2/PTP coupling (SHP2), not Src C-lobe/αC.",
            "Active PDB = 6CRF (open E76K). 6MCF is NOT SHP2 — historical typo corrected before first Pass/Fail stamp.",
            "Perturbation arms run on 6CRF only (OOD active coordinate set).",
        ],
    }
    if archive_arrays:
        report["out_effect"] = {
            "2SHP": oe_in.tolist(),
            "6CRF": oe_act.tolist(),
        }
        report["resseq_maps"] = {
            "2SHP": {str(k): int(v) for k, v in map_in.items()},
            "6CRF": {str(k): int(v) for k, v in map_act.items()},
        }
    return report


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
        "--skip-perturbation",
        action="store_true",
        help="Skip cutoff/truncate arms (CI smoke); forces Fail on that arm.",
    )
    p.add_argument("--no-archive-arrays", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    bars = dict(DEFAULT_BARS)
    if args.prereg.is_file():
        prereg = json.loads(args.prereg.read_text())
        bars.update({k: float(v) for k, v in (prereg.get("bars") or {}).items()})

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")

    report = grade(
        checkpoint=args.checkpoint,
        pdb_dir=args.pdb_dir,
        device=args.device,
        bars=bars,
        archive_arrays=not args.no_archive_arrays,
        skip_perturbation=bool(args.skip_perturbation),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps(report["state_transition"], indent=2))
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
