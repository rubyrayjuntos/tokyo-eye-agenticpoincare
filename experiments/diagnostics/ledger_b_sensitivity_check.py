#!/usr/bin/env python3
"""Ledger B sensitivity check — 2SHP focal hubs (R32, I310, N308, V457).

Pre-registered bars (locked before run):
  - Coordinate jitter: Spearman(rank_base, rank_jitter) on full out_effect > 0.85
  - Sparsity trajectory: focal hubs remain in top-10% across ep46/48/50 (same λ=0.0075 run)
  - Wrapping: report ρ/τ; underwrapped (ρ < τ) is supporting evidence, not a hard Pass gate

Does not change Ledger B interface sets or the 0.25 recall bar.
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
from scipy.stats import spearmanr
from torch_geometric.data import Data

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.diagnostics.kras_knockout_causal import knockout_scan
from experiments.training.v66._data import EDGE_CUTOFF, load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.dtie.common.residue_features import TAU
from science.dtie.v5.gnn.model import precompute_clustering
from science.training.gnn_lineage import load_model_from_checkpoint

FOCAL_HUBS = (32, 310, 308, 457)  # R32, I310, N308, V457
JITTER_SIGMA_A = 0.1
JITTER_SPEARMAN_BAR = 0.85
DEFAULT_OUT = Path(
    "checkpoints/v66/diagnostics/routing_sparsity/ledger_b_sensitivity_check.json"
)


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    """1-based ranks; ties broken by first occurrence (stable argsort)."""
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def _rebuild_ca_graph(prot: dict[str, Any], ca_np: np.ndarray) -> dict[str, Any]:
    """Deep-copy prot and rebuild Cα contact edges from perturbed coordinates."""
    out = copy.deepcopy(prot)
    ca_np = np.asarray(ca_np, dtype=np.float64)
    n = ca_np.shape[0]
    data0 = out["data"]
    x = data0.x.detach().cpu().clone()
    if x.shape[0] != n:
        raise ValueError(f"ca/x length mismatch: {n} vs {x.shape[0]}")

    dists = cdist(ca_np, ca_np)
    src, dst = np.where((dists < EDGE_CUTOFF) & (dists > 0.1))
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


def _focal_report(
    out_effect: np.ndarray,
    idx_map: dict[int, int],
    *,
    k_frac: float = 0.10,
) -> dict[str, Any]:
    ranks = _ranks_desc(out_effect)
    k = max(1, int(np.ceil(k_frac * out_effect.size)))
    rows = {}
    for rs in FOCAL_HUBS:
        if rs not in idx_map:
            rows[str(rs)] = {"present": False}
            continue
        i = idx_map[rs]
        r = int(ranks[i])
        rows[str(rs)] = {
            "present": True,
            "graph_index": int(i),
            "out_effect": float(out_effect[i]),
            "rank": r,
            "in_top_10pct": r <= k,
            "top_k": int(k),
        }
    return rows


def _load_prot(pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy("2SHP", "A", pdb_dir)
    if prot is None:
        raise FileNotFoundError("2SHP:A load failed")
    return prot


def _scan(
    checkpoint: Path,
    prot: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, dict[str, int], int]:
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    idx_map = residue_index_map(list(prot.get("residue_ids") or [])[:n])
    oe = np.asarray(knockout_scan(model, prot, device)["out_effect"], dtype=np.float64)[:n]
    return oe, idx_map, n


def arm_jitter(
    *,
    checkpoint: Path,
    prot0: dict[str, Any],
    oe0: np.ndarray,
    idx_map: dict[int, int],
    device: str,
    n_seeds: int,
    sigma: float,
) -> dict[str, Any]:
    ca0 = prot0["ca_coords"]
    ca0 = ca0.detach().cpu().numpy() if torch.is_tensor(ca0) else np.asarray(ca0)
    ca0 = ca0[: oe0.size]
    ranks0 = _ranks_desc(oe0)
    seed_rows = []
    rhos = []
    focal_rank_series: dict[str, list[int]] = {str(r): [] for r in FOCAL_HUBS}

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, size=ca0.shape)
        prot_j = _rebuild_ca_graph(prot0, ca0 + noise)
        print(f"  jitter seed={seed} σ={sigma} Å ...", flush=True)
        oe_j, _, _ = _scan(checkpoint, prot_j, device)
        ranks_j = _ranks_desc(oe_j)
        rho = float(spearmanr(ranks0, ranks_j).correlation)
        rhos.append(rho)
        foc = _focal_report(oe_j, idx_map)
        for rs in FOCAL_HUBS:
            key = str(rs)
            if foc[key].get("present"):
                focal_rank_series[key].append(int(foc[key]["rank"]))
        seed_rows.append(
            {
                "seed": seed,
                "spearman_rank_vs_baseline": rho,
                "pass_rho": bool(rho > JITTER_SPEARMAN_BAR),
                "focal": foc,
            }
        )

    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    return {
        "sigma_angstrom": sigma,
        "n_seeds": n_seeds,
        "bar_spearman": JITTER_SPEARMAN_BAR,
        "spearman_per_seed": rhos,
        "spearman_mean": mean_rho,
        "pass": bool(mean_rho > JITTER_SPEARMAN_BAR and all(r > JITTER_SPEARMAN_BAR for r in rhos)),
        "pass_rule": "mean and all seeds Spearman(rank_base, rank_jitter) > 0.85",
        "focal_rank_series": focal_rank_series,
        "focal_rank_variance": {
            k: float(np.var(v)) if len(v) > 1 else 0.0 for k, v in focal_rank_series.items()
        },
        "seeds": seed_rows,
    }


def arm_sparsity_trajectory(
    *,
    run_dir: Path,
    epochs: list[int],
    prot0: dict[str, Any],
    idx_map: dict[int, int],
    device: str,
) -> dict[str, Any]:
    """Fixed λ=0.0075 run; sweep epochs with different realized routing entropy."""
    rows = []
    for ep in epochs:
        ckpt = run_dir / "epochs" / f"epoch_{ep:03d}.pt"
        if not ckpt.is_file():
            rows.append({"epoch": ep, "error": f"missing {ckpt}"})
            continue
        print(f"  sparsity trajectory epoch={ep} ...", flush=True)
        oe, _, _ = _scan(ckpt, copy.deepcopy(prot0), device)
        foc = _focal_report(oe, idx_map)
        rows.append(
            {
                "epoch": ep,
                "checkpoint": str(ckpt),
                "focal": foc,
                "all_focal_in_top10": all(
                    foc[str(r)].get("in_top_10pct") for r in FOCAL_HUBS if foc[str(r)].get("present")
                ),
            }
        )

    present_rows = [r for r in rows if "focal" in r]
    persist = (
        all(r["all_focal_in_top10"] for r in present_rows) if present_rows else False
    )
    # Hub-signature variance: rank std across epochs per focal site
    sig_var = {}
    for rs in FOCAL_HUBS:
        ranks = [
            r["focal"][str(rs)]["rank"]
            for r in present_rows
            if r["focal"][str(rs)].get("present")
        ]
        sig_var[str(rs)] = {
            "ranks": ranks,
            "rank_std": float(np.std(ranks)) if ranks else float("nan"),
            "rank_range": int(max(ranks) - min(ranks)) if ranks else None,
        }

    return {
        "note": (
            "λ_sparse fixed at 0.0075 for this run; sweep is realized sparsity "
            "(routing entropy trajectory) across epochs — not a λ rematch retrain."
        ),
        "lambda_sparse_train": 0.0075,
        "epochs": epochs,
        "panel": rows,
        "pass": persist,
        "pass_rule": "all focal hubs remain in top-10% out_effect on every available epoch",
        "hub_signature_rank_stats": sig_var,
    }


def arm_wrapping(
    *,
    prot0: dict[str, Any],
    idx_map: dict[int, int],
    oe0: np.ndarray,
) -> dict[str, Any]:
    n = oe0.size
    rho = prot0["target_rho"].detach().cpu().numpy().reshape(-1)[:n]
    tau_flag = prot0["target_dehydron"].detach().cpu().numpy().reshape(-1)[:n]
    bf = prot0.get("b_factor_ca")
    bf_np = (
        bf.detach().cpu().numpy().reshape(-1)[:n]
        if bf is not None
        else np.full(n, np.nan)
    )
    ranks = _ranks_desc(oe0)
    rows = {}
    for rs in FOCAL_HUBS:
        if rs not in idx_map:
            rows[str(rs)] = {"present": False}
            continue
        i = idx_map[rs]
        r_rho = float(rho[i])
        under = bool(r_rho < float(TAU))
        rows[str(rs)] = {
            "present": True,
            "wrapping_rho": r_rho,
            "tau_threshold": float(TAU),
            "tau_flag": float(tau_flag[i]),
            "underwrapped": under,
            "b_factor_ca": float(bf_np[i]) if np.isfinite(bf_np[i]) else None,
            "out_effect": float(oe0[i]),
            "rank": int(ranks[i]),
            "source_leak_soft": bool(under and ranks[i] <= max(1, int(np.ceil(0.10 * n)))),
        }

    # Corpus contrast: mean ρ of hubs vs non-hubs (top 10%)
    k = max(1, int(np.ceil(0.10 * n)))
    hub_mask = ranks <= k
    return {
        "tau": float(TAU),
        "focal": rows,
        "top10_mean_rho": float(np.nanmean(rho[hub_mask])),
        "non_top10_mean_rho": float(np.nanmean(rho[~hub_mask])),
        "focal_underwrapped_count": sum(
            1 for v in rows.values() if v.get("underwrapped")
        ),
        "note": (
            "Underwrapping supports biophysical motivation; not a hard Pass gate. "
            "B-factors reported to flag possible surface/disorder noise."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        FIX1_SPARSITY_CHAMPION_RUN_DIR,
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
    p.add_argument("--run-dir", type=Path, default=FIX1_SPARSITY_CHAMPION_RUN_DIR)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--jitter-seeds", type=int, default=3)
    p.add_argument("--jitter-sigma", type=float, default=JITTER_SIGMA_A)
    p.add_argument("--epochs", default="46,48,50")
    p.add_argument("--skip-jitter", action="store_true")
    p.add_argument("--skip-trajectory", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    epochs = [int(x) for x in str(args.epochs).split(",") if x.strip()]

    print("baseline 2SHP knockout ...", flush=True)
    prot0 = _load_prot(args.pdb_dir)
    oe0, idx_map, n = _scan(args.checkpoint, copy.deepcopy(prot0), args.device)
    baseline_focal = _focal_report(oe0, idx_map)

    wrapping = arm_wrapping(prot0=prot0, idx_map=idx_map, oe0=oe0)

    if args.skip_jitter:
        jitter = {"skipped": True}
    else:
        print("arm: coordinate jitter ...", flush=True)
        jitter = arm_jitter(
            checkpoint=args.checkpoint,
            prot0=prot0,
            oe0=oe0,
            idx_map=idx_map,
            device=args.device,
            n_seeds=int(args.jitter_seeds),
            sigma=float(args.jitter_sigma),
        )

    if args.skip_trajectory:
        trajectory = {"skipped": True}
    else:
        print("arm: sparsity trajectory ...", flush=True)
        trajectory = arm_sparsity_trajectory(
            run_dir=args.run_dir,
            epochs=epochs,
            prot0=prot0,
            idx_map=idx_map,
            device=args.device,
        )

    hard_pass = True
    if not jitter.get("skipped"):
        hard_pass = hard_pass and bool(jitter.get("pass"))
    if not trajectory.get("skipped"):
        hard_pass = hard_pass and bool(trajectory.get("pass"))

    out = {
        "schema_version": 1,
        "probe": "ledger_b_sensitivity_check",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "pdb_id": "2SHP",
        "chain": "A",
        "checkpoint": str(args.checkpoint),
        "focal_hubs": list(FOCAL_HUBS),
        "n_residues": int(n),
        "pre_registered_bars": {
            "jitter_spearman_min": JITTER_SPEARMAN_BAR,
            "jitter_sigma_angstrom": JITTER_SIGMA_A,
            "trajectory_rule": "focal hubs stay in top-10% across sparsity epochs",
            "wrapping": "report-only (underwrap supports; not hard gate)",
        },
        "baseline": {
            "focal": baseline_focal,
            "top10_k": max(1, int(np.ceil(0.10 * n))),
        },
        "coordinate_jitter": jitter,
        "sparsity_trajectory": trajectory,
        "wrapping_audit": wrapping,
        "verdict": {
            "pass": hard_pass,
            "outcome": "Pass" if hard_pass else "Fail",
            "hard_gates": ["coordinate_jitter", "sparsity_trajectory"],
            "report_only": ["wrapping_audit"],
        },
        "notes": [
            "Bars locked before run: Spearman ρ > 0.85 on jitter; trajectory top-10% persistence.",
            "Does not alter Ledger B recall threshold or pre-reg I.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["verdict"], indent=2))
    print(f"wrote {args.output}")
    return 0 if hard_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
