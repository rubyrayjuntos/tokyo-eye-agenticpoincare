#!/usr/bin/env python3
"""v7 KRAS G12D hub migration — forward knockout on x_hyp (Hyp MP trunk).

Pre-reg: docs/specs/tokyo-eye-v7/kras-g12d-hub-migration-prereg.md

Jacobian forbidden (z-norm-on). Disc is report-only. Pass: R_out_163(4DSO) > R(4OBE).
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
from geoopt.manifolds.stereographic import math as pmath

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.diagnostics.kras_hub_migration_ood import (
    EXPECTED,
    HUB_FROM,
    HUB_TO,
    load_prot,
    resolve_hub_indices,
)
from experiments.diagnostics.kras_knockout_causal import _pdb_residue_names
from experiments.training.v6._data import _download_pdb
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.training.gnn_lineage import load_model_from_checkpoint

SPEC = "docs/specs/tokyo-eye-v7/kras-g12d-hub-migration-prereg.md"
DEFAULT_OUT = Path(
    "checkpoints/v7/diagnostics/hub_migration/kras_hub_migration_4obe_4dso.json"
)
FIX1_COMPARE = Path(
    "checkpoints/v66/diagnostics/routing_sparsity/kras_hub_migration_4obe_4dso.json"
)


def _curvature_c(model: torch.nn.Module, out: dict[str, Any]) -> float:
    c_t = (out.get("audit_trail") or {}).get("curvature_value")
    if c_t is not None:
        return float(c_t.detach().cpu()) if torch.is_tensor(c_t) else float(c_t)
    return float(model.curvature.detach().cpu().reshape(-1)[0])


def _as_k(c: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.tensor(float(c), device=device, dtype=dtype)
    if float(k.item()) > 0:
        k = -k.abs()
    return k


def _forward_x_hyp(
    model: torch.nn.Module,
    data: Any,
) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        out = model(data)
    x = out["x_hyp"]
    return x, _curvature_c(model, out)


def knockout_scan_x_hyp(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Zero each input row; geodesic out-effect on x_hyp."""
    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    x0, c = _forward_x_hyp(model, data0)
    x0 = x0[:n]
    k = _as_k(c, device=x0.device, dtype=x0.dtype)

    out_effect = np.zeros(n, dtype=np.float64)
    delta_rows: list[np.ndarray] = []

    for i in range(n):
        data = prepare_training_batch(model, prot, device)
        data.x = data.x.clone()
        data.x[i] = 0.0
        x, _ = _forward_x_hyp(model, data)
        x = x[:n]
        # Pairwise geodesic displacement vs baseline (per residue).
        with torch.no_grad():
            dist = pmath.dist(x0, x, k=k).detach().float().cpu().numpy().reshape(-1)
        delta = np.asarray(dist, dtype=np.float64)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out_effect[i] = float(np.mean(delta[mask]))
        delta_rows.append(delta)

    return {
        "n_residues": n,
        "curvature_c": float(c),
        "baseline_x_hyp_dist0_mean": float(
            pmath.dist0(x0, k=k).detach().float().cpu().numpy().mean()
        ),
        "out_effect": out_effect,
        "delta_rows": delta_rows,
        "readout": "x_hyp_geodesic",
    }


def score_knockout(
    out_effect: np.ndarray,
    delta_rows: list[np.ndarray],
    indices: dict[int, int],
) -> dict[str, float]:
    out_effect = np.asarray(out_effect, dtype=np.float64)
    finite = out_effect[np.isfinite(out_effect)]
    median_out = float(np.median(finite))
    if median_out <= 0 or not np.isfinite(median_out):
        raise ValueError(f"median out-effect not positive: {median_out}")

    i163 = indices[HUB_TO]
    i151 = indices[HUB_FROM]
    i12 = indices[12]

    out_163 = float(out_effect[i163])
    out_151 = float(out_effect[i151])
    r_out_163 = out_163 / median_out

    in_to_163 = np.array([row[i163] for row in delta_rows], dtype=np.float64)
    in_to_163[i163] = np.nan
    finite_in = in_to_163[np.isfinite(in_to_163)]
    median_in = float(np.median(finite_in))
    in_from_12 = float(in_to_163[i12])
    in_from_151 = float(in_to_163[i151])

    order = np.argsort(-np.nan_to_num(out_effect, nan=-np.inf))
    rank_out_163 = int(np.where(order == i163)[0][0]) + 1
    rank_out_151 = int(np.where(order == i151)[0][0]) + 1

    return {
        "out_163_raw": out_163,
        "out_151_raw": out_151,
        "median_out": median_out,
        "R_out_163": r_out_163,
        "H_out_163_over_151": out_163 / out_151 if abs(out_151) > 1e-30 else float("nan"),
        "rank_out_163": float(rank_out_163),
        "rank_out_151": float(rank_out_151),
        "in_163_from_12_raw": in_from_12,
        "in_163_from_151_raw": in_from_151,
        "median_in_to_163": median_in,
        "R_in_163_from_12": (
            in_from_12 / median_in if median_in > 0 else float("nan")
        ),
        "n_residues": float(out_effect.size),
    }


def _fix1_soft_monitor() -> dict[str, Any] | None:
    if not FIX1_COMPARE.is_file():
        return None
    raw = json.loads(FIX1_COMPARE.read_text())
    v = raw.get("verdict") or {}
    return {
        "artifact": str(FIX1_COMPARE),
        "note": "compare-only; Jacobian/knockout lineage as stamped in artifact",
        "R_4OBE": v.get("R_4OBE"),
        "R_4DSO": v.get("R_4DSO"),
        "delta_R": v.get("delta_R"),
        "outcome": v.get("outcome"),
        "rank_163_4OBE": (v.get("secondary") or {}).get("rank_163_4OBE"),
        "rank_163_4DSO": (v.get("secondary") or {}).get("rank_163_4DSO"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
    )
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    structures: dict[str, Any] = {}
    for pdb_id, expected in EXPECTED.items():
        chain = "A"
        prot = load_prot(
            pdb_id,
            chain=chain,
            pdb_dir=args.pdb_dir,
            corpus_cache=args.corpus_cache if pdb_id == "4OBE" else None,
        )
        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.is_file():
            pdb_path = Path(_download_pdb(pdb_id, args.pdb_dir))
        indices = resolve_hub_indices(
            list(prot["residue_ids"]),
            _pdb_residue_names(pdb_path, chain),
            expected,
        )
        print(f"v7 knockout x_hyp {pdb_id} on {args.checkpoint} ...", flush=True)
        model = load_model_from_checkpoint(args.checkpoint, args.device)
        model.eval()
        scan = knockout_scan_x_hyp(model, prot, args.device)
        score = score_knockout(scan["out_effect"], scan["delta_rows"], indices)
        structures[pdb_id] = {
            "alignment": {
                str(r): {
                    "pdb_resname": expected[r],
                    "graph_index": indices[r],
                }
                for r in expected
            },
            "curvature_c": scan["curvature_c"],
            "readout": scan["readout"],
            "baseline_x_hyp_dist0_mean": scan["baseline_x_hyp_dist0_mean"],
            "trunk_knockout": score,
        }
        print(
            f"  R_out_163={score['R_out_163']:.4f} "
            f"rank={int(score['rank_out_163'])} "
            f"raw={score['out_163_raw']:.6f}",
            flush=True,
        )

    r_obe = structures["4OBE"]["trunk_knockout"]["R_out_163"]
    r_dso = structures["4DSO"]["trunk_knockout"]["R_out_163"]
    primary_pass = bool(r_dso > r_obe)
    raw_obe = structures["4OBE"]["trunk_knockout"]["out_163_raw"]
    raw_dso = structures["4DSO"]["trunk_knockout"]["out_163_raw"]
    verdict = {
        "primary_metric": (
            "R_out_163 = mean_j≠163 d_B(x_hyp_knock163[j], x_hyp0[j]) "
            "/ median_i out_effect(i)"
        ),
        "method": "forward_pass_input_knockout_x_hyp_geodesic_no_gradients",
        "R_out_4OBE": r_obe,
        "R_out_4DSO": r_dso,
        "delta_R_out": r_dso - r_obe,
        "pass": primary_pass,
        "outcome": "Pass" if primary_pass else "Fail",
        "raw_logged_for_attribution": {
            "4OBE_out_163_raw": raw_obe,
            "4OBE_median_out": structures["4OBE"]["trunk_knockout"]["median_out"],
            "4DSO_out_163_raw": raw_dso,
            "4DSO_median_out": structures["4DSO"]["trunk_knockout"]["median_out"],
            "raw_out_163_rises": bool(raw_dso > raw_obe),
        },
        "secondary": {
            "rank_out_163_4OBE": structures["4OBE"]["trunk_knockout"]["rank_out_163"],
            "rank_out_163_4DSO": structures["4DSO"]["trunk_knockout"]["rank_out_163"],
            "H_out_4OBE": structures["4OBE"]["trunk_knockout"]["H_out_163_over_151"],
            "H_out_4DSO": structures["4DSO"]["trunk_knockout"]["H_out_163_over_151"],
        },
    }

    out = {
        "schema_version": 1,
        "probe": "tokyo_eye_v7_kras_g12d_hub_migration",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": SPEC,
        "checkpoint": str(args.checkpoint),
        "excluded": [
            "jacobian_flow_influence",
            "disc_pass_criterion",
            "nig_uncertainty",
        ],
        "structures": structures,
        "verdict": verdict,
        "fix1_compare_only": _fix1_soft_monitor(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.output}")
    return 0 if primary_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
