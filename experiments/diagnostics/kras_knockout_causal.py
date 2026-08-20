#!/usr/bin/env python3
"""Forward-pass residue knockout — gradient-free causal cross-check.

Triangulates the Jacobian hub-migration result: zero ``data.x[i]``, forward,
measure trunk ``encoder_h`` displacement on other residues. No backward pass.

Pre-registration companion to
``docs/specs/learned-flow-influence/ablation.md`` §knockout cross-check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser

from experiments.diagnostics.kras_hub_migration_ood import (
    EXPECTED,
    HUB_FROM,
    HUB_TO,
    load_prot,
    resolve_hub_indices,
)
from experiments.diagnostics.jacobian_flow_influence import _capture_layers
from experiments.training.v6._data import _download_pdb
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.training.gnn_lineage import load_model_from_checkpoint


def _pdb_residue_names(pdb_path: Path, chain: str) -> dict[int, str]:
    structure = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    out: dict[int, str] = {}
    for residue in model[chain]:
        het, resseq, _ = residue.id
        if het.strip():
            continue
        out[int(resseq)] = residue.get_resname().upper()
    return out


def _forward_encoder_h(
    model: nn.Module,
    data: Any,
) -> np.ndarray:
    handles, captured = _capture_layers(model)
    try:
        with torch.no_grad():
            _ = model(data)
    finally:
        for handle in handles:
            handle.remove()
    if "encoder_h" not in captured:
        raise RuntimeError("failed to capture encoder_h")
    return captured["encoder_h"].detach().float().cpu().numpy()


def knockout_scan(
    model: nn.Module,
    prot: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Single-residue input knockouts → out-effect and per-source Δ rows."""
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    h0 = _forward_encoder_h(model, data0)[:n]

    out_effect = np.zeros(n, dtype=np.float64)
    delta_rows: list[np.ndarray] = []

    for i in range(n):
        data = prepare_training_batch(model, prot, device)
        data.x = data.x.clone()
        data.x[i] = 0.0
        h = _forward_encoder_h(model, data)[:n]
        delta = np.linalg.norm(h - h0, axis=1)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out_effect[i] = float(np.mean(delta[mask]))
        delta_rows.append(delta)

    return {
        "n_residues": n,
        "baseline_encoder_h_norm_mean": float(np.linalg.norm(h0, axis=1).mean()),
        "out_effect": out_effect,
        "delta_rows": delta_rows,
    }


def score_knockout(
    out_effect: np.ndarray,
    delta_rows: list[np.ndarray],
    indices: dict[int, int],
) -> dict[str, float]:
    """Raw + normalized causal scores at 151/163."""
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

    # In-effect at 163 when knocking source i: delta_rows[i][i163]
    in_to_163 = np.array([row[i163] for row in delta_rows], dtype=np.float64)
    in_to_163[i163] = np.nan  # self-knock undefined for "into"
    finite_in = in_to_163[np.isfinite(in_to_163)]
    median_in = float(np.median(finite_in))
    in_from_12 = float(in_to_163[i12])
    in_from_151 = float(in_to_163[i151])
    r_in_from_12 = in_from_12 / median_in if median_in > 0 else float("nan")

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
        "R_in_163_from_12": r_in_from_12,
        "n_residues": float(out_effect.size),
    }


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import HEALTHY_FIX1_CKPT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=HEALTHY_FIX1_CKPT,
    )
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("pdb_cache"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/learned_flow_influence/"
            "kras_knockout_4obe_4dso.json"
        ),
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.eval()

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
        print(f"knockout-scan {pdb_id} on {args.checkpoint} ...", flush=True)
        # Fresh model weights each structure (no autograd state, but clean).
        model = load_model_from_checkpoint(args.checkpoint, args.device)
        model.eval()
        scan = knockout_scan(model, prot, args.device)
        score = score_knockout(scan["out_effect"], scan["delta_rows"], indices)
        structures[pdb_id] = {
            "alignment": {
                str(r): {
                    "pdb_resname": expected[r],
                    "graph_index": indices[r],
                }
                for r in expected
            },
            "trunk_knockout": score,
        }
        del scan  # drop large delta_rows from memory before JSON

    r_obe = structures["4OBE"]["trunk_knockout"]["R_out_163"]
    r_dso = structures["4DSO"]["trunk_knockout"]["R_out_163"]
    primary_pass = bool(r_dso > r_obe)
    verdict = {
        "primary_metric": (
            "R_out_163 = mean_j≠163 ||Δencoder_h[j]|| after zeroing x[163] "
            "/ median_i out_effect(i)"
        ),
        "method": "forward_pass_input_knockout_no_gradients",
        "R_out_4OBE": r_obe,
        "R_out_4DSO": r_dso,
        "delta_R_out": r_dso - r_obe,
        "pass": primary_pass,
        "outcome": "Pass" if primary_pass else "Fail",
        "raw_logged_for_attribution": {
            "4OBE_out_163_raw": structures["4OBE"]["trunk_knockout"]["out_163_raw"],
            "4OBE_median_out": structures["4OBE"]["trunk_knockout"]["median_out"],
            "4DSO_out_163_raw": structures["4DSO"]["trunk_knockout"]["out_163_raw"],
            "4DSO_median_out": structures["4DSO"]["trunk_knockout"]["median_out"],
        },
        "secondary_in_to_163": {
            "4OBE_from_12": structures["4OBE"]["trunk_knockout"]["in_163_from_12_raw"],
            "4DSO_from_12": structures["4DSO"]["trunk_knockout"]["in_163_from_12_raw"],
            "4OBE_R_in_from_12": structures["4OBE"]["trunk_knockout"]["R_in_163_from_12"],
            "4DSO_R_in_from_12": structures["4DSO"]["trunk_knockout"]["R_in_163_from_12"],
        },
        "triangulation_read": (
            "If knockout also fails/flat while Jacobian R-pass was median-driven: "
            "strengthens 'trunk lacks directional causal hubs' over 'Jacobian "
            "underestimates'. If knockout shows large raw out(163) rise on 4DSO: "
            "revisit Jacobian negatives / saturation before trusting them."
        ),
    }

    out = {
        "schema_version": 1,
        "probe": "kras_knockout_causal",
        "preregistration": (
            "docs/specs/learned-flow-influence/ablation.md §knockout cross-check"
        ),
        "checkpoint": str(args.checkpoint),
        "structures": structures,
        "verdict": verdict,
    }
    # Drop any accidental large arrays
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
