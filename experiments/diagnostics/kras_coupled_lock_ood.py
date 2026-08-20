#!/usr/bin/env python3
"""Forward-only OOD check of the KRAS G12D coupled-lock residues.

The check compares matched baseline and dehydron-scalar checkpoints on residues
12, 32, 60, and 61. OOD sidecars are normalized with the locked Stage A-12
statistics; normalization is never recomputed on the target structure.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.PDB import PDBParser

from experiments.training.v6._data import (
    _download_pdb,
    attach_dehydron_barcode_features,
    load_protein_graph_from_pdb_legacy,
)
from experiments.training.v6.precompute_dehydron_barcodes import (
    precompute_single_pdb,
)
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.common.hyperbolic_lorentz_ops import poincare_distance
from science.training.gnn_lineage import load_model_from_checkpoint

TARGET_IDENTITIES = {12: "ASP", 32: "TYR", 60: "GLY", 61: "GLN"}
TARGET_RESIDUES = tuple(TARGET_IDENTITIES)
LOCK_PAIRS = ((12, 32), (12, 60), (12, 61))


def resolve_target_indices(
    residue_ids: Sequence[str],
    pdb_resnames: Mapping[int, str],
) -> dict[int, int]:
    """Validate PDB identities and map deposited residue numbers to graph rows."""
    graph_by_resnum: dict[int, int] = {}
    for idx, residue_id in enumerate(residue_ids):
        parts = str(residue_id).split(":")
        if len(parts) < 2:
            raise ValueError(f"Unparseable graph residue id: {residue_id!r}")
        resnum = int(parts[1])
        if resnum in graph_by_resnum:
            raise ValueError(f"Duplicate graph residue number {resnum}")
        graph_by_resnum[resnum] = idx

    resolved: dict[int, int] = {}
    for resnum, expected_name in TARGET_IDENTITIES.items():
        actual_name = str(pdb_resnames.get(resnum, "<missing>")).upper()
        if actual_name != expected_name:
            raise ValueError(
                f"Residue {resnum} expected {expected_name}, found {actual_name}; "
                "refusing coupled-lock scoring due to numbering/identity mismatch"
            )
        if resnum not in graph_by_resnum:
            raise ValueError(
                f"Residue {resnum} ({expected_name}) is absent from model graph"
            )
        resolved[resnum] = graph_by_resnum[resnum]
    return resolved


def summarize_motif_distances(
    disc: np.ndarray,
    cone_depth: np.ndarray,
    target_indices: Mapping[int, int],
    *,
    curvature: float,
) -> dict[str, Any]:
    """Compute invariant pair distances and global-scale-normalized compactness.

    Distances are Poincaré geodesics on ``hyp_projections_2d`` (final disc).
    """
    disc = np.asarray(disc, dtype=np.float64)
    cone_depth = np.asarray(cone_depth, dtype=np.float64).reshape(-1)
    if disc.ndim != 2 or disc.shape[0] != cone_depth.shape[0]:
        raise ValueError("disc and cone_depth row counts must match")

    all_distances = [
        poincare_distance(disc[i], disc[j], curvature)
        for i, j in combinations(range(disc.shape[0]), 2)
    ]
    all_pair_median = float(np.median(all_distances))
    if not np.isfinite(all_pair_median) or all_pair_median <= 0.0:
        raise ValueError("All-residue Poincare distance median is not positive")

    pair_rows: dict[str, dict[str, float]] = {}
    for left, right in combinations(TARGET_RESIDUES, 2):
        i, j = target_indices[left], target_indices[right]
        pair_rows[f"{left}-{right}"] = {
            "poincare": float(poincare_distance(disc[i], disc[j], curvature)),
            "cone_depth_delta": float(abs(cone_depth[i] - cone_depth[j])),
        }

    lock_rows = {f"{left}-{right}": pair_rows[f"{left}-{right}"] for left, right in LOCK_PAIRS}
    lock_mean = float(np.mean([row["poincare"] for row in lock_rows.values()]))
    return {
        "layer": "hyp_projections_2d",
        "metric": "poincare",
        "lock_pairs": lock_rows,
        "all_target_pairs": pair_rows,
        "lock_mean_poincare": lock_mean,
        "all_pair_median_poincare": all_pair_median,
        "lock_mean_poincare_normalized": lock_mean / all_pair_median,
    }


def summarize_euclidean_motif_distances(
    trunk: np.ndarray,
    target_indices: Mapping[int, int],
) -> dict[str, Any]:
    """Euclidean lock-pair distances on post-MP ``encoder_h`` (T1a trunk)."""
    trunk = np.asarray(trunk, dtype=np.float64)
    if trunk.ndim != 2:
        raise ValueError(f"encoder_h must be [N, D], got {trunk.shape}")

    all_distances = [
        float(np.linalg.norm(trunk[i] - trunk[j]))
        for i, j in combinations(range(trunk.shape[0]), 2)
    ]
    all_pair_median = float(np.median(all_distances))
    if not np.isfinite(all_pair_median) or all_pair_median <= 0.0:
        raise ValueError("All-residue Euclidean distance median is not positive")

    pair_rows: dict[str, dict[str, float]] = {}
    for left, right in combinations(TARGET_RESIDUES, 2):
        i, j = target_indices[left], target_indices[right]
        pair_rows[f"{left}-{right}"] = {
            "euclidean": float(np.linalg.norm(trunk[i] - trunk[j])),
        }

    lock_rows = {f"{left}-{right}": pair_rows[f"{left}-{right}"] for left, right in LOCK_PAIRS}
    lock_mean = float(np.mean([row["euclidean"] for row in lock_rows.values()]))
    return {
        "layer": "encoder_h",
        "metric": "euclidean_l2",
        "lock_pairs": lock_rows,
        "all_target_pairs": pair_rows,
        "lock_mean_euclidean": lock_mean,
        "all_pair_median_euclidean": all_pair_median,
        "lock_mean_euclidean_normalized": lock_mean / all_pair_median,
    }


def _pdb_residue_names(pdb_path: Path, chain: str) -> dict[int, str]:
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    if chain not in model:
        raise ValueError(f"Chain {chain!r} not found in {pdb_path}")
    return {
        int(residue.id[1]): residue.get_resname().strip().upper()
        for residue in model[chain].get_residues()
        if residue.id[0] == " "
    }


def _capture_encoder_h(model: torch.nn.Module) -> tuple[list[Any], dict[str, torch.Tensor]]:
    """Hook radial/angular heads the same way as T1a trunk occupancy."""
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _radial_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()

    def _angular_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        if "encoder_h" not in captured and inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()

    handles.append(model.radial_head.register_forward_hook(_radial_hook))
    if hasattr(model, "angular_head"):
        handles.append(model.angular_head.register_forward_hook(_angular_hook))
    return handles, captured


def _forward(
    checkpoint: Path,
    prot: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    handles, captured = _capture_encoder_h(model)
    try:
        data = prepare_training_batch(model, prot, device)
        expected_width = int(model.node_emb.in_features)
        actual_width = int(data.x.size(1))
        if actual_width != expected_width:
            raise ValueError(
                f"{checkpoint}: model expects node width {expected_width}, got {actual_width}"
            )
        with torch.no_grad():
            out = model(data)
    finally:
        for handle in handles:
            handle.remove()
    if "encoder_h" not in captured:
        raise RuntimeError(f"{checkpoint}: failed to capture encoder_h via radial/angular hooks")
    disc = out["hyp_projections_2d"].detach().cpu().numpy()
    cone_depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)
    encoder_h = captured["encoder_h"].cpu().numpy()
    curvature = float(model.curvature.detach().cpu().item())
    return disc, cone_depth, encoder_h, curvature


def evaluate_structure(
    *,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    sidecar_dir: Path,
    baseline_checkpoint: Path,
    scalars_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    pdb_id = pdb_id.upper()
    pdb_path = _download_pdb(pdb_id, pdb_dir)
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Failed to build graph for {pdb_id}:{chain}")

    pdb_resnames = _pdb_residue_names(pdb_path, chain)
    indices = resolve_target_indices(prot["residue_ids"], pdb_resnames)
    alignment = {
        str(resnum): {
            "pdb_resname": pdb_resnames[resnum],
            "graph_index": indices[resnum],
            "graph_residue_id": prot["residue_ids"][indices[resnum]],
        }
        for resnum in TARGET_RESIDUES
    }

    base_disc, base_cone, base_trunk, base_c = _forward(
        baseline_checkpoint,
        prot,
        device,
    )
    scalar_prot = copy.deepcopy(prot)
    attach_dehydron_barcode_features(
        scalar_prot,
        barcode_dir=sidecar_dir,
        use_binned=False,
    )
    scalar_disc, scalar_cone, scalar_trunk, scalar_c = _forward(
        scalars_checkpoint,
        scalar_prot,
        device,
    )

    baseline_disc = summarize_motif_distances(
        base_disc,
        base_cone,
        indices,
        curvature=base_c,
    )
    scalars_disc = summarize_motif_distances(
        scalar_disc,
        scalar_cone,
        indices,
        curvature=scalar_c,
    )
    baseline_trunk = summarize_euclidean_motif_distances(base_trunk, indices)
    scalars_trunk = summarize_euclidean_motif_distances(scalar_trunk, indices)

    disc_pair_delta = {
        pair: scalars_disc["lock_pairs"][pair]["poincare"]
        - baseline_disc["lock_pairs"][pair]["poincare"]
        for pair in baseline_disc["lock_pairs"]
    }
    trunk_pair_delta = {
        pair: scalars_trunk["lock_pairs"][pair]["euclidean"]
        - baseline_trunk["lock_pairs"][pair]["euclidean"]
        for pair in baseline_trunk["lock_pairs"]
    }
    trunk_norm_delta = (
        scalars_trunk["lock_mean_euclidean_normalized"]
        - baseline_trunk["lock_mean_euclidean_normalized"]
    )
    disc_norm_delta = (
        scalars_disc["lock_mean_poincare_normalized"]
        - baseline_disc["lock_mean_poincare_normalized"]
    )
    if abs(trunk_norm_delta) < 0.02 and abs(disc_norm_delta) >= 0.05:
        layer_verdict = "projection_only"
    elif abs(trunk_norm_delta) >= 0.05 and np.sign(trunk_norm_delta) == np.sign(disc_norm_delta):
        layer_verdict = "trunk_and_disc_same_sign"
    elif abs(trunk_norm_delta) >= 0.05 and np.sign(trunk_norm_delta) != np.sign(disc_norm_delta):
        layer_verdict = "trunk_disc_disagree"
    else:
        layer_verdict = "ambiguous_small"

    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "pdb_path": str(pdb_path),
        "residue_alignment": alignment,
        "baseline": {
            "curvature": base_c,
            "disc": baseline_disc,
            "trunk": baseline_trunk,
            # Backward-compatible flat disc fields for previous report consumers.
            **baseline_disc,
        },
        "scalars": {
            "curvature": scalar_c,
            "disc": scalars_disc,
            "trunk": scalars_trunk,
            **scalars_disc,
        },
        "delta_scalars_minus_baseline": {
            "lock_pair_poincare": disc_pair_delta,
            "lock_mean_poincare": (
                scalars_disc["lock_mean_poincare"] - baseline_disc["lock_mean_poincare"]
            ),
            "lock_mean_poincare_normalized": disc_norm_delta,
            "lock_pair_euclidean_trunk": trunk_pair_delta,
            "lock_mean_euclidean_trunk": (
                scalars_trunk["lock_mean_euclidean"]
                - baseline_trunk["lock_mean_euclidean"]
            ),
            "lock_mean_euclidean_trunk_normalized": trunk_norm_delta,
            "layer_verdict": layer_verdict,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-id", action="append", dest="pdb_ids")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/kras_coupled_lock_ood/sidecars"),
    )
    parser.add_argument(
        "--zscore-stats",
        type=Path,
        default=Path("checkpoints/v65/dehydron_barcode_v1/corpus_zscore_stats.json"),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("checkpoints/v66/runs/dbh_ablation_baseline_v1/phase_2.pt"),
    )
    parser.add_argument(
        "--scalars-checkpoint",
        type=Path,
        default=Path("checkpoints/v66/runs/dbh_ablation_scalars_v1/phase_2.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/kras_coupled_lock_ood/report.json"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.environ["GNN_INPUT_MODE"] = "topology_three_vector"
    os.environ["TRAINING_LOAD_FROM_PDB"] = "1"
    pdb_ids = args.pdb_ids or ["1AGP", "4DSO"]
    args.pdb_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for pdb_id in pdb_ids:
        precompute_single_pdb(
            pdb_id=pdb_id,
            chain=args.chain,
            pdb_dir=args.pdb_dir,
            out_dir=args.sidecar_dir,
            zscore_stats_path=args.zscore_stats,
            use_binned=False,
        )
        results.append(
            evaluate_structure(
                pdb_id=pdb_id,
                chain=args.chain,
                pdb_dir=args.pdb_dir,
                sidecar_dir=args.sidecar_dir,
                baseline_checkpoint=args.baseline_checkpoint,
                scalars_checkpoint=args.scalars_checkpoint,
                device=args.device,
            )
        )

    report = {
        "schema_version": 2,
        "interpretation": (
            "Forward-only OOD hypothesis generator. Distances are reported at two "
            "layers: disc = Poincare on hyp_projections_2d; trunk = Euclidean L2 on "
            "encoder_h (post-MP, same hook as T1a). Negative deltas mean Scalars "
            "places the G12D lock residues closer."
        ),
        "zscore_stats": str(args.zscore_stats),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "scalars_checkpoint": str(args.scalars_checkpoint),
        "structures": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
