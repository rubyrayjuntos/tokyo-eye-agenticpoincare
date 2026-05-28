"""
export_for_viewer.py — Export v5 checkpoint embeddings for the React viewer
============================================================================
Eidetix Bio | 2026-05-20

Runs inference on a trained v5 checkpoint and exports the embeddings as
JSON that the Tokyo Eyes viewer can directly consume.

Usage:
    python export_for_viewer.py \
        --checkpoint ./checkpoints_v5/best_checkpoint.pt \
        --pdb_id 4OBE --chain A \
        --pdb_dir /tmp/dtie_pdb_cache \
        --output ./viewer_data.json

Output JSON schema matches the viewer's NodeData/EdgeData interfaces:
{
  "metadata": { "pdb_id", "curvature", "version", "n_residues", ... },
  "nodes": [{ "id", "position", "position2D", "color", "domain", "isOutlier", "value", "depth", "expertId" }],
  "edges": [{ "source", "target", "distance" }]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ORCHESTRATION_DIR = _THIS_DIR.parent

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ORCHESTRATION_DIR / "TokyoEyesv4"))

from Gnnv5 import GOSPConeMapper, precompute_clustering
from train_v4 import load_protein_graph, TRAINING_TARGETS

# KRAS domain annotations for coloring
KRAS_DOMAINS = {
    "P-loop": (range(10, 18), "#ff0055"),
    "Switch-I": (range(25, 41), "#00ffcc"),
    "Switch-II": (range(57, 76), "#aa00ff"),
    "α3-helix": (range(87, 105), "#ffaa00"),
    "α4-helix": (range(116, 127), "#00aaff"),
    "C-terminal": (range(145, 170), "#ff6b6b"),
}


def get_domain_info(resnum: int) -> tuple:
    """Returns (domain_name, color) for a KRAS residue number."""
    for name, (rng, color) in KRAS_DOMAINS.items():
        if resnum in rng:
            return name, color
    return "other", "#666666"


def export_protein(
    model: GOSPConeMapper,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str = "cpu",
) -> Dict:
    """Run inference and build viewer-compatible JSON."""
    from geoopt.manifolds.stereographic import math as pmath

    prot = load_protein_graph(pdb_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Failed to load {pdb_id} chain {chain}")

    model.eval()
    with torch.no_grad():
        data = prot["data"].to(device)
        output = model(data)

    # Extract embeddings
    hyp_3d = output["hyp_projections_3d"].cpu().numpy()  # [N, 3]
    hyp_2d = output["hyp_projections_2d"].cpu().numpy()  # [N, 2]
    cone_depth = output["cone_depth"].squeeze().cpu().numpy()  # [N]
    expert_weights = output["expert_weights"].cpu().numpy()  # [N, num_experts]
    epistemic = output["uncertainty"]["epistemic"].squeeze().cpu().numpy()
    aleatoric = output["uncertainty"]["aleatoric"].squeeze().cpu().numpy()
    rho = prot["target_rho"].squeeze().numpy()

    c = model.curvature.item()
    n_residues = len(prot["residue_ids"])

    # Build nodes
    nodes = []
    for i in range(n_residues):
        res_id = prot["residue_ids"][i]
        resnum = int(res_id.split(":")[1])
        domain_name, domain_color = get_domain_info(resnum)

        # Outlier detection: high aleatoric + low epistemic = genuine void
        is_outlier = bool(
            aleatoric[i] > np.median(aleatoric) * 1.5 and
            epistemic[i] < np.median(epistemic)
        )

        nodes.append({
            "id": f"res_{resnum}",
            "position": [float(hyp_3d[i, 0]), float(hyp_3d[i, 1]), float(hyp_3d[i, 2])],
            "position2D": [float(hyp_2d[i, 0]), float(hyp_2d[i, 1])],
            "color": domain_color,
            "domain": domain_name,
            "isOutlier": is_outlier,
            "value": float(rho[i] / 30.0),
            "depth": float(cone_depth[i]),
            "expertId": int(expert_weights[i].argmax()),
        })

    # Build edges: k-nearest in 3D hyperbolic space
    k = 4
    edges = []

    # Compute pairwise hyperbolic distances from 3D projections
    # Using the Poincaré ball distance formula
    for i in range(n_residues):
        dists = []
        for j in range(n_residues):
            if i == j:
                dists.append((j, float('inf')))
                continue
            # Poincaré distance in 3D
            xi = hyp_3d[i]
            xj = hyp_3d[j]
            norm_xi_sq = np.sum(xi ** 2)
            norm_xj_sq = np.sum(xj ** 2)
            diff_sq = np.sum((xi - xj) ** 2)
            denom = (1 - c * norm_xi_sq) * (1 - c * norm_xj_sq)
            if denom < 1e-10:
                dists.append((j, float('inf')))
                continue
            z = 1.0 + 2.0 * c * diff_sq / max(denom, 1e-10)
            z = max(z, 1.0 + 1e-10)
            d = (1.0 / np.sqrt(c)) * np.arccosh(z)
            dists.append((j, float(d)))

        dists.sort(key=lambda x: x[1])
        for j_idx, d in dists[:k]:
            if j_idx > i:  # avoid duplicates
                edges.append({
                    "source": f"res_{int(prot['residue_ids'][i].split(':')[1])}",
                    "target": f"res_{int(prot['residue_ids'][j_idx].split(':')[1])}",
                    "distance": d,
                })

    metadata = {
        "pdb_id": pdb_id,
        "chain": chain,
        "curvature": c,
        "version": "v5",
        "architecture": "decoupled_radial_angular",
        "n_residues": n_residues,
        "radial_scale": model.radial_head.radial_scale.item(),
        "depth_range": [float(cone_depth.min()), float(cone_depth.max())],
        "depth_std": float(cone_depth.std()),
    }

    return {
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(description="Export v5 embeddings for viewer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pdb_id", type=str, default="4OBE")
    parser.add_argument("--chain", type=str, default="A")
    parser.add_argument("--pdb_dir", type=str, default="/tmp/dtie_pdb_cache")
    parser.add_argument("--output", type=str, default="./viewer_data.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
    arch = ckpt.get("architecture", {})

    model = GOSPConeMapper(
        node_dim=4,
        hidden=arch.get("hidden", 128),
        num_layers=arch.get("num_layers", 6),
        num_experts=arch.get("num_experts", 4),
        projection_dim=arch.get("projection_dim", 64),
        hyp_proj_dim_2d=arch.get("hyp_proj_dim_2d", 2),
        hyp_proj_dim_3d=arch.get("hyp_proj_dim_3d", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Curvature: {model.curvature.item():.4f}")
    print(f"  Radial scale: {model.radial_head.radial_scale.item():.4f}")

    # Export
    result = export_protein(
        model=model,
        pdb_id=args.pdb_id,
        chain=args.chain,
        pdb_dir=Path(args.pdb_dir),
        device=args.device,
    )

    # Write JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nExported: {output_path}")
    print(f"  Nodes: {len(result['nodes'])}")
    print(f"  Edges: {len(result['edges'])}")
    print(f"  Curvature: {result['metadata']['curvature']:.4f}")
    print(f"  Depth range: {result['metadata']['depth_range']}")


if __name__ == "__main__":
    main()
