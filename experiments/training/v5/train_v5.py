"""
train_v5.py — Tokyo Eyes v5 Training Script
=============================================
Eidetix Bio | 2026-05-20

Training regime for the decoupled radial-angular architecture.

Key differences from v4 training:
  1. No radial/angular conflict — losses are architecturally isolated
  2. Faster convergence expected — each head optimizes independently
  3. 3-stage curriculum:
     Stage 1: Radial calibration (cone loss dominant, angular frozen)
     Stage 2: Angular structure (domain sep + angular diversity, radial frozen)
     Stage 3: Joint fine-tuning (all losses, all parameters)
  4. Monitors gradient norms per pathway to verify isolation

Usage:
    python train_v5.py --pdb_dir /path/to/pdbs --output_dir ./checkpoints_v5
"""

"""
Expected directory layout when deployed:

  /home/rswan/Documents/adk-samples/python/agents/data-science/
    DTIE_GNN_ORCHESTRATION/
      TokyoEyesv4/
        Gnnv4.py
        train_v4.py          <- load_protein_graph, TRAINING_TARGETS imported from here
        diagnose_v4.py
        ...
      TokyoEyesv5/
        Gnnv5.py             <- v5 model
        train_v5.py          <- THIS FILE
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA

# Resolve paths relative to this file's location in the DTIE orchestration tree:
# /home/rswan/Documents/adk-samples/python/agents/data-science/
#   DTIE_GNN_ORCHESTRATION/
#     TokyoEyesv5/   <- this file lives here
#     TokyoEyesv4/   <- v4 data loading utilities live here
_THIS_DIR = Path(__file__).resolve().parent
_ORCHESTRATION_DIR = _THIS_DIR.parent

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ORCHESTRATION_DIR / "TokyoEyesv4"))

from Gnnv5 import (
    GOSPConeMapper,
    gosp_loss_v5,
    build_optimizer,
    precompute_clustering,
)
from train_v4 import (
    TRAINING_TARGETS,
    load_protein_graph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_v5")


def evaluate_disc_structure(
    model: GOSPConeMapper,
    proteins: List[Dict],
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Evaluate disc and ball structure quality per protein."""
    model.eval()
    results = {}

    with torch.no_grad():
        for prot in proteins:
            data = prot["data"].to(device)
            output = model(data)

            hyp_proj_2d = output["hyp_projections_2d"].cpu().numpy()
            hyp_proj_3d = output["hyp_projections_3d"].cpu().numpy()
            x_hyp = output["x_routed_hyp"].cpu().numpy()
            cone_depth = output["cone_depth"].cpu().squeeze()
            radial_feat = output["radial_features"].cpu().squeeze().numpy()

            # PC1 variance of the full ball embedding
            if x_hyp.shape[0] > 3:
                pca = PCA(n_components=min(3, x_hyp.shape[1]))
                pca.fit(x_hyp)
                pc1_var = float(pca.explained_variance_ratio_[0])
            else:
                pc1_var = 1.0

            # Disc and ball coordinate stats
            disc_norms = np.linalg.norm(hyp_proj_2d, axis=1)
            ball_norms = np.linalg.norm(hyp_proj_3d, axis=1)
            has_2d = pc1_var < 0.90

            # Expert usage
            expert_w = output["expert_weights"].cpu().numpy().mean(axis=0)

            # Radial health
            radial_std = float(np.std(radial_feat))
            radial_range = float(np.max(radial_feat) - np.min(radial_feat))

            # Correlation with target rho
            cone_rho_corr = 0.0
            if "target_rho" in prot:
                rho = prot["target_rho"].squeeze().numpy()
                d = cone_depth.numpy()
                if d.std() > 1e-6:
                    cone_rho_corr = float(np.corrcoef(rho, d)[0, 1])

            results[prot["pdb_id"]] = {
                "pc1_variance": pc1_var,
                "has_2d_structure": has_2d,
                "disc_norm_mean": float(disc_norms.mean()),
                "disc_norm_max": float(disc_norms.max()),
                "ball_norm_mean": float(ball_norms.mean()),
                "ball_norm_max": float(ball_norms.max()),
                "disc_inside_ball": bool((disc_norms < 1.0).all()),
                "ball_inside_ball": bool((ball_norms < 1.0).all()),
                "expert_usage": expert_w.tolist(),
                "n_residues": prot["n_residues"],
                "radial_std": radial_std,
                "radial_range": radial_range,
                "cone_rho_correlation": cone_rho_corr,
            }

    return results


def train_epoch(
    model: GOSPConeMapper,
    optimizer,
    proteins: List[Dict],
    loss_coeffs: Dict[str, float],
    freeze_radial: bool = False,
    freeze_angular: bool = False,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train one epoch with configurable pathway freezing."""
    model.train()

    # Freeze/unfreeze pathways
    for p in model.radial_head.parameters():
        p.requires_grad = not freeze_radial
    for p in model.angular_head.parameters():
        p.requires_grad = not freeze_angular

    epoch_losses = {
        "total": [], "evidential": [], "balance": [],
        "cone_consistency": [], "neighborhood_consistency": [],
        "angular_diversity": [], "domain_separation_2d": [],
        "domain_separation_3d": [],
    }
    grad_norms = {"radial": [], "angular": [], "backbone": []}

    for prot in proteins:
        data = prot["data"].to(device)
        target_rho = prot["target_rho"].to(device)
        target_dehydron = prot["target_dehydron"].to(device)
        ca_coords = prot["ca_coords"].to(device)
        domain_labels = prot.get("domain_labels")
        if domain_labels is not None:
            domain_labels = domain_labels.to(device)

        optimizer.zero_grad()
        output = model(data)

        losses = gosp_loss_v5(
            output=output,
            target_rho=target_rho,
            target_dehydron=target_dehydron,
            ca_coords=ca_coords,
            domain_labels=domain_labels,
            **loss_coeffs,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track gradient norms per pathway
        r_norm = sum(
            p.grad.norm().item() for p in model.radial_head.parameters()
            if p.grad is not None
        )
        a_norm = sum(
            p.grad.norm().item() for p in model.angular_head.parameters()
            if p.grad is not None
        )
        b_norm = sum(
            p.grad.norm().item() for p in model.convs.parameters()
            if p.grad is not None
        )
        grad_norms["radial"].append(r_norm)
        grad_norms["angular"].append(a_norm)
        grad_norms["backbone"].append(b_norm)

        for k in epoch_losses:
            if k in losses:
                val = losses[k]
                epoch_losses[k].append(val.item() if torch.is_tensor(val) else float(val))

    result = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    result["grad_radial"] = np.mean(grad_norms["radial"])
    result["grad_angular"] = np.mean(grad_norms["angular"])
    result["grad_backbone"] = np.mean(grad_norms["backbone"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Train Tokyo Eyes v5")
    parser.add_argument("--pdb_dir", type=str, default="/tmp/dtie_pdb_cache")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_v5")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    logger.info(f"Device: {device}")

    # ── Load proteins ─────────────────────────────────────────────────────
    logger.info("Loading training proteins...")
    all_proteins = {}
    for pdb_id, info in TRAINING_TARGETS.items():
        prot = load_protein_graph(pdb_id, info["chain"], pdb_dir)
        if prot is not None:
            all_proteins[pdb_id] = prot
            logger.info(f"  {pdb_id} ({info['gene']} {info['desc']}): "
                       f"{prot['n_residues']} residues")

    if not all_proteins:
        logger.error("No proteins loaded.")
        sys.exit(1)

    all_protein_list = list(all_proteins.values())
    logger.info(f"Loaded {len(all_proteins)} proteins")

    # ── Initialize model ──────────────────────────────────────────────────
    model = GOSPConeMapper(
        node_dim=4,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        projection_dim=64,
        hyp_proj_dim_2d=2,
        hyp_proj_dim_3d=3,
        depth_conditioning=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    radial_params = sum(p.numel() for p in model.radial_head.parameters())
    angular_params = sum(p.numel() for p in model.angular_head.parameters())
    logger.info(f"Model: {total_params:,} params "
               f"(radial: {radial_params:,}, angular: {angular_params:,})")
    logger.info(f"Initial curvature: {model.curvature.item():.4f}")

    # ── 3-Stage Curriculum ────────────────────────────────────────────────
    # Stage 1: Radial calibration — teach the model burial depth
    # Stage 2: Angular structure — teach domain clustering
    # Stage 3: Joint — refine both together with full loss

    stages = [
        {
            "name": "Stage 1: Radial calibration",
            "epochs": 20,
            "lr": args.lr,
            "freeze_radial": False,
            "freeze_angular": True,  # Angular frozen — only radial learns
            "coeffs": {
                "evidential_coeff": 0.005,
                "balance_coeff": 0.01,
                "cone_coeff": 0.30,          # High — this is the focus
                "neighborhood_coeff": 0.10,   # Low — just prevent collapse
                "angular_coeff": 0.0,         # Off — angular is frozen
                "domain_sep_2d_coeff": 0.0,   # Off
                "domain_sep_3d_coeff": 0.0,   # Off
            },
        },
        {
            "name": "Stage 2: Angular structure",
            "epochs": 30,
            "lr": args.lr,
            "freeze_radial": True,   # Radial frozen — preserve burial ordering
            "freeze_angular": False,
            "coeffs": {
                "evidential_coeff": 0.005,
                "balance_coeff": 0.01,
                "cone_coeff": 0.0,            # Off — radial is frozen
                "neighborhood_coeff": 0.30,
                "angular_coeff": 0.30,
                "domain_sep_2d_coeff": 0.40,  # High — this is the focus
                "domain_sep_3d_coeff": 0.40,
            },
        },
        {
            "name": "Stage 3: Joint fine-tuning",
            "epochs": 30,
            "lr": args.lr * 0.3,
            "freeze_radial": False,
            "freeze_angular": False,
            "coeffs": {
                "evidential_coeff": 0.005,
                "balance_coeff": 0.005,
                "cone_coeff": 0.15,
                "neighborhood_coeff": 0.25,
                "angular_coeff": 0.20,
                "domain_sep_2d_coeff": 0.30,
                "domain_sep_3d_coeff": 0.30,
            },
        },
    ]

    metrics_log = []
    global_epoch = 0
    best_score = 0.0

    for stage in stages:
        logger.info(f"\n{'='*70}")
        logger.info(f"{stage['name']} ({stage['epochs']} epochs, lr={stage['lr']})")
        logger.info(f"  freeze_radial={stage['freeze_radial']} "
                   f"freeze_angular={stage['freeze_angular']}")
        logger.info(f"  coeffs: {stage['coeffs']}")
        logger.info(f"{'='*70}")

        optimizer = build_optimizer(model, lr=stage["lr"])

        for epoch in range(stage["epochs"]):
            global_epoch += 1
            t0 = time.time()

            losses = train_epoch(
                model=model,
                optimizer=optimizer,
                proteins=all_protein_list,
                loss_coeffs=stage["coeffs"],
                freeze_radial=stage["freeze_radial"],
                freeze_angular=stage["freeze_angular"],
                device=device,
            )

            elapsed = time.time() - t0
            c_val = model.curvature.item()
            r_scale = model.radial_head.radial_scale.item()

            # Eval
            disc_eval = evaluate_disc_structure(model, all_protein_list, device)
            n_2d = sum(1 for v in disc_eval.values() if v["has_2d_structure"])
            pc1_mean = np.mean([v["pc1_variance"] for v in disc_eval.values()])
            radial_std_mean = np.mean([v["radial_std"] for v in disc_eval.values()])
            corr_mean = np.mean([v["cone_rho_correlation"] for v in disc_eval.values()])

            logger.info(
                f"  Epoch {global_epoch:3d} | "
                f"loss={losses['total']:.4f} "
                f"(cone={losses['cone_consistency']:.4f} "
                f"ang={losses['angular_diversity']:.4f} "
                f"nbr={losses['neighborhood_consistency']:.4f} "
                f"dom2d={losses['domain_separation_2d']:.4f} "
                f"dom3d={losses['domain_separation_3d']:.4f}) | "
                f"c={c_val:.4f} r_scale={r_scale:.4f} | "
                f"PC1={pc1_mean:.3f} 2D={n_2d}/{len(disc_eval)} | "
                f"rad_std={radial_std_mean:.4f} corr={corr_mean:.3f} | "
                f"∇rad={losses['grad_radial']:.4f} "
                f"∇ang={losses['grad_angular']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Track best (composite score: 2D count + correlation)
            score = n_2d + max(0, corr_mean)
            if score > best_score:
                best_score = score
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "curvature": c_val,
                    "radial_scale": r_scale,
                    "stage": stage["name"],
                    "global_epoch": global_epoch,
                    "score": score,
                    "architecture": {
                        "version": "v5",
                        "node_dim": 4, "hidden": args.hidden,
                        "num_layers": args.num_layers,
                        "num_experts": args.num_experts,
                        "projection_dim": 64,
                        "hyp_proj_dim_2d": 2,
                        "hyp_proj_dim_3d": 3,
                    },
                }, output_dir / "best_checkpoint.pt")

            metrics_log.append({
                "global_epoch": global_epoch,
                "stage": stage["name"],
                "losses": losses,
                "curvature": c_val,
                "radial_scale": r_scale,
                "pc1_mean": pc1_mean,
                "n_2d": n_2d,
                "radial_std_mean": radial_std_mean,
                "cone_rho_correlation": corr_mean,
                "elapsed": elapsed,
            })

            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2, default=str)

        # Save stage checkpoint
        stage_name = stage["name"].split(":")[0].strip().lower().replace(" ", "_")
        torch.save({
            "model_state_dict": model.state_dict(),
            "curvature": c_val,
            "radial_scale": model.radial_head.radial_scale.item(),
            "stage": stage["name"],
            "global_epoch": global_epoch,
            "architecture": {
                "version": "v5",
                "node_dim": 4, "hidden": args.hidden,
                "num_layers": args.num_layers,
                "num_experts": args.num_experts,
                "projection_dim": 64,
                "hyp_proj_dim_2d": 2,
                "hyp_proj_dim_3d": 3,
            },
        }, output_dir / f"checkpoint_{stage_name}.pt")
        logger.info(f"  Saved: checkpoint_{stage_name}.pt")

    # ── Final ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best score: {best_score:.2f}")
    logger.info(f"  Final curvature: {model.curvature.item():.4f}")
    logger.info(f"  Final radial_scale: {model.radial_head.radial_scale.item():.4f}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
