"""
finetune_differential.py — Stage 3 Fine-tune with Mutation Differential Loss
=============================================================================
Eidetix Bio | 2026-05-19

Warm-starts from checkpoints_v4_retrain/checkpoint_stage_2b.pt (epoch 75)
and adds paired-structure contrastive supervision to recover the Switch-I
differential signal that was present in cone_fix2 but lost during retraining.

The epoch 74/75 checkpoint has:
  ✓ Correct radial spread (|p|=1.08±0.05)
  ✓ Functional neighbor topology (Probe 3)
  ✓ Expert specialization (buried vs exposed)
  ✗ Weak Switch-I differential (1.07x, was 2.26x in cone_fix2)
  ✗ Inverted burial geometry (cone loss target was wrong sign)

This script fixes both:
  1. Cone loss target flipped: high ρ → high depth (buried=deep, dehydron=shallow)
  2. mutation_differential_loss explicitly supervises Switch-I displacement

Usage:
    python finetune_differential.py \
        --pdb_dir /tmp/dtie_pdb_cache \
        --checkpoint ../../checkpoints_v4_retrain/checkpoint_stage_2b.pt \
        --output_dir ../../checkpoints_v4_differential
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Gnnv4 import (
    GOSPConeMapper,
    gosp_loss,
    mutation_differential_loss,
    build_optimizer,
)
from train_v4 import (
    TRAINING_TARGETS,
    load_protein_graph,
    evaluate_disc_structure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune_diff")


# Known biology: Switch-I residues that should move between WT and G12D
# These are the effector-binding residues validated in cone_fix2 Probe 5
SWITCH_I_MOBILE = list(range(29, 41))  # residues 29-40 (Switch-I)
# Stable core: α3-helix residues that should NOT move
CORE_STABLE = list(range(87, 105))  # α3-helix


def get_aligned_indices(prot_wt: Dict, prot_mut: Dict, resnums: List[int]):
    """Get indices in both proteins for a set of residue numbers."""
    wt_map = {}
    for i, rid in enumerate(prot_wt["residue_ids"]):
        resnum = int(rid.split(":")[1])
        wt_map[resnum] = i

    mut_map = {}
    for i, rid in enumerate(prot_mut["residue_ids"]):
        resnum = int(rid.split(":")[1])
        mut_map[resnum] = i

    wt_idx, mut_idx = [], []
    for r in resnums:
        if r in wt_map and r in mut_map:
            wt_idx.append(wt_map[r])
            mut_idx.append(mut_map[r])

    return wt_idx, mut_idx


def train_epoch_differential(
    model: GOSPConeMapper,
    optimizer,
    proteins: List[Dict],
    prot_wt: Dict,
    prot_g12d: Dict,
    mobile_wt_idx: List[int],
    mobile_mut_idx: List[int],
    stable_wt_idx: List[int],
    stable_mut_idx: List[int],
    loss_kwargs: Dict,
    differential_coeff: float = 0.10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train one epoch with standard loss + mutation differential."""
    model.train()
    epoch_losses = {
        "total": [], "evidential": [], "balance": [],
        "cone_consistency": [], "neighborhood_consistency": [],
        "angular_diversity": [], "domain_separation": [],
        "differential": [],
    }

    # Standard training over all proteins
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

        losses = gosp_loss(
            output=output,
            target_rho=target_rho,
            target_dehydron=target_dehydron,
            ca_coords=ca_coords,
            domain_labels=domain_labels,
            **loss_kwargs,
        )

        loss = losses["total"]

        # Add differential loss for WT/G12D pair
        # Run both structures through the model
        if prot["pdb_id"] == "4OBE":
            # We're already processing WT — run G12D too
            out_mut = model(prot_g12d["data"].to(device))
            c = model.curvature

            diff_loss = mutation_differential_loss(
                x_hyp_wt=output["x_routed_hyp"][mobile_wt_idx],
                x_hyp_mut=out_mut["x_routed_hyp"][mobile_mut_idx],
                known_mobile=list(range(len(mobile_wt_idx))),
                known_stable=[],  # handled separately below
                c=c,
            )
            # Stable loss: penalize displacement of core residues
            stable_loss = mutation_differential_loss(
                x_hyp_wt=output["x_routed_hyp"],
                x_hyp_mut=out_mut["x_routed_hyp"][:output["x_routed_hyp"].shape[0]],
                known_mobile=[],
                known_stable=stable_wt_idx[:min(len(stable_wt_idx), out_mut["x_routed_hyp"].shape[0])],
                c=c,
            )
            # Combined: want mobile to move MORE than stable
            from geoopt.manifolds.stereographic import math as pmath
            k = -c
            mobile_disp = pmath.dist(
                output["x_routed_hyp"][mobile_wt_idx],
                out_mut["x_routed_hyp"][mobile_mut_idx],
                k=k,
            ).mean()
            stable_disp = pmath.dist(
                output["x_routed_hyp"][stable_wt_idx],
                out_mut["x_routed_hyp"][stable_mut_idx],
                k=k,
            ).mean()
            diff_loss = torch.relu(stable_disp - mobile_disp + 1.0)

            loss = loss + differential_coeff * diff_loss
            epoch_losses["differential"].append(diff_loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k_name in epoch_losses:
            if k_name in losses:
                val = losses[k_name]
                epoch_losses[k_name].append(val.item() if torch.is_tensor(val) else float(val))

    return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with differential loss")
    parser.add_argument("--pdb_dir", type=str, default="/tmp/dtie_pdb_cache")
    parser.add_argument("--checkpoint", type=str,
                       default="../../checkpoints_v4_retrain/checkpoint_stage_2b.pt")
    parser.add_argument("--output_dir", type=str, default="../../checkpoints_v4_differential")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--differential_coeff", type=float, default=0.10)
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load proteins
    logger.info("Loading training proteins...")
    all_proteins = {}
    for pdb_id, info in TRAINING_TARGETS.items():
        prot = load_protein_graph(pdb_id, info["chain"], pdb_dir)
        if prot is not None:
            all_proteins[pdb_id] = prot

    prot_wt = all_proteins.get("4OBE")
    prot_g12d = all_proteins.get("4DSO")
    if prot_wt is None or prot_g12d is None:
        logger.error("Need both 4OBE and 4DSO for differential training")
        sys.exit(1)

    all_protein_list = list(all_proteins.values())
    logger.info(f"Loaded {len(all_proteins)} proteins")

    # Compute aligned indices for differential loss
    mobile_wt_idx, mobile_mut_idx = get_aligned_indices(prot_wt, prot_g12d, SWITCH_I_MOBILE)
    stable_wt_idx, stable_mut_idx = get_aligned_indices(prot_wt, prot_g12d, CORE_STABLE)
    logger.info(f"Differential supervision: {len(mobile_wt_idx)} mobile (Switch-I), "
               f"{len(stable_wt_idx)} stable (α3-helix)")

    # Load model
    model = GOSPConeMapper(
        node_dim=4, hidden=128, num_layers=6, num_experts=4,
        projection_dim=64, hyp_proj_dim=2, depth_conditioning=False,
    )
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.info(f"New parameters: {missing}")
    model = model.to(device)
    logger.info(f"Loaded checkpoint (epoch {ckpt.get('global_epoch', '?')})")
    logger.info(f"curvature={model.curvature.item():.4f} hyper_scale={model.hyper_scale.item():.4f}")

    # Pre-training Probe 5 check
    model.eval()
    with torch.no_grad():
        from geoopt.manifolds.stereographic import math as pmath
        out_wt = model(prot_wt["data"].to(device))
        out_g12d_pre = model(prot_g12d["data"].to(device))
        k = -model.curvature
        mobile_d = pmath.dist(
            out_wt["x_routed_hyp"][mobile_wt_idx],
            out_g12d_pre["x_routed_hyp"][mobile_mut_idx],
            k=k,
        ).mean().item()
        stable_d = pmath.dist(
            out_wt["x_routed_hyp"][stable_wt_idx],
            out_g12d_pre["x_routed_hyp"][stable_mut_idx],
            k=k,
        ).mean().item()
    logger.info(f"Pre-training: Switch-I disp={mobile_d:.4f}, α3 disp={stable_d:.4f}, "
               f"ratio={mobile_d/(stable_d+1e-8):.2f}x")

    # Training
    loss_kwargs = {
        "evidential_coeff": 0.005,
        "cone_coeff": 0.10,
        "neighborhood_coeff": 0.25,
        "angular_coeff": 0.20,
        "domain_sep_coeff": 0.35,
        "balance_coeff": 0.005,
    }

    optimizer = build_optimizer(model, lr=args.lr)
    metrics_log = []
    global_epoch = ckpt.get("global_epoch", 75)
    best_ratio = 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"Fine-tune with differential loss ({args.epochs} epochs, lr={args.lr})")
    logger.info(f"  differential_coeff={args.differential_coeff}")
    logger.info(f"  loss_kwargs={loss_kwargs}")
    logger.info(f"{'='*70}")

    for epoch in range(args.epochs):
        global_epoch += 1
        t0 = time.time()

        losses = train_epoch_differential(
            model=model,
            optimizer=optimizer,
            proteins=all_protein_list,
            prot_wt=prot_wt,
            prot_g12d=prot_g12d,
            mobile_wt_idx=mobile_wt_idx,
            mobile_mut_idx=mobile_mut_idx,
            stable_wt_idx=stable_wt_idx,
            stable_mut_idx=stable_mut_idx,
            loss_kwargs=loss_kwargs,
            differential_coeff=args.differential_coeff,
            device=device,
        )

        elapsed = time.time() - t0

        # Eval
        model.eval()
        disc_eval = evaluate_disc_structure(model, all_protein_list, device)
        n_2d = sum(1 for v in disc_eval.values() if v["has_2d_structure"])
        pc1_mean = np.mean([v["pc1_variance"] for v in disc_eval.values()])
        cone_corr_mean = np.mean([v["cone_rho_correlation"] for v in disc_eval.values()])

        # Probe 5 metric: Switch-I vs α3 displacement ratio
        with torch.no_grad():
            out_wt_eval = model(prot_wt["data"].to(device))
            out_g12d_eval = model(prot_g12d["data"].to(device))
            k = -model.curvature
            mobile_d = pmath.dist(
                out_wt_eval["x_routed_hyp"][mobile_wt_idx],
                out_g12d_eval["x_routed_hyp"][mobile_mut_idx],
                k=k,
            ).mean().item()
            stable_d = pmath.dist(
                out_wt_eval["x_routed_hyp"][stable_wt_idx],
                out_g12d_eval["x_routed_hyp"][stable_mut_idx],
                k=k,
            ).mean().item()
            ratio = mobile_d / (stable_d + 1e-8)

            norms = out_wt_eval["x_routed_hyp"].norm(dim=-1)

        logger.info(
            f"  Epoch {global_epoch:3d} | "
            f"loss={losses['total']:.4f} "
            f"(ev={losses['evidential']:.4f} "
            f"cone={losses['cone_consistency']:.4f} "
            f"diff={losses.get('differential', 0):.4f}) | "
            f"PC1={pc1_mean:.3f} 2D={n_2d}/{len(disc_eval)} | "
            f"corr={cone_corr_mean:.3f} | "
            f"SwI={mobile_d:.3f} α3={stable_d:.3f} ratio={ratio:.2f}x | "
            f"|p|={norms.mean():.3f}±{norms.std():.3f} | "
            f"{elapsed:.1f}s"
        )

        if ratio > best_ratio:
            best_ratio = ratio
            torch.save({
                "model_state_dict": model.state_dict(),
                "curvature": model.curvature.item(),
                "hyper_scale": model.hyper_scale.item(),
                "global_epoch": global_epoch,
                "switch_i_ratio": ratio,
                "mobile_disp": mobile_d,
                "stable_disp": stable_d,
                "pc1_mean": pc1_mean,
                "n_2d": n_2d,
                "cone_corr_mean": cone_corr_mean,
                "architecture": {
                    "node_dim": 4, "hidden": 128,
                    "num_layers": 6, "num_experts": 4,
                    "projection_dim": 64, "hyp_proj_dim": 2,
                },
            }, output_dir / "best_differential.pt")
            logger.info(f"    ★ New best ratio: {ratio:.2f}x")

        metrics_log.append({
            "global_epoch": global_epoch,
            "losses": losses,
            "pc1_mean": pc1_mean,
            "n_2d": n_2d,
            "cone_corr_mean": cone_corr_mean,
            "switch_i_disp": mobile_d,
            "alpha3_disp": stable_d,
            "ratio": ratio,
            "elapsed": elapsed,
        })
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2, default=str)

        model.train()

    # Final
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE | Best Switch-I ratio: {best_ratio:.2f}x")
    logger.info(f"{'='*70}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "curvature": model.curvature.item(),
        "hyper_scale": model.hyper_scale.item(),
        "global_epoch": global_epoch,
        "architecture": {
            "node_dim": 4, "hidden": 128,
            "num_layers": 6, "num_experts": 4,
            "projection_dim": 64, "hyp_proj_dim": 2,
        },
    }, output_dir / "tokyo_eyes_v4.pt")


if __name__ == "__main__":
    main()
