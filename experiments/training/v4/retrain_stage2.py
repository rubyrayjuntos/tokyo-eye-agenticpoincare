"""
retrain_stage2.py — Tokyo Eyes v4 Stage 2 Retraining
=====================================================
Eidetix Bio | 2026-05-19

Warm-starts from checkpoints_v4_cone_fix2/checkpoint_stage_1.pt (epoch 35)
and runs a corrected Stage 2 with the following fixes based on training analysis:

DIAGNOSIS (from cone_fix2 run):
  - Radial loss drifted 0.0024 over 38 epochs — gradients can't reach weights
    through saturated tanh in expmap0 when points are at |p|=1.156 (99.6% boundary)
  - Correlation ρ↔depth was NEGATIVE (-0.86→-0.79) — biology inverted
  - Cone loss was fighting radial supervision (cone wins at 0.40 vs implicit radial)
  - Epoch 36: domain_sep ramp caused instant 11/11 2D — proves angular structure
    is one push away, but collapsed back when domain_sep wasn't sustained

FIXES:
  1. Remove radial_burial loss entirely (wrong sign, saturated gradients)
  2. Reduce evidential_coeff: 0.01 → 0.005 (overfitting)
  3. Reduce cone_coeff: 0.10 → 0.10 (keep — it's the right value now)
  4. Increase domain_sep_coeff: 0.15 → 0.40 (epoch 36 proved this works)
  5. hyper_scale initialized fresh at 0.5 (not loaded from checkpoint)
     — pulls points off boundary over 5-10 epochs, gives domain_sep room
  6. Keep domain_sep_coeff HIGH through ALL of Stage 2 (not just first epoch)
  7. Freeze backbone for first 5 epochs to let hyper_scale adapt first

Usage:
    python retrain_stage2.py --pdb_dir /tmp/dtie_pdb_cache \
        --checkpoint ../checkpoints_v4_cone_fix2/checkpoint_stage_1.pt \
        --output_dir ../checkpoints_v4_retrain
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
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Gnnv4 import (
    GOSPConeMapper,
    gosp_loss,
    build_optimizer,
    precompute_clustering,
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
logger = logging.getLogger("retrain_stage2")


def train_epoch_corrected(
    model: GOSPConeMapper,
    optimizer,
    proteins: List[Dict],
    evidential_coeff: float,
    cone_coeff: float,
    neighborhood_coeff: float,
    angular_coeff: float,
    domain_sep_coeff: float,
    balance_coeff: float,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train one epoch with corrected loss coefficients."""
    model.train()
    epoch_losses = {
        "total": [], "evidential": [], "balance": [],
        "cone_consistency": [], "neighborhood_consistency": [],
        "angular_diversity": [], "domain_separation": [],
    }

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
            evidential_coeff=evidential_coeff,
            balance_coeff=balance_coeff,
            cone_coeff=cone_coeff,
            neighborhood_coeff=neighborhood_coeff,
            angular_coeff=angular_coeff,
            domain_sep_coeff=domain_sep_coeff,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in epoch_losses:
            if k in losses:
                val = losses[k]
                epoch_losses[k].append(val.item() if torch.is_tensor(val) else float(val))

    return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Retrain Tokyo Eyes v4 Stage 2")
    parser.add_argument("--pdb_dir", type=str, default="/tmp/dtie_pdb_cache")
    parser.add_argument("--checkpoint", type=str,
                       default="../checkpoints_v4_cone_fix2/checkpoint_stage_1.pt",
                       help="Stage 1 checkpoint to warm-start from")
    parser.add_argument("--output_dir", type=str, default="../checkpoints_v4_retrain")
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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Device: {device}")
    logger.info(f"Warm-start checkpoint: {checkpoint_path}")
    logger.info(f"Output dir: {output_dir}")

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
    logger.info(f"Loaded {len(all_proteins)} proteins "
               f"({sum(p['n_residues'] for p in all_proteins.values())} total residues)")

    # ── Initialize model and load checkpoint ──────────────────────────────
    model = GOSPConeMapper(
        node_dim=4,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        projection_dim=64,
        hyp_proj_dim=2,
        depth_conditioning=False,
    )

    # Load Stage 1 checkpoint with strict=False so hyper_scale initializes fresh
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(f"Loaded checkpoint (epoch {ckpt.get('global_epoch', '?')})")
    if missing:
        logger.info(f"  New parameters (fresh init): {missing}")
    if unexpected:
        logger.info(f"  Unexpected keys (ignored): {unexpected}")

    # CRITICAL: Reset hyper_scale to 0.5 regardless of checkpoint
    # This is the architectural fix — it pulls points off the boundary
    # over 5-10 epochs, giving domain_sep room to create persistent 2D structure
    with torch.no_grad():
        model.hyper_scale.fill_(0.5)
    logger.info(f"  hyper_scale reset to: {model.hyper_scale.item():.4f}")
    logger.info(f"  curvature: {model.curvature.item():.4f}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")

    # ── Pre-training diagnostic ───────────────────────────────────────────
    logger.info("\n── Pre-training state (from Stage 1 checkpoint) ──")
    pre_eval = evaluate_disc_structure(model, all_protein_list, device)
    n_2d_pre = sum(1 for v in pre_eval.values() if v["has_2d_structure"])
    pc1_pre = np.mean([v["pc1_variance"] for v in pre_eval.values()])
    corr_pre = np.mean([v["cone_rho_correlation"] for v in pre_eval.values()])
    logger.info(f"  PC1={pc1_pre:.3f} | 2D={n_2d_pre}/{len(pre_eval)} | corr={corr_pre:.3f}")

    # Check radial distribution
    model.eval()
    with torch.no_grad():
        sample_data = all_protein_list[0]["data"].to(device)
        sample_out = model(sample_data)
        norms = sample_out["x_hyp"].norm(dim=-1)
        logger.info(f"  x_hyp norms: mean={norms.mean():.4f} std={norms.std():.4f} "
                   f"max={norms.max():.4f}")

    # ── Training stages ───────────────────────────────────────────────────
    # Stage 2 corrected: sustained high domain_sep, no radial loss,
    # reduced evidential, hyper_scale adapts freely
    stages = [
        {
            "name": "Stage 2a: Scale adaptation + domain push",
            "epochs": 15,
            "lr": args.lr,
            "freeze_backbone": True,  # Let hyper_scale adapt first
            "evidential_coeff": 0.005,
            "cone_coeff": 0.10,
            "neighborhood_coeff": 0.30,
            "angular_coeff": 0.25,
            "domain_sep_coeff": 0.40,
            "balance_coeff": 0.01,
        },
        {
            "name": "Stage 2b: Full model + sustained domain sep",
            "epochs": 25,
            "lr": args.lr,
            "freeze_backbone": False,
            "evidential_coeff": 0.005,
            "cone_coeff": 0.10,
            "neighborhood_coeff": 0.30,
            "angular_coeff": 0.25,
            "domain_sep_coeff": 0.40,
            "balance_coeff": 0.01,
        },
        {
            "name": "Stage 3: Fine-tune",
            "epochs": 20,
            "lr": args.lr * 0.2,
            "freeze_backbone": False,
            "evidential_coeff": 0.005,
            "cone_coeff": 0.10,
            "neighborhood_coeff": 0.25,
            "angular_coeff": 0.20,
            "domain_sep_coeff": 0.35,
            "balance_coeff": 0.005,
        },
    ]

    metrics_log = []
    global_epoch = ckpt.get("global_epoch", 35)  # Continue from where Stage 1 left off
    best_2d_count = 0
    best_epoch = 0

    for stage in stages:
        logger.info(f"\n{'='*70}")
        logger.info(f"{stage['name']} ({stage['epochs']} epochs, lr={stage['lr']})")
        logger.info(f"  evidential={stage['evidential_coeff']} cone={stage['cone_coeff']} "
                   f"nbr={stage['neighborhood_coeff']} ang={stage['angular_coeff']} "
                   f"dom={stage['domain_sep_coeff']} bal={stage['balance_coeff']}")
        logger.info(f"  freeze_backbone={stage['freeze_backbone']}")
        logger.info(f"{'='*70}")

        # Freeze/unfreeze backbone
        for p in model.convs.parameters():
            p.requires_grad = not stage["freeze_backbone"]
        for p in model.norms.parameters():
            p.requires_grad = not stage["freeze_backbone"]

        # VERIFY: hyper_scale must always be unfrozen — it's the architectural lever
        model.hyper_scale.requires_grad = True
        if stage["freeze_backbone"]:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            logger.info(f"  Trainable params (backbone frozen): {len(trainable)}")
            assert "hyper_scale" in trainable, "hyper_scale must be trainable!"

        optimizer = build_optimizer(model, lr=stage["lr"])

        for epoch in range(stage["epochs"]):
            global_epoch += 1
            t0 = time.time()

            losses = train_epoch_corrected(
                model=model,
                optimizer=optimizer,
                proteins=all_protein_list,
                evidential_coeff=stage["evidential_coeff"],
                cone_coeff=stage["cone_coeff"],
                neighborhood_coeff=stage["neighborhood_coeff"],
                angular_coeff=stage["angular_coeff"],
                domain_sep_coeff=stage["domain_sep_coeff"],
                balance_coeff=stage["balance_coeff"],
                device=device,
            )

            elapsed = time.time() - t0
            c_val = model.curvature.item()
            scale_val = model.hyper_scale.item()

            # Eval
            disc_eval = evaluate_disc_structure(model, all_protein_list, device)
            n_2d = sum(1 for v in disc_eval.values() if v["has_2d_structure"])
            pc1_mean = np.mean([v["pc1_variance"] for v in disc_eval.values()])
            cone_std_mean = np.mean([v["cone_depth_std"] for v in disc_eval.values()])
            cone_corr_mean = np.mean([v["cone_rho_correlation"] for v in disc_eval.values()])

            # Track radial distribution — min/max matter as much as mean
            # If min stays above 1.10, points aren't actually moving off boundary
            model.eval()
            with torch.no_grad():
                sample_out = model(all_protein_list[0]["data"].to(device))
                norms = sample_out["x_routed_hyp"].norm(dim=-1)
                norm_mean = norms.mean().item()
                norm_std = norms.std().item()
                norm_min = norms.min().item()
                norm_max = norms.max().item()
            model.train()

            # Expert usage
            first_eval = next(iter(disc_eval.values()))
            expert_str = " ".join(f"{w:.3f}" for w in first_eval["expert_usage"])

            logger.info(
                f"  Epoch {global_epoch:3d} | "
                f"loss={losses['total']:.4f} "
                f"(ev={losses['evidential']:.4f} "
                f"cone={losses['cone_consistency']:.4f} "
                f"ang={losses.get('angular_diversity', 0):.4f} "
                f"nbr={losses['neighborhood_consistency']:.4f} "
                f"dom={losses.get('domain_separation', 0):.4f} "
                f"bal={losses['balance']:.4f}) | "
                f"c={c_val:.4f} scale={scale_val:.4f} | "
                f"|p|={norm_mean:.3f}±{norm_std:.3f} "
                f"[{norm_min:.3f}, {norm_max:.3f}] | "
                f"PC1={pc1_mean:.3f} 2D={n_2d}/{len(disc_eval)} | "
                f"cone_std={cone_std_mean:.3f} corr={cone_corr_mean:.3f} | "
                f"[{expert_str}] | {elapsed:.1f}s"
            )

            # Track best
            if n_2d > best_2d_count:
                best_2d_count = n_2d
                best_epoch = global_epoch
                # Save best checkpoint
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "curvature": c_val,
                    "hyper_scale": scale_val,
                    "stage": stage["name"],
                    "global_epoch": global_epoch,
                    "n_2d": n_2d,
                    "pc1_mean": pc1_mean,
                    "cone_corr_mean": cone_corr_mean,
                    "architecture": {
                        "node_dim": 4, "hidden": args.hidden,
                        "num_layers": args.num_layers, "num_experts": args.num_experts,
                        "projection_dim": 64, "hyp_proj_dim": 2,
                    },
                }, output_dir / "best_checkpoint.pt")
                logger.info(f"    ★ New best: {n_2d}/{len(disc_eval)} 2D (epoch {global_epoch})")

            # Log metrics
            entry = {
                "global_epoch": global_epoch,
                "stage": stage["name"],
                "losses": losses,
                "curvature": c_val,
                "hyper_scale": scale_val,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
                "norm_min": norm_min,
                "norm_max": norm_max,
                "pc1_variance_mean": pc1_mean,
                "n_2d_structure": n_2d,
                "n_proteins": len(disc_eval),
                "cone_depth_std_mean": cone_std_mean,
                "cone_rho_correlation_mean": cone_corr_mean,
                "expert_usage": first_eval["expert_usage"],
                "elapsed_sec": elapsed,
            }
            metrics_log.append(entry)

            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2, default=str)

            # Early exit from 2a: if 2D is stable and radial spread achieved,
            # don't wait — move to 2b before projection heads overfit angular task
            if stage["freeze_backbone"] and epoch >= 9:
                if n_2d >= len(disc_eval) and norm_std > 0.04:
                    logger.info(
                        f"    ★ Early exit from 2a: 2D={n_2d}/{len(disc_eval)} stable, "
                        f"|p| std={norm_std:.4f} > 0.04. Moving to 2b."
                    )
                    break

            # Stall detection: if after 5 epochs |p| std < 0.005, hyper_scale isn't moving
            if stage["freeze_backbone"] and epoch == 4:
                if norm_std < 0.005:
                    logger.warning(
                        f"    ⚠ STALL: |p| std={norm_std:.6f} after 5 epochs. "
                        f"hyper_scale={scale_val:.4f}. Check if it's being optimized."
                    )

            # Early warning: if 2D collapses back after achieving it, log it
            if n_2d < best_2d_count and global_epoch > best_epoch + 3:
                logger.warning(
                    f"    ⚠ 2D collapsed: was {best_2d_count} at epoch {best_epoch}, "
                    f"now {n_2d}. domain_sep may need increase."
                )

        # Save stage checkpoint
        stage_name = stage["name"].split(":")[0].strip().lower().replace(" ", "_")
        ckpt_path = output_dir / f"checkpoint_{stage_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "curvature": c_val,
            "hyper_scale": model.hyper_scale.item(),
            "stage": stage["name"],
            "global_epoch": global_epoch,
            "architecture": {
                "node_dim": 4, "hidden": args.hidden,
                "num_layers": args.num_layers, "num_experts": args.num_experts,
                "projection_dim": 64, "hyp_proj_dim": 2,
            },
        }, ckpt_path)
        logger.info(f"  Saved: {ckpt_path}")

    # ── Final evaluation ──────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("FINAL EVALUATION")
    logger.info(f"{'='*70}")

    final_eval = evaluate_disc_structure(model, all_protein_list, device)
    for pdb_id, metrics in final_eval.items():
        gene = TRAINING_TARGETS.get(pdb_id, {}).get("gene", "?")
        status = "✓ 2D" if metrics["has_2d_structure"] else "✗ 1D"
        logger.info(
            f"  {pdb_id} ({gene:8s}): PC1={metrics['pc1_variance']:.3f} "
            f"cone_std={metrics['cone_depth_std']:.3f} "
            f"corr={metrics['cone_rho_correlation']:.3f} "
            f"{status}"
        )

    n_2d_final = sum(1 for v in final_eval.values() if v["has_2d_structure"])
    logger.info(f"\n  Final: {n_2d_final}/{len(final_eval)} 2D | "
               f"Best: {best_2d_count}/{len(final_eval)} at epoch {best_epoch}")
    logger.info(f"  hyper_scale final: {model.hyper_scale.item():.4f}")
    logger.info(f"  curvature final: {model.curvature.item():.4f}")

    # Save final
    final_path = output_dir / "tokyo_eyes_v4.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "curvature": model.curvature.item(),
        "hyper_scale": model.hyper_scale.item(),
        "stage": "final_retrain",
        "global_epoch": global_epoch,
        "best_2d_count": best_2d_count,
        "best_epoch": best_epoch,
        "architecture": {
            "node_dim": 4, "hidden": args.hidden,
            "num_layers": args.num_layers, "num_experts": args.num_experts,
            "projection_dim": 64, "hyp_proj_dim": 2,
        },
        "training_targets": list(TRAINING_TARGETS.keys()),
        "training_notes": (
            "Retrained from cone_fix2/checkpoint_stage_1.pt. "
            "Removed radial_burial loss (wrong sign, saturated gradients). "
            "Reduced evidential_coeff to 0.005. "
            "Sustained domain_sep_coeff=0.40 through all of Stage 2. "
            "hyper_scale initialized fresh at 0.5 to pull points off boundary."
        ),
    }, final_path)
    logger.info(f"\n  Final checkpoint: {final_path}")
    logger.info(f"  Total training time: {sum(e['elapsed_sec'] for e in metrics_log):.1f}s")


if __name__ == "__main__":
    main()
