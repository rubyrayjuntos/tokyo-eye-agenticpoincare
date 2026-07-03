"""
launch_training.py — V6 GNN training launcher with MLflow tracking.

Usage:
    python -m experiments.training.v6.launch_training \\
        --corpus manifests/v6_corpus_120.json \\
        --output-dir checkpoints/v6/runs/run001 \\
        --device cpu \\
        --phase 1

    make train-v6 STAGE=1 CORPUS=manifests/v6_corpus_120.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from science.training.config import TrainingConfig
from science.training.corpus_governance import STAGE_A_MAX_RESIDUES
from science.training.tracking import TrainingTracker
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.stage_runner import StageRunner
from experiments.training.v6.train_loop import (
    init_v6_radial_scale,
    resolve_default_warm_start,
    warm_start_v5_backbone,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launch_v6")


def resolve_prior_checkpoint(output_dir: Path, phase: int, protein_count: int) -> Path | None:
    """Pick v6_best.pt or prior phase end checkpoint for curriculum continuity."""
    best = output_dir / "v6_best.pt"
    if best.is_file():
        return best
    if phase >= 2:
        prev = output_dir / f"v6_phase{phase - 1}_{protein_count}prot.pt"
        if prev.is_file():
            return prev
    return None


def build_model(config: TrainingConfig) -> torch.nn.Module:
    from science.dtie.v6.gnn.model import GOSPConeMapperV6

    model = GOSPConeMapperV6(
        node_dim=4,
        hidden=config.hidden,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        capacity_threshold=config.capacity_threshold,
        expert_dropout_p=0.0,
        min_usage=config.min_usage,
        topology_only_gate=config.topology_only_gate,
        hyperbolic_gate=config.hyperbolic_gate,
        hyperbolic_expert_mix=config.hyperbolic_expert_mix,
        gate_disc_scale=config.gate_disc_scale,
        gate_gumbel=config.gate_gumbel,
        deep_hyperbolic_gate=config.deep_hyperbolic_gate,
        legacy_disc_projection=config.legacy_disc_projection,
        radial_angular_recombine=config.radial_angular_recombine,
        disc_radial_source=config.disc_radial_source,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 GNN training launcher")
    parser.add_argument("--corpus", type=Path, default=Path("manifests/v6_corpus_120.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/v6/runs/default"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--warm-start-v5", type=Path, default=None)
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip v5/v6 warm-start even when a default checkpoint exists",
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument(
        "--max-residues",
        type=int,
        default=STAGE_A_MAX_RESIDUES,
        help="Skip structures larger than this (GPU guard; Stage A floor is STAGE_A_MAX_RESIDUES)",
    )
    parser.add_argument("--no-corpus-cache", action="store_true")
    parser.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "file:/app/mlruns"))
    parser.add_argument("--mlflow-experiment", default="tokyo-eyes-v6")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-hyperbolic-gate", action="store_true", help="Use legacy tangent MLP gate")
    parser.add_argument("--hyperbolic-expert-mix", action="store_true", help="Stage 3: mix expert outputs on ball")
    parser.add_argument("--epochs", type=int, default=None, help="Override phase epoch count (smoke tests)")
    parser.add_argument("--v2-teacher-checkpoint", type=Path, default=None, help="Frozen v3 teacher .pt")
    parser.add_argument(
        "--v2-teacher-depth-coeff",
        type=float,
        default=None,
        help="V2 teacher depth distill weight (default 0.20; 0.30 for --p1c)",
    )
    parser.add_argument("--v2-teacher-epistemic-coeff", type=float, default=0.10)
    parser.add_argument("--gentle-phase2", action="store_true", help="P2: lower LR + freeze radial 5 epochs")
    parser.add_argument(
        "--phase2-lr",
        type=float,
        default=None,
        help="Override P2 learning rate (default: base_lr*0.2 when --gentle-phase2)",
    )
    parser.add_argument(
        "--save-epoch-snapshots",
        action="store_true",
        help="Save epochs/epoch_NNN.pt each epoch for training filmstrip viz",
    )
    parser.add_argument(
        "--p1b",
        action="store_true",
        help="Run Phase 1b only (angular unfreeze, gentle shell coeffs)",
    )
    parser.add_argument("--p1b-lr", type=float, default=1e-4, help="LR for --p1b (default 1e-4)")
    parser.add_argument(
        "--p1c",
        action="store_true",
        help="Run Phase 1c only (disc expansion: ramped angular + disc spread)",
    )
    parser.add_argument("--p1c-lr", type=float, default=1e-4, help="LR for --p1c (default 1e-4)")
    parser.add_argument(
        "--p1d",
        action="store_true",
        help="Run Phase 1d only (disc-depth scale loss on hyp_proj_head_2d)",
    )
    parser.add_argument("--p1d-lr", type=float, default=1e-4, help="LR for --p1d (default 1e-4)")
    parser.add_argument(
        "--p1d-extend",
        action="store_true",
        help="P1d extension preset (slower disc target ramp, higher scale coeff)",
    )
    parser.add_argument("--p1d-disc-depth-scale-coeff", type=float, default=None)
    parser.add_argument("--p1d-disc-target-start", type=float, default=None)
    parser.add_argument("--p1d-disc-target-end", type=float, default=None)
    parser.add_argument("--p1d-freeze-radial-epochs", type=int, default=None)
    parser.add_argument("--p1d-min-probe-r-depth-sasa", type=float, default=None)
    parser.add_argument("--p1d-disc-spread-min-std", type=float, default=None)
    parser.add_argument(
        "--p2-bridge",
        action="store_true",
        help="P2 bridge from saturated P1 (relaxed routing save + disc scale + shell guard)",
    )
    parser.add_argument("--p2-bridge-lr", type=float, default=5e-5, help="LR for --p2-bridge")
    parser.add_argument(
        "--p2-bridge-ramp-epochs",
        type=int,
        default=None,
        help="Routing save ceiling ramp length for --p2-bridge (default 10)",
    )
    parser.add_argument(
        "--p2-hypmix",
        action="store_true",
        help="P2 hypmix preset: stronger MoE pressure + shell guard 0.65",
    )
    parser.add_argument(
        "--p2-hypmix3",
        action="store_true",
        help="P2 hypmix3: amplify MoE from hypmix2 champion (dropout 0.15, capacity 0.008)",
    )
    parser.add_argument(
        "--p2-hypmix-final",
        action="store_true",
        help="P2 hypmix final lock-in (dropout 0.18, capacity 0.01)",
    )
    parser.add_argument(
        "--p2-disc-occupancy",
        action="store_true",
        help="P2 disc occupancy recovery: penalize rank-1 hyp_projections_2d (warm-start)",
    )
    parser.add_argument(
        "--p2-disc-occupancy-v2",
        action="store_true",
        help="Refined disc occupancy: stronger σ₂/σ₁ + PC2 repulsion + shell floor",
    )
    parser.add_argument(
        "--p2-disc-occupancy-v3",
        action="store_true",
        help="Cluster-break push: coeff 3.5, PC repulsion 2.5, LR 1e-5, disc_r_std≥0.045",
    )
    parser.add_argument(
        "--p2-disc-occupancy-v4",
        action="store_true",
        help="Target-corpus disc push: 11QE/4OBE/1IVO + eff_rank loss (warm-start v2b)",
    )
    parser.add_argument(
        "--p2-disc-occupancy-v5",
        action="store_true",
        help="Batch diversity repulsion on target corpus (PC2-residual pairwise)",
    )
    parser.add_argument(
        "--p2-disc-gentle-arch",
        action="store_true",
        help="Gentle new-path retrain from early recovery (visual gates + low occupancy)",
    )
    parser.add_argument(
        "--p2-disc-proj-recovery",
        action="store_true",
        help="Projection head recovery (thickness/span floors + legacy teacher)",
    )
    parser.add_argument(
        "--p2-disc-proj-recovery-v2",
        action="store_true",
        help="v2 recovery: no path_align, stronger thickness floor, mlp_fusion default",
    )
    parser.add_argument(
        "--p2-disc-proj-recovery-v3",
        action="store_true",
        help="v3 recovery: unfreeze angular + fusion + disc head (fix x_hyp wedge)",
    )
    parser.add_argument(
        "--p2-disc-proj-recovery-v4",
        action="store_true",
        help="v4 recovery: unfreeze radial+angular+fusion + x_hyp spread loss",
    )
    parser.add_argument(
        "--p2-disc-proj-recovery-v5",
        action="store_true",
        help="v5 recovery: angular_lift (direction-only expmap) + v4 lift-path training",
    )
    parser.add_argument(
        "--p2-rec-ablation",
        action="store_true",
        help="Radial×angular fusion ablation (train fusion MLP + disc head; bridge losses only)",
    )
    parser.add_argument(
        "--radial-angular-recombine",
        choices=("multiply", "mlp_fusion", "angular_lift"),
        default="multiply",
        help="Tangent lift recombination (mlp_fusion / angular_lift ablations)",
    )
    parser.add_argument(
        "--disc-radial-source",
        choices=("mobius", "radial_depth", "dist0_x_hyp"),
        default="mobius",
        help="Pre-routing 2D radial authority (Lever A: radial_depth | dist0_x_hyp)",
    )
    parser.add_argument(
        "--p2-disc-path-align",
        action="store_true",
        help="Legacy-teacher path alignment (no occupancy; train disc proj + gate readout)",
    )
    parser.add_argument("--p2-disc-occupancy-coeff", type=float, default=None)
    parser.add_argument("--p2-disc-path-align-coeff", type=float, default=None)
    parser.add_argument("--p2-disc-eff-rank-coeff", type=float, default=None)
    parser.add_argument("--p2-disc-batch-diversity-coeff", type=float, default=None)
    parser.add_argument(
        "--p4-epistemic-decoupling",
        action="store_true",
        help="Phase 4: B-factor residual epistemic-depth decoupling (warm-start lever_a)",
    )
    parser.add_argument("--p4-epistemic-lr", type=float, default=1e-4)
    parser.add_argument(
        "--epistemic-decoupling-holdouts",
        default="1IVO,4MNE",
        help="Comma-separated PDB IDs excluded from B-factor decoupling loss",
    )
    parser.add_argument(
        "--epistemic-bf-align-coeff",
        type=float,
        default=None,
        help="λ₁ final (B-factor alignment); default 0.22",
    )
    parser.add_argument(
        "--epistemic-sasa-pen-coeff",
        type=float,
        default=None,
        help="λ₂ final (SASA partial penalty); default 0.246 (0.10 when --p4-epistemic-staged)",
    )
    parser.add_argument(
        "--shell-corr-epi-sasa-weight",
        type=float,
        default=None,
        help="shell_corr epi_sasa_weight override (v3 ablation: 0.0)",
    )
    parser.add_argument(
        "--p4-epistemic-staged",
        action="store_true",
        help="Option B: λ₁-only epochs 1-10, capped/log λ₂ ramp thereafter",
    )
    parser.add_argument(
        "--legacy-disc-projection",
        action="store_true",
        help="Use post-routing hard-clamp disc path (old checkpoints / inference compat)",
    )
    parser.add_argument(
        "--no-legacy-disc-projection",
        action="store_true",
        help="Use pre-routing soft disc path (default for new training)",
    )
    parser.add_argument("--p2-radial-freeze-epochs", type=int, default=None)
    parser.add_argument("--p2-disc-r-std-floor", type=float, default=None)
    parser.add_argument("--p2-disc-line-thickness-floor", type=float, default=None)
    parser.add_argument(
        "--disc-scatter-interval",
        type=int,
        default=0,
        help="Export 11QE disc scatter every N epochs (0=off)",
    )
    parser.add_argument(
        "--full-hyp-moe-test",
        action="store_true",
        help="Full hyperbolic MoE theory test from hypmix_final lock-in (deep gate, disc×2.5, Gumbel)",
    )
    parser.add_argument(
        "--theory-test-lr",
        type=float,
        default=3e-5,
        help="LR for --full-hyp-moe-test (default 3e-5)",
    )
    parser.add_argument(
        "--deep-hyperbolic-gate",
        action="store_true",
        help="Third Mobius layer in hyperbolic prototype gate",
    )
    parser.add_argument(
        "--gate-disc-scale",
        type=float,
        default=None,
        help="Scale disc_x/y/r features in hyperbolic gate (default 2.0 with --p2-hypmix)",
    )
    parser.add_argument(
        "--gate-gumbel",
        action="store_true",
        help="Gumbel-Softmax hard routing in hyperbolic gate",
    )
    args = parser.parse_args()

    gate_disc_scale = args.gate_disc_scale
    if gate_disc_scale is None:
        if args.full_hyp_moe_test or args.p2_hypmix_final or args.p2_hypmix3:
            gate_disc_scale = 2.5
        elif args.p2_hypmix:
            gate_disc_scale = 2.0
        else:
            gate_disc_scale = 1.0
    gate_gumbel = (
        args.gate_gumbel
        or args.p2_hypmix
        or args.p2_hypmix3
        or args.p2_hypmix_final
        or args.full_hyp_moe_test
    )
    hyperbolic_expert_mix = args.hyperbolic_expert_mix or args.full_hyp_moe_test
    deep_hyperbolic_gate = args.deep_hyperbolic_gate or args.full_hyp_moe_test
    capacity_threshold = 0.35 if args.full_hyp_moe_test else 0.4

    legacy_disc_projection = False
    if args.legacy_disc_projection:
        legacy_disc_projection = True
    if args.no_legacy_disc_projection:
        legacy_disc_projection = False

    if args.p2_bridge and (args.p1d or args.p1c or args.p1b):
        logger.warning("Multiple phase presets set; using P2 bridge")
    elif args.p1d and (args.p1c or args.p1b):
        logger.warning("Multiple phase presets set; using P1d (disc-depth scale)")
    elif args.p1c and args.p1b:
        logger.warning("Both --p1c and --p1b set; using P1c (disc expansion preset)")

    v2_depth_coeff = args.v2_teacher_depth_coeff
    if v2_depth_coeff is None:
        v2_depth_coeff = 0.30 if (args.p1c or args.p1d or args.p2_bridge) else 0.20

    v2_ckpt = args.v2_teacher_checkpoint
    if v2_ckpt is None:
        from experiments.training.v6.v2_teacher import resolve_default_v2_teacher_checkpoint

        v2_ckpt = resolve_default_v2_teacher_checkpoint()

    config = TrainingConfig(
        device=args.device,
        lr=args.lr,
        output_dir=args.output_dir,
        pdb_dir=args.pdb_dir,
        corpus_manifest=args.corpus,
        phase=args.phase,
        resume=args.resume,
        warm_start_v5=args.warm_start_v5,
        max_proteins=args.max_proteins,
        max_residues=args.max_residues,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        hyperbolic_gate=not args.no_hyperbolic_gate,
        hyperbolic_expert_mix=hyperbolic_expert_mix,
        capacity_threshold=capacity_threshold,
        deep_hyperbolic_gate=deep_hyperbolic_gate,
        theory_test_lr=args.theory_test_lr,
        epochs_override=args.epochs,
        v2_teacher_checkpoint=v2_ckpt,
        v2_teacher_depth_coeff=v2_depth_coeff,
        v2_teacher_epistemic_coeff=args.v2_teacher_epistemic_coeff,
        gentle_phase2=args.gentle_phase2,
        phase2_lr=args.phase2_lr,
        save_epoch_snapshots=args.save_epoch_snapshots,
        p1b=args.p1b and not args.p1c and not args.p1d and not args.p2_bridge,
        p1b_lr=args.p1b_lr,
        p1c=args.p1c and not args.p1d and not args.p2_bridge,
        p1c_lr=args.p1c_lr,
        p1d=args.p1d and not args.p2_bridge,
        p1d_lr=args.p1d_lr,
        p1d_extend=args.p1d_extend,
        p1d_disc_depth_scale_coeff=args.p1d_disc_depth_scale_coeff,
        p1d_disc_target_start=args.p1d_disc_target_start,
        p1d_disc_target_end=args.p1d_disc_target_end,
        p1d_freeze_radial_epochs=args.p1d_freeze_radial_epochs,
        p1d_min_probe_r_depth_sasa=args.p1d_min_probe_r_depth_sasa,
        p1d_disc_spread_min_std=args.p1d_disc_spread_min_std,
        p2_bridge=(
            args.p2_bridge
            or args.p2_hypmix
            or args.p2_hypmix3
            or args.p2_hypmix_final
            or args.p2_disc_occupancy
            or args.p2_disc_occupancy_v2
            or args.p2_disc_occupancy_v3
            or args.p2_disc_occupancy_v4
            or args.p2_disc_occupancy_v5
            or args.p2_disc_gentle_arch
            or args.p2_disc_path_align
            or args.p2_disc_proj_recovery
            or args.p2_disc_proj_recovery_v2
            or args.p2_disc_proj_recovery_v3
            or args.p2_disc_proj_recovery_v4
            or args.p2_disc_proj_recovery_v5
            or args.p2_rec_ablation
            or args.full_hyp_moe_test
        ),
        p2_bridge_lr=args.p2_bridge_lr,
        p2_bridge_ramp_epochs=args.p2_bridge_ramp_epochs,
        p2_hypmix=args.p2_hypmix and not args.p2_hypmix3 and not args.p2_hypmix_final and not args.p2_disc_occupancy and not args.p2_disc_occupancy_v2 and not args.p2_disc_occupancy_v3 and not args.p2_disc_occupancy_v4 and not args.p2_disc_occupancy_v5 and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_hypmix3=args.p2_hypmix3 and not args.p2_hypmix_final and not args.p2_disc_occupancy and not args.p2_disc_occupancy_v2 and not args.p2_disc_occupancy_v3 and not args.p2_disc_occupancy_v4 and not args.p2_disc_occupancy_v5 and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_hypmix_final=args.p2_hypmix_final and not args.p2_disc_occupancy and not args.p2_disc_occupancy_v2 and not args.p2_disc_occupancy_v3 and not args.p2_disc_occupancy_v4 and not args.p2_disc_occupancy_v5 and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_occupancy=(args.p2_disc_occupancy or args.p2_disc_occupancy_v2 or args.p2_disc_occupancy_v3 or args.p2_disc_occupancy_v4 or args.p2_disc_occupancy_v5) and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_occupancy_v2=args.p2_disc_occupancy_v2 and not args.p2_disc_occupancy_v3 and not args.p2_disc_occupancy_v4 and not args.p2_disc_occupancy_v5 and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_occupancy_v3=args.p2_disc_occupancy_v3 and not args.p2_disc_occupancy_v4 and not args.p2_disc_occupancy_v5 and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_occupancy_v4=args.p2_disc_occupancy_v4 and not args.p2_disc_occupancy_v5 and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_occupancy_v5=args.p2_disc_occupancy_v5 and not args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_gentle_arch=args.p2_disc_gentle_arch and not args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_path_align=args.p2_disc_path_align and not args.p2_disc_proj_recovery and not args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_proj_recovery=(
            args.p2_disc_proj_recovery
            or args.p2_disc_proj_recovery_v2
            or args.p2_disc_proj_recovery_v3
            or args.p2_disc_proj_recovery_v4
            or args.p2_disc_proj_recovery_v5
        ) and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v2=args.p2_disc_proj_recovery_v2 and not args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v3=args.p2_disc_proj_recovery_v3 and not args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v4=args.p2_disc_proj_recovery_v4 and not args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v5=args.p2_disc_proj_recovery_v5 and not args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_rec_ablation=args.p2_rec_ablation and not args.full_hyp_moe_test,
        p2_disc_occupancy_coeff=args.p2_disc_occupancy_coeff,
        p2_disc_path_align_coeff=args.p2_disc_path_align_coeff,
        p2_disc_eff_rank_coeff=args.p2_disc_eff_rank_coeff,
        p2_disc_batch_diversity_coeff=args.p2_disc_batch_diversity_coeff,
        p2_radial_freeze_epochs=args.p2_radial_freeze_epochs,
        p2_disc_r_std_floor=args.p2_disc_r_std_floor,
        p2_disc_line_thickness_floor=args.p2_disc_line_thickness_floor,
        disc_scatter_interval_epochs=args.disc_scatter_interval,
        full_hyp_moe_test=args.full_hyp_moe_test,
        gate_disc_scale=gate_disc_scale,
        gate_gumbel=gate_gumbel,
        legacy_disc_projection=legacy_disc_projection,
        radial_angular_recombine=(
            "angular_lift"
            if args.p2_disc_proj_recovery_v5
            else "mlp_fusion"
            if (
                args.p2_rec_ablation
                or args.p2_disc_proj_recovery_v2
                or args.p2_disc_proj_recovery_v3
                or args.p2_disc_proj_recovery_v4
            )
            else args.radial_angular_recombine
        ),
        disc_radial_source=args.disc_radial_source,
        p4_epistemic_decoupling=args.p4_epistemic_decoupling,
        p4_epistemic_lr=args.p4_epistemic_lr,
        epistemic_decoupling_holdouts=args.epistemic_decoupling_holdouts,
        epistemic_bf_align_coeff=args.epistemic_bf_align_coeff,
        epistemic_sasa_pen_coeff=args.epistemic_sasa_pen_coeff,
        shell_corr_epi_sasa_weight=args.shell_corr_epi_sasa_weight,
        p4_epistemic_staged=args.p4_epistemic_staged,
    )
    if (
        config.p2_disc_proj_recovery
        or config.p2_disc_proj_recovery_v2
        or config.p2_disc_proj_recovery_v3
        or config.p2_disc_proj_recovery_v4
        or config.p2_disc_proj_recovery_v5
    ):
        config.legacy_disc_projection = False
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.pdb_dir.is_dir():
        config.pdb_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus from %s", config.corpus_manifest)
    proteins, failed = load_training_proteins(
        config.pdb_dir,
        config.corpus_manifest,
        max_proteins=config.max_proteins,
        max_residues=config.max_residues,
        use_cache=not args.no_corpus_cache,
    )
    if not proteins:
        logger.error("No proteins loaded (%d failed). Check network / manifest.", failed)
        sys.exit(1)
    logger.info("Loaded %d proteins (%d failed)", len(proteins), failed)

    model = build_model(config)
    device = config.device
    if device != "cpu" and torch.cuda.is_available():
        model.to(device)
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        config.device = "cpu"
        model.to(device)

    init_v6_radial_scale(model)

    if config.phase and config.phase >= 2 and (config.resume is None or not Path(config.resume).is_file()):
        auto_resume = resolve_prior_checkpoint(config.output_dir, config.phase, len(proteins))
        if auto_resume is not None:
            config.resume = auto_resume
            logger.info("Auto-resuming phase %d from %s", config.phase, auto_resume)

    resume_state = None
    if config.resume and config.resume.is_file():
        from science.training.checkpoint import CheckpointManager

        resume_state = CheckpointManager(config.output_dir, len(proteins)).load(config.resume, device)
        from science.dtie.v6.gnn.model import load_v6_state_dict

        missing, unexpected = load_v6_state_dict(model, resume_state.model_state_dict)
        if missing:
            logger.info("Resume: %d missing keys (new modules init from scratch)", len(missing))
        if unexpected:
            logger.warning("Resume: %d unexpected keys ignored", len(unexpected))
        logger.info(
            "Resumed from %s (epoch %d, prior best score=%.4f)",
            config.resume,
            resume_state.global_epoch,
            resume_state.score,
        )
    else:
        if args.no_warm_start:
            logger.info("Warm-start disabled — training from initialized weights only")
        else:
            warm_path = config.warm_start_v5
            if warm_path is None:
                warm_path = resolve_default_warm_start()
            if warm_path is not None and Path(warm_path).is_file():
                n = warm_start_v5_backbone(model, str(warm_path), device)
                logger.info("Warm-started %d tensors from checkpoint %s", n, warm_path)
            else:
                logger.warning(
                    "No warm-start checkpoint found — training from initialized weights only. "
                    "Pass --warm-start-v5 /app/checkpoints/... for faster convergence."
                )

    tracker: TrainingTracker | None = None
    if not args.no_mlflow:
        tracker = TrainingTracker(config, run_name=config.output_dir.name)

    v2_teacher = None
    if config.v2_teacher_checkpoint is not None and Path(config.v2_teacher_checkpoint).is_file():
        from experiments.training.v6.v2_teacher import V2Teacher

        logger.info("Loading V2 teacher from %s", config.v2_teacher_checkpoint)
        v2_teacher = V2Teacher(
            config.v2_teacher_checkpoint,
            device=device,
        )
        v2_teacher.precompute(proteins)
    elif config.v2_teacher_checkpoint is not None:
        logger.warning("V2 teacher checkpoint not found: %s — training without distill", config.v2_teacher_checkpoint)

    runner = StageRunner(
        model,
        proteins,
        config,
        tracker=tracker,
        resume_state=resume_state,
        v2_teacher=v2_teacher,
    )

    if tracker:
        with tracker.start_run(proteins=proteins, phases=runner._phases()):
            result = runner.run()
            from science.training.mlflow_governance import finalize_governance_run

            finalize_governance_run(
                tracker,
                model,
                config,
                proteins,
                device=device,
            )
            tracker.log_artifact(config.output_dir / "metrics.json")
            focus = result.get("focus_summary")
            if focus:
                tracker.log_focus_summary(
                    focus,
                    config.output_dir / "focus_summary.json",
                )
    else:
        result = runner.run()

    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
