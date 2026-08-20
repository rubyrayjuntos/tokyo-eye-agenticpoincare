"""
launch_training.py — V6.6 GNN training launcher (standalone lineage).

Does **not** import ``experiments.training.v6`` or ``science.dtie.v6`` / ``v65``.

Usage:
    python -m experiments.training.v66.launch_training \\
        --corpus manifests/v6_corpus_120.json \\
        --output-dir checkpoints/v66/runs/run001 \\
        --device cpu \\
        --phase 1

    make train-v66 STAGE=1 CORPUS=manifests/v6_corpus_120.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from experiments.training.v66.corpus import load_training_proteins
from experiments.training.v66.stage_runner import StageRunner
from experiments.training.v66.train_loop import (
    init_v6_radial_scale,
    resolve_default_warm_start,
    warm_start_v5_backbone,
)
from science.training.config import TrainingConfig
from science.training.corpus_governance import STAGE_A_MAX_RESIDUES
from science.training.tracking import TrainingTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launch_v66")


def resolve_prior_checkpoint(
    output_dir: Path,
    phase: int,
    protein_count: int,
    *,
    checkpoint_prefix: str = "v6",
) -> Path | None:
    from science.training.gnn_lineage import resolve_prior_checkpoint as _resolve

    return _resolve(
        output_dir,
        checkpoint_prefix=checkpoint_prefix,
        phase=phase,
        protein_count=protein_count,
    )


def _viewer_checkpoint_label(output_dir: Path, checkpoint_prefix: str) -> Path:
    """Pick a checkpoint filename for viewer footers (post-train model is in memory).

    Prefer mature phase / best-score artifacts over ``*_best_disc.pt``. The disc
    saver can freeze an early origin-collapsed epoch (high σ₂/σ₁, disc_r_std≈0);
    labeling (or regenerating) from that file makes healthy runs look like a
    single center dot.
    """
    for name in (
        f"{checkpoint_prefix}_best_route.pt",
        "phase_12.pt",
        f"{checkpoint_prefix}_phase12_12prot.pt",
        "phase_2.pt",
        f"{checkpoint_prefix}_phase2_12prot.pt",
        f"{checkpoint_prefix}_best.pt",
        f"{checkpoint_prefix}_best_disc.pt",  # last — may be early origin-collapse
    ):
        path = output_dir / name
        if path.is_file():
            return path
    return output_dir / f"{checkpoint_prefix}_phase_latest.pt"


def build_model(config: TrainingConfig, node_dim: int | None = None) -> torch.nn.Module:
    from science.training.gnn_lineage import build_model as _build

    return _build(config, node_dim=node_dim)


def _node_dim_from_loaded_graphs(proteins: list[dict]) -> int | None:
    if not proteins:
        return None

    node_dim = int(proteins[0]["data"].x.size(1))
    mismatches = [
        f"{str(prot.get('pdb_id', '?')).upper()}:{prot.get('chain', 'A')}={int(prot['data'].x.size(1))}"
        for prot in proteins
        if int(prot["data"].x.size(1)) != node_dim
    ]
    if mismatches:
        raise ValueError(
            f"Loaded graph node feature dim mismatch; expected {node_dim}, got "
            + ", ".join(mismatches)
        )
    return node_dim


def _resolve_training_node_dim(
    proteins: list[dict],
    *,
    use_dehydron_barcode: bool = False,
    use_binned_dehydron: bool = False,
) -> int:
    """Size ``node_emb`` from input contract + optional barcode, not stale cache.

    Cached graphs built under ``legacy_four_vector`` previously overrode a
    live ``topology_three_vector`` env and silently trained ``Linear(4→128)``.
    Barcode arms append dehydron scalars (and optional bins); expected width
    must include those channels or cold scalars exits with a false mismatch.
    """
    from science.dtie.common.residue_features import (
        gnn_feature_set_id_for_barcode,
        gnn_input_dim_for_barcode,
    )

    expected = int(
        gnn_input_dim_for_barcode(
            use_barcode=use_dehydron_barcode,
            use_binned=use_binned_dehydron,
        )
    )
    feature_set = gnn_feature_set_id_for_barcode(
        use_barcode=use_dehydron_barcode,
        use_binned=use_binned_dehydron,
    )
    loaded = _node_dim_from_loaded_graphs(proteins)
    if loaded is None:
        return expected
    if int(loaded) != expected:
        raise ValueError(
            f"Loaded graphs have node_dim={loaded} but input contract expects "
            f"{expected} ({feature_set}). Corpus cache was built under a "
            "different input mode / barcode flag — rebuild cache (key now "
            "includes feature_set) or delete the stale graphs_*.pt under "
            "pdb_dir/corpus_cache."
        )
    return expected


def _set_run_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 GNN training launcher")
    parser.add_argument(
        "--corpus", type=Path, default=Path("manifests/v6_corpus_120.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/v66/runs/default")
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (python/numpy/torch) for reproducible cold inits",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--warm-start-v5", type=Path, default=None)
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip v5/v6 warm-start even when a default checkpoint exists",
    )
    parser.add_argument(
        "--master-cold-lineage",
        action="store_true",
        help="MASTER-feature cold-start lineage root (MLflow: warm_start=none, parent_run_id=null)",
    )
    parser.add_argument(
        "--slim-moe-structural-ssot",
        action="store_true",
        help="Cold-start: frozen structural disc SSOT + slim MoE (inference-aligned layout)",
    )
    parser.add_argument(
        "--v66-feeler-lineage",
        action="store_true",
        help="v6.6 feeler: learned MP→geometry→MoE, 20-ep P1, minimal losses, timeout@45%",
    )
    parser.add_argument(
        "--v66-feeler-angular-lift",
        action="store_true",
        help="v6.6 feeler: angular_lift + lift-path recovery (resume P3; no disc occupancy)",
    )
    parser.add_argument(
        "--v66-feeler-coupling",
        action="store_true",
        help="v6.6 feeler: cross-subgraph coupling edges (resume P3; no disc occupancy)",
    )
    parser.add_argument(
        "--no-dehydron-exclusivity",
        action="store_true",
        help="Allow packing/spoke/ribbon on dehydron pairs (ablation; use with --v66-feeler-no-exclusivity)",
    )
    parser.add_argument(
        "--v66-feeler-no-exclusivity",
        action="store_true",
        help="v6.6 feeler: multi-relation dehydron pairs off P3 (no disc occupancy)",
    )
    parser.add_argument(
        "--v66-feeler-dehydron-angular",
        action="store_true",
        help="v6.6 feeler: scale down dehydron angular SH off P3 (no disc occupancy)",
    )
    parser.add_argument(
        "--v66-feeler-rim-decouple",
        action="store_true",
        help="v6.6 feeler: no dehydron exclusivity + dehydron angular scale (combined Exp1+2)",
    )
    parser.add_argument(
        "--v66-feeler-p3-geom-edges",
        action="store_true",
        help="v6.6 feeler: P3 geometry-fill + local dehydron edge barcode (resume p3_geom)",
    )
    parser.add_argument(
        "--v66-feeler-p3-geom",
        action="store_true",
        help="v6.6 feeler: continue P3 geometry-fill from best_disc (no barcode)",
    )
    parser.add_argument(
        "--v66-feeler-p3-geom-half-stack",
        action="store_true",
        help="With --v66-feeler-p3-geom: halve disc occupancy stack coeffs (0.5×)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-model",
        action="store_true",
        help="v6.6 feeler: model-level rim fan-out + spoke/ribbon boost (resume P4 ep58)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-cold-curriculum",
        action="store_true",
        help="With --v66-feeler-rim-fanout-model: P1+P2 angular fill then P12 (not P12-only cold)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-polish",
        action="store_true",
        help="With --v66-feeler-rim-fanout-model: gentle polish (depth-scale + quarter stack)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-angular",
        action="store_true",
        help="With --v66-feeler-rim-fanout-model: short angular-fill (close blank disc wedges)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-angular-v2",
        action="store_true",
        help="With --v66-feeler-rim-fanout-model: angular-fill v2 (min_r=0.12, stronger rim_*)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-radius",
        action="store_true",
        help="With --v66-feeler-rim-fanout-model: radius push (depth_scale_target=0.55)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-coverage",
        action="store_true",
        help="With --v66-feeler-rim-fanout-model: angular coverage (empty-sector bin floor)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-antibarrier",
        action="store_true",
        help="With --v66-feeler-rim-fanout-model: anti-barrier (expert θ diversity + recruit)",
    )
    parser.add_argument(
        "--v66-feeler-rim-fanout-expert-arc",
        action="store_true",
        help=(
            "With --v66-feeler-rim-fanout-model: mild expert-arc θ diversity "
            "(no sector recruit; soft circular-R + separation)"
        ),
    )
    parser.add_argument(
        "--v66-feeler-geom-angular-prior",
        action="store_true",
        help=(
            "With --v66-feeler-rim-fanout-model: geometric dehydron/peptide disc θ prior "
            "+ α tanh residual (Fix 1)"
        ),
    )
    parser.add_argument(
        "--hyperbolic-mp-graph",
        action="store_true",
        help=(
            "S4: hyperbolic disc k-NN for message passing during training "
            "(no SSOT lift freeze). Exclusive with role edges."
        ),
    )
    parser.add_argument(
        "--input-feature-zscore",
        action="store_true",
        help="T1a: corpus z-score of node features (ρ,τ,ss,sasa) before node_emb",
    )
    parser.add_argument(
        "--replace-tau-abs-dist",
        action="store_true",
        help="T1a optional: replace binary tau_flag with |ρ−TAU| before z-score",
    )
    parser.add_argument(
        "--gate-logit-softplus-init",
        type=float,
        default=None,
        help="Init softplus(gate.logit_scale) to this value (L1≈2.65 = 2× IBU baseline)",
    )
    parser.add_argument(
        "--gate-logit-softplus-floor",
        type=float,
        default=None,
        help="Clamp softplus(gate.logit_scale) to at least this during forward",
    )
    parser.add_argument(
        "--gate-include-sasa",
        action="store_true",
        help=(
            "Explicit normalized SASA on topology gate board "
            "(pre-registered dehydron×exposure enrichment)"
        ),
    )
    parser.add_argument(
        "--routing-entropy-sparsity-coeff",
        type=float,
        default=0.0,
        help=(
            "Peak λ for mean-residue routing entropy sparsity "
            "(λ * mean_i H(p_i); warmup via --routing-entropy-sparsity-warmup-epochs). "
            "0=off. Fix-1 default peak 0.0075 (= 0.5 × feeler balance_coeff)"
        ),
    )
    parser.add_argument(
        "--routing-entropy-sparsity-warmup-epochs",
        type=int,
        default=8,
        help="Linear warmup epochs for routing entropy sparsity λ (default 8)",
    )
    parser.add_argument(
        "--sparsity-style-save",
        action="store_true",
        help=(
            "Use sparsity-style checkpoint save (mean_residue_H band + max_share; "
            "no legacy H(f̄)≤1.21 ceiling). Does not enable sparsity λ."
        ),
    )
    parser.add_argument(
        "--routing-entropy-mean-residue-min-save",
        type=float,
        default=None,
        help="Lower save bound for mean_i H(p_i) when sparsity-style save is on",
    )
    parser.add_argument(
        "--routing-entropy-mean-residue-max-save",
        type=float,
        default=None,
        help="Upper save bound for mean_i H(p_i) when sparsity-style save is on",
    )
    parser.add_argument(
        "--disc-occupancy-coeff",
        type=float,
        default=None,
        dest="disc_occupancy_coeff_override",
        help="Override phase disc_occupancy_coeff (mild bump for B′ disc health)",
    )
    parser.add_argument(
        "--disc-depth-scale-coeff",
        type=float,
        default=None,
        dest="disc_depth_scale_coeff_override",
        help="Override phase disc_depth_scale_coeff",
    )
    parser.add_argument(
        "--disc-depth-scale-target",
        type=float,
        default=None,
        dest="disc_depth_scale_target_override",
        help="Override phase disc_depth_scale_target",
    )
    parser.add_argument(
        "--core-radial-floor-coeff",
        type=float,
        default=None,
        dest="core_radial_floor_coeff_override",
        help="Soft min-r for high-ρ low-τ residues (e1 origin drag; 0=off)",
    )
    parser.add_argument(
        "--core-radial-floor-min-r",
        type=float,
        default=None,
        dest="core_radial_floor_min_r_override",
        help="Near-origin floor for core_radial_floor (default phase 0.15)",
    )
    parser.add_argument(
        "--prototype-repulsion-coeff",
        type=float,
        default=0.0,
        help=(
            "Nearest-pair prototype hyp-distance hinge coeff "
            "(relu(m − min d); pre-reg PROTO_SEP_*). 0=off"
        ),
    )
    parser.add_argument(
        "--prototype-repulsion-margin",
        type=float,
        default=0.25,
        help="Margin m for prototype nearest-pair repulsion hinge (default 0.25 = L2 floor)",
    )
    parser.add_argument(
        "--directionality-asym-coeff",
        type=float,
        default=0.0,
        help=(
            "Path 2 directionality asym reward λ "
            "(maximize 1−asym on encoder_h; diam≤9 mask). 0=off"
        ),
    )
    parser.add_argument(
        "--prototype-gram-logdet-coeff",
        type=float,
        default=0.0,
        help=(
            "Full-bank Gram logdet hinge coeff "
            "(ReLU(τ − logdet)²; pre-reg GRAM_COND_*). 0=off"
        ),
    )
    parser.add_argument(
        "--prototype-gram-logdet-tau",
        type=float,
        default=-1.15,
        help="Saturating logdet target τ for Gram hinge (default -1.15 = bank floor)",
    )
    parser.add_argument(
        "--majority-committed-share-coeff",
        type=float,
        default=0.0,
        help=(
            "Majority-conditional committed-share hinge coeff "
            "(STE hard share; pre-reg MAJORITY_SPLIT_*). 0=off"
        ),
    )
    parser.add_argument(
        "--majority-committed-share-tau",
        type=float,
        default=0.56,
        help="Hinge threshold on committed hard majority share (default 0.56)",
    )
    parser.add_argument(
        "--core-majority-committed-share-coeff",
        type=float,
        default=0.0,
        help=(
            "Core-only (dehydron=0) committed-share hinge coeff "
            "(pre-reg CORE_MAJORITY_SPLIT_*). 0=off; keep agnostic majority coeff at 0"
        ),
    )
    parser.add_argument(
        "--core-majority-committed-share-tau",
        type=float,
        default=0.56,
        help="Hinge threshold on core committed hard majority share (default 0.56)",
    )
    parser.add_argument(
        "--core-capacity-quota-tau",
        type=float,
        default=0.0,
        help=(
            "Core (dehydron=0) capacity quota τ_cap (pre-reg CORE_QUOTA_*). "
            "0=off; registered value 0.40. Agnostic/core share coeffs must be 0"
        ),
    )
    parser.add_argument(
        "--geometric-angular-kappa",
        type=float,
        default=1.0,
        help="Fixed dehydron→peptide handoff κ (default 1.0)",
    )
    parser.add_argument(
        "--geometric-angular-alpha",
        type=float,
        default=0.7853981633974483,
        help="Max |Δθ| residual bound in radians (default π/4)",
    )
    parser.add_argument(
        "--epoch-anchor-pdb-ids",
        type=str,
        default="",
        help=(
            "Comma-separated PDB IDs forced first each epoch (spread anchors). "
            "Default with --v66-feeler-rim-fanout-expert-arc / "
            "--v66-feeler-geom-angular-prior: 1F88"
        ),
    )
    parser.add_argument(
        "--rim-fanout-min-r",
        type=float,
        default=0.35,
        help="RimFanoutSpread minimum disc radius (lower = act earlier on mid-disc)",
    )
    parser.add_argument(
        "--rim-fanout-forward",
        action="store_true",
        help="Apply RimFanoutSpread in GNN forward (pre-routing 2D disc)",
    )
    parser.add_argument(
        "--rim-fanout-strength",
        type=float,
        default=0.12,
        help="Initial rim fan-out spread strength (learnable when --rim-fanout-forward)",
    )
    parser.add_argument(
        "--spoke-edge-scale",
        type=float,
        default=1.5,
        help="Multiplier on spoke role-edge ρ affinity and conv messages",
    )
    parser.add_argument(
        "--ribbon-edge-scale",
        type=float,
        default=1.35,
        help="Multiplier on ribbon role-edge conv messages",
    )
    parser.add_argument(
        "--dehydron-angular-scale",
        type=float,
        default=0.3,
        help="Scale l=1 spherical harmonics on dehydron relation edges (default 0.3)",
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument(
        "--max-residues",
        type=int,
        default=STAGE_A_MAX_RESIDUES,
        help="Skip structures larger than this (GPU guard; Stage A floor is STAGE_A_MAX_RESIDUES)",
    )
    parser.add_argument("--no-corpus-cache", action="store_true")
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    parser.add_argument("--mlflow-experiment", default="tokyo-eyes-v66")
    parser.add_argument(
        "--gnn-lineage",
        choices=["v6", "v6.5", "v6.6", "v7"],
        default="v6.6",
        help="Architecture package line (v66 launcher defaults to v6.6; use v7 via experiments.training.v7)",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--skip-p-feature-01-gate",
        action="store_true",
        help="Dev only: bypass P_FEATURE_01 DB gate stamp (sets SKIP_P_FEATURE_01_GATE)",
    )
    parser.add_argument(
        "--no-hyperbolic-gate", action="store_true", help="Use legacy tangent MLP gate"
    )
    parser.add_argument(
        "--hyperbolic-expert-mix",
        action="store_true",
        help="Stage 3: mix expert outputs on ball",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override phase epoch count (smoke tests)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=4,
        help="MoE expert count (gate + expert MLP head only; backbone unchanged)",
    )
    parser.add_argument(
        "--v2-teacher-checkpoint", type=Path, default=None, help="Frozen v3 teacher .pt"
    )
    parser.add_argument(
        "--v2-teacher-depth-coeff",
        type=float,
        default=None,
        help="V2 teacher depth distill weight (default 0.20; 0.30 for --p1c)",
    )
    parser.add_argument("--v2-teacher-epistemic-coeff", type=float, default=0.10)
    parser.add_argument(
        "--gentle-phase2",
        action="store_true",
        help="P2: lower LR + freeze radial 5 epochs",
    )
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
        "--no-stage-a-stop",
        action="store_true",
        help="Disable P2 inference-routing stop enforcement on locked Stage A corpus",
    )
    parser.add_argument(
        "--routing-load-floor",
        action="store_true",
        help="Phase 2: min-expert routing floor loss (Stage A contested-routing fix)",
    )
    parser.add_argument(
        "--routing-load-floor-coeff",
        type=float,
        default=10.0,
        help="λ for routing load floor (default 10.0; conservative)",
    )
    parser.add_argument(
        "--routing-load-floor-min",
        type=float,
        default=0.05,
        help="Min per-expert share hinge (default 0.05; matches inference stop)",
    )
    parser.add_argument(
        "--p1b",
        action="store_true",
        help="Run Phase 1b only (angular unfreeze, gentle shell coeffs)",
    )
    parser.add_argument(
        "--p1b-lr", type=float, default=1e-4, help="LR for --p1b (default 1e-4)"
    )
    parser.add_argument(
        "--p1c",
        action="store_true",
        help="Run Phase 1c only (disc expansion: ramped angular + disc spread)",
    )
    parser.add_argument(
        "--p1c-lr", type=float, default=1e-4, help="LR for --p1c (default 1e-4)"
    )
    parser.add_argument(
        "--p1d",
        action="store_true",
        help="Run Phase 1d only (disc-depth scale loss on hyp_proj_head_2d)",
    )
    parser.add_argument(
        "--p1d-lr", type=float, default=1e-4, help="LR for --p1d (default 1e-4)"
    )
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
    parser.add_argument(
        "--p2-bridge-lr", type=float, default=5e-5, help="LR for --p2-bridge"
    )
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
        "--p4-uncertainty-calibration",
        action="store_true",
        help="Phase 4 uncertainty calibration from rs2_post_p4 (decouple epi/ale, gated saves)",
    )
    parser.add_argument(
        "--v7-bprime-uncertainty-heads",
        action="store_true",
        help=(
            "Tokyo Eye v7 B′: heads-only ale/epi recovery from sealed health "
            "(forces epistemic_uncertainty_only_train; skips feeler geom phase)"
        ),
    )
    parser.add_argument(
        "--v7-bprime-uncertainty-heads-rematch",
        action="store_true",
        help=(
            "Pre-authorized rematch-1: higher anticollapse/decorrelation, "
            "still uncertainty_head-only"
        ),
    )
    parser.add_argument(
        "--min-disc-r-mean-hold",
        type=float,
        default=None,
        help="Abort if disc_r_mean stays below this for 2 consecutive epochs",
    )
    parser.add_argument(
        "--p4-head-decouple",
        action="store_true",
        help="Phase 4: split evidential epi/ale trunks + r(epi,ale) loss (rs2 warm-start)",
    )
    parser.add_argument(
        "--p4-head-decouple-decorr-only",
        action="store_true",
        help="G3 ablation A: head decouple + decorrelation only (no B-factor/SASA supervision)",
    )
    parser.add_argument(
        "--p4-v3-aleatoric-shaping",
        action="store_true",
        help="Phase 4 + v3 var_penalty/hinge on G4 train mask (holdout P8 eval)",
    )
    parser.add_argument(
        "--p4-g4-shaping-only-isolation",
        action="store_true",
        help="G4 isolation: uncertainty-head-only with only v3 shaping loss active",
    )
    parser.add_argument(
        "--p4-g4-ale-only-unshaped",
        action="store_true",
        help="G4 isolation: aleatoric branch only, no shaping losses, minimal evidential objective",
    )
    parser.add_argument(
        "--w-var-penalty",
        type=float,
        default=None,
        help="Override v3 var_penalty weight (G4 coefficient sweep)",
    )
    parser.add_argument(
        "--max-probe-r-epi-sasa-save",
        type=float,
        default=None,
        help="Relax v6_best gate: probe_r_epi_sasa must be <= this (default 0.78 for head decouple)",
    )
    parser.add_argument(
        "--p4-gate-promotion",
        action="store_true",
        help="Gate-only pass after head decouple: reduce routing H, preserve uncertainty gates",
    )
    parser.add_argument(
        "--p4-corpus25-gate-promotion",
        action="store_true",
        help="Gate-only on locked 25-protein Stage A (moderate MoE pressure, 30 ep default)",
    )
    parser.add_argument(
        "--p4-corpus25-touchup-extended",
        action="store_true",
        help="Extended routed uncertainty touchup (25 ep, stronger SASA recal)",
    )
    parser.add_argument(
        "--p4-gate-uncertainty-touchup",
        action="store_true",
        help="Uncertainty-only recalibration after gate promotion (re-lock epi/ale/sasa)",
    )
    parser.add_argument(
        "--p4-gate-balance-coeff",
        type=float,
        default=None,
        help="Override MoE balance_coeff for --p4-gate-promotion (default 0.06)",
    )
    parser.add_argument(
        "--p4-gate-load-floor-coeff",
        type=float,
        default=None,
        help="Override routing load floor λ for --p4-gate-promotion (default 12.0)",
    )
    parser.add_argument(
        "--p4-gate-load-floor-min",
        type=float,
        default=None,
        help="Override min expert share hinge for --p4-gate-promotion (default 0.10)",
    )
    parser.add_argument(
        "--residue-stage1",
        action="store_true",
        help="ResidueStage1: pocket + interface BCE heads (warm-start small corpus)",
    )
    parser.add_argument("--residue-stage1-lr", type=float, default=1e-4)
    parser.add_argument(
        "--residue-stage1-epochs",
        type=int,
        default=None,
        help="ResidueStage1 epoch count (default 30)",
    )
    parser.add_argument(
        "--residue-stage2",
        action="store_true",
        help="ResidueStage2: pipeline cryptic pocket + source-leak BCE heads",
    )
    parser.add_argument("--residue-stage2-lr", type=float, default=1e-4)
    parser.add_argument(
        "--residue-stage2-epochs",
        type=int,
        default=None,
        help="ResidueStage2 epoch count (default 30)",
    )
    parser.add_argument(
        "--dehydron-rim-recovery",
        action="store_true",
        help="τ→rim cone recovery warm-start (freeze backbone+gate, SASA shell off)",
    )
    parser.add_argument(
        "--dehydron-rim-recovery-lr",
        type=float,
        default=2e-5,
        help="LR for --dehydron-rim-recovery (default 2e-5)",
    )
    parser.add_argument(
        "--structural-disc-frozen",
        action="store_true",
        help="Attach structural SSOT disc at forward (resume-compatible with --topology-routing-recovery)",
    )
    parser.add_argument(
        "--topology-routing-recovery",
        action="store_true",
        help="P2 routing extension off topology checkpoint (τ-depth, low balance, tight route ceiling)",
    )
    parser.add_argument(
        "--topology-routing-recovery-lr",
        type=float,
        default=3e-5,
        help="LR for --topology-routing-recovery (default 3e-5)",
    )
    parser.add_argument(
        "--topology-gate-disc-recovery",
        action="store_true",
        help="Gate + light disc recovery off route_v3 (freeze expert_depth_bias, 20ep default)",
    )
    parser.add_argument(
        "--topology-gate-disc-recovery-lr",
        type=float,
        default=2e-5,
        help="LR for --topology-gate-disc-recovery (default 2e-5)",
    )
    parser.add_argument(
        "--topology-crescent-recovery",
        action="store_true",
        help="Open collapsed 1D crescent: angular+disc wedge recovery (25ep default)",
    )
    parser.add_argument(
        "--topology-crescent-recovery-lr",
        type=float,
        default=2e-5,
        help="LR for --topology-crescent-recovery (default 2e-5)",
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
    parser.add_argument(
        "--use-dehydron-barcode",
        action="store_true",
        help="Append precomputed dehydron barcode sidecar features to topology node inputs",
    )
    parser.add_argument(
        "--use-binned-dehydron",
        action="store_true",
        help="Append full binned dehydron barcode features (requires --use-dehydron-barcode)",
    )
    parser.add_argument(
        "--dehydron-barcode-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing current-version "
            "{PDB}_{chain}_{BARCODE_FEATURE_VERSION}.pt sidecars"
        ),
    )
    parser.add_argument(
        "--dehydron-edge-barcode",
        action="store_true",
        help="Inject local dehydron witness barcode on dehydron role edges (not node broadcast)",
    )
    parser.add_argument(
        "--chem-edge-mp",
        action="store_true",
        help=(
            "Chem-MVP: append disulf/covale edges from fact_covalent_bond "
            "(requires role_edge_mp / v6.6 feeler; train-side only)"
        ),
    )
    parser.add_argument(
        "--ha-edge-mp",
        action="store_true",
        help=(
            "ha_edges_v1: heavy-atom packing existence + packing/dehydron strength aux "
            "(requires role_edge_mp; same relation IDs; train-side only)"
        ),
    )
    parser.add_argument(
        "--containment-edge-mp",
        action="store_true",
        help=(
            "Path B containment: SSE parent nodes + contain_up/down edges "
            "(requires role_edge_mp + chem_edge_mp; train-side only)"
        ),
    )
    parser.add_argument(
        "--euclidean-shortcut-mp",
        action="store_true",
        help=(
            "Euclidean reach: spatial shortcuts (hop>6 + seq≥10, rel 7) "
            "(requires role_edge_mp + chem_edge_mp; exclusive with "
            "--containment-edge-mp; train-side only)"
        ),
    )
    parser.add_argument(
        "--allow-dead-feature-channel",
        action="store_true",
        help="Debug only: allow --use-dehydron-barcode with slim MoE / frozen backbone",
    )
    parser.add_argument(
        "--feature-liveness-probe",
        action="store_true",
        help="Probe that barcode/MP channels move inference outputs each epoch",
    )
    parser.add_argument(
        "--no-feature-liveness-fail",
        action="store_true",
        help="Log liveness failures as warnings instead of aborting the run",
    )
    args = parser.parse_args()
    if args.seed is not None:
        _set_run_seed(int(args.seed))
        logger.info("RNG seed set to %s", args.seed)
    from science.training.gnn_lineage import get_lineage

    lineage_spec = get_lineage(args.gnn_lineage)
    if args.gnn_lineage in {"v6.5", "v6.6", "v7"}:
        if Path(args.output_dir).as_posix() in {
            "checkpoints/v6/runs/default",
            "checkpoints/v66/runs/default",
        }:
            args.output_dir = lineage_spec.checkpoint_root / "default"
        if args.mlflow_experiment in {"tokyo-eyes-v6", "tokyo-eyes-v66"} and args.gnn_lineage == "v7":
            args.mlflow_experiment = lineage_spec.mlflow_experiment
        elif args.mlflow_experiment == "tokyo-eyes-v6":
            args.mlflow_experiment = lineage_spec.mlflow_experiment

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
    if args.dehydron_rim_recovery or args.topology_routing_recovery:
        v2_depth_coeff = 0.0
    elif v2_depth_coeff is None:
        v2_depth_coeff = 0.30 if (args.p1c or args.p1d or args.p2_bridge) else 0.20

    v2_ckpt = args.v2_teacher_checkpoint
    if args.dehydron_rim_recovery or args.topology_routing_recovery:
        v2_ckpt = None
    elif v2_ckpt is None:
        from experiments.training.v66.v2_teacher import (
            resolve_default_v2_teacher_checkpoint,
        )

        v2_ckpt = resolve_default_v2_teacher_checkpoint()

    config = TrainingConfig(
        device=args.device,
        lr=args.lr,
        num_experts=args.num_experts,
        gnn_lineage=args.gnn_lineage,
        model_version=lineage_spec.model_version,
        init_seed=int(args.seed) if args.seed is not None else None,
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
        v2_teacher_epistemic_coeff=(
            0.0
            if (args.dehydron_rim_recovery or args.topology_routing_recovery)
            else args.v2_teacher_epistemic_coeff
        ),
        gentle_phase2=args.gentle_phase2,
        phase2_lr=args.phase2_lr,
        save_epoch_snapshots=args.save_epoch_snapshots,
        enforce_stage_a_stop=not args.no_stage_a_stop,
        routing_load_floor=args.routing_load_floor,
        routing_load_floor_coeff=args.routing_load_floor_coeff,
        routing_load_floor_min=args.routing_load_floor_min,
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
        p2_hypmix=args.p2_hypmix
        and not args.p2_hypmix3
        and not args.p2_hypmix_final
        and not args.p2_disc_occupancy
        and not args.p2_disc_occupancy_v2
        and not args.p2_disc_occupancy_v3
        and not args.p2_disc_occupancy_v4
        and not args.p2_disc_occupancy_v5
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_hypmix3=args.p2_hypmix3
        and not args.p2_hypmix_final
        and not args.p2_disc_occupancy
        and not args.p2_disc_occupancy_v2
        and not args.p2_disc_occupancy_v3
        and not args.p2_disc_occupancy_v4
        and not args.p2_disc_occupancy_v5
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_hypmix_final=args.p2_hypmix_final
        and not args.p2_disc_occupancy
        and not args.p2_disc_occupancy_v2
        and not args.p2_disc_occupancy_v3
        and not args.p2_disc_occupancy_v4
        and not args.p2_disc_occupancy_v5
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_occupancy=(
            args.p2_disc_occupancy
            or args.p2_disc_occupancy_v2
            or args.p2_disc_occupancy_v3
            or args.p2_disc_occupancy_v4
            or args.p2_disc_occupancy_v5
        )
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_occupancy_v2=args.p2_disc_occupancy_v2
        and not args.p2_disc_occupancy_v3
        and not args.p2_disc_occupancy_v4
        and not args.p2_disc_occupancy_v5
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_occupancy_v3=args.p2_disc_occupancy_v3
        and not args.p2_disc_occupancy_v4
        and not args.p2_disc_occupancy_v5
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_occupancy_v4=args.p2_disc_occupancy_v4
        and not args.p2_disc_occupancy_v5
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_occupancy_v5=args.p2_disc_occupancy_v5
        and not args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_gentle_arch=args.p2_disc_gentle_arch
        and not args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_path_align=args.p2_disc_path_align
        and not args.p2_disc_proj_recovery
        and not args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_proj_recovery=(
            args.p2_disc_proj_recovery
            or args.p2_disc_proj_recovery_v2
            or args.p2_disc_proj_recovery_v3
            or args.p2_disc_proj_recovery_v4
            or args.p2_disc_proj_recovery_v5
        )
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v2=args.p2_disc_proj_recovery_v2
        and not args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v3=args.p2_disc_proj_recovery_v3
        and not args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v4=args.p2_disc_proj_recovery_v4
        and not args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
        p2_disc_proj_recovery_v5=args.p2_disc_proj_recovery_v5
        and not args.p2_rec_ablation
        and not args.full_hyp_moe_test,
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
            if args.p2_disc_proj_recovery_v5 or args.v66_feeler_angular_lift
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
        p4_uncertainty_calibration=args.p4_uncertainty_calibration,
        v7_bprime_uncertainty_heads=bool(
            getattr(args, "v7_bprime_uncertainty_heads", False)
        ),
        v7_bprime_uncertainty_heads_rematch=bool(
            getattr(args, "v7_bprime_uncertainty_heads_rematch", False)
        ),
        min_disc_r_mean_hold=getattr(args, "min_disc_r_mean_hold", None),
        p4_head_decouple=args.p4_head_decouple,
        p4_head_decouple_decorr_only=args.p4_head_decouple_decorr_only,
        p4_v3_aleatoric_shaping=args.p4_v3_aleatoric_shaping,
        p4_g4_shaping_only_isolation=args.p4_g4_shaping_only_isolation,
        p4_g4_ale_only_unshaped=args.p4_g4_ale_only_unshaped,
        w_var_penalty=args.w_var_penalty,
        p4_gate_promotion=args.p4_gate_promotion,
        p4_corpus25_gate_promotion=args.p4_corpus25_gate_promotion,
        p4_corpus25_touchup_extended=args.p4_corpus25_touchup_extended,
        p4_gate_uncertainty_touchup=args.p4_gate_uncertainty_touchup,
        p4_gate_balance_coeff=args.p4_gate_balance_coeff,
        p4_gate_load_floor_coeff=args.p4_gate_load_floor_coeff,
        p4_gate_load_floor_min=args.p4_gate_load_floor_min,
        decoupled_uncertainty_heads=(
            args.p4_head_decouple
            or args.p4_head_decouple_decorr_only
            or args.p4_v3_aleatoric_shaping
            or args.p4_g4_shaping_only_isolation
            or args.p4_g4_ale_only_unshaped
            or args.p4_gate_uncertainty_touchup
            or args.p4_corpus25_touchup_extended
        ),
        max_probe_r_epi_sasa_save=args.max_probe_r_epi_sasa_save,
        residue_stage1=args.residue_stage1,
        residue_stage1_lr=args.residue_stage1_lr,
        residue_stage1_epochs=args.residue_stage1_epochs,
        residue_stage2=args.residue_stage2,
        residue_stage2_lr=args.residue_stage2_lr,
        residue_stage2_epochs=args.residue_stage2_epochs,
        master_cold_lineage=args.master_cold_lineage,
        v66_feeler_lineage=args.v66_feeler_lineage,
        v66_feeler_angular_lift=args.v66_feeler_angular_lift,
        v66_feeler_coupling=args.v66_feeler_coupling,
        v66_feeler_no_exclusivity=args.v66_feeler_no_exclusivity or args.v66_feeler_rim_decouple,
        v66_feeler_dehydron_angular=args.v66_feeler_dehydron_angular or args.v66_feeler_rim_decouple,
        v66_feeler_rim_decouple=args.v66_feeler_rim_decouple,
        v66_feeler_p3_geom_edges=args.v66_feeler_p3_geom_edges,
        v66_feeler_p3_geom=args.v66_feeler_p3_geom,
        v66_feeler_p3_geom_half_stack=args.v66_feeler_p3_geom_half_stack,
        v66_feeler_rim_fanout_model=args.v66_feeler_rim_fanout_model,
        v66_feeler_rim_fanout_cold_curriculum=args.v66_feeler_rim_fanout_cold_curriculum,
        v66_feeler_rim_fanout_polish=args.v66_feeler_rim_fanout_polish,
        v66_feeler_rim_fanout_angular=args.v66_feeler_rim_fanout_angular
        and not args.v66_feeler_rim_fanout_angular_v2
        and not args.v66_feeler_rim_fanout_radius
        and not args.v66_feeler_rim_fanout_coverage
        and not args.v66_feeler_rim_fanout_antibarrier
        and not args.v66_feeler_rim_fanout_expert_arc
        and not args.v66_feeler_geom_angular_prior,
        v66_feeler_rim_fanout_angular_v2=args.v66_feeler_rim_fanout_angular_v2
        and not args.v66_feeler_rim_fanout_radius
        and not args.v66_feeler_rim_fanout_coverage
        and not args.v66_feeler_rim_fanout_antibarrier
        and not args.v66_feeler_rim_fanout_expert_arc
        and not args.v66_feeler_geom_angular_prior,
        v66_feeler_rim_fanout_radius=args.v66_feeler_rim_fanout_radius
        and not args.v66_feeler_rim_fanout_coverage
        and not args.v66_feeler_rim_fanout_antibarrier
        and not args.v66_feeler_rim_fanout_expert_arc
        and not args.v66_feeler_geom_angular_prior,
        v66_feeler_rim_fanout_coverage=args.v66_feeler_rim_fanout_coverage
        and not args.v66_feeler_rim_fanout_antibarrier
        and not args.v66_feeler_rim_fanout_expert_arc
        and not args.v66_feeler_geom_angular_prior,
        v66_feeler_rim_fanout_antibarrier=args.v66_feeler_rim_fanout_antibarrier
        and not args.v66_feeler_rim_fanout_expert_arc
        and not args.v66_feeler_geom_angular_prior,
        v66_feeler_rim_fanout_expert_arc=args.v66_feeler_rim_fanout_expert_arc
        and not args.v66_feeler_geom_angular_prior,
        v66_feeler_geom_angular_prior=args.v66_feeler_geom_angular_prior,
        geometric_angular_prior=args.v66_feeler_geom_angular_prior
        or getattr(args, "geometric_angular_prior", False),
        geometric_angular_kappa=args.geometric_angular_kappa,
        geometric_angular_alpha=args.geometric_angular_alpha,
        hyperbolic_mp_graph=bool(getattr(args, "hyperbolic_mp_graph", False)),
        input_feature_zscore=bool(getattr(args, "input_feature_zscore", False)),
        replace_tau_with_abs_dist=bool(getattr(args, "replace_tau_abs_dist", False)),
        gate_logit_softplus_init=getattr(args, "gate_logit_softplus_init", None),
        gate_logit_softplus_floor=getattr(args, "gate_logit_softplus_floor", None),
        gate_include_sasa=bool(getattr(args, "gate_include_sasa", False)),
        routing_entropy_sparsity_coeff=float(
            getattr(args, "routing_entropy_sparsity_coeff", 0.0) or 0.0
        ),
        routing_entropy_sparsity_warmup_epochs=int(
            getattr(args, "routing_entropy_sparsity_warmup_epochs", 8)
            if getattr(args, "routing_entropy_sparsity_warmup_epochs", None) is not None
            else 8
        ),
        sparsity_style_save=bool(getattr(args, "sparsity_style_save", False)),
        routing_entropy_mean_residue_min_save=getattr(
            args, "routing_entropy_mean_residue_min_save", None
        ),
        routing_entropy_mean_residue_max_save=getattr(
            args, "routing_entropy_mean_residue_max_save", None
        ),
        disc_occupancy_coeff_override=getattr(
            args, "disc_occupancy_coeff_override", None
        ),
        disc_depth_scale_coeff_override=getattr(
            args, "disc_depth_scale_coeff_override", None
        ),
        disc_depth_scale_target_override=getattr(
            args, "disc_depth_scale_target_override", None
        ),
        core_radial_floor_coeff_override=getattr(
            args, "core_radial_floor_coeff_override", None
        ),
        core_radial_floor_min_r_override=getattr(
            args, "core_radial_floor_min_r_override", None
        ),
        prototype_repulsion_coeff=float(
            getattr(args, "prototype_repulsion_coeff", 0.0) or 0.0
        ),
        prototype_repulsion_margin=float(
            getattr(args, "prototype_repulsion_margin", 0.25) or 0.25
        ),
        directionality_asym_coeff=float(
            getattr(args, "directionality_asym_coeff", 0.0) or 0.0
        ),
        prototype_gram_logdet_coeff=float(
            getattr(args, "prototype_gram_logdet_coeff", 0.0) or 0.0
        ),
        prototype_gram_logdet_tau=float(
            getattr(args, "prototype_gram_logdet_tau", -1.15) or -1.15
        ),
        majority_committed_share_coeff=float(
            getattr(args, "majority_committed_share_coeff", 0.0) or 0.0
        ),
        majority_committed_share_tau=float(
            getattr(args, "majority_committed_share_tau", 0.56) or 0.56
        ),
        core_majority_committed_share_coeff=float(
            getattr(args, "core_majority_committed_share_coeff", 0.0) or 0.0
        ),
        core_majority_committed_share_tau=float(
            getattr(args, "core_majority_committed_share_tau", 0.56) or 0.56
        ),
        core_capacity_quota_tau=float(
            getattr(args, "core_capacity_quota_tau", 0.0) or 0.0
        ),
        epoch_anchor_pdb_ids=(
            [p.strip().upper() for p in str(args.epoch_anchor_pdb_ids).split(",") if p.strip()]
            if str(args.epoch_anchor_pdb_ids).strip()
            else (
                ["1F88"]
                if (
                    args.v66_feeler_rim_fanout_expert_arc
                    or args.v66_feeler_geom_angular_prior
                )
                else []
            )
        ),
        rim_fanout_forward=args.rim_fanout_forward or args.v66_feeler_rim_fanout_model,
        rim_fanout_strength=args.rim_fanout_strength,
        rim_fanout_min_r=(
            0.20
            if (
                (
                    args.v66_feeler_rim_fanout_angular_v2
                    or args.v66_feeler_rim_fanout_radius
                    or args.v66_feeler_rim_fanout_coverage
                    or args.v66_feeler_rim_fanout_antibarrier
                    or args.v66_feeler_rim_fanout_expert_arc
                    or args.v66_feeler_geom_angular_prior
                )
                and args.rim_fanout_min_r == 0.35
            )
            else args.rim_fanout_min_r
        ),
        spoke_edge_scale=args.spoke_edge_scale,
        ribbon_edge_scale=args.ribbon_edge_scale,
        dehydron_exclusivity=not (args.no_dehydron_exclusivity or args.v66_feeler_rim_decouple),
        dehydron_angular_scale=(
            args.dehydron_angular_scale
            if args.v66_feeler_dehydron_angular or args.v66_feeler_rim_decouple
            else 1.0
        ),
        slim_moe_structural_ssot=args.slim_moe_structural_ssot,
        dehydron_rim_recovery=args.dehydron_rim_recovery,
        dehydron_rim_recovery_lr=args.dehydron_rim_recovery_lr,
        topology_routing_recovery=args.topology_routing_recovery,
        topology_routing_recovery_lr=args.topology_routing_recovery_lr,
        topology_gate_disc_recovery=args.topology_gate_disc_recovery,
        topology_gate_disc_recovery_lr=args.topology_gate_disc_recovery_lr,
        topology_crescent_recovery=args.topology_crescent_recovery,
        topology_crescent_recovery_lr=args.topology_crescent_recovery_lr,
        structural_disc_frozen=args.structural_disc_frozen,
        use_dehydron_barcode=args.use_dehydron_barcode,
        use_binned_dehydron=args.use_binned_dehydron,
        dehydron_barcode_dir=args.dehydron_barcode_dir,
        dehydron_edge_barcode=args.dehydron_edge_barcode,
        chem_edge_mp=args.chem_edge_mp,
        ha_edge_mp=args.ha_edge_mp,
        containment_edge_mp=args.containment_edge_mp,
        euclidean_shortcut_mp=args.euclidean_shortcut_mp,
        allow_dead_feature_channel=args.allow_dead_feature_channel,
        feature_liveness_probe=args.feature_liveness_probe,
        feature_liveness_fail_if_dead=not args.no_feature_liveness_fail,
    )
    if config.use_binned_dehydron and not config.use_dehydron_barcode:
        logger.error("--use-binned-dehydron requires --use-dehydron-barcode")
        sys.exit(2)
    if config.dehydron_edge_barcode and config.use_dehydron_barcode:
        logger.error(
            "Refusing --dehydron-edge-barcode with --use-dehydron-barcode: "
            "edge channel is local-only; node-global broadcast caused disc collapse."
        )
        sys.exit(2)
    if config.dehydron_edge_barcode and not config.dehydron_barcode_dir:
        logger.error("--dehydron-edge-barcode requires --dehydron-barcode-dir")
        sys.exit(2)
    if config.use_dehydron_barcode and not config.allow_dead_feature_channel:
        if config.slim_moe_structural_ssot:
            logger.error(
                "Refusing --use-dehydron-barcode with --slim-moe-structural-ssot: "
                "node_emb is frozen and structural SSOT bypasses MP→geometry, so "
                "barcode cannot affect inference. Use --master-cold-lineage "
                "(learned GNN) or pass --allow-dead-feature-channel for debugging."
            )
            sys.exit(2)
    if config.topology_crescent_recovery:
        from science.training.config import apply_topology_crescent_recovery_config

        config = apply_topology_crescent_recovery_config(config)
        logger.info(
            "Topology crescent recovery: angular+fusion+hyp_proj_2d trainable; "
            "radial+expert_depth_bias frozen; disc thickness/PC2/eff_rank floors"
        )
    elif config.topology_gate_disc_recovery:
        from science.training.config import apply_topology_gate_disc_recovery_config

        config = apply_topology_gate_disc_recovery_config(config)
        logger.info(
            "Topology gate+disc recovery: gate + hyp_proj_2d trainable, "
            "expert_depth_bias frozen, disc_occupancy=0.35"
        )
    elif config.topology_routing_recovery:
        from science.training.config import apply_topology_routing_recovery_config

        config = apply_topology_routing_recovery_config(config)
        logger.info(
            "Topology routing recovery: topology_only_gate=True, soft routing, "
            "expert_depth_decouple=%s, structure_gate=%s, track_v6_best_route=True",
            config.expert_depth_decouple,
            config.structure_gate,
        )
        if config.structural_disc_frozen:
            logger.info(
                "Structural disc SSOT frozen — MoE routing + uncertainty only (resume-compatible)"
            )
    if config.v66_feeler_lineage and (
        config.slim_moe_structural_ssot or config.master_cold_lineage
    ):
        logger.error(
            "--v66-feeler-lineage is mutually exclusive with "
            "--slim-moe-structural-ssot and --master-cold-lineage"
        )
        sys.exit(2)
    if config.v66_feeler_p3_geom_half_stack and not config.v66_feeler_p3_geom:
        logger.error("--v66-feeler-p3-geom-half-stack requires --v66-feeler-p3-geom")
        sys.exit(2)
    if config.v66_feeler_p3_geom and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-p3-geom requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_p3_geom_edges and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-p3-geom-edges requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_p3_geom_edges and not config.dehydron_edge_barcode:
        logger.error(
            "--v66-feeler-p3-geom-edges requires --dehydron-edge-barcode "
            "(local witness barcode on dehydron edges)"
        )
        sys.exit(2)
    if config.v66_feeler_angular_lift and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-angular-lift requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_angular_lift and config.p2_disc_proj_recovery_v5:
        logger.error(
            "--v66-feeler-angular-lift is mutually exclusive with --p2-disc-proj-recovery-v5"
        )
        sys.exit(2)
    if config.v66_feeler_coupling and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-coupling requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_coupling and config.v66_feeler_angular_lift:
        logger.error(
            "--v66-feeler-coupling is mutually exclusive with --v66-feeler-angular-lift"
        )
        sys.exit(2)
    feeler_ablation_flags = (
        config.v66_feeler_p3_geom,
        config.v66_feeler_p3_geom_edges,
        config.v66_feeler_rim_fanout_model,
        config.v66_feeler_coupling,
        config.v66_feeler_rim_decouple,
        config.v66_feeler_angular_lift,
        config.v66_feeler_no_exclusivity and not config.v66_feeler_dehydron_angular,
        config.v66_feeler_dehydron_angular and not config.v66_feeler_no_exclusivity,
    )
    if sum(int(x) for x in feeler_ablation_flags) > 1:
        logger.error(
            "At most one feeler ablation mode: --v66-feeler-p3-geom, --v66-feeler-p3-geom-edges, "
            "--v66-feeler-rim-fanout-model, --v66-feeler-coupling, --v66-feeler-rim-decouple, "
            "--v66-feeler-angular-lift, --v66-feeler-no-exclusivity (alone), "
            "--v66-feeler-dehydron-angular (alone)"
        )
        sys.exit(2)
    if config.v66_feeler_rim_fanout_model and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-rim-fanout-model requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_rim_fanout_cold_curriculum and not config.v66_feeler_rim_fanout_model:
        logger.error(
            "--v66-feeler-rim-fanout-cold-curriculum requires --v66-feeler-rim-fanout-model"
        )
        sys.exit(2)
    if config.v66_feeler_rim_fanout_polish and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-rim-fanout-polish requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    if config.v66_feeler_rim_fanout_angular and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-rim-fanout-angular requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    if config.v66_feeler_rim_fanout_angular_v2 and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-rim-fanout-angular-v2 requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    if config.v66_feeler_rim_fanout_radius and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-rim-fanout-radius requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    if config.v66_feeler_rim_fanout_coverage and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-rim-fanout-coverage requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    if config.v66_feeler_rim_fanout_antibarrier and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-rim-fanout-antibarrier requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    if config.v66_feeler_rim_fanout_expert_arc and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-rim-fanout-expert-arc requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    if config.v66_feeler_geom_angular_prior and not config.v66_feeler_rim_fanout_model:
        logger.error("--v66-feeler-geom-angular-prior requires --v66-feeler-rim-fanout-model")
        sys.exit(2)
    _rim_modes = (
        config.v66_feeler_rim_fanout_cold_curriculum,
        config.v66_feeler_rim_fanout_polish,
        config.v66_feeler_rim_fanout_angular,
        config.v66_feeler_rim_fanout_angular_v2,
        config.v66_feeler_rim_fanout_radius,
        config.v66_feeler_rim_fanout_coverage,
        config.v66_feeler_rim_fanout_antibarrier,
        config.v66_feeler_rim_fanout_expert_arc,
        config.v66_feeler_geom_angular_prior,
    )
    if sum(int(x) for x in _rim_modes) > 1:
        logger.error(
            "At most one of --v66-feeler-rim-fanout-cold-curriculum, "
            "--v66-feeler-rim-fanout-polish, --v66-feeler-rim-fanout-angular, "
            "--v66-feeler-rim-fanout-angular-v2, --v66-feeler-rim-fanout-radius, "
            "--v66-feeler-rim-fanout-coverage, --v66-feeler-rim-fanout-antibarrier, "
            "--v66-feeler-rim-fanout-expert-arc, --v66-feeler-geom-angular-prior"
        )
        sys.exit(2)
    if config.v66_feeler_rim_decouple and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-rim-decouple requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_no_exclusivity and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-no-exclusivity requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_dehydron_angular and not config.v66_feeler_lineage:
        logger.error("--v66-feeler-dehydron-angular requires --v66-feeler-lineage")
        sys.exit(2)
    if config.v66_feeler_rim_decouple:
        logger.info(
            "rim decouple: dehydron exclusivity OFF + angular_scale=%.2f",
            config.dehydron_angular_scale,
        )
    elif config.v66_feeler_no_exclusivity:
        config = config.model_copy(update={"dehydron_exclusivity": False})
        logger.info("dehydron exclusivity OFF — dehydron pairs may carry packing/ribbon/spoke")
    if config.v66_feeler_dehydron_angular:
        logger.info(
            "dehydron angular scale=%.2f on dehydron relation SH l=1",
            config.dehydron_angular_scale,
        )
    if config.v66_feeler_coupling and not config.dehydron_exclusivity:
        logger.info(
            "dehydron exclusivity OFF — packing/spoke may coexist with dehydron pairs"
        )
    if config.slim_moe_structural_ssot and config.master_cold_lineage:
        logger.error(
            "Use --slim-moe-structural-ssot OR --master-cold-lineage, not both"
        )
        sys.exit(2)
    if config.v66_feeler_lineage:
        if not args.no_warm_start and config.resume is None:
            logger.error(
                "--v66-feeler-lineage requires --no-warm-start (no weight transfer) "
                "unless continuing via --resume"
            )
            sys.exit(2)
        if config.resume is not None:
            # Continue path (P2+): weights come from feeler checkpoint, not v5 warm-start.
            args.no_warm_start = True
        from science.training.config import apply_v66_feeler_config

        config = apply_v66_feeler_config(config)
        if config.hyperbolic_mp_graph:
            config = config.model_copy(
                update={
                    "role_edge_mp": False,
                    "multi_rel_edge_mp": False,
                    "thermo_edge_features": False,
                }
            )
            logger.info(
                "S4 hyperbolic_mp_graph: disabling role/multi-rel/thermo edge paths "
                "(mutually exclusive with hyp disc k-NN)"
            )
        if config.chem_edge_mp and not config.role_edge_mp:
            logger.error(
                "--chem-edge-mp requires role_edge_mp "
                "(enabled by --v66-feeler-lineage unless S4 hyp-MP disables it)"
            )
            sys.exit(2)
        if config.ha_edge_mp and not config.role_edge_mp:
            logger.error(
                "--ha-edge-mp requires role_edge_mp "
                "(enabled by --v66-feeler-lineage unless S4 hyp-MP disables it)"
            )
            sys.exit(2)
        if config.containment_edge_mp and not (
            config.role_edge_mp and config.chem_edge_mp
        ):
            logger.error(
                "--containment-edge-mp requires role_edge_mp + chem_edge_mp "
                "(matched Path B arm; enabled by --v66-feeler-lineage + --chem-edge-mp)"
            )
            sys.exit(2)
        if config.euclidean_shortcut_mp and not (
            config.role_edge_mp and config.chem_edge_mp
        ):
            logger.error(
                "--euclidean-shortcut-mp requires role_edge_mp + chem_edge_mp "
                "(matched Euclidean reach arm)"
            )
            sys.exit(2)
        if config.euclidean_shortcut_mp and config.containment_edge_mp:
            logger.error(
                "--euclidean-shortcut-mp and --containment-edge-mp are "
                "mutually exclusive"
            )
            sys.exit(2)
        if config.v66_feeler_coupling:
            config = config.model_copy(update={"role_coupling_edges": True})
        if config.resume is not None:
            logger.info(
                "v6.6 feeler continue from %s (phase=%s, timeout@45%%/1ep)",
                config.resume,
                config.phase,
            )
        else:
            logger.info(
                "v6.6 feeler: learned MP→geometry→MoE, minimal losses, expert timeout@45%%/1ep"
            )
    elif config.slim_moe_structural_ssot:
        # Resume is allowed for P2/P3 continuation from a prior slim MoE P1.
        # Still skip v5 warm-start — weights come from --resume when present.
        args.no_warm_start = True
        from science.training.config import apply_slim_moe_structural_ssot_config

        config = apply_slim_moe_structural_ssot_config(config)
        logger.info(
            "Slim MoE structural SSOT: frozen disc from ρ/τ/Cα, train MoE routing + uncertainty only"
        )
        if config.resume is not None:
            logger.info(
                "Slim MoE resume from %s (no expert timeout in P2+)", config.resume
            )
    elif config.master_cold_lineage:
        # Resume is allowed for recipe-preserving matched continues (P1 barcode
        # baseline/scalars). This is not a cold-start claim: weights come from
        # --resume; master-cold only keeps topology_only_gate + dehydron phases.
        if config.resume is not None:
            logger.info(
                "MASTER cold recipe-preserving continue from %s "
                "(topology_only_gate + dehydron phases retained; not a cold-start)",
                config.resume,
            )
        args.no_warm_start = True
        from science.training.config import apply_master_cold_dehydron_config

        config = apply_master_cold_dehydron_config(config)
        logger.info(
            "MASTER cold dehydron ablation: topology_only_gate=True, V2 teacher disabled"
        )
    if (
        config.p2_disc_proj_recovery
        or config.p2_disc_proj_recovery_v2
        or config.p2_disc_proj_recovery_v3
        or config.p2_disc_proj_recovery_v4
        or config.p2_disc_proj_recovery_v5
    ):
        config.legacy_disc_projection = False
    from science.training.gnn_lineage import apply_lineage_defaults

    config = apply_lineage_defaults(config)
    lineage_spec = get_lineage(config.gnn_lineage)
    logger.info(
        "GNN lineage %s → %s_*.pt under %s, MLflow experiment %s",
        config.gnn_lineage,
        lineage_spec.checkpoint_prefix,
        config.output_dir,
        config.mlflow_experiment,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.pdb_dir.is_dir():
        config.pdb_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_p_feature_01_gate:
        os.environ["SKIP_P_FEATURE_01_GATE"] = "1"
    from science.training.p_feature_01_gate import (
        DEFAULT_STAMP_PATH,
        require_p_feature_01_for_training,
    )

    try:
        require_p_feature_01_for_training(
            Path(config.corpus_manifest),
            stamp_path=DEFAULT_STAMP_PATH,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(
            "P_FEATURE_01 DB gate not satisfied: %s — run `make gate-p-feature-01` first",
            exc,
        )
        sys.exit(2)

    logger.info("Loading corpus from %s", config.corpus_manifest)
    proteins, failed = load_training_proteins(
        config.pdb_dir,
        config.corpus_manifest,
        max_proteins=config.max_proteins,
        max_residues=config.max_residues,
        use_cache=not args.no_corpus_cache,
        use_dehydron_barcode=config.use_dehydron_barcode,
        use_binned_dehydron=config.use_binned_dehydron,
        dehydron_barcode_dir=config.dehydron_barcode_dir,
        dehydron_edge_barcode=config.dehydron_edge_barcode,
    )
    if not proteins:
        logger.error(
            "No proteins loaded (%d failed). Check network / manifest.", failed
        )
        sys.exit(1)
    logger.info("Loaded %d proteins (%d failed)", len(proteins), failed)
    try:
        node_dim = _resolve_training_node_dim(
            proteins,
            use_dehydron_barcode=config.use_dehydron_barcode,
            use_binned_dehydron=config.use_binned_dehydron,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    logger.info(
        "Training node_dim=%d (GNN_INPUT_MODE / feature_set aligned with loaded graphs)",
        node_dim,
    )

    if config.chem_edge_mp and not config.role_edge_mp:
        logger.error(
            "--chem-edge-mp requires role_edge_mp "
            "(use --v66-feeler-lineage, or an explicit role-edge lineage)"
        )
        sys.exit(2)
    if config.ha_edge_mp and not config.role_edge_mp:
        logger.error(
            "--ha-edge-mp requires role_edge_mp "
            "(use --v66-feeler-lineage, or an explicit role-edge lineage)"
        )
        sys.exit(2)
    if config.containment_edge_mp and not (
        config.role_edge_mp and config.chem_edge_mp
    ):
        logger.error(
            "--containment-edge-mp requires role_edge_mp + chem_edge_mp "
            "(matched Path B arm)"
        )
        sys.exit(2)
    if config.euclidean_shortcut_mp and not (
        config.role_edge_mp and config.chem_edge_mp
    ):
        logger.error(
            "--euclidean-shortcut-mp requires role_edge_mp + chem_edge_mp "
            "(matched Euclidean reach arm)"
        )
        sys.exit(2)
    if config.euclidean_shortcut_mp and config.containment_edge_mp:
        logger.error(
            "--euclidean-shortcut-mp and --containment-edge-mp are "
            "mutually exclusive"
        )
        sys.exit(2)

    if proteins and (
        config.master_cold_lineage
        or config.v66_feeler_lineage
        or config.slim_moe_structural_ssot
        or config.topology_routing_recovery
    ):
        scatter_pdb = config.disc_scatter_structure.split(":")[0].upper()
        loaded_pdbs = {str(p.get("pdb_id", "")).upper() for p in proteins}
        if scatter_pdb not in loaded_pdbs:
            old_spec = config.disc_scatter_structure
            p0 = proteins[0]
            new_spec = f"{p0['pdb_id']}:{p0.get('chain', 'A')}"
            config = config.model_copy(update={"disc_scatter_structure": new_spec})
            logger.info(
                "MASTER cold lineage: disc_scatter_structure %s not in corpus — using %s",
                old_spec,
                new_spec,
            )

    if config.resume and config.resume.is_file():
        from science.dtie.v66.gnn.evidential import uncertainty_head_is_decoupled

        resume_blob = torch.load(config.resume, map_location="cpu", weights_only=False)
        resume_sd = resume_blob.get("model_state_dict", resume_blob)
        if uncertainty_head_is_decoupled(resume_sd):
            config = config.model_copy(update={"decoupled_uncertainty_heads": True})
            logger.info("Resume checkpoint uses decoupled uncertainty head")

    model = build_model(config, node_dim=node_dim)
    device = config.device
    if device != "cpu" and torch.cuda.is_available():
        model.to(device)
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        config.device = "cpu"
        model.to(device)

    init_v6_radial_scale(model)

    if config.gate_logit_softplus_init is not None or config.gate_logit_softplus_floor is not None:
        import math

        import torch.nn.functional as F

        gate = getattr(model, "gate", None)
        if gate is None or not hasattr(gate, "logit_scale"):
            logger.warning(
                "gate_logit_softplus_* set but model.gate has no logit_scale — skipping"
            )
        else:
            target = config.gate_logit_softplus_init
            if target is not None:
                y = float(max(target, 1e-8))
                init_param = y if y > 20.0 else math.log(math.expm1(y))
                with torch.no_grad():
                    gate.logit_scale.fill_(init_param)
            if config.gate_logit_softplus_floor is not None:
                gate.logit_softplus_floor = float(config.gate_logit_softplus_floor)
            sp = float(F.softplus(gate.logit_scale.detach()).cpu())
            floor = getattr(gate, "logit_softplus_floor", None)
            logger.info(
                "Gate logit_scale: softplus=%.4f (param=%.4f) floor=%s",
                sp,
                float(gate.logit_scale.detach().cpu()),
                floor,
            )

    if config.input_feature_zscore or config.replace_tau_with_abs_dist:
        from science.dtie.common.input_feature_norm import fit_and_install_input_feature_norm

        stats = fit_and_install_input_feature_norm(
            model,
            proteins,
            zscore=bool(config.input_feature_zscore)
            or bool(config.replace_tau_with_abs_dist),
            replace_tau_with_abs_dist=bool(config.replace_tau_with_abs_dist),
        )
        logger.info(
            "T1a input feature norm: zscore=%s replace_tau_abs_dist=%s mean=%s std=%s",
            stats["input_feature_zscore"],
            stats["replace_tau_with_abs_dist"],
            [f"{x:.4g}" for x in stats["mean"]],
            [f"{x:.4g}" for x in stats["std"]],
        )
        try:
            import json as _json

            config.output_dir.mkdir(parents=True, exist_ok=True)
            (config.output_dir / "t1a_input_feature_norm.json").write_text(
                _json.dumps(stats, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not write t1a_input_feature_norm.json: %s", exc)

    if (
        config.phase
        and config.phase >= 2
        and (config.resume is None or not Path(config.resume).is_file())
    ):
        auto_resume = resolve_prior_checkpoint(
            config.output_dir,
            config.phase,
            len(proteins),
            checkpoint_prefix=lineage_spec.checkpoint_prefix,
        )
        if auto_resume is not None:
            config.resume = auto_resume
            logger.info("Auto-resuming phase %d from %s", config.phase, auto_resume)

    resume_state = None
    if config.resume and config.resume.is_file():
        import importlib

        from science.training.checkpoint import CheckpointManager

        resume_state = CheckpointManager(
            config.output_dir,
            len(proteins),
            checkpoint_prefix=lineage_spec.checkpoint_prefix,
            architecture_version=lineage_spec.architecture_version,
        ).load(config.resume, device)
        module = importlib.import_module(lineage_spec.package)
        load_state = getattr(
            module, f"load_{lineage_spec.checkpoint_prefix}_state_dict"
        )
        missing, unexpected = load_state(model, resume_state.model_state_dict)
        if missing:
            logger.info(
                "Resume: %d missing keys (new modules init from scratch)", len(missing)
            )
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
    if (
        config.v2_teacher_checkpoint is not None
        and Path(config.v2_teacher_checkpoint).is_file()
    ):
        from experiments.training.v66.v2_teacher import V2Teacher

        logger.info("Loading V2 teacher from %s", config.v2_teacher_checkpoint)
        v2_teacher = V2Teacher(
            config.v2_teacher_checkpoint,
            device=device,
        )
        v2_teacher.precompute(proteins)
    elif config.v2_teacher_checkpoint is not None:
        logger.warning(
            "V2 teacher checkpoint not found: %s — training without distill",
            config.v2_teacher_checkpoint,
        )

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

    from experiments.training.v66.export_corpus_viewers import (
        export_training_run_viewers,
    )

    try:
        viewer_paths = export_training_run_viewers(
            model,
            proteins,
            output_dir=config.output_dir,
            pdb_dir=config.pdb_dir,
            device=device,
            structural_disc_frozen=config.structural_disc_frozen,
            checkpoint_label=str(
                _viewer_checkpoint_label(config.output_dir, lineage_spec.checkpoint_prefix)
            ),
        )
        if viewer_paths:
            logger.info(
                "Exported %d interactive viewer sets → %s/viewers/",
                len(viewer_paths),
                config.output_dir,
            )
            if tracker and tracker._active:
                manifest = config.output_dir / "viewers" / "viewer_manifest.json"
                if manifest.is_file():
                    tracker.log_artifact(manifest, artifact_path="viewers")
    except Exception as exc:
        logger.warning("Training viewer export failed (non-fatal): %s", exc)

    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
