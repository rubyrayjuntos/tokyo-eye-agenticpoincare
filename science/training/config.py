"""Pydantic training configuration for v6 GNN lifecycle runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from science.training.corpus_governance import STAGE_A_MAX_RESIDUES


class LossCoeffs(BaseModel):
    """Per-phase loss coefficients for gosp_loss_v6."""

    evidential_coeff: float = 0.001
    balance_coeff: float = 0.01
    cone_coeff: float = 0.15
    neighborhood_coeff: float = 0.25
    angular_coeff: float = 0.20
    domain_sep_2d_coeff: float = 0.30
    domain_sep_3d_coeff: float = 0.30
    cone_depth_anticollapse_coeff: float = 0.50
    shell_corr_coeff: float = 0.25
    cone_depth_min_std: float = 0.02
    shell_corr_depth_sasa_weight: float = 1.0
    shell_corr_epi_sasa_weight: float = 0.5
    shell_corr_proj_depth_weight: float = 1.0
    shell_corr_disc_spread_weight: float = 0.5
    shell_corr_disc_sasa_weight: float = 0.3
    disc_spread_min_std: float = 0.15
    proj_violation_coeff: float = 2.0
    disc_depth_scale_coeff: float = 0.0
    disc_depth_scale_target: float = 0.45
    disc_occupancy_coeff: float = 0.0
    disc_occupancy_min_sigma_ratio: float = 0.35
    disc_min_r_mean: float = 0.05
    disc_r_collapse_scale: float = 10.0
    disc_pc_repulsion_coeff: float = 0.0
    disc_pc2_min_std: float = 0.08
    disc_eff_rank_coeff: float = 0.0
    disc_eff_rank_min: float = 1.6
    disc_batch_diversity_coeff: float = 0.0
    disc_batch_min_pairwise_dist: float = 0.035
    rim_angular_repulsion_coeff: float = 0.0
    rim_angular_min_r: float = 0.35
    rim_angular_min_sep: float = 0.12
    rim_angular_spatial_exempt: float = 8.0
    rim_pc2_floor_coeff: float = 0.0
    rim_pc2_min_r: float = 0.35
    rim_pc2_min_std: float = 0.06
    # Soft angular-bin coverage: penalize under-filled θ sectors (empty-wedge pressure).
    disc_angular_coverage_coeff: float = 0.0
    disc_angular_coverage_min_r: float = 0.12
    disc_angular_coverage_n_bins: int = 12
    disc_angular_coverage_min_bin_frac: float = 0.40
    disc_angular_coverage_temperature: float = 0.20
    # Anti-barrier: untie crest experts from shared θ rays / recruit into empty bins.
    expert_angular_diversity_coeff: float = 0.0
    expert_angular_max_R: float = 0.55
    expert_angular_min_mean_sep: float = 0.55
    expert_sector_recruit_coeff: float = 0.0
    expert_sector_recruit_min_bin_frac: float = 0.25
    geometric_angular_fidelity_coeff: float = 0.0
    disc_path_align_coeff: float = 0.0
    disc_thickness_floor_coeff: float = 0.0
    disc_thickness_floor_min: float = 0.025
    disc_origin_span_floor_coeff: float = 0.0
    disc_origin_span_min_spread: float = 0.12
    x_hyp_thickness_floor_coeff: float = 0.0
    x_hyp_thickness_floor_min: float = 0.18
    shell_floor_coeff: float = 0.0
    shell_floor_min_r_depth_sasa: float = 0.60
    epistemic_decoupling_coeff: float = 0.0
    epi_ale_decorrelation_coeff: float = 0.0
    epistemic_bf_align_coeff: float = 0.22
    epistemic_sasa_pen_coeff: float = 0.246
    epistemic_anticollapse_coeff: float = 0.05
    epistemic_min_epi_std: float = 0.02
    routing_load_floor_coeff: float = 0.0
    routing_load_floor_min: float = 0.05
    routing_load_ceiling_coeff: float = 0.0
    routing_load_ceiling_max: float = 0.45
    # Mean-residue routing entropy sparsity: λ * mean_i H(p_i). Peak + warmup on
    # TrainingConfig; stage_runner writes the scheduled λ into this coeff each epoch.
    routing_entropy_sparsity_coeff: float = 0.0
    routing_entropy_sparsity_warmup_epochs: int = 8
    # Nearest-pair prototype repulsion: relu(m − min_{i<j} d_H(p_i, p_j)).
    prototype_repulsion_coeff: float = 0.0
    prototype_repulsion_margin: float = 0.25
    # Full-bank Gram logdet hinge: ReLU(τ − logdet(G+εI))² (pre-reg GRAM_COND_*).
    prototype_gram_logdet_coeff: float = 0.0
    prototype_gram_logdet_tau: float = -1.15
    # Majority-conditional committed-share hinge (local monopole; STE hard share).
    majority_committed_share_coeff: float = 0.0
    majority_committed_share_tau: float = 0.56
    majority_committed_share_commit_thr: float = 0.60
    majority_committed_share_min_n: int = 20
    # Core-only majority hinge (dehydron=0 eligible; pre-reg CORE_MAJORITY_SPLIT_*).
    core_majority_committed_share_coeff: float = 0.0
    core_majority_committed_share_tau: float = 0.56
    core_majority_committed_share_commit_thr: float = 0.60
    core_majority_committed_share_min_n: int = 20
    # Path 2 directionality asym reward (diam≤9 mask applied in train_loop).
    directionality_asym_coeff: float = 0.0
    pocket_bce_coeff: float = 0.0
    interface_bce_coeff: float = 0.0
    leak_bce_coeff: float = 0.0
    cone_target_mode: Literal["rho_wrap", "tau_dehydron_rim", "rho_rim"] = "rho_wrap"
    v3_aleatoric_shaping_coeff: float = 0.0
    w_var_penalty: float = 2.8
    w_aleatoric_hinge: float = 4.8
    aleatoric_hinge_target: float = 1.0
    aleatoric_shaping_holdout_fraction: float = 0.20
    aleatoric_shaping_holdout_mode: Literal["protein", "residue_stratified"] = "protein"


class PhaseConfig(BaseModel):
    """Single training phase (v6 MoE specialization schedule)."""

    phase: int
    name: str
    epochs: int
    lr: float
    freeze_radial: bool = False
    freeze_angular: bool = False
    freeze_backbone: bool = False
    freeze_gate: bool = False
    # Slim MoE: freeze specific expert MLPs by index (e.g. [2] locks the dominant expert).
    freeze_experts: list[int] = Field(default_factory=list)
    # Soft timeout overrides (None → stage_runner defaults: max_share=0.50, all experts).
    expert_timeout_max_share: float | None = None
    expert_timeout_eligible_experts: list[int] | None = None
    expert_dropout_p: float = 0.0
    freeze_radial_epochs: int = 0  # freeze radial for first N epochs within phase
    coeffs: LossCoeffs = Field(default_factory=LossCoeffs)
    # Optional linear coeff ramp (epoch 0 → coeff_ramp_epochs)
    coeff_ramp_epochs: int = 0
    angular_coeff_final: float | None = None
    shell_corr_disc_spread_weight_final: float | None = None
    disc_spread_min_std_final: float | None = None
    disc_depth_scale_target_final: float | None = None
    min_probe_r_depth_sasa: float | None = None  # abort if r(d,s) below for 2 epochs
    min_probe_r_depth_sasa_save: float | None = None  # ineligible v6_best if below
    min_disc_sigma2_sigma1_save: float | None = (
        None  # ineligible v6_best if disc streak
    )
    min_disc_r_std_save: float | None = None  # ineligible if radial spread collapsed
    min_disc_line_thickness_save: float | None = (
        None  # ineligible if rank-1 streak (visual)
    )
    min_disc_effective_rank_save: float | None = None
    # P2 bridge: relaxed routing save ceiling ramp (saturated P1 → standard P2)
    p2_bridge: bool = False
    routing_save_ceiling_start: float | None = None
    routing_save_ceiling_final: float | None = None
    routing_save_ceiling_ramp_epochs: int = 0
    expert_dropout_ramp_epochs: int = (
        0  # hold dropout at 0, then ramp to expert_dropout_p
    )
    angular_ramp_epochs: int = (
        0  # P2 bridge angular/domain ramp length (overrides default P2 warmup)
    )
    path_alignment_train: bool = False  # only train disc projection + gate disc readout
    rec_ablation_train: bool = (
        False  # only train radial_angular_fusion + hyp_proj_head_2d
    )
    projection_recovery_train: bool = (
        False  # train hyp_proj_head_2d + gate disc readout
    )
    fusion_path_recovery_train: bool = (
        False  # angular + fusion + disc proj + gate disc readout
    )
    lift_path_recovery_train: bool = (
        False  # radial + angular + fusion + disc (fix x_hyp wedge)
    )
    epistemic_decoupling_ramp_epochs: int = 0
    epistemic_bf_align_coeff_final: float | None = None
    epistemic_sasa_pen_coeff_final: float | None = None
    epistemic_uncertainty_only_train: bool = False
    aleatoric_only_train: bool = False
    gate_only_train: bool = False
    topology_gate_disc_recovery_train: bool = False
    topology_crescent_recovery_train: bool = False
    slim_moe_structural_ssot_train: bool = False
    # Option B staged decoupling: λ₁-only phase, then capped/log λ₂ ramp
    epistemic_staged_decoupling: bool = False
    epistemic_bf_only_epochs: int = 10
    epistemic_sasa_pen_cap: float = 0.05
    epistemic_sasa_pen_cap_epochs: int = 5
    max_probe_r_epi_ale_save: float | None = None
    max_probe_r_epi_sasa_save: float | None = None
    min_probe_r_epi_sasa_save: float | None = None  # reject inverted epi×SASA
    min_epistemic_std_save: float | None = None
    min_aleatoric_std_save: float | None = None
    require_tau_ale_elevation_save: bool = False
    require_g4_holdout_p8_save: bool = False
    # MoE save gates (slim SSOT v2): eval-mode hard routing + starvation
    min_eval_routing_fraction_save: float | None = None
    max_eval_routing_fraction_save: float | None = None
    max_expert_starvation_save: int | None = (
        None  # reject if starve > this (0 → starve≥1)
    )
    routing_entropy_min_save: float | None = None


def routing_save_max_for_epoch(
    phase_cfg: PhaseConfig,
    epoch: int,
    *,
    num_experts: int = 4,
) -> float | None:
    """Interpolate routing_H save ceiling across a P2 bridge phase."""
    if phase_cfg.routing_save_ceiling_start is None:
        return None
    from science.training.checkpoint_score import ROUTING_ENTROPY_SAVE_MAX
    from science.training.routing_gate_bounds import scale_routing_entropy_ceiling

    start = scale_routing_entropy_ceiling(
        phase_cfg.routing_save_ceiling_start, num_experts
    )
    final_raw = phase_cfg.routing_save_ceiling_final or ROUTING_ENTROPY_SAVE_MAX
    final = scale_routing_entropy_ceiling(final_raw, num_experts)
    n = phase_cfg.routing_save_ceiling_ramp_epochs or phase_cfg.epochs
    if n <= 1:
        return final
    t = min(epoch, n - 1) / (n - 1)
    return start + t * (final - start)


def routing_save_ceiling_for_display(
    phase_cfg: PhaseConfig,
    epoch: int,
    *,
    num_experts: int = 4,
) -> tuple[float, str]:
    """Return (value, label) for epoch logs — never NaN.

    ``route_ceil`` when the phase defines a save ramp; ``route_ref`` otherwise
    (P1 has no save ceiling — use promotion reference only).
    """
    ceiling = routing_save_max_for_epoch(phase_cfg, epoch, num_experts=num_experts)
    if ceiling is not None:
        return ceiling, "route_ceil"
    from science.training.checkpoint_score import ROUTING_ENTROPY_PROMOTE_MAX

    if phase_cfg.phase >= 2:
        return ROUTING_ENTROPY_PROMOTE_MAX + 0.2, "route_ref"
    return ROUTING_ENTROPY_PROMOTE_MAX, "route_ref"


class TrainingConfig(BaseModel):
    """Full v6 training run configuration."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str = "GOSPConeMapper-v6"
    gnn_lineage: Literal["v6", "v6.5", "v6.6"] = "v6"
    device: str = "cpu"
    lr: float = 5e-4
    hidden: int = 128
    num_layers: int = 6
    num_experts: int = 4
    # When set (from --seed), gate/prototype bank init uses an isolated RNG keyed
    # only by this value — stable across node_emb width (3 vs 4). See isolated_init.py.
    init_seed: int | None = None
    output_dir: Path = Path("checkpoints/v6/runs")
    pdb_dir: Path = Path("/tmp/dtie_pdb_cache")
    corpus_manifest: Path = Path("manifests/v6_corpus_120.json")
    phase: int | None = None  # None = full 3-phase curriculum
    resume: Path | None = None
    warm_start_v5: Path | None = None
    mlflow_experiment: str = "tokyo-eyes-v6"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    max_proteins: int | None = None
    max_residues: int = STAGE_A_MAX_RESIDUES
    topology_only_gate: bool = False
    gate_include_sasa: bool = False
    # Mean-residue routing entropy sparsity peak λ + linear warmup (stage_runner schedules).
    routing_entropy_sparsity_coeff: float = 0.0
    routing_entropy_sparsity_warmup_epochs: int = 8
    # Prototype nearest-pair repulsion (pre-reg PROTO_SEP_*); applied to all phases when >0.
    prototype_repulsion_coeff: float = 0.0
    prototype_repulsion_margin: float = 0.25
    # Full-bank Gram logdet hinge (pre-reg GRAM_COND_*); applied to all phases when >0.
    prototype_gram_logdet_coeff: float = 0.0
    prototype_gram_logdet_tau: float = -1.15
    # Majority-conditional committed-share hinge (pre-reg MAJORITY_SPLIT_*).
    majority_committed_share_coeff: float = 0.0
    majority_committed_share_tau: float = 0.56
    majority_committed_share_commit_thr: float = 0.60
    majority_committed_share_min_n: int = 20
    # Core-only majority hinge (pre-reg CORE_MAJORITY_SPLIT_*).
    core_majority_committed_share_coeff: float = 0.0
    core_majority_committed_share_tau: float = 0.56
    core_majority_committed_share_commit_thr: float = 0.60
    core_majority_committed_share_min_n: int = 20
    # Path 2: explicit directionality asym on diam≤9 (Move 3).
    directionality_asym_coeff: float = 0.0
    # Core capacity quotas (pre-reg CORE_QUOTA_*); 0 = off.
    core_capacity_quota_tau: float = 0.0
    hyperbolic_gate: bool = True
    hyperbolic_expert_mix: bool = False
    expert_dropout_p: float = 0.15
    capacity_threshold: float = 0.4
    min_usage: float = 0.05
    v2_teacher_checkpoint: Path | None = None
    v2_teacher_depth_coeff: float = 0.20
    v2_teacher_epistemic_coeff: float = 0.10
    epochs_override: int | None = None
    gentle_phase2: bool = False
    phase2_lr: float | None = None
    save_epoch_snapshots: bool = False
    enforce_stage_a_stop: bool = True
    routing_load_floor: bool = False
    routing_load_floor_coeff: float = 10.0
    routing_load_floor_min: float = 0.05
    p1b: bool = False
    p1b_lr: float = 1e-4
    p1c: bool = False
    p1c_lr: float = 1e-4
    p1c_v2_teacher_depth_coeff: float = 0.30
    p1d: bool = False
    p1d_lr: float = 1e-4
    p1d_extend: bool = False
    p1d_disc_depth_scale_coeff: float | None = None
    p1d_disc_target_start: float | None = None
    p1d_disc_target_end: float | None = None
    p1d_freeze_radial_epochs: int | None = None
    p1d_min_probe_r_depth_sasa: float | None = None
    p1d_disc_spread_min_std: float | None = None
    p2_bridge: bool = False
    p2_bridge_lr: float = 5e-5
    p2_bridge_ramp_epochs: int | None = None
    p2_hypmix: bool = False
    p2_hypmix3: bool = False
    p2_hypmix_final: bool = False
    p2_disc_occupancy: bool = False
    p2_disc_occupancy_v2: bool = False
    p2_disc_occupancy_v3: bool = False
    p2_disc_occupancy_v4: bool = False
    p2_disc_occupancy_v5: bool = False
    p2_disc_gentle_arch: bool = False
    p2_disc_path_align: bool = False
    p2_disc_path_align_coeff: float | None = None
    p2_rec_ablation: bool = False
    p2_disc_proj_recovery: bool = False
    p2_disc_proj_recovery_v2: bool = False
    p2_disc_proj_recovery_v3: bool = False
    p2_disc_proj_recovery_v4: bool = False
    p2_disc_proj_recovery_v5: bool = False
    p2_disc_eff_rank_coeff: float | None = None
    p2_disc_batch_diversity_coeff: float | None = None
    p2_disc_occupancy_coeff: float | None = None
    p2_radial_freeze_epochs: int | None = None
    p2_disc_r_std_floor: float | None = None
    p2_disc_line_thickness_floor: float | None = None
    disc_scatter_interval_epochs: int = 0
    disc_scatter_structure: str = "11QE:A"
    full_hyp_moe_test: bool = False
    deep_hyperbolic_gate: bool = False
    theory_test_lr: float = 3e-5
    gate_disc_scale: float = 1.0
    gate_gumbel: bool = False
    legacy_disc_projection: bool = False
    radial_angular_recombine: str = "multiply"
    disc_radial_source: str = "mobius"
    p4_epistemic_decoupling: bool = False
    p4_epistemic_lr: float = 1e-4
    epistemic_decoupling_holdouts: str = "1IVO,4MNE"
    epistemic_bf_align_coeff: float | None = None
    epistemic_sasa_pen_coeff: float | None = None
    shell_corr_epi_sasa_weight: float | None = None
    p4_epistemic_staged: bool = False
    p4_uncertainty_calibration: bool = False
    p4_head_decouple: bool = False
    p4_head_decouple_decorr_only: bool = False
    p4_v3_aleatoric_shaping: bool = False
    p4_g4_shaping_only_isolation: bool = False
    p4_g4_ale_only_unshaped: bool = False
    w_var_penalty: float | None = None
    p4_gate_promotion: bool = False
    p4_corpus25_gate_promotion: bool = False
    p4_corpus25_touchup_extended: bool = False
    p4_gate_uncertainty_touchup: bool = False
    p4_gate_balance_coeff: float | None = None
    p4_gate_load_floor_coeff: float | None = None
    p4_gate_load_floor_min: float | None = None
    decoupled_uncertainty_heads: bool = False
    max_probe_r_epi_sasa_save: float | None = None
    residue_stage1: bool = False
    residue_stage1_lr: float = 1e-4
    residue_stage1_epochs: int | None = None
    residue_stage2: bool = False
    residue_stage2_lr: float = 1e-4
    residue_stage2_epochs: int | None = None
    master_cold_lineage: bool = False
    v66_feeler_lineage: bool = False
    v66_feeler_angular_lift: bool = False
    v66_feeler_coupling: bool = False
    v66_feeler_no_exclusivity: bool = False
    v66_feeler_dehydron_angular: bool = False
    v66_feeler_rim_decouple: bool = False
    v66_feeler_p3_geom_edges: bool = False
    v66_feeler_p3_geom: bool = False
    v66_feeler_p3_geom_half_stack: bool = False
    v66_feeler_rim_fanout_model: bool = False
    v66_feeler_rim_fanout_cold_curriculum: bool = False
    v66_feeler_rim_fanout_polish: bool = False
    v66_feeler_rim_fanout_angular: bool = False
    v66_feeler_rim_fanout_angular_v2: bool = False
    v66_feeler_rim_fanout_radius: bool = False
    v66_feeler_rim_fanout_coverage: bool = False
    v66_feeler_rim_fanout_antibarrier: bool = False
    v66_feeler_rim_fanout_expert_arc: bool = False
    v66_feeler_geom_angular_prior: bool = False
    geometric_angular_prior: bool = False
    geometric_angular_kappa: float = 1.0
    geometric_angular_alpha: float = 0.7853981633974483  # π/4
    # S4: hyp disc k-NN for MP during training (no SSOT freeze). Exclusive with role_edge_mp.
    hyperbolic_mp_graph: bool = False
    # T1a: per-channel z-score of node features before node_emb (corpus-fit mean/std).
    input_feature_zscore: bool = False
    # T1a optional: replace binary tau_flag with |ρ−TAU| before z-score.
    replace_tau_with_abs_dist: bool = False
    # Gate inverse-temperature: init softplus(logit_scale) target (None = default  softplus(1)≈1.31).
    gate_logit_softplus_init: float | None = None
    # Minimum softplus(logit_scale) during forward (None = no floor).
    gate_logit_softplus_floor: float | None = None
    # PDB IDs forced first each epoch (spread anchors); empty = corpus order.
    epoch_anchor_pdb_ids: list[str] = Field(default_factory=list)
    rim_fanout_forward: bool = False
    rim_fanout_strength: float = 0.12
    rim_fanout_min_r: float = 0.35
    spoke_edge_scale: float = 1.0
    ribbon_edge_scale: float = 1.0
    # Training-only: append thermo affinity to edge_attr (type one-hots + ρ).
    # Does not touch GraphBuilder / Normalizer / fact_graph_edge.
    thermo_edge_features: bool = False
    # Training-only: per-relation radial MP (generic / wrapped / dehydron).
    # Requires thermo_edge_features for type labels; supersedes thermo gate.
    multi_rel_edge_mp: bool = False
    # Training-only: replace Cα contact graph with packing/dehydron/spoke/ribbon[/coupling].
    role_edge_mp: bool = False
    role_coupling_edges: bool = False
    # Training-only Chem-MVP: append disulf/covale rows from fact_covalent_bond.
    # Requires role_edge_mp; does not write Normalizer / fact_graph_edge.
    chem_edge_mp: bool = False
    # Training-only ha_edges_v1: heavy-atom packing existence + dehydron/packing strength aux.
    # Requires role_edge_mp; same relation IDs as chem-MVP; does not write Normalizer.
    ha_edge_mp: bool = False
    # Training-only Path B: SSE parent nodes + contain_up/down relations (9 total).
    # Requires chem_edge_mp; does not write Normalizer / fact_graph_edge.
    containment_edge_mp: bool = False
    # Training-only Euclidean reach: spatial shortcuts with hop>6 + seq≥10 (rel 7).
    # Requires chem_edge_mp; mutually exclusive with containment_edge_mp; no Normalizer writes.
    euclidean_shortcut_mp: bool = False
    dehydron_exclusivity: bool = True
    dehydron_angular_scale: float = 1.0
    dehydron_rim_recovery: bool = False
    dehydron_rim_recovery_lr: float = 2e-5
    topology_routing_recovery: bool = False
    topology_routing_recovery_lr: float = 3e-5
    topology_gate_disc_recovery: bool = False
    topology_gate_disc_recovery_lr: float = 2e-5
    topology_crescent_recovery: bool = False
    topology_crescent_recovery_lr: float = 2e-5
    expert_depth_decouple: bool = False
    structure_gate: bool = False
    track_v6_best_route: bool = False
    structural_disc_frozen: bool = False
    slim_moe_structural_ssot: bool = False
    use_dehydron_barcode: bool = False
    use_binned_dehydron: bool = False
    dehydron_barcode_dir: Path | None = None
    # Local witness barcode on dehydron role edges (not node-global broadcast).
    dehydron_edge_barcode: bool = False
    # Guardrails: refuse dead barcode+slim combos unless explicitly allowed.
    allow_dead_feature_channel: bool = False
    feature_liveness_probe: bool = False
    feature_liveness_fail_if_dead: bool = True
    feature_liveness_min_epochs: int = 2
    disc_layout_source: str | None = None

    def model_post_init(self, __context: object) -> None:
        self.output_dir = Path(self.output_dir)
        self.pdb_dir = Path(self.pdb_dir)
        self.corpus_manifest = Path(self.corpus_manifest)
        if self.resume is not None:
            self.resume = Path(self.resume)
        if self.v2_teacher_checkpoint is not None:
            self.v2_teacher_checkpoint = Path(self.v2_teacher_checkpoint)
        if self.dehydron_barcode_dir is not None:
            self.dehydron_barcode_dir = Path(self.dehydron_barcode_dir)

    def phase_preset_name(self) -> str | None:
        """Stable curriculum preset id for MLflow tags."""
        if self.v66_feeler_lineage:
            return "v66_feeler_p1"
        if self.slim_moe_structural_ssot:
            return "slim_moe_structural_ssot"
        if self.master_cold_lineage:
            return "stage_a_small_master_cold"
        if self.topology_routing_recovery:
            return "topology_routing_recovery"
        if self.topology_gate_disc_recovery:
            return "topology_gate_disc_recovery"
        if self.topology_crescent_recovery:
            return "topology_crescent_recovery"
        if self.dehydron_rim_recovery:
            return "dehydron_rim_recovery"
        if self.full_hyp_moe_test:
            return "full_hyp_moe_test"
        if self.residue_stage2:
            return "residue_stage2"
        if self.residue_stage1:
            return "residue_stage1"
        if self.p4_epistemic_decoupling:
            return "p4_epistemic_decoupling"
        if self.p4_uncertainty_calibration:
            return "p4_uncertainty_calibration"
        if self.p4_head_decouple_decorr_only:
            return "p4_head_decouple_decorr_only"
        if self.p4_v3_aleatoric_shaping:
            return "p4_v3_aleatoric_shaping"
        if self.p4_g4_shaping_only_isolation:
            return "p4_g4_shaping_only_isolation"
        if self.p4_g4_ale_only_unshaped:
            return "p4_g4_ale_only_unshaped"
        if self.p4_head_decouple:
            return "p4_head_decouple"
        if self.p4_corpus25_gate_promotion:
            return "p4_corpus25_gate_promotion"
        if self.p4_gate_promotion:
            return "p4_gate_promotion"
        if self.p4_corpus25_touchup_extended:
            return "p4_corpus25_touchup_extended"
        if self.p4_gate_uncertainty_touchup:
            return "p4_gate_uncertainty_touchup"
        if self.p2_hypmix_final:
            return "p2_hypmix_final"
        if (
            self.p2_disc_proj_recovery
            or self.p2_disc_proj_recovery_v2
            or self.p2_disc_proj_recovery_v3
            or self.p2_disc_proj_recovery_v4
            or self.p2_disc_proj_recovery_v5
        ):
            return "p2_disc_proj_recovery"
        if self.p2_rec_ablation:
            return "p2_rec_ablation"
        if self.p2_disc_path_align:
            return "p2_disc_path_align"
        if self.p2_disc_gentle_arch:
            return "p2_disc_gentle_arch"
        if self.p2_disc_occupancy_v5:
            return "p2_disc_occupancy_v5"
        if self.p2_disc_occupancy_v4:
            return "p2_disc_occupancy_v4"
        if self.p2_disc_occupancy_v3:
            return "p2_disc_occupancy_v3"
        if self.p2_disc_occupancy_v2:
            return "p2_disc_occupancy_v2"
        if self.p2_disc_occupancy:
            return "p2_disc_occupancy"
        if self.p2_hypmix3:
            return "p2_hypmix3"
        if self.p2_hypmix:
            return "p2_hypmix"
        if self.p2_bridge:
            return "p2_bridge"
        if self.p1d:
            return "p1d_extend" if self.p1d_extend else "p1d"
        if self.p1c:
            return "p1c"
        if self.p1b:
            return "p1b"
        if self.gentle_phase2:
            return "gentle_phase2"
        if self.phase is not None:
            return f"phase_{self.phase}"
        return "curriculum"

    def to_mlflow_params(self) -> dict[str, str | int | float | bool]:
        """Flatten config for MLflow param logging."""
        data = self.model_dump(mode="json")
        flat: dict[str, str | int | float | bool] = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, bool):
                flat[key] = "true" if value else "false"
            elif isinstance(value, Path):
                flat[key] = str(value)
            else:
                flat[key] = value
        return flat


def default_v6_phases(
    base_lr: float = 5e-4,
    *,
    gentle_phase2: bool = False,
    phase2_lr: float | None = None,
) -> list[PhaseConfig]:
    """Three-phase MoE specialization schedule per v6-moe spec."""
    p2_lr = (
        phase2_lr
        if phase2_lr is not None
        else (base_lr * 0.2 if gentle_phase2 else base_lr)
    )
    p2_radial_warmup = 5 if gentle_phase2 else 0
    return [
        PhaseConfig(
            phase=1,
            name="Phase 1: Representation stabilization",
            epochs=50,
            lr=base_lr,
            freeze_radial=False,
            freeze_angular=True,
            freeze_backbone=False,
            freeze_gate=False,
            expert_dropout_p=0.0,
            coeffs=LossCoeffs(
                balance_coeff=0.1,
                cone_coeff=0.30,
                neighborhood_coeff=0.10,
                angular_coeff=0.0,
                domain_sep_2d_coeff=0.0,
                domain_sep_3d_coeff=0.0,
                evidential_coeff=0.001,
                cone_depth_anticollapse_coeff=0.75,
                shell_corr_coeff=0.35,
            ),
        ),
        PhaseConfig(
            phase=2,
            name="Phase 2: Expert specialization",
            epochs=150,
            lr=p2_lr,
            freeze_radial=False,
            freeze_angular=False,
            freeze_backbone=False,
            freeze_gate=False,
            expert_dropout_p=0.15,
            freeze_radial_epochs=p2_radial_warmup,
            coeffs=LossCoeffs(
                balance_coeff=0.001,
                cone_coeff=0.25,
                neighborhood_coeff=0.20,
                angular_coeff=0.20,
                domain_sep_2d_coeff=0.15,
                domain_sep_3d_coeff=0.15,
                evidential_coeff=0.001,
                cone_depth_anticollapse_coeff=0.50,
                shell_corr_coeff=0.25,
                disc_occupancy_coeff=0.5,
                disc_occupancy_min_sigma_ratio=0.35,
            ),
        ),
        PhaseConfig(
            phase=3,
            name="Phase 3: Gate-locked expert fine-tune",
            epochs=50,
            lr=base_lr * 0.3,
            freeze_radial=False,
            freeze_angular=False,
            freeze_backbone=False,
            freeze_gate=True,
            expert_dropout_p=0.0,
            coeffs=LossCoeffs(
                balance_coeff=0.001,
                cone_coeff=0.15,
                neighborhood_coeff=0.25,
                angular_coeff=0.20,
                domain_sep_2d_coeff=0.30,
                domain_sep_3d_coeff=0.30,
                evidential_coeff=0.001,
                cone_depth_anticollapse_coeff=0.40,
                shell_corr_coeff=0.20,
            ),
        ),
    ]


def apply_routing_load_floor_phase2(
    phases: list[PhaseConfig],
    *,
    coeff: float,
    min_fraction: float = 0.05,
) -> list[PhaseConfig]:
    """Enable per-structure min-expert routing floor on Phase 2 only (Stage A contested routing)."""
    out: list[PhaseConfig] = []
    for phase_cfg in phases:
        if phase_cfg.phase != 2:
            out.append(phase_cfg)
            continue
        new_coeffs = phase_cfg.coeffs.model_copy(
            update={
                "routing_load_floor_coeff": coeff,
                "routing_load_floor_min": min_fraction,
            }
        )
        out.append(phase_cfg.model_copy(update={"coeffs": new_coeffs}))
    return out


def apply_prototype_nearest_pair_repulsion(
    phases: list[PhaseConfig],
    *,
    coeff: float,
    margin: float = 0.25,
) -> list[PhaseConfig]:
    """Enable nearest-pair prototype hyp-distance hinge on every phase (one coeff, no anneal)."""
    if coeff <= 0:
        return phases
    out: list[PhaseConfig] = []
    for phase_cfg in phases:
        new_coeffs = phase_cfg.coeffs.model_copy(
            update={
                "prototype_repulsion_coeff": float(coeff),
                "prototype_repulsion_margin": float(margin),
            }
        )
        out.append(phase_cfg.model_copy(update={"coeffs": new_coeffs}))
    return out


def apply_directionality_asym_reward(
    phases: list[PhaseConfig],
    *,
    coeff: float,
) -> list[PhaseConfig]:
    """Enable Path 2 directionality asym reward on every phase (diam mask in train_loop)."""
    if coeff <= 0:
        return phases
    out: list[PhaseConfig] = []
    for phase_cfg in phases:
        new_coeffs = phase_cfg.coeffs.model_copy(
            update={"directionality_asym_coeff": float(coeff)}
        )
        out.append(phase_cfg.model_copy(update={"coeffs": new_coeffs}))
    return out


def apply_prototype_gram_logdet_hinge(
    phases: list[PhaseConfig],
    *,
    coeff: float,
    tau_logdet: float = -1.15,
) -> list[PhaseConfig]:
    """Enable saturating full-bank Gram logdet hinge on every phase (one coeff)."""
    if coeff <= 0:
        return phases
    out: list[PhaseConfig] = []
    for phase_cfg in phases:
        new_coeffs = phase_cfg.coeffs.model_copy(
            update={
                "prototype_gram_logdet_coeff": float(coeff),
                "prototype_gram_logdet_tau": float(tau_logdet),
            }
        )
        out.append(phase_cfg.model_copy(update={"coeffs": new_coeffs}))
    return out


def apply_majority_committed_share_hinge(
    phases: list[PhaseConfig],
    *,
    coeff: float,
    tau: float = 0.56,
    commit_thr: float = 0.60,
    min_n: int = 20,
) -> list[PhaseConfig]:
    """Enable majority-conditional committed-share hinge on every phase (one coeff)."""
    if coeff <= 0:
        return phases
    out: list[PhaseConfig] = []
    for phase_cfg in phases:
        new_coeffs = phase_cfg.coeffs.model_copy(
            update={
                "majority_committed_share_coeff": float(coeff),
                "majority_committed_share_tau": float(tau),
                "majority_committed_share_commit_thr": float(commit_thr),
                "majority_committed_share_min_n": int(min_n),
            }
        )
        out.append(phase_cfg.model_copy(update={"coeffs": new_coeffs}))
    return out


def apply_core_majority_committed_share_hinge(
    phases: list[PhaseConfig],
    *,
    coeff: float,
    tau: float = 0.56,
    commit_thr: float = 0.60,
    min_n: int = 20,
) -> list[PhaseConfig]:
    """Enable core-only (dehydron=0) committed-share hinge on every phase."""
    if coeff <= 0:
        return phases
    out: list[PhaseConfig] = []
    for phase_cfg in phases:
        new_coeffs = phase_cfg.coeffs.model_copy(
            update={
                "core_majority_committed_share_coeff": float(coeff),
                "core_majority_committed_share_tau": float(tau),
                "core_majority_committed_share_commit_thr": float(commit_thr),
                "core_majority_committed_share_min_n": int(min_n),
            }
        )
        out.append(phase_cfg.model_copy(update={"coeffs": new_coeffs}))
    return out


def apply_core_capacity_quota_config(
    config: TrainingConfig,
    *,
    tau_cap: float = 0.40,
) -> TrainingConfig:
    """Enable train-time core capacity quotas (pre-reg CORE_QUOTA_*)."""
    return config.model_copy(update={"core_capacity_quota_tau": float(tau_cap)})


def dehydron_rim_recovery_phase_config(
    lr: float = 2e-5,
    epochs: int = 12,
    *,
    min_disc_line_thickness_save: float = 0.025,
) -> PhaseConfig:
    """Warm-start off production v6: τ→rim cone, SASA shell off, backbone+gate frozen."""
    return PhaseConfig(
        phase=1,
        name="Dehydron rim recovery (τ→cone, SASA shell off)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=True,
        freeze_gate=True,
        expert_dropout_p=0.0,
        min_disc_line_thickness_save=min_disc_line_thickness_save,
        min_disc_sigma2_sigma1_save=0.35,
        coeffs=LossCoeffs(
            balance_coeff=0.02,
            cone_coeff=0.45,
            neighborhood_coeff=0.08,
            angular_coeff=0.05,
            cone_target_mode="tau_dehydron_rim",
            cone_depth_anticollapse_coeff=0.75,
            cone_depth_min_std=0.06,
            shell_corr_coeff=0.12,
            shell_corr_depth_sasa_weight=0.0,
            shell_corr_disc_sasa_weight=0.0,
            shell_floor_coeff=0.0,
            disc_occupancy_coeff=0.8,
            disc_occupancy_min_sigma_ratio=0.35,
        ),
    )


def topology_routing_recovery_phase_config(
    lr: float = 3e-5,
    epochs: int = 40,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
) -> PhaseConfig:
    """P2 extension: Gumbel routing + expert depth decouple + structure gate."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else epochs
    )
    return PhaseConfig(
        phase=2,
        name="Topology P2 routing recovery",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.12,
        expert_dropout_ramp_epochs=6,
        freeze_radial_epochs=2,
        coeff_ramp_epochs=min(ramp, 15),
        angular_coeff_final=0.18,
        p2_bridge=True,
        routing_save_ceiling_start=1.20,
        routing_save_ceiling_final=1.14,
        routing_save_ceiling_ramp_epochs=ramp,
        angular_ramp_epochs=min(ramp, 15),
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=LossCoeffs(
            balance_coeff=0.001,
            cone_coeff=0.22,
            cone_target_mode="tau_dehydron_rim",
            neighborhood_coeff=0.10,
            angular_coeff=0.10,
            domain_sep_2d_coeff=0.08,
            domain_sep_3d_coeff=0.08,
            evidential_coeff=0.0003,
            cone_depth_anticollapse_coeff=0.55,
            cone_depth_min_std=0.08,
            shell_corr_coeff=0.12,
            shell_corr_depth_sasa_weight=0.0,
            shell_corr_disc_sasa_weight=0.0,
            shell_corr_epi_sasa_weight=0.0,
            shell_floor_coeff=0.0,
            routing_load_floor_coeff=0.12,
            routing_load_floor_min=0.08,
            disc_occupancy_coeff=0.35,
            disc_occupancy_min_sigma_ratio=0.35,
            disc_depth_scale_coeff=1.2,
            disc_depth_scale_target=0.40,
        ),
    )


def apply_topology_routing_recovery_config(config: TrainingConfig) -> TrainingConfig:
    """Topology lineage + MoE routing recovery (resume-compatible, keep N=4).

    Avoid Gumbel-hard and expert-count expansion on ep169 resume — both raised H and
    starved experts in route_v2. Use structure_gate + depth decouple at 4 experts first;
    try NUM_EXPERTS=6 only as a fresh cold start.
    """
    return apply_master_cold_dehydron_config(config).model_copy(
        update={
            "gate_gumbel": False,
            "expert_depth_decouple": True,
            "structure_gate": True,
            "track_v6_best_route": True,
        }
    )


def topology_gate_disc_recovery_phase_config(
    lr: float = 2e-5,
    epochs: int = 20,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
) -> PhaseConfig:
    """Gate + light disc projection touch-up; depth biases frozen (post route_v3)."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else epochs
    )
    return PhaseConfig(
        phase=2,
        name="Topology gate + disc recovery",
        epochs=epochs,
        lr=lr,
        freeze_radial=True,
        freeze_angular=True,
        freeze_backbone=True,
        freeze_gate=False,
        topology_gate_disc_recovery_train=True,
        expert_dropout_p=0.08,
        expert_dropout_ramp_epochs=3,
        p2_bridge=True,
        routing_save_ceiling_start=1.22,
        routing_save_ceiling_final=1.14,
        routing_save_ceiling_ramp_epochs=ramp,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        coeffs=LossCoeffs(
            balance_coeff=0.001,
            cone_coeff=0.05,
            cone_target_mode="tau_dehydron_rim",
            neighborhood_coeff=0.04,
            angular_coeff=0.0,
            domain_sep_2d_coeff=0.0,
            domain_sep_3d_coeff=0.0,
            evidential_coeff=0.0,
            cone_depth_anticollapse_coeff=0.0,
            shell_corr_coeff=0.08,
            shell_corr_depth_sasa_weight=0.0,
            shell_corr_disc_sasa_weight=0.0,
            shell_floor_coeff=0.0,
            routing_load_floor_coeff=0.12,
            routing_load_floor_min=0.08,
            disc_occupancy_coeff=0.35,
            disc_occupancy_min_sigma_ratio=0.32,
            disc_depth_scale_coeff=0.0,
        ),
    )


def apply_topology_gate_disc_recovery_config(config: TrainingConfig) -> TrainingConfig:
    """Resume route_v3: train gate + hyp_proj_2d only; freeze expert_depth_bias."""
    base = apply_topology_routing_recovery_config(config)
    return base.model_copy(
        update={
            "topology_routing_recovery": False,
            "topology_gate_disc_recovery": True,
        }
    )


def topology_crescent_recovery_phase_config(
    lr: float = 2e-5,
    epochs: int = 25,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_thickness_floor_min: float = 0.03,
) -> PhaseConfig:
    """
    Open a collapsed 1D crescent without moving burial depth.

    Trains angular_head + fusion + hyp_proj_2d (+ gate disc readout); radial frozen.
    Strong PC2/thickness/eff_rank floors spread mass perpendicular to the streak.
    """
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else epochs
    )
    base = p2_disc_proj_recovery_v3_phase_config(
        lr=lr,
        epochs=epochs,
        disc_thickness_floor_min=disc_thickness_floor_min,
        disc_thickness_floor_coeff=6.0,
    )
    return base.model_copy(
        update={
            "name": "Topology crescent recovery (angular + disc wedge)",
            "topology_crescent_recovery_train": True,
            "fusion_path_recovery_train": False,
            "projection_recovery_train": False,
            "freeze_gate": True,
            "min_disc_line_thickness_save": disc_thickness_floor_min,
            "min_disc_effective_rank_save": 1.35,
            "min_disc_sigma2_sigma1_save": 0.22,
            "routing_save_ceiling_start": 1.22,
            "routing_save_ceiling_final": 1.14,
            "routing_save_ceiling_ramp_epochs": ramp,
            "p2_bridge": True,
            "coeffs": base.coeffs.model_copy(
                update={
                    "cone_coeff": 0.08,
                    "cone_target_mode": "tau_dehydron_rim",
                    "cone_depth_anticollapse_coeff": 0.35,
                    "angular_coeff": 0.14,
                    "neighborhood_coeff": 0.08,
                    "shell_corr_coeff": 0.08,
                    "shell_corr_depth_sasa_weight": 0.0,
                    "shell_corr_disc_sasa_weight": 0.0,
                    "disc_occupancy_coeff": 0.45,
                    "disc_occupancy_min_sigma_ratio": 0.32,
                    "disc_pc_repulsion_coeff": 2.5,
                    "disc_pc2_min_std": 0.07,
                    "disc_eff_rank_coeff": 1.2,
                    "disc_eff_rank_min": 1.35,
                    "disc_batch_diversity_coeff": 0.8,
                    "disc_batch_min_pairwise_dist": 0.03,
                    "disc_thickness_floor_coeff": 6.0,
                    "disc_thickness_floor_min": disc_thickness_floor_min,
                    "disc_origin_span_floor_coeff": 2.5,
                    "disc_origin_span_min_spread": 0.14,
                    "balance_coeff": 0.0,
                    "routing_load_floor_coeff": 0.0,
                }
            ),
        }
    )


def apply_topology_crescent_recovery_config(config: TrainingConfig) -> TrainingConfig:
    """Resume after route_v3 / n6: spread angular wedge; keep radial depth + routing frozen."""
    base = apply_topology_routing_recovery_config(config)
    return base.model_copy(
        update={
            "topology_routing_recovery": False,
            "topology_crescent_recovery": True,
        }
    )


def apply_master_cold_dehydron_phases(phases: list[PhaseConfig]) -> list[PhaseConfig]:
    """Dehydron-rim cone contract for MASTER cold-start lineage (P_DEHYDRON_CONE_01)."""
    out: list[PhaseConfig] = []
    for phase_cfg in phases:
        coeff_updates: dict[str, float | str] = {
            "cone_target_mode": "tau_dehydron_rim",
            "shell_corr_depth_sasa_weight": 0.0,
            "shell_corr_disc_sasa_weight": 0.0,
            "shell_corr_epi_sasa_weight": 0.0,
            "shell_floor_coeff": 0.0,
            "epistemic_sasa_pen_coeff": 0.0,
        }
        if phase_cfg.phase == 1:
            coeff_updates["shell_corr_coeff"] = 0.12
        if phase_cfg.phase == 2:
            coeff_updates.update(
                {
                    "disc_thickness_floor_coeff": max(
                        phase_cfg.coeffs.disc_thickness_floor_coeff, 1.2
                    ),
                    "disc_thickness_floor_min": max(
                        phase_cfg.coeffs.disc_thickness_floor_min, 0.022
                    ),
                    "disc_pc_repulsion_coeff": max(
                        phase_cfg.coeffs.disc_pc_repulsion_coeff, 0.35
                    ),
                    "disc_eff_rank_coeff": max(
                        phase_cfg.coeffs.disc_eff_rank_coeff, 0.4
                    ),
                }
            )
        new_coeffs = phase_cfg.coeffs.model_copy(update=coeff_updates)
        phase_updates: dict[str, Any] = {
            "coeffs": new_coeffs,
            "min_probe_r_depth_sasa": None,
            "min_probe_r_depth_sasa_save": None,
            "max_probe_r_epi_sasa_save": None,
        }
        if phase_cfg.phase == 2:
            phase_updates.update(
                {
                    "routing_save_ceiling_start": 1.35,
                    "routing_save_ceiling_final": 1.21,
                    "routing_save_ceiling_ramp_epochs": min(phase_cfg.epochs, 20),
                    "min_disc_line_thickness_save": 0.022,
                }
            )
        out.append(phase_cfg.model_copy(update=phase_updates))
    return out


def apply_master_cold_dehydron_config(config: TrainingConfig) -> TrainingConfig:
    """Model/training overrides for MASTER cold dehydron ablation (topology-only gate, no V2 teacher)."""
    return config.model_copy(
        update={
            "topology_only_gate": True,
            "v2_teacher_checkpoint": None,
            "v2_teacher_depth_coeff": 0.0,
            "v2_teacher_epistemic_coeff": 0.0,
            "structural_disc_frozen": False,
            "disc_layout_source": "gnn_learned",
        }
    )


def _v66_feeler_base_coeffs(**overrides: float | str) -> LossCoeffs:
    """Minimal feeler whitelist — evidential/shell/epistemic/BCE off."""
    base = dict(
        evidential_coeff=0.0,
        balance_coeff=0.05,
        cone_coeff=0.30,
        neighborhood_coeff=0.10,
        angular_coeff=0.0,
        domain_sep_2d_coeff=0.0,
        domain_sep_3d_coeff=0.0,
        cone_depth_anticollapse_coeff=0.50,
        shell_corr_coeff=0.0,
        shell_corr_depth_sasa_weight=0.0,
        shell_corr_epi_sasa_weight=0.0,
        shell_corr_proj_depth_weight=0.0,
        shell_corr_disc_spread_weight=0.0,
        shell_corr_disc_sasa_weight=0.0,
        proj_violation_coeff=2.0,
        disc_depth_scale_coeff=0.0,
        disc_occupancy_coeff=0.35,
        disc_occupancy_min_sigma_ratio=0.35,
        disc_pc_repulsion_coeff=0.25,
        disc_pc2_min_std=0.08,
        disc_eff_rank_coeff=0.0,
        disc_batch_diversity_coeff=0.0,
        disc_path_align_coeff=0.0,
        disc_thickness_floor_coeff=0.80,
        disc_thickness_floor_min=0.022,
        disc_origin_span_floor_coeff=0.0,
        x_hyp_thickness_floor_coeff=0.0,
        shell_floor_coeff=0.0,
        epistemic_decoupling_coeff=0.0,
        epi_ale_decorrelation_coeff=0.0,
        epistemic_bf_align_coeff=0.0,
        epistemic_sasa_pen_coeff=0.0,
        epistemic_anticollapse_coeff=0.0,
        routing_load_floor_coeff=0.0,
        routing_load_ceiling_coeff=0.0,
        pocket_bce_coeff=0.0,
        interface_bce_coeff=0.0,
        leak_bce_coeff=0.0,
        v3_aleatoric_shaping_coeff=0.0,
        cone_target_mode="rho_rim",
    )
    base.update(overrides)
    return LossCoeffs(**base)


def apply_v66_feeler_phases(
    phases: list[PhaseConfig] | None = None,
    *,
    base_lr: float = 5e-4,
    epochs: int = 20,
    p2_epochs: int | None = None,
    p3_epochs: int | None = None,
    p4_epochs: int | None = None,
) -> list[PhaseConfig]:
    """v6.6 feeler curriculum: P1 radial → P2 angular → P3 edge barcode → P4 rim fan-out.

    Expert timeout (share>45% → ban 1 epoch, all experts) is the anti-dominance
    lever — routing floor/ceiling coeffs stay off.
    """
    del phases  # feeler replaces the curriculum entirely
    p2_ep = p2_epochs if p2_epochs is not None else epochs
    p3_ep = p3_epochs if p3_epochs is not None else epochs
    p4_ep = p4_epochs if p4_epochs is not None else max(10, epochs // 2)
    common = dict(
        freeze_radial=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
    )
    return [
        PhaseConfig(
            phase=1,
            name="Phase 1: v6.6 feeler (minimal losses, timeout@45%)",
            epochs=epochs,
            lr=base_lr,
            freeze_angular=True,
            coeffs=_v66_feeler_base_coeffs(),
            **common,
        ),
        PhaseConfig(
            phase=2,
            name="Phase 2: v6.6 feeler (unfreeze angular, soft routing)",
            epochs=p2_ep,
            lr=base_lr * 0.5,
            freeze_angular=False,
            # Light angular — open disc occupancy without crushing router_H.
            # Softer balance so experts can commit past ~25% soft share.
            coeffs=_v66_feeler_base_coeffs(
                balance_coeff=0.02,
                cone_coeff=0.15,
                neighborhood_coeff=0.15,
                angular_coeff=0.10,
                cone_depth_anticollapse_coeff=0.30,
                disc_occupancy_coeff=0.45,
                disc_pc_repulsion_coeff=0.35,
            ),
            **common,
        ),
        PhaseConfig(
            phase=3,
            name="Phase 3: v6.6 feeler (dehydron edge barcode, no disc occupancy)",
            epochs=p3_ep,
            lr=base_lr * 0.25,
            freeze_angular=False,
            # P2 recipe + local edge barcode; zero disc occupancy stack (crescent trap).
            coeffs=_v66_feeler_base_coeffs(
                balance_coeff=0.02,
                cone_coeff=0.15,
                neighborhood_coeff=0.15,
                angular_coeff=0.10,
                cone_depth_anticollapse_coeff=0.30,
                disc_occupancy_coeff=0.0,
                disc_pc_repulsion_coeff=0.0,
                disc_thickness_floor_coeff=0.0,
                disc_origin_span_floor_coeff=0.0,
                disc_eff_rank_coeff=0.0,
            ),
            **common,
        ),
        PhaseConfig(
            phase=4,
            name="Phase 4: v6.6 feeler (rim fan-out, edge barcode)",
            epochs=p4_ep,
            lr=base_lr * 0.20,
            freeze_angular=False,
            # P3 recipe + rim-only angular/PC2 spread (no global disc occupancy).
            coeffs=_v66_feeler_base_coeffs(
                balance_coeff=0.02,
                cone_coeff=0.15,
                neighborhood_coeff=0.15,
                angular_coeff=0.10,
                cone_depth_anticollapse_coeff=0.30,
                disc_occupancy_coeff=0.0,
                disc_pc_repulsion_coeff=0.0,
                disc_thickness_floor_coeff=0.0,
                disc_origin_span_floor_coeff=0.0,
                disc_eff_rank_coeff=0.0,
                rim_angular_repulsion_coeff=0.25,
                rim_angular_min_r=0.35,
                rim_angular_min_sep=0.12,
                rim_pc2_floor_coeff=0.15,
                rim_pc2_min_r=0.35,
                rim_pc2_min_std=0.06,
            ),
            **common,
        ),
    ]


def v66_feeler_angular_lift_phase_config(
    *,
    lr: float = 1.25e-4,
    epochs: int = 15,
) -> PhaseConfig:
    """v6.6 feeler: angular_lift + lift-path recovery; P3 recipe without disc occupancy."""
    return PhaseConfig(
        phase=5,
        name="Phase 5: v6.6 feeler angular_lift (lift-path recovery, no disc occupancy)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=True,
        freeze_gate=True,
        expert_dropout_p=0.0,
        lift_path_recovery_train=True,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_base_coeffs(
            balance_coeff=0.02,
            cone_coeff=0.15,
            neighborhood_coeff=0.15,
            angular_coeff=0.10,
            cone_depth_anticollapse_coeff=0.30,
            disc_occupancy_coeff=0.0,
            disc_pc_repulsion_coeff=0.0,
            disc_thickness_floor_coeff=0.0,
            disc_origin_span_floor_coeff=0.0,
            disc_eff_rank_coeff=0.0,
            x_hyp_thickness_floor_coeff=0.0,
        ),
    )


def _v66_feeler_p3_coeffs() -> LossCoeffs:
    return _v66_feeler_base_coeffs(
        balance_coeff=0.02,
        cone_coeff=0.15,
        neighborhood_coeff=0.15,
        angular_coeff=0.10,
        cone_depth_anticollapse_coeff=0.30,
        disc_occupancy_coeff=0.0,
        disc_pc_repulsion_coeff=0.0,
        disc_thickness_floor_coeff=0.0,
        disc_origin_span_floor_coeff=0.0,
        disc_eff_rank_coeff=0.0,
    )


def _v66_feeler_p3_geom_coeffs(*, stack_scale: float = 1.0) -> LossCoeffs:
    """P3 geometry-fill recipe (angular span + 2D occupancy) from feeler_expand_23_p3_geom_v1."""
    s = stack_scale
    return _v66_feeler_base_coeffs(
        balance_coeff=0.015,
        cone_coeff=0.10,
        neighborhood_coeff=0.10,
        angular_coeff=0.15,
        cone_depth_anticollapse_coeff=0.25,
        disc_occupancy_coeff=0.65 * s,
        disc_occupancy_min_sigma_ratio=0.40,
        disc_pc_repulsion_coeff=0.50 * s,
        disc_pc2_min_std=0.10,
        disc_thickness_floor_coeff=1.20 * s,
        disc_thickness_floor_min=0.028,
        disc_origin_span_floor_coeff=2.5 * s,
        disc_origin_span_min_spread=0.18,
        disc_eff_rank_coeff=0.35 * s,
        disc_eff_rank_min=1.55,
    )


def v66_feeler_p3_geom_phase_config(
    *,
    lr: float = 1.25e-4,
    epochs: int = 20,
    stack_scale: float = 1.0,
) -> PhaseConfig:
    """Continue p3_geom champion — angular span + 2D fill, no barcode."""
    label = "half occupancy stack" if stack_scale < 1.0 else "no barcode"
    return PhaseConfig(
        phase=11,
        name=f"Phase 11: v6.6 feeler (P3 geom continue, {label})",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_radial_epochs=3,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_p3_geom_coeffs(stack_scale=stack_scale),
    )


def v66_feeler_p3_geom_edges_phase_config(
    *,
    lr: float = 1.0e-4,
    epochs: int = 15,
) -> PhaseConfig:
    """Resume p3_geom champion + local dehydron edge barcode (geometry losses on)."""
    return PhaseConfig(
        phase=10,
        name="Phase 10: v6.6 feeler (P3 geom + dehydron edge barcode)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_radial_epochs=3,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_p3_geom_coeffs(),
    )


def v66_feeler_no_exclusivity_phase_config(
    *,
    lr: float = 1.0e-4,
    epochs: int = 12,
) -> PhaseConfig:
    """v6.6 feeler: allow packing/ribbon on dehydron pairs (P3 recipe)."""
    return PhaseConfig(
        phase=7,
        name="Phase 7: v6.6 feeler no dehydron exclusivity (multi-relation pairs)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_p3_coeffs(),
    )


def v66_feeler_dehydron_angular_phase_config(
    *,
    lr: float = 1.0e-4,
    epochs: int = 12,
) -> PhaseConfig:
    """v6.6 feeler: weaken dehydron angular SH (P3 recipe)."""
    return PhaseConfig(
        phase=8,
        name="Phase 8: v6.6 feeler dehydron angular scale (radial-strong, θ-weak)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_p3_coeffs(),
    )


def v66_feeler_rim_decouple_phase_config(
    *,
    lr: float = 1.0e-4,
    epochs: int = 12,
) -> PhaseConfig:
    """v6.6 feeler: no dehydron exclusivity + weakened dehydron angular SH."""
    return PhaseConfig(
        phase=9,
        name="Phase 9: v6.6 feeler rim decouple (multi-rel pairs + dehydron θ-scale)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_p3_coeffs(),
    )


def v66_feeler_coupling_phase_config(
    *,
    lr: float = 1.0e-4,
    epochs: int = 12,
) -> PhaseConfig:
    """v6.6 feeler: cross-subgraph coupling edges; P3 recipe without disc occupancy."""
    return PhaseConfig(
        phase=6,
        name="Phase 6: v6.6 feeler coupling (propagation edges, no disc occupancy)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_p3_coeffs(),
    )


def v66_feeler_rim_fanout_model_phase_config(
    *,
    lr: float = 1.0e-4,
    epochs: int = 12,
) -> PhaseConfig:
    """Model-level rim fan-out + spoke/ribbon boost.

    Rim angular spread is handled in forward (RimFanoutSpread), not loss-only rim_* terms.
    Disc occupancy / thickness / origin-span floors stay on — forward fan-out is a no-op
    until r >= min_r_rim, so cold starts still need 2D spread pressure.
    """
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out forward + spoke/ribbon boost)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=_v66_feeler_p3_geom_coeffs(stack_scale=1.0),
    )


def v66_feeler_rim_fanout_warm_phase_config(
    *,
    lr: float = 1.0e-4,
    epochs: int = 15,
) -> PhaseConfig:
    """Warm P4 resume → P12: forward rim fan-out on a disc that already reaches the rim.

    Half P3 geom stack preserves probe_r_proj_depth; loss-only rim_* terms backstop
    RimFanoutSpread on 4OBE where rim_frac is already ~25–50%.
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.5).model_copy(
        update={
            "rim_angular_repulsion_coeff": 0.25,
            "rim_angular_min_r": 0.35,
            "rim_angular_min_sep": 0.12,
            "rim_pc2_floor_coeff": 0.15,
            "rim_pc2_min_r": 0.35,
            "rim_pc2_min_std": 0.06,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out warm, P4 resume)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_rim_fanout_polish_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 10,
) -> PhaseConfig:
    """Continue warm rim-fanout from a mid-run sweet spot without burning probe.

    Quarter P3 geom stack + disc_depth_scale to re-lock r↔depth; lighter rim_* losses.
    Intended resume: p4_v2 epoch_062 (probe≈0.50, σ₂/σ₁≈0.83).
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.25).model_copy(
        update={
            "disc_depth_scale_coeff": 1.0,
            "disc_depth_scale_target": 0.45,
            "rim_angular_repulsion_coeff": 0.15,
            "rim_angular_min_r": 0.35,
            "rim_angular_min_sep": 0.12,
            "rim_pc2_floor_coeff": 0.10,
            "rim_pc2_min_r": 0.35,
            "rim_pc2_min_std": 0.06,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out polish, depth-lock)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_rim_fanout_angular_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 8,
) -> PhaseConfig:
    """Short angular-fill polish: close blank Poincaré wedges before radius push.

    Mid-disc rim_* floors (min_r=0.20) + stronger occupancy/PC2/origin-span, with
    disc_depth_scale keeping probe_r_proj_depth armed under the feeler probe guard.
    Intended resume: polish_v1 latest (probe≈0.90).
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.75).model_copy(
        update={
            "angular_coeff": 0.20,
            "disc_depth_scale_coeff": 1.2,
            "disc_depth_scale_target": 0.45,
            "rim_angular_repulsion_coeff": 0.35,
            "rim_angular_min_r": 0.20,
            "rim_angular_min_sep": 0.10,
            "rim_pc2_floor_coeff": 0.25,
            "rim_pc2_min_r": 0.20,
            "rim_pc2_min_std": 0.08,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out angular-fill)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_rim_fanout_angular_v2_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 8,
) -> PhaseConfig:
    """Angular-fill v2: close remaining ~30° wedges; min_r=0.12 + stronger rim_*.

    Resume from angular_v1 ep80 (probe≈0.92). Keep disc_depth_scale ≥1.2 so the
    feeler probe guard stays meaningful. Success: 1F88 sparse wedge ≤~20° or stall.
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.75).model_copy(
        update={
            "angular_coeff": 0.22,
            "disc_depth_scale_coeff": 1.25,
            "disc_depth_scale_target": 0.45,
            "rim_angular_repulsion_coeff": 0.40,
            "rim_angular_min_r": 0.12,
            "rim_angular_min_sep": 0.09,
            "rim_pc2_floor_coeff": 0.30,
            "rim_pc2_min_r": 0.12,
            "rim_pc2_min_std": 0.08,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out angular-fill v2)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_rim_fanout_radius_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 8,
) -> PhaseConfig:
    """Radius push after angular-fill: raise depth→r target; hold occupancy stack.

    Resume from angular_v2 ep88 (probe≈0.95). disc_depth_scale_target 0.45→0.55 is the
    main lever; rim_* stay at angular_v2 (wedge lock) without cranking occupancy.
    Keep disc_depth_scale_coeff ≥1.2 so the feeler probe guard stays armed.
    Gate: rim_frac ↑ (~40% on 4OBE/1F88) with probe ≥ baseline−0.08; don't reopen wedge.
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.75).model_copy(
        update={
            "angular_coeff": 0.22,
            "disc_depth_scale_coeff": 1.25,
            "disc_depth_scale_target": 0.55,
            "rim_angular_repulsion_coeff": 0.40,
            "rim_angular_min_r": 0.12,
            "rim_angular_min_sep": 0.09,
            "rim_pc2_floor_coeff": 0.30,
            "rim_pc2_min_r": 0.12,
            "rim_pc2_min_std": 0.08,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out radius push)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_rim_fanout_coverage_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 8,
) -> PhaseConfig:
    """Angular coverage: soft empty-sector floor; hold depth/probe lock.

    Resume from radius_v1 ep96 (probe≈0.97). Adds disc_angular_coverage so mass can
    enter blank θ wedges (rim repulsion cannot invent new rays). Keeps
    disc_depth_scale ≥1.2 + target 0.55. Gate: 1F88 largest gap ↓ with probe ≥ baseline−0.08.
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.75).model_copy(
        update={
            "angular_coeff": 0.22,
            "disc_depth_scale_coeff": 1.25,
            "disc_depth_scale_target": 0.55,
            "rim_angular_repulsion_coeff": 0.35,
            "rim_angular_min_r": 0.12,
            "rim_angular_min_sep": 0.09,
            "rim_pc2_floor_coeff": 0.25,
            "rim_pc2_min_r": 0.12,
            "rim_pc2_min_std": 0.08,
            "disc_angular_coverage_coeff": 0.45,
            "disc_angular_coverage_min_r": 0.12,
            "disc_angular_coverage_n_bins": 12,
            "disc_angular_coverage_min_bin_frac": 0.40,
            "disc_angular_coverage_temperature": 0.20,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out angular coverage)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_rim_fanout_antibarrier_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 25,
) -> PhaseConfig:
    """Anti-barrier: untie crest experts from shared θ rays; recruit into empty bins.

    Resume from coverage_v2 ep164. Eases global coverage slightly; adds expert circular
    diversity (R ceiling + mean separation) and per-expert floors on underfilled bins.
    Hold disc_depth_scale. Gate: gap ↓, e1 wall_share ↓, r̄ ≳ 0.18, probe ≥ baseline−0.08.
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.75).model_copy(
        update={
            "angular_coeff": 0.22,
            "disc_depth_scale_coeff": 1.25,
            "disc_depth_scale_target": 0.55,
            "rim_angular_repulsion_coeff": 0.30,
            "rim_angular_min_r": 0.12,
            "rim_angular_min_sep": 0.09,
            "rim_pc2_floor_coeff": 0.22,
            "rim_pc2_min_r": 0.12,
            "rim_pc2_min_std": 0.08,
            "disc_angular_coverage_coeff": 0.30,
            "disc_angular_coverage_min_r": 0.12,
            "disc_angular_coverage_n_bins": 12,
            "disc_angular_coverage_min_bin_frac": 0.35,
            "disc_angular_coverage_temperature": 0.20,
            "expert_angular_diversity_coeff": 0.40,
            "expert_angular_max_R": 0.55,
            "expert_angular_min_mean_sep": 0.55,
            "expert_sector_recruit_coeff": 0.35,
            "expert_sector_recruit_min_bin_frac": 0.25,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out anti-barrier)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_rim_fanout_expert_arc_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 20,
) -> PhaseConfig:
    """Mild expert-arc specialization: soft θ diversity without sector recruit.

    Resume from coverage_v2 ep164. Keeps coverage floor + depth_scale hold; adds
    mild expert circular R ceiling + mean separation (no expert_sector_recruit —
    that overdrove anti-barrier). Use with epoch_anchor_pdb_ids including 1F88.
    Gates: probe ≥ baseline−0.08; 1F88 gap not worse by >10°; 4OBE circ-R ≲ 0.60.
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.75).model_copy(
        update={
            "angular_coeff": 0.22,
            "disc_depth_scale_coeff": 1.25,
            "disc_depth_scale_target": 0.55,
            "rim_angular_repulsion_coeff": 0.35,
            "rim_angular_min_r": 0.12,
            "rim_angular_min_sep": 0.09,
            "rim_pc2_floor_coeff": 0.25,
            "rim_pc2_min_r": 0.12,
            "rim_pc2_min_std": 0.08,
            "disc_angular_coverage_coeff": 0.40,
            "disc_angular_coverage_min_r": 0.12,
            "disc_angular_coverage_n_bins": 12,
            "disc_angular_coverage_min_bin_frac": 0.38,
            "disc_angular_coverage_temperature": 0.20,
            # Mild vs antibarrier (0.40 / 0.55 / 0.55 + recruit 0.35).
            "expert_angular_diversity_coeff": 0.18,
            "expert_angular_max_R": 0.60,
            "expert_angular_min_mean_sep": 0.40,
            "expert_sector_recruit_coeff": 0.0,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (rim fan-out mild expert-arc)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def v66_feeler_geom_angular_prior_phase_config(
    *,
    lr: float = 5.0e-5,
    epochs: int = 20,
) -> PhaseConfig:
    """Geometric angular prior sibling: grounded disc θ + light fidelity.

    Resume from coverage_v2 ep164 with ``geometric_angular_prior`` model flag.
    Keeps coverage / depth_scale; no sector recruit. Gates: probe hold, 1F88 gap,
    4OBE circ-R.
    """
    coeffs = _v66_feeler_p3_geom_coeffs(stack_scale=0.75).model_copy(
        update={
            "angular_coeff": 0.22,
            "disc_depth_scale_coeff": 1.25,
            "disc_depth_scale_target": 0.55,
            "rim_angular_repulsion_coeff": 0.35,
            "rim_angular_min_r": 0.12,
            "rim_angular_min_sep": 0.09,
            "rim_pc2_floor_coeff": 0.25,
            "rim_pc2_min_r": 0.12,
            "rim_pc2_min_std": 0.08,
            "disc_angular_coverage_coeff": 0.40,
            "disc_angular_coverage_min_r": 0.12,
            "disc_angular_coverage_n_bins": 12,
            "disc_angular_coverage_min_bin_frac": 0.38,
            "disc_angular_coverage_temperature": 0.20,
            "expert_angular_diversity_coeff": 0.0,
            "expert_sector_recruit_coeff": 0.0,
            "geometric_angular_fidelity_coeff": 0.10,
        }
    )
    return PhaseConfig(
        phase=12,
        name="Phase 12: v6.6 feeler (geometric angular prior)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
        coeffs=coeffs,
    )


def apply_v66_feeler_rim_fanout_cold_phases(
    *,
    base_lr: float = 5e-4,
    p1_epochs: int = 12,
    p2_epochs: int = 15,
    p12_epochs: int = 10,
    p12_lr: float = 1.0e-4,
) -> list[PhaseConfig]:
    """Cold rim fan-out: P1 radial → P2 angular/2D fill → P12 model fan-out.

    Single-phase cold (P12 only) learns depth→radius but rank-1 disc filaments —
    RimFanoutSpread is inactive until r >= min_r_rim, which P12 alone rarely reaches.
    """
    common = dict(
        freeze_radial=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        expert_timeout_max_share=0.45,
        expert_timeout_eligible_experts=None,
        slim_moe_structural_ssot_train=False,
        min_probe_r_depth_sasa=None,
        min_probe_r_depth_sasa_save=None,
        max_probe_r_epi_sasa_save=None,
    )
    return [
        PhaseConfig(
            phase=1,
            name="Phase 1: v6.6 rim fan-out cold (radial shell)",
            epochs=p1_epochs,
            lr=base_lr,
            freeze_angular=True,
            coeffs=_v66_feeler_base_coeffs(),
            **common,
        ),
        PhaseConfig(
            phase=2,
            name="Phase 2: v6.6 rim fan-out cold (angular + 2D occupancy)",
            epochs=p2_epochs,
            lr=base_lr * 0.5,
            freeze_angular=False,
            coeffs=_v66_feeler_base_coeffs(
                balance_coeff=0.02,
                cone_coeff=0.15,
                neighborhood_coeff=0.15,
                angular_coeff=0.10,
                cone_depth_anticollapse_coeff=0.30,
                disc_occupancy_coeff=0.45,
                disc_pc_repulsion_coeff=0.35,
                disc_thickness_floor_coeff=0.40,
                disc_origin_span_floor_coeff=1.0,
                disc_eff_rank_coeff=0.25,
            ),
            **common,
        ),
        v66_feeler_rim_fanout_model_phase_config(lr=p12_lr, epochs=p12_epochs),
    ]


def apply_v66_feeler_config(config: TrainingConfig) -> TrainingConfig:
    """Learned cold-start feeler: no SSOT freeze, topology gate, no V2 teacher."""
    return apply_master_cold_dehydron_config(config).model_copy(
        update={
            "v66_feeler_lineage": True,
            "master_cold_lineage": False,
            "slim_moe_structural_ssot": False,
            "structural_disc_frozen": False,
            "disc_layout_source": "gnn_learned",
            "gnn_lineage": "v6.6",
            "model_version": "GOSPConeMapper-v6.6",
            "mlflow_experiment": "tokyo-eyes-v66",
            "thermo_edge_features": False,
            "multi_rel_edge_mp": True,
            "role_edge_mp": True,
        }
    )


def apply_v66_feeler_lineage(
    config: TrainingConfig,
    phases: list[PhaseConfig] | None = None,
    *,
    epochs: int = 20,
) -> tuple[TrainingConfig, list[PhaseConfig]]:
    """v6.6 feeler: learned GNN + minimal losses + expert timeout@45%/1ep."""
    cfg = apply_v66_feeler_config(config)
    return cfg, apply_v66_feeler_phases(phases, base_lr=cfg.lr, epochs=epochs)


def apply_master_cold_dehydron_lineage(
    config: TrainingConfig,
    phases: list[PhaseConfig],
) -> tuple[TrainingConfig, list[PhaseConfig]]:
    """
    MASTER cold dehydron ablation (n6_v4+): τ-rim cone phases + topology-only gate + no V2 teacher.

    Topology-only routing removes the 128D x_hyp Mobius trunk from gate logits so τ/ρ/degree/ss
    drive prototypes instead of fold-blind backbone embeddings.
    """
    return apply_master_cold_dehydron_config(config), apply_master_cold_dehydron_phases(
        phases
    )


_SLIM_MOE_ZERO_DISC_COEFFS: dict[str, float] = {
    "disc_occupancy_coeff": 0.0,
    "disc_thickness_floor_coeff": 0.0,
    "disc_pc_repulsion_coeff": 0.0,
    "disc_eff_rank_coeff": 0.0,
    "disc_batch_diversity_coeff": 0.0,
    "disc_path_align_coeff": 0.0,
    "disc_depth_scale_coeff": 0.0,
    "disc_origin_span_floor_coeff": 0.0,
    "disc_angular_coverage_coeff": 0.0,
    "expert_angular_diversity_coeff": 0.0,
    "expert_sector_recruit_coeff": 0.0,
    "geometric_angular_fidelity_coeff": 0.0,
    "x_hyp_thickness_floor_coeff": 0.0,
    "angular_coeff": 0.0,
    "shell_corr_disc_sasa_weight": 0.0,
    "shell_corr_disc_spread_weight": 0.0,
    "shell_corr_proj_depth_weight": 0.0,
}


def _slim_moe_ssot_coeffs(**overrides: float | str) -> LossCoeffs:
    """Loss coeffs for slim MoE SSOT: zero disc geometry pressure + MoE overrides."""
    return LossCoeffs(
        **{
            **_SLIM_MOE_ZERO_DISC_COEFFS,
            "cone_target_mode": "tau_dehydron_rim",
            "shell_corr_depth_sasa_weight": 0.0,
            "shell_corr_epi_sasa_weight": 0.0,
            "shell_floor_coeff": 0.0,
            "domain_sep_2d_coeff": 0.0,
            "domain_sep_3d_coeff": 0.0,
            "evidential_coeff": 0.001,
            **overrides,
        }
    )


def apply_slim_moe_structural_ssot_phases(
    phases: list[PhaseConfig],
) -> list[PhaseConfig]:
    """Frozen structural disc SSOT — MoE-alive curriculum (cold_start_v8+).

    Geometry is given (radial/angular/backbone frozen). Pressure is on gate +
    experts + uncertainty. Floor/ceiling stay off in P2/P3; soft expert timeout
    (share>50%, 1 epoch) is the anti-dominance safety net.

    * P1 (40): light balance; soft floor/ceiling rarely fire; timeout@50%/1ep
    * P2 (160): specialize; no floor/ceiling; timeout@50%/1ep
    * P3 (50): gate unfrozen; freeze e2 weights; e2-only soft timeout@45%/1ep
      (consolidate after partition); relaxed eval save gates; H≤1.30
    """
    base_lr = next((p.lr for p in phases if p.phase == 1), 5e-4)
    moe_save: dict[str, Any] = {
        "slim_moe_structural_ssot_train": True,
        "freeze_radial": True,
        "freeze_angular": True,
        "freeze_backbone": True,
        "freeze_radial_epochs": 0,
        "min_probe_r_depth_sasa": None,
        "min_probe_r_depth_sasa_save": None,
        "min_disc_line_thickness_save": None,
        "min_disc_effective_rank_save": None,
        "min_disc_sigma2_sigma1_save": None,
        "min_disc_r_std_save": None,
        "max_expert_starvation_save": 0,
        "min_eval_routing_fraction_save": 0.08,
        "max_eval_routing_fraction_save": 0.50,
        "routing_entropy_min_save": 0.90,
        # Allow mild specialization (H~1.20–1.25); still reject near-uniform (~1.386).
        "routing_save_ceiling_start": 1.30,
        "routing_save_ceiling_final": 1.30,
        "routing_save_ceiling_ramp_epochs": 1,
        "min_probe_r_epi_sasa_save": 0.0,
    }
    # P3 continue: allow the locked-routing eval band that blocked gate-frozen saves.
    moe_save_p3: dict[str, Any] = {
        **moe_save,
        "min_eval_routing_fraction_save": 0.05,
        "max_eval_routing_fraction_save": 0.55,
    }
    return [
        PhaseConfig(
            phase=1,
            name="Phase 1: MoE alive (structural disc SSOT)",
            epochs=40,
            lr=base_lr,
            freeze_gate=False,
            expert_dropout_p=0.0,
            coeffs=_slim_moe_ssot_coeffs(
                balance_coeff=0.02,
                cone_coeff=0.20,
                neighborhood_coeff=0.05,
                cone_depth_anticollapse_coeff=0.40,
                shell_corr_coeff=0.12,
                # Soft nets only: fire on soft-collapse / extreme dominance.
                # Hard timeout@30% is the primary anti-dominance lever.
                routing_load_floor_coeff=1.0,
                routing_load_floor_min=0.05,
                routing_load_ceiling_coeff=2.0,
                routing_load_ceiling_max=0.55,
                epistemic_sasa_pen_coeff=0.0,
                epi_ale_decorrelation_coeff=0.0,
            ),
            max_probe_r_epi_ale_save=0.95,
            **moe_save,
        ),
        PhaseConfig(
            phase=2,
            name="Phase 2: Expert specialize (structural disc SSOT)",
            epochs=160,
            lr=base_lr,
            freeze_gate=False,
            expert_dropout_p=0.11,
            expert_dropout_ramp_epochs=8,
            coeffs=_slim_moe_ssot_coeffs(
                balance_coeff=0.01,
                cone_coeff=0.18,
                neighborhood_coeff=0.10,
                cone_depth_anticollapse_coeff=0.35,
                shell_corr_coeff=0.10,
                # No floor/ceiling — soft timeout@50% (1 epoch) is the anti-dominance lever.
                routing_load_floor_coeff=0.0,
                routing_load_floor_min=0.05,
                routing_load_ceiling_coeff=0.0,
                routing_load_ceiling_max=0.60,
                epistemic_sasa_pen_coeff=0.05,
                epi_ale_decorrelation_coeff=0.55,
                epistemic_anticollapse_coeff=0.05,
            ),
            max_probe_r_epi_ale_save=0.85,
            max_probe_r_epi_sasa_save=0.90,
            min_epistemic_std_save=0.02,
            min_aleatoric_std_save=0.05,
            **moe_save,
        ),
        PhaseConfig(
            phase=3,
            name="Phase 3: routing consolidate (freeze e2, e2-only timeout@45%)",
            epochs=50,
            lr=base_lr * 0.5,
            freeze_gate=False,
            freeze_experts=[2],
            # Mild safety net only — 0.42 caused ban ping-pong after e2 shed to ~0.39.
            expert_timeout_max_share=0.45,
            expert_timeout_eligible_experts=[2],
            expert_dropout_p=0.0,
            coeffs=_slim_moe_ssot_coeffs(
                balance_coeff=0.005,
                cone_coeff=0.15,
                neighborhood_coeff=0.08,
                cone_depth_anticollapse_coeff=0.30,
                shell_corr_coeff=0.08,
                routing_load_floor_coeff=0.0,
                routing_load_floor_min=0.05,
                routing_load_ceiling_coeff=0.0,
                routing_load_ceiling_max=0.65,
                epistemic_sasa_pen_coeff=0.04,
                epi_ale_decorrelation_coeff=0.45,
                epistemic_anticollapse_coeff=0.05,
            ),
            max_probe_r_epi_ale_save=0.85,
            max_probe_r_epi_sasa_save=0.90,
            min_epistemic_std_save=0.02,
            min_aleatoric_std_save=0.05,
            **moe_save_p3,
        ),
    ]


def apply_slim_moe_structural_ssot_config(config: TrainingConfig) -> TrainingConfig:
    """Cold-start preset aligned with ingest: structural disc layout + slim MoE head."""
    return apply_master_cold_dehydron_config(config).model_copy(
        update={
            "structural_disc_frozen": True,
            "slim_moe_structural_ssot": True,
            "expert_depth_decouple": True,
            "structure_gate": True,
            "gate_gumbel": False,
            "track_v6_best_route": True,
            "disc_layout_source": "structural_ssot_frozen",
        }
    )


def apply_slim_moe_structural_ssot_lineage(
    config: TrainingConfig,
    phases: list[PhaseConfig],
) -> tuple[TrainingConfig, list[PhaseConfig]]:
    """Config + phase coeffs for structural SSOT cold training (inference-aligned)."""
    cfg = apply_slim_moe_structural_ssot_config(config)
    return cfg, apply_slim_moe_structural_ssot_phases(phases)


def apply_phase_coeff_ramp(
    coeffs: dict[str, float],
    phase_cfg: PhaseConfig,
    epoch: int,
) -> dict[str, float]:
    """Linearly ramp selected loss coeffs over the first N epochs of a phase."""
    out = dict(coeffs)
    n = phase_cfg.coeff_ramp_epochs
    if n > 0:
        t = min(epoch + 1, n) / n
        if phase_cfg.angular_coeff_final is not None:
            start = phase_cfg.coeffs.angular_coeff
            out["angular_coeff"] = start + t * (phase_cfg.angular_coeff_final - start)
        if phase_cfg.shell_corr_disc_spread_weight_final is not None:
            start = phase_cfg.coeffs.shell_corr_disc_spread_weight
            end = phase_cfg.shell_corr_disc_spread_weight_final
            out["shell_corr_disc_spread_weight"] = start + t * (end - start)
        if phase_cfg.disc_spread_min_std_final is not None:
            start = phase_cfg.coeffs.disc_spread_min_std
            end = phase_cfg.disc_spread_min_std_final
            out["disc_spread_min_std"] = start + t * (end - start)
        if phase_cfg.disc_depth_scale_target_final is not None:
            start = phase_cfg.coeffs.disc_depth_scale_target
            end = phase_cfg.disc_depth_scale_target_final
            out["disc_depth_scale_target"] = start + t * (end - start)
        if (
            phase_cfg.angular_coeff_final is not None
            and phase_cfg.coeffs.domain_sep_2d_coeff > 0
        ):
            sep_scale = 0.5 + 0.5 * t
            out["domain_sep_2d_coeff"] = (
                phase_cfg.coeffs.domain_sep_2d_coeff * sep_scale
            )
            out["domain_sep_3d_coeff"] = (
                phase_cfg.coeffs.domain_sep_3d_coeff * sep_scale
            )
    n_epi = phase_cfg.epistemic_decoupling_ramp_epochs
    ep = epoch + 1
    if phase_cfg.epistemic_staged_decoupling:
        bf_final = phase_cfg.epistemic_bf_align_coeff_final or 0.22
        sasa_final = phase_cfg.epistemic_sasa_pen_coeff_final or 0.10
        bf_ramp = max(n_epi, 1)
        bf_t = min(ep, bf_ramp) / bf_ramp
        out["epistemic_bf_align_coeff"] = bf_t * bf_final
        bf_only = phase_cfg.epistemic_bf_only_epochs
        if ep <= bf_only:
            out["epistemic_sasa_pen_coeff"] = 0.0
        else:
            sasa_ep = ep - bf_only
            cap = phase_cfg.epistemic_sasa_pen_cap
            cap_n = max(phase_cfg.epistemic_sasa_pen_cap_epochs, 1)
            if sasa_ep <= cap_n:
                out["epistemic_sasa_pen_coeff"] = (sasa_ep / cap_n) * cap
            else:
                remain = max(phase_cfg.epochs - bf_only - cap_n, 1)
                t_log = min(sasa_ep - cap_n, remain) / remain
                # log curve: cap → sasa_final over remaining phase-2 epochs
                log_frac = math.log1p(9.0 * t_log) / math.log(10.0)
                out["epistemic_sasa_pen_coeff"] = cap + log_frac * (sasa_final - cap)
    elif n_epi > 0:
        t_epi = min(ep, n_epi) / n_epi
        if phase_cfg.epistemic_bf_align_coeff_final is not None:
            out["epistemic_bf_align_coeff"] = (
                t_epi * phase_cfg.epistemic_bf_align_coeff_final
            )
        if phase_cfg.epistemic_sasa_pen_coeff_final is not None:
            out["epistemic_sasa_pen_coeff"] = (
                t_epi * phase_cfg.epistemic_sasa_pen_coeff_final
            )
    return out


def p1b_phase_config(lr: float = 1e-4, epochs: int = 3) -> PhaseConfig:
    """Controlled angular unfreeze after P1 geometry (resume from v6_best)."""
    return PhaseConfig(
        phase=1,
        name="Phase 1b: Angular unfreeze",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        coeffs=LossCoeffs(
            balance_coeff=0.05,
            cone_coeff=0.25,
            neighborhood_coeff=0.10,
            angular_coeff=0.05,
            domain_sep_2d_coeff=0.05,
            domain_sep_3d_coeff=0.05,
            evidential_coeff=0.0005,
            cone_depth_anticollapse_coeff=1.5,
            cone_depth_min_std=0.08,
            shell_corr_coeff=0.5,
            shell_corr_disc_spread_weight=0.5,
            shell_corr_disc_sasa_weight=0.3,
            disc_spread_min_std=0.15,
            proj_violation_coeff=0.5,
        ),
    )


def p1c_phase_config(lr: float = 1e-4, epochs: int = 5) -> PhaseConfig:
    """Disc expansion after P1b radial shell (resume from shell_p1b/p1c v6_best)."""
    return PhaseConfig(
        phase=1,
        name="Phase 1c: Disc expansion",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        coeff_ramp_epochs=epochs,
        angular_coeff_final=0.15,
        shell_corr_disc_spread_weight_final=1.0,
        disc_spread_min_std_final=0.15,
        min_probe_r_depth_sasa=0.38,
        coeffs=LossCoeffs(
            balance_coeff=0.04,
            cone_coeff=0.20,
            neighborhood_coeff=0.08,
            angular_coeff=0.05,
            domain_sep_2d_coeff=0.08,
            domain_sep_3d_coeff=0.08,
            evidential_coeff=0.0004,
            cone_depth_anticollapse_coeff=1.2,
            cone_depth_min_std=0.08,
            shell_corr_coeff=0.6,
            shell_corr_disc_spread_weight=0.5,
            shell_corr_disc_sasa_weight=0.4,
            disc_spread_min_std=0.05,
            proj_violation_coeff=0.5,
        ),
    )


def p1d_phase_config(
    lr: float = 1e-4,
    epochs: int = 5,
    *,
    disc_depth_scale_coeff: float = 1.5,
    disc_target_start: float = 0.20,
    disc_target_end: float = 0.45,
    freeze_radial_epochs: int = 2,
    min_probe_r_depth_sasa: float = 0.44,
    disc_spread_min_std: float = 0.08,
    extend: bool = False,
) -> PhaseConfig:
    """Scale hyp_proj_2d radius to match cone_depth (MobiusLinear head)."""
    label = "Phase 1d: Disc-depth scale"
    if extend:
        label = "Phase 1d: Disc-depth scale (extend)"
    return PhaseConfig(
        phase=1,
        name=label,
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.0,
        freeze_radial_epochs=freeze_radial_epochs,
        coeff_ramp_epochs=epochs,
        disc_depth_scale_target_final=disc_target_end,
        min_probe_r_depth_sasa=min_probe_r_depth_sasa,
        coeffs=LossCoeffs(
            balance_coeff=0.03,
            cone_coeff=0.12,
            neighborhood_coeff=0.06,
            angular_coeff=0.08,
            domain_sep_2d_coeff=0.05,
            domain_sep_3d_coeff=0.05,
            evidential_coeff=0.0003,
            cone_depth_anticollapse_coeff=0.8,
            cone_depth_min_std=0.08,
            shell_corr_coeff=0.35,
            shell_corr_disc_spread_weight=0.9 if extend else 0.8,
            shell_corr_disc_sasa_weight=0.35,
            disc_spread_min_std=disc_spread_min_std,
            proj_violation_coeff=0.4,
            disc_depth_scale_coeff=disc_depth_scale_coeff,
            disc_depth_scale_target=disc_target_start,
        ),
    )


def p1d_extend_defaults(epochs: int = 8) -> dict[str, float | int]:
    """Tuned overrides for shell_p1d_disc2 continuation from P1d best."""
    return {
        "epochs": epochs,
        "disc_depth_scale_coeff": 2.2,
        "disc_target_start": 0.20,
        "disc_target_end": 0.35,
        "freeze_radial_epochs": 3,
        "min_probe_r_depth_sasa": 0.48,
        "disc_spread_min_std": 0.10,
        "extend": True,
    }


def p2_bridge_phase_config(
    lr: float = 5e-5,
    epochs: int = 12,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
) -> PhaseConfig:
    """MoE bridge from saturated-P1 routing (H~1.386) with shell + disc protection."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 10
    )
    return PhaseConfig(
        phase=2,
        name="Phase 2 bridge: MoE from saturated P1",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=False,
        expert_dropout_p=0.05,
        expert_dropout_ramp_epochs=5,
        freeze_radial_epochs=3,
        coeff_ramp_epochs=epochs,
        angular_coeff_final=0.20,
        disc_depth_scale_target_final=0.50,
        min_probe_r_depth_sasa=0.60,
        min_probe_r_depth_sasa_save=0.60,
        p2_bridge=True,
        routing_save_ceiling_start=1.40,
        routing_save_ceiling_final=1.21,
        routing_save_ceiling_ramp_epochs=ramp,
        angular_ramp_epochs=ramp,
        coeffs=LossCoeffs(
            balance_coeff=0.001,
            cone_coeff=0.18,
            neighborhood_coeff=0.12,
            angular_coeff=0.08,
            domain_sep_2d_coeff=0.10,
            domain_sep_3d_coeff=0.10,
            evidential_coeff=0.0003,
            cone_depth_anticollapse_coeff=0.6,
            cone_depth_min_std=0.08,
            shell_corr_coeff=0.30,
            shell_corr_disc_spread_weight=0.8,
            shell_corr_disc_sasa_weight=0.35,
            disc_spread_min_std=0.10,
            proj_violation_coeff=0.4,
            disc_depth_scale_coeff=2.2,
            disc_depth_scale_target=0.35,
        ),
    )


def p2_hypmix_phase_config(
    lr: float = 5e-5,
    epochs: int = 15,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
) -> PhaseConfig:
    """P2 bridge + stronger MoE pressure (hypmix2): higher dropout, shell guard 0.65."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 15
    )
    base = p2_bridge_phase_config(
        lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp
    )
    return base.model_copy(
        update={
            "name": "Phase 2 hypmix: disc gate + ball mix + MoE pressure",
            "expert_dropout_p": 0.12,
            "expert_dropout_ramp_epochs": 3,
            "min_probe_r_depth_sasa": 0.60,
            "min_probe_r_depth_sasa_save": 0.65,
            "coeffs": base.coeffs.model_copy(update={"balance_coeff": 0.005}),
        }
    )


def p2_hypmix3_phase_config(
    lr: float = 5e-5,
    epochs: int = 12,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
) -> PhaseConfig:
    """Hypmix3: amplify MoE from hypmix2 champion (dropout 0.15, capacity 0.008)."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 15
    )
    base = p2_hypmix_phase_config(
        lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp
    )
    return base.model_copy(
        update={
            "name": "Phase 2 hypmix3: lock MoE specialization",
            "expert_dropout_p": 0.15,
            "expert_dropout_ramp_epochs": 2,
            "coeffs": base.coeffs.model_copy(update={"balance_coeff": 0.008}),
        }
    )


def p2_hypmix_final_phase_config(
    lr: float = 5e-5,
    epochs: int = 10,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
) -> PhaseConfig:
    """Final lock-in: max MoE pressure before full hyperbolic re-arch (dropout 0.18, capacity 0.01)."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 15
    )
    base = p2_hypmix_phase_config(
        lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp
    )
    return base.model_copy(
        update={
            "name": "Phase 2 hypmix final: lock MoE + shell",
            "expert_dropout_p": 0.18,
            "expert_dropout_ramp_epochs": 2,
            "coeffs": base.coeffs.model_copy(update={"balance_coeff": 0.01}),
        }
    )


def full_hyp_moe_theory_config(
    lr: float = 3e-5,
    epochs: int = 15,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
) -> PhaseConfig:
    """Full hyperbolic MoE theory test from hypmix_final lock-in (ep78 warm-start)."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 15
    )
    base = p2_hypmix_final_phase_config(
        lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp
    )
    return base.model_copy(
        update={
            "name": "Full hyperbolic MoE theory test",
            "lr": lr,
            "expert_dropout_p": 0.15,
            "expert_dropout_ramp_epochs": 0,
            "freeze_radial_epochs": 2,
            "routing_save_ceiling_start": 1.32,
            "routing_save_ceiling_final": 1.18,
            "routing_save_ceiling_ramp_epochs": ramp,
            "angular_ramp_epochs": ramp,
            "coeffs": base.coeffs.model_copy(update={"balance_coeff": 0.008}),
        }
    )


def p2_disc_occupancy_phase_config(
    lr: float = 3e-5,
    epochs: int = 15,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_occupancy_coeff: float = 2.5,
    freeze_radial_epochs: int = 3,
    min_disc_sigma2_sigma1_save: float = 0.35,
    name: str = "Phase 2 disc occupancy recovery",
) -> PhaseConfig:
    """Warm-start recovery: keep shell scalars, penalize rank-1 hyp_projections_2d."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 12
    )
    base = p2_hypmix_final_phase_config(
        lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp
    )
    return base.model_copy(
        update={
            "name": name,
            "lr": lr,
            "epochs": epochs,
            "freeze_radial_epochs": freeze_radial_epochs,
            "min_probe_r_depth_sasa": None,
            "min_probe_r_depth_sasa_save": 0.55,
            "min_disc_sigma2_sigma1_save": min_disc_sigma2_sigma1_save,
            "routing_save_ceiling_start": 1.30,
            "routing_save_ceiling_final": 1.20,
            "routing_save_ceiling_ramp_epochs": ramp,
            "angular_ramp_epochs": ramp,
            "expert_dropout_p": 0.10,
            "expert_dropout_ramp_epochs": 2,
            "coeffs": base.coeffs.model_copy(
                update={
                    "shell_corr_coeff": 0.15,
                    "angular_coeff": 0.10,
                    "domain_sep_2d_coeff": 0.40,
                    "domain_sep_3d_coeff": 0.15,
                    "disc_depth_scale_coeff": 0.0,
                    "disc_occupancy_coeff": disc_occupancy_coeff,
                    "disc_occupancy_min_sigma_ratio": 0.35,
                }
            ),
        }
    )


def p2_rec_ablation_phase_config(
    lr: float = 2e-5,
    epochs: int = 5,
) -> PhaseConfig:
    """Test residual radial×angular fusion — bridge losses only, no occupancy pressure."""
    return PhaseConfig(
        phase=2,
        name="Phase 2 radial×angular fusion ablation",
        epochs=epochs,
        lr=lr,
        freeze_radial=True,
        freeze_angular=True,
        freeze_backbone=True,
        freeze_gate=True,
        expert_dropout_p=0.0,
        rec_ablation_train=True,
        min_probe_r_depth_sasa_save=None,
        min_disc_sigma2_sigma1_save=None,
        min_disc_r_std_save=None,
        min_disc_effective_rank_save=None,
        min_disc_line_thickness_save=None,
        p2_bridge=True,
        routing_save_ceiling_start=1.35,
        routing_save_ceiling_final=1.35,
        routing_save_ceiling_ramp_epochs=1,
        coeffs=LossCoeffs(
            evidential_coeff=0.0003,
            balance_coeff=0.001,
            cone_coeff=0.18,
            neighborhood_coeff=0.12,
            angular_coeff=0.08,
            domain_sep_2d_coeff=0.10,
            domain_sep_3d_coeff=0.10,
            cone_depth_anticollapse_coeff=0.6,
            shell_corr_coeff=0.30,
            disc_depth_scale_coeff=0.0,
            disc_occupancy_coeff=0.0,
            disc_pc_repulsion_coeff=0.0,
            disc_eff_rank_coeff=0.0,
            disc_batch_diversity_coeff=0.0,
            disc_path_align_coeff=0.0,
            proj_violation_coeff=0.0,
        ),
    )


def p2_disc_proj_recovery_phase_config(
    lr: float = 2e-5,
    epochs: int = 20,
    *,
    disc_thickness_floor_min: float = 0.025,
    disc_origin_span_min_spread: float = 0.12,
    disc_pc_repulsion_coeff: float = 2.0,
    disc_path_align_coeff: float = 2.0,
    disc_thickness_floor_coeff: float = 3.0,
) -> PhaseConfig:
    """Projection head recovery — thickness/span floors on pre-routing disc + legacy teacher."""
    return PhaseConfig(
        phase=2,
        name="Phase 2 disc projection recovery",
        epochs=epochs,
        lr=lr,
        freeze_radial=True,
        freeze_angular=True,
        freeze_backbone=True,
        freeze_gate=True,
        expert_dropout_p=0.0,
        projection_recovery_train=True,
        min_probe_r_depth_sasa_save=None,
        min_disc_sigma2_sigma1_save=None,
        min_disc_r_std_save=None,
        min_disc_effective_rank_save=None,
        min_disc_line_thickness_save=disc_thickness_floor_min,
        p2_bridge=True,
        routing_save_ceiling_start=1.35,
        routing_save_ceiling_final=1.35,
        routing_save_ceiling_ramp_epochs=1,
        coeffs=LossCoeffs(
            evidential_coeff=0.0,
            balance_coeff=0.0,
            cone_coeff=0.10,
            neighborhood_coeff=0.0,
            angular_coeff=0.0,
            domain_sep_2d_coeff=0.0,
            domain_sep_3d_coeff=0.0,
            cone_depth_anticollapse_coeff=0.4,
            shell_corr_coeff=0.15,
            disc_depth_scale_coeff=0.0,
            disc_occupancy_coeff=0.0,
            disc_pc_repulsion_coeff=disc_pc_repulsion_coeff,
            disc_pc2_min_std=0.06,
            disc_eff_rank_coeff=0.0,
            disc_batch_diversity_coeff=0.0,
            disc_path_align_coeff=disc_path_align_coeff,
            disc_thickness_floor_coeff=disc_thickness_floor_coeff,
            disc_thickness_floor_min=disc_thickness_floor_min,
            disc_origin_span_floor_coeff=2.0,
            disc_origin_span_min_spread=disc_origin_span_min_spread,
            proj_violation_coeff=0.0,
        ),
    )


def p2_disc_proj_recovery_v2_phase_config(
    lr: float = 2e-5,
    epochs: int = 20,
    *,
    disc_thickness_floor_min: float = 0.025,
    disc_thickness_floor_coeff: float = 9.0,
) -> PhaseConfig:
    """v2 recovery — no path_align (fights pre-routing thickness), stronger thickness floor."""
    return p2_disc_proj_recovery_phase_config(
        lr=lr,
        epochs=epochs,
        disc_thickness_floor_min=disc_thickness_floor_min,
        disc_path_align_coeff=0.0,
        disc_thickness_floor_coeff=disc_thickness_floor_coeff,
        disc_pc_repulsion_coeff=2.5,
    )


def p2_disc_proj_recovery_v3_phase_config(
    lr: float = 2e-5,
    epochs: int = 20,
    *,
    disc_thickness_floor_min: float = 0.025,
    disc_thickness_floor_coeff: float = 9.0,
) -> PhaseConfig:
    """v3 recovery — train angular + fusion + disc head (fix x_hyp before projection)."""
    cfg = p2_disc_proj_recovery_v2_phase_config(
        lr=lr,
        epochs=epochs,
        disc_thickness_floor_min=disc_thickness_floor_min,
        disc_thickness_floor_coeff=disc_thickness_floor_coeff,
    )
    return cfg.model_copy(
        update={
            "name": "Phase 2 disc fusion-path recovery",
            "projection_recovery_train": False,
            "fusion_path_recovery_train": True,
            "coeffs": cfg.coeffs.model_copy(
                update={
                    "angular_coeff": 0.06,
                    "neighborhood_coeff": 0.06,
                }
            ),
        }
    )


def p2_disc_proj_recovery_v4_phase_config(
    lr: float = 2e-5,
    epochs: int = 20,
    *,
    disc_thickness_floor_min: float = 0.025,
    x_hyp_thickness_floor_min: float = 0.18,
) -> PhaseConfig:
    """v4 recovery — unfreeze radial + lift path; penalize thin x_hyp ball spread."""
    cfg = p2_disc_proj_recovery_v3_phase_config(
        lr=lr,
        epochs=epochs,
        disc_thickness_floor_min=disc_thickness_floor_min,
    )
    return cfg.model_copy(
        update={
            "name": "Phase 2 disc lift-path recovery (radial+angular+fusion)",
            "fusion_path_recovery_train": False,
            "lift_path_recovery_train": True,
            "freeze_radial": False,
            "coeffs": cfg.coeffs.model_copy(
                update={
                    "cone_coeff": 0.12,
                    "cone_depth_anticollapse_coeff": 0.55,
                    "x_hyp_thickness_floor_coeff": 6.0,
                    "x_hyp_thickness_floor_min": x_hyp_thickness_floor_min,
                }
            ),
        }
    )


def p2_disc_proj_recovery_v5_phase_config(
    lr: float = 3e-5,
    epochs: int = 20,
    *,
    disc_thickness_floor_min: float = 0.025,
    x_hyp_thickness_floor_min: float = 0.18,
) -> PhaseConfig:
    """v5 lift ablation — angular_lift (no cone multiply) + v4 lift-path training."""
    cfg = p2_disc_proj_recovery_v4_phase_config(
        lr=lr,
        epochs=epochs,
        disc_thickness_floor_min=disc_thickness_floor_min,
        x_hyp_thickness_floor_min=x_hyp_thickness_floor_min,
    )
    return cfg.model_copy(
        update={
            "name": "Phase 2 angular_lift disc recovery",
            "coeffs": cfg.coeffs.model_copy(
                update={
                    "angular_coeff": 0.10,
                    "cone_coeff": 0.15,
                }
            ),
        }
    )


def p2_disc_path_align_phase_config(
    lr: float = 1e-5,
    epochs: int = 8,
    *,
    disc_path_align_coeff: float = 3.0,
    min_disc_line_thickness_save: float = 0.02,
) -> PhaseConfig:
    """Align new pre-routing disc path to legacy teacher spread — no occupancy pressure."""
    return PhaseConfig(
        phase=2,
        name="Phase 2 disc path alignment (legacy teacher → new path)",
        epochs=epochs,
        lr=lr,
        freeze_radial=True,
        freeze_angular=True,
        freeze_backbone=True,
        freeze_gate=True,
        expert_dropout_p=0.0,
        path_alignment_train=True,
        min_probe_r_depth_sasa_save=None,
        min_disc_sigma2_sigma1_save=None,
        min_disc_r_std_save=None,
        min_disc_effective_rank_save=None,
        min_disc_line_thickness_save=min_disc_line_thickness_save,
        p2_bridge=True,
        routing_save_ceiling_start=1.35,
        routing_save_ceiling_final=1.35,
        routing_save_ceiling_ramp_epochs=1,
        coeffs=LossCoeffs(
            evidential_coeff=0.0,
            balance_coeff=0.0,
            cone_coeff=0.0,
            neighborhood_coeff=0.0,
            angular_coeff=0.0,
            domain_sep_2d_coeff=0.0,
            domain_sep_3d_coeff=0.0,
            cone_depth_anticollapse_coeff=0.0,
            shell_corr_coeff=0.0,
            disc_depth_scale_coeff=0.0,
            disc_occupancy_coeff=0.0,
            disc_pc_repulsion_coeff=0.0,
            disc_eff_rank_coeff=0.0,
            disc_batch_diversity_coeff=0.0,
            disc_path_align_coeff=disc_path_align_coeff,
            proj_violation_coeff=0.0,
        ),
    )


def p2_disc_gentle_arch_phase_config(
    lr: float = 5e-6,
    epochs: int = 10,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_occupancy_coeff: float = 1.2,
    freeze_radial_epochs: int = 4,
    min_disc_r_std_save: float = 0.04,
    min_disc_line_thickness_save: float = 0.02,
) -> PhaseConfig:
    """Preserve early-recovery visual spread under pre-routing disc path — gentle occupancy."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 8
    )
    base = p2_disc_occupancy_phase_config(
        lr=lr,
        epochs=epochs,
        routing_save_ceiling_ramp_epochs=ramp,
        disc_occupancy_coeff=disc_occupancy_coeff,
        freeze_radial_epochs=freeze_radial_epochs,
        min_disc_sigma2_sigma1_save=0.32,
        name="Phase 2 gentle disc arch (early recovery → new path)",
    )
    return base.model_copy(
        update={
            "min_probe_r_depth_sasa_save": 0.55,
            "min_disc_r_std_save": min_disc_r_std_save,
            "min_disc_line_thickness_save": min_disc_line_thickness_save,
            "min_disc_effective_rank_save": None,
            "routing_save_ceiling_start": 1.28,
            "routing_save_ceiling_final": 1.20,
            "expert_dropout_p": 0.08,
            "expert_dropout_ramp_epochs": 2,
            "coeffs": base.coeffs.model_copy(
                update={
                    "shell_corr_coeff": 0.12,
                    "angular_coeff": 0.06,
                    "domain_sep_2d_coeff": 0.35,
                    "domain_sep_3d_coeff": 0.10,
                    "disc_occupancy_coeff": disc_occupancy_coeff,
                    "disc_occupancy_min_sigma_ratio": 0.32,
                    "disc_pc_repulsion_coeff": 0.8,
                    "disc_pc2_min_std": 0.06,
                    "disc_eff_rank_coeff": 0.0,
                    "disc_batch_diversity_coeff": 0.0,
                    "disc_min_r_mean": 0.08,
                    "disc_r_collapse_scale": 8.0,
                }
            ),
        }
    )


def p2_disc_occupancy_v2_phase_config(
    lr: float = 2e-5,
    epochs: int = 12,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_occupancy_coeff: float = 2.8,
    freeze_radial_epochs: int = 4,
    min_disc_r_std_save: float = 0.02,
) -> PhaseConfig:
    """Refined recovery: stronger occupancy + PC2 repulsion + soft shell floor."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 10
    )
    base = p2_disc_occupancy_phase_config(
        lr=lr,
        epochs=epochs,
        routing_save_ceiling_ramp_epochs=ramp,
        disc_occupancy_coeff=disc_occupancy_coeff,
        freeze_radial_epochs=freeze_radial_epochs,
        min_disc_sigma2_sigma1_save=0.40,
        name="Phase 2 disc occupancy recovery v2",
    )
    return base.model_copy(
        update={
            "min_probe_r_depth_sasa_save": 0.58,
            "min_disc_r_std_save": min_disc_r_std_save,
            "routing_save_ceiling_start": 1.28,
            "routing_save_ceiling_final": 1.18,
            "expert_dropout_p": 0.08,
            "expert_dropout_ramp_epochs": 3,
            "coeffs": base.coeffs.model_copy(
                update={
                    "shell_corr_coeff": 0.20,
                    "angular_coeff": 0.08,
                    "domain_sep_2d_coeff": 0.45,
                    "disc_occupancy_min_sigma_ratio": 0.40,
                    "disc_min_r_mean": 0.05,
                    "disc_r_collapse_scale": 10.0,
                    "disc_pc_repulsion_coeff": 1.5,
                    "disc_pc2_min_std": 0.10,
                    "shell_floor_coeff": 0.50,
                    "shell_floor_min_r_depth_sasa": 0.60,
                }
            ),
        }
    )


def p2_disc_occupancy_v3_phase_config(
    lr: float = 1e-5,
    epochs: int = 12,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_occupancy_coeff: float = 3.5,
    freeze_radial_epochs: int = 5,
    min_disc_r_std_save: float = 0.045,
) -> PhaseConfig:
    """Cluster-break push: stronger occupancy + PC2 repulsion, finer LR, tighter radial gate."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 10
    )
    base = p2_disc_occupancy_v2_phase_config(
        lr=lr,
        epochs=epochs,
        routing_save_ceiling_ramp_epochs=ramp,
        disc_occupancy_coeff=disc_occupancy_coeff,
        freeze_radial_epochs=freeze_radial_epochs,
        min_disc_r_std_save=min_disc_r_std_save,
    )
    return base.model_copy(
        update={
            "name": "Phase 2 disc occupancy recovery v3",
            "min_disc_sigma2_sigma1_save": 0.45,
            "min_probe_r_depth_sasa_save": 0.58,
            "freeze_radial_epochs": freeze_radial_epochs,
            "coeffs": base.coeffs.model_copy(
                update={
                    "disc_pc_repulsion_coeff": 2.5,
                    "disc_pc2_min_std": 0.12,
                    "shell_floor_coeff": 0.55,
                    "shell_floor_min_r_depth_sasa": 0.58,
                }
            ),
        }
    )


def p2_disc_occupancy_v4_phase_config(
    lr: float = 1e-5,
    epochs: int = 12,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_occupancy_coeff: float = 3.5,
    disc_eff_rank_coeff: float = 1.0,
    freeze_radial_epochs: int = 5,
    min_disc_r_std_save: float = 0.045,
    min_disc_effective_rank_save: float = 1.5,
) -> PhaseConfig:
    """Target-structure corpus push: occupancy + eff_rank on gate proteins."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 10
    )
    base = p2_disc_occupancy_v3_phase_config(
        lr=lr,
        epochs=epochs,
        routing_save_ceiling_ramp_epochs=ramp,
        disc_occupancy_coeff=disc_occupancy_coeff,
        freeze_radial_epochs=freeze_radial_epochs,
        min_disc_r_std_save=min_disc_r_std_save,
    )
    return base.model_copy(
        update={
            "name": "Phase 2 disc occupancy recovery v4 (target corpus)",
            "min_disc_sigma2_sigma1_save": 0.45,
            "min_disc_effective_rank_save": min_disc_effective_rank_save,
            "coeffs": base.coeffs.model_copy(
                update={
                    "disc_eff_rank_coeff": disc_eff_rank_coeff,
                    "disc_eff_rank_min": 1.6,
                    "disc_occupancy_min_sigma_ratio": 0.45,
                }
            ),
        }
    )


def p2_disc_occupancy_v5_phase_config(
    lr: float = 1e-5,
    epochs: int = 12,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_occupancy_coeff: float = 3.5,
    disc_eff_rank_coeff: float = 1.0,
    disc_batch_diversity_coeff: float = 1.25,
    freeze_radial_epochs: int = 5,
    min_disc_r_std_save: float = 0.045,
    min_disc_effective_rank_save: float = 1.5,
) -> PhaseConfig:
    """Batch diversity repulsion on target corpus — break lower-left cluster collinearity."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 10
    )
    base = p2_disc_occupancy_v4_phase_config(
        lr=lr,
        epochs=epochs,
        routing_save_ceiling_ramp_epochs=ramp,
        disc_occupancy_coeff=disc_occupancy_coeff,
        disc_eff_rank_coeff=disc_eff_rank_coeff,
        freeze_radial_epochs=freeze_radial_epochs,
        min_disc_r_std_save=min_disc_r_std_save,
        min_disc_effective_rank_save=min_disc_effective_rank_save,
    )
    return base.model_copy(
        update={
            "name": "Phase 2 disc occupancy recovery v5 (batch diversity)",
            "coeffs": base.coeffs.model_copy(
                update={
                    "disc_batch_diversity_coeff": disc_batch_diversity_coeff,
                    "disc_batch_min_pairwise_dist": 0.035,
                }
            ),
        }
    )


def p4_epistemic_decoupling_phase_config(
    lr: float = 1e-4,
    epochs: int = 30,
    *,
    routing_save_ceiling_ramp_epochs: int | None = None,
    disc_occupancy_coeff: float = 3.5,
    disc_eff_rank_coeff: float = 1.0,
    disc_line_thickness_floor_coeff: float = 0.5,
    disc_thickness_floor_min: float = 0.18,
    epistemic_bf_align_coeff: float = 0.22,
    epistemic_sasa_pen_coeff: float = 0.246,
    epistemic_anticollapse_coeff: float = 0.05,
    shell_corr_epi_sasa_weight: float = 0.5,
    staged_decoupling: bool = False,
) -> PhaseConfig:
    """Phase 4: B-factor residual epistemic decoupling on lever_a shell foundation."""
    ramp = (
        routing_save_ceiling_ramp_epochs
        if routing_save_ceiling_ramp_epochs is not None
        else 10
    )
    base = p2_disc_occupancy_v4_phase_config(
        lr=lr,
        epochs=epochs,
        routing_save_ceiling_ramp_epochs=ramp,
        disc_occupancy_coeff=disc_occupancy_coeff,
        disc_eff_rank_coeff=disc_eff_rank_coeff,
    )
    sasa_final = 0.10 if staged_decoupling else epistemic_sasa_pen_coeff
    return base.model_copy(
        update={
            "phase": 4,
            "name": (
                "Phase 4 epistemic-depth decoupling (B-factor residual, staged λ)"
                if staged_decoupling
                else "Phase 4 epistemic-depth decoupling (B-factor residual)"
            ),
            "lr": lr,
            "epochs": epochs,
            "freeze_radial_epochs": 0,
            "min_disc_line_thickness_save": disc_thickness_floor_min,
            "min_probe_r_depth_sasa_save": 0.65,
            "epistemic_decoupling_ramp_epochs": 5,
            "epistemic_bf_align_coeff_final": epistemic_bf_align_coeff,
            "epistemic_sasa_pen_coeff_final": sasa_final,
            "epistemic_staged_decoupling": staged_decoupling,
            "epistemic_bf_only_epochs": 10 if staged_decoupling else 0,
            "epistemic_sasa_pen_cap": 0.05,
            "epistemic_sasa_pen_cap_epochs": 5,
            "epistemic_uncertainty_only_train": True,
            "coeffs": base.coeffs.model_copy(
                update={
                    "epistemic_decoupling_coeff": 1.0,
                    "epistemic_bf_align_coeff": 0.0,
                    "epistemic_sasa_pen_coeff": 0.0,
                    "epistemic_anticollapse_coeff": epistemic_anticollapse_coeff,
                    "disc_thickness_floor_coeff": disc_line_thickness_floor_coeff,
                    "disc_thickness_floor_min": disc_thickness_floor_min,
                    "shell_corr_epi_sasa_weight": shell_corr_epi_sasa_weight,
                }
            ),
        }
    )


def p4_uncertainty_calibration_phase_config(
    lr: float = 5e-5,
    epochs: int = 25,
) -> PhaseConfig:
    """P4 from rs2_post_p4: decouple ν_epi/ν_ale without heavy disc-occupancy pressure."""
    return PhaseConfig(
        phase=4,
        name="Phase 4 uncertainty calibration (staged decoupling, rs2 warm-start)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=True,
        expert_dropout_p=0.0,
        epistemic_decoupling_ramp_epochs=5,
        epistemic_bf_align_coeff_final=0.28,
        epistemic_sasa_pen_coeff_final=0.08,
        epistemic_staged_decoupling=True,
        epistemic_bf_only_epochs=8,
        epistemic_sasa_pen_cap=0.04,
        epistemic_sasa_pen_cap_epochs=5,
        epistemic_uncertainty_only_train=True,
        min_disc_line_thickness_save=0.04,
        min_probe_r_depth_sasa_save=0.55,
        max_probe_r_epi_ale_save=0.85,
        max_probe_r_epi_sasa_save=0.78,
        min_epistemic_std_save=0.05,
        min_aleatoric_std_save=0.5,
        routing_save_ceiling_start=1.39,
        routing_save_ceiling_final=1.22,
        routing_save_ceiling_ramp_epochs=10,
        coeffs=LossCoeffs(
            evidential_coeff=0.01,
            balance_coeff=0.001,
            cone_coeff=0.05,
            neighborhood_coeff=0.05,
            angular_coeff=0.02,
            domain_sep_2d_coeff=0.02,
            domain_sep_3d_coeff=0.02,
            cone_depth_anticollapse_coeff=0.15,
            shell_corr_coeff=0.05,
            shell_corr_epi_sasa_weight=0.0,
            disc_occupancy_coeff=0.10,
            disc_occupancy_min_sigma_ratio=0.30,
            epistemic_decoupling_coeff=1.0,
            epistemic_bf_align_coeff=0.0,
            epistemic_sasa_pen_coeff=0.0,
            epistemic_anticollapse_coeff=0.15,
            epistemic_min_epi_std=0.03,
        ),
    )


def p4_head_decouple_phase_config(
    lr: float = 5e-5,
    epochs: int = 20,
    *,
    max_probe_r_epi_sasa_save: float = 0.78,
    phase_name: str = "Phase 4 head decouple split epi ale trunks rs2 warm-start",
) -> PhaseConfig:
    """P4 with split epi/ale trunks + direct r(epi,ale) penalty (rs2 warm-start)."""
    return PhaseConfig(
        phase=4,
        name=phase_name,
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=True,
        expert_dropout_p=0.0,
        epistemic_decoupling_ramp_epochs=5,
        epistemic_bf_align_coeff_final=0.28,
        epistemic_sasa_pen_coeff_final=0.08,
        epistemic_staged_decoupling=True,
        epistemic_bf_only_epochs=6,
        epistemic_sasa_pen_cap=0.04,
        epistemic_sasa_pen_cap_epochs=5,
        epistemic_uncertainty_only_train=True,
        min_disc_line_thickness_save=0.04,
        min_probe_r_depth_sasa_save=0.55,
        max_probe_r_epi_ale_save=0.70,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        min_epistemic_std_save=0.05,
        min_aleatoric_std_save=0.5,
        require_tau_ale_elevation_save=True,
        routing_save_ceiling_start=1.39,
        routing_save_ceiling_final=1.22,
        routing_save_ceiling_ramp_epochs=8,
        coeffs=LossCoeffs(
            evidential_coeff=0.01,
            balance_coeff=0.001,
            cone_coeff=0.05,
            neighborhood_coeff=0.05,
            angular_coeff=0.02,
            domain_sep_2d_coeff=0.02,
            domain_sep_3d_coeff=0.02,
            cone_depth_anticollapse_coeff=0.15,
            shell_corr_coeff=0.05,
            shell_corr_epi_sasa_weight=0.0,
            disc_occupancy_coeff=0.08,
            disc_occupancy_min_sigma_ratio=0.30,
            epistemic_decoupling_coeff=1.0,
            epi_ale_decorrelation_coeff=0.75,
            epistemic_bf_align_coeff=0.0,
            epistemic_sasa_pen_coeff=0.0,
            epistemic_anticollapse_coeff=0.15,
            epistemic_min_epi_std=0.03,
        ),
    )


def p4_v3_aleatoric_shaping_phase_config(
    lr: float = 5e-5,
    epochs: int = 20,
    *,
    max_probe_r_epi_sasa_save: float = 0.78,
    holdout_fraction: float = 0.20,
    w_var_penalty: float = 2.8,
) -> PhaseConfig:
    """Phase 4 + v3 var_penalty/hinge on train mask only; P8 eval on holdout (G4)."""
    cfg = p4_head_decouple_phase_config(
        lr=lr,
        epochs=epochs,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        phase_name="Phase 4 v3 aleatoric shaping (G4 holdout)",
    )
    return cfg.model_copy(
        update={
            "name": "Phase 4 v3 aleatoric shaping (G4 holdout)",
            "require_tau_ale_elevation_save": False,
            "require_g4_holdout_p8_save": True,
            "coeffs": cfg.coeffs.model_copy(
                update={
                    "v3_aleatoric_shaping_coeff": 1.0,
                    "w_var_penalty": w_var_penalty,
                    "w_aleatoric_hinge": 4.8,
                    "aleatoric_hinge_target": 1.0,
                    "aleatoric_shaping_holdout_fraction": holdout_fraction,
                    "aleatoric_shaping_holdout_mode": "protein",
                }
            ),
        }
    )


def p4_g4_shaping_only_isolation_phase_config(
    lr: float = 5e-5,
    epochs: int = 12,
    *,
    max_probe_r_epi_sasa_save: float = 0.78,
    holdout_fraction: float = 0.20,
    w_var_penalty: float = 2.8,
) -> PhaseConfig:
    """G4 isolation: uncertainty-head-only + v3 shaping losses, all other losses off."""
    cfg = p4_head_decouple_phase_config(
        lr=lr,
        epochs=epochs,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        phase_name="Phase 4 G4 shaping-only isolation",
    )
    return cfg.model_copy(
        update={
            "name": "Phase 4 G4 shaping-only isolation",
            "require_tau_ale_elevation_save": False,
            "require_g4_holdout_p8_save": True,
            "coeffs": cfg.coeffs.model_copy(
                update={
                    "evidential_coeff": 0.0,
                    "balance_coeff": 0.0,
                    "cone_coeff": 0.0,
                    "neighborhood_coeff": 0.0,
                    "angular_coeff": 0.0,
                    "domain_sep_2d_coeff": 0.0,
                    "domain_sep_3d_coeff": 0.0,
                    "cone_depth_anticollapse_coeff": 0.0,
                    "shell_corr_coeff": 0.0,
                    "shell_corr_epi_sasa_weight": 0.0,
                    "disc_occupancy_coeff": 0.0,
                    "disc_pc_repulsion_coeff": 0.0,
                    "disc_eff_rank_coeff": 0.0,
                    "disc_batch_diversity_coeff": 0.0,
                    "disc_path_align_coeff": 0.0,
                    "disc_thickness_floor_coeff": 0.0,
                    "disc_origin_span_floor_coeff": 0.0,
                    "x_hyp_thickness_floor_coeff": 0.0,
                    "routing_load_floor_coeff": 0.0,
                    "epistemic_decoupling_coeff": 0.0,
                    "epi_ale_decorrelation_coeff": 0.0,
                    "epistemic_bf_align_coeff": 0.0,
                    "epistemic_sasa_pen_coeff": 0.0,
                    "epistemic_anticollapse_coeff": 0.0,
                    "v3_aleatoric_shaping_coeff": 1.0,
                    "w_var_penalty": w_var_penalty,
                    "w_aleatoric_hinge": 4.8,
                    "aleatoric_hinge_target": 1.0,
                    "aleatoric_shaping_holdout_fraction": holdout_fraction,
                    "aleatoric_shaping_holdout_mode": "protein",
                }
            ),
        }
    )


def p4_g4_ale_only_unshaped_phase_config(
    lr: float = 5e-5,
    epochs: int = 12,
    *,
    max_probe_r_epi_sasa_save: float = 0.78,
) -> PhaseConfig:
    """G4 isolation: aleatoric subpath only, no shaping losses, minimal evidential objective."""
    cfg = p4_head_decouple_phase_config(
        lr=lr,
        epochs=epochs,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        phase_name="Phase 4 G4 ale-only unshaped isolation",
    )
    return cfg.model_copy(
        update={
            "name": "Phase 4 G4 ale-only unshaped isolation",
            "require_tau_ale_elevation_save": False,
            "require_g4_holdout_p8_save": False,
            "aleatoric_only_train": True,
            "coeffs": cfg.coeffs.model_copy(
                update={
                    "evidential_coeff": 0.01,
                    "balance_coeff": 0.0,
                    "cone_coeff": 0.0,
                    "neighborhood_coeff": 0.0,
                    "angular_coeff": 0.0,
                    "domain_sep_2d_coeff": 0.0,
                    "domain_sep_3d_coeff": 0.0,
                    "cone_depth_anticollapse_coeff": 0.0,
                    "shell_corr_coeff": 0.0,
                    "shell_corr_epi_sasa_weight": 0.0,
                    "disc_occupancy_coeff": 0.0,
                    "disc_pc_repulsion_coeff": 0.0,
                    "disc_eff_rank_coeff": 0.0,
                    "disc_batch_diversity_coeff": 0.0,
                    "disc_path_align_coeff": 0.0,
                    "disc_thickness_floor_coeff": 0.0,
                    "disc_origin_span_floor_coeff": 0.0,
                    "x_hyp_thickness_floor_coeff": 0.0,
                    "routing_load_floor_coeff": 0.0,
                    "epistemic_decoupling_coeff": 0.0,
                    "epi_ale_decorrelation_coeff": 0.0,
                    "epistemic_bf_align_coeff": 0.0,
                    "epistemic_sasa_pen_coeff": 0.0,
                    "epistemic_anticollapse_coeff": 0.0,
                    "v3_aleatoric_shaping_coeff": 0.0,
                }
            ),
        }
    )


def p4_head_decouple_decorr_only_phase_config(
    lr: float = 5e-5,
    epochs: int = 20,
    *,
    max_probe_r_epi_sasa_save: float = 0.78,
) -> PhaseConfig:
    """G3 ablation A: split epi/ale trunks + r(epi,ale) penalty — NO B-factor/SASA supervision."""
    cfg = p4_head_decouple_phase_config(
        lr=lr,
        epochs=epochs,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        phase_name="Phase 4 head decouple decorr-only (G3 ablation A)",
    )
    return cfg.model_copy(
        update={
            "name": "Phase 4 head decouple decorr-only (G3 ablation A)",
            "coeffs": cfg.coeffs.model_copy(
                update={
                    "epistemic_decoupling_coeff": 0.0,
                    "epistemic_bf_align_coeff": 0.0,
                    "epistemic_sasa_pen_coeff": 0.0,
                }
            ),
            "require_tau_ale_elevation_save": True,
        }
    )


def p4_gate_promotion_phase_config(
    lr: float = 3e-5,
    epochs: int = 20,
    *,
    max_probe_r_epi_sasa_save: float = 0.79,
    balance_coeff: float = 0.06,
    routing_load_floor_coeff: float = 12.0,
    routing_load_floor_min: float = 0.10,
) -> PhaseConfig:
    """Gate-only pass after head decouple: reduce routing H while preserving uncertainty gates."""
    return PhaseConfig(
        phase=2,
        name="Phase 2 gate promotion after head decouple",
        epochs=epochs,
        lr=lr,
        freeze_radial=True,
        freeze_angular=True,
        freeze_backbone=True,
        freeze_gate=False,
        gate_only_train=True,
        expert_dropout_p=0.12,
        expert_dropout_ramp_epochs=5,
        min_probe_r_depth_sasa_save=0.55,
        min_disc_line_thickness_save=0.04,
        max_probe_r_epi_ale_save=0.70,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        min_epistemic_std_save=0.05,
        min_aleatoric_std_save=0.5,
        routing_save_ceiling_start=1.38,
        routing_save_ceiling_final=1.18,
        routing_save_ceiling_ramp_epochs=epochs,
        p2_bridge=True,
        coeffs=LossCoeffs(
            balance_coeff=balance_coeff,
            routing_load_floor_coeff=routing_load_floor_coeff,
            routing_load_floor_min=routing_load_floor_min,
            evidential_coeff=0.0,
            cone_coeff=0.02,
            neighborhood_coeff=0.02,
            angular_coeff=0.0,
            domain_sep_2d_coeff=0.0,
            domain_sep_3d_coeff=0.0,
            cone_depth_anticollapse_coeff=0.05,
            shell_corr_coeff=0.05,
            disc_occupancy_coeff=0.05,
            disc_occupancy_min_sigma_ratio=0.30,
            epistemic_decoupling_coeff=0.0,
            epi_ale_decorrelation_coeff=0.0,
        ),
    )


def p4_corpus25_gate_phase_config(
    lr: float = 3e-5,
    epochs: int = 30,
    *,
    max_probe_r_epi_sasa_save: float = 0.79,
    balance_coeff: float = 0.08,
    routing_load_floor_coeff: float = 14.0,
    routing_load_floor_min: float = 0.10,
) -> PhaseConfig:
    """Gate-only on locked 25-protein Stage A: moderate MoE pressure vs v5_strong."""
    base = p4_gate_promotion_phase_config(
        lr=lr,
        epochs=epochs,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        balance_coeff=balance_coeff,
        routing_load_floor_coeff=routing_load_floor_coeff,
        routing_load_floor_min=routing_load_floor_min,
    )
    return base.model_copy(
        update={
            "name": "Phase 2 corpus-25 gate promotion",
        }
    )


def p4_gate_uncertainty_touchup_phase_config(
    lr: float = 5e-5,
    epochs: int = 15,
    *,
    max_probe_r_epi_sasa_save: float = 0.79,
) -> PhaseConfig:
    """Re-lock uncertainty after gate routing shift (uncertainty head only, gate frozen)."""
    return PhaseConfig(
        phase=4,
        name="Phase 4 gate touchup uncertainty recalibration",
        epochs=epochs,
        lr=lr,
        freeze_radial=True,
        freeze_angular=True,
        freeze_backbone=True,
        freeze_gate=True,
        expert_dropout_p=0.0,
        epistemic_decoupling_ramp_epochs=4,
        epistemic_bf_align_coeff_final=0.22,
        epistemic_sasa_pen_coeff_final=0.12,
        epistemic_staged_decoupling=True,
        epistemic_bf_only_epochs=5,
        epistemic_sasa_pen_cap=0.06,
        epistemic_sasa_pen_cap_epochs=6,
        epistemic_uncertainty_only_train=True,
        min_disc_line_thickness_save=0.04,
        min_probe_r_depth_sasa_save=0.55,
        max_probe_r_epi_ale_save=0.70,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
        min_epistemic_std_save=0.05,
        min_aleatoric_std_save=0.5,
        # Gate frozen — do not block touchup saves on routing H.
        routing_save_ceiling_start=1.45,
        routing_save_ceiling_final=1.45,
        routing_save_ceiling_ramp_epochs=1,
        coeffs=LossCoeffs(
            evidential_coeff=0.005,
            balance_coeff=0.0,
            cone_coeff=0.0,
            neighborhood_coeff=0.0,
            angular_coeff=0.0,
            domain_sep_2d_coeff=0.0,
            domain_sep_3d_coeff=0.0,
            epistemic_decoupling_coeff=1.0,
            epi_ale_decorrelation_coeff=0.6,
            epistemic_bf_align_coeff=0.0,
            epistemic_sasa_pen_coeff=0.0,
            epistemic_anticollapse_coeff=0.10,
            epistemic_min_epi_std=0.03,
        ),
    )


def p4_corpus25_touchup_extended_phase_config(
    lr: float = 5e-5,
    epochs: int = 25,
    *,
    max_probe_r_epi_sasa_save: float = 0.79,
) -> PhaseConfig:
    """Extended routed uncertainty recal: rebuild r(epi,sasa) while gate stays frozen."""
    base = p4_gate_uncertainty_touchup_phase_config(
        lr=lr,
        epochs=epochs,
        max_probe_r_epi_sasa_save=max_probe_r_epi_sasa_save,
    )
    return base.model_copy(
        update={
            "name": "Phase 4 corpus expand extended uncertainty touchup",
            "epistemic_decoupling_ramp_epochs": 6,
            "epistemic_bf_only_epochs": 6,
            "epistemic_sasa_pen_cap_epochs": 10,
            "epistemic_sasa_pen_cap": 0.10,
            "epistemic_sasa_pen_coeff_final": 0.18,
            "coeffs": base.coeffs.model_copy(
                update={
                    "epi_ale_decorrelation_coeff": 0.7,
                    "epistemic_sasa_pen_coeff": 0.0,
                }
            ),
        }
    )


def residue_stage2_phase_config(
    lr: float = 1e-4,
    epochs: int = 30,
) -> PhaseConfig:
    """ResidueStage2: pipeline cryptic pocket + source-leak BCE (gate frozen)."""
    return PhaseConfig(
        phase=1,
        name="ResidueStage2: pipeline cryptic + source-leak BCE (gate frozen)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=True,
        expert_dropout_p=0.0,
        coeffs=LossCoeffs(
            evidential_coeff=0.001,
            balance_coeff=0.001,
            cone_coeff=0.10,
            neighborhood_coeff=0.10,
            angular_coeff=0.05,
            domain_sep_2d_coeff=0.05,
            domain_sep_3d_coeff=0.05,
            cone_depth_anticollapse_coeff=0.30,
            shell_corr_coeff=0.15,
            disc_occupancy_coeff=0.25,
            disc_occupancy_min_sigma_ratio=0.35,
            pocket_bce_coeff=0.5,
            interface_bce_coeff=0.25,
            leak_bce_coeff=1.5,
        ),
    )


def residue_stage1_phase_config(
    lr: float = 1e-4,
    epochs: int = 30,
) -> PhaseConfig:
    """ResidueStage1: pocket + interface BCE on stable small-corpus manifold."""
    return PhaseConfig(
        phase=1,
        name="ResidueStage1: pocket + interface BCE (gate frozen)",
        epochs=epochs,
        lr=lr,
        freeze_radial=False,
        freeze_angular=False,
        freeze_backbone=False,
        freeze_gate=True,
        expert_dropout_p=0.0,
        coeffs=LossCoeffs(
            evidential_coeff=0.001,
            balance_coeff=0.001,
            cone_coeff=0.10,
            neighborhood_coeff=0.10,
            angular_coeff=0.05,
            domain_sep_2d_coeff=0.05,
            domain_sep_3d_coeff=0.05,
            cone_depth_anticollapse_coeff=0.30,
            shell_corr_coeff=0.15,
            disc_occupancy_coeff=0.25,
            disc_occupancy_min_sigma_ratio=0.35,
            pocket_bce_coeff=1.0,
            interface_bce_coeff=0.5,
        ),
    )


class PromotionConfig(BaseModel):
    """Checkpoint promotion into onboard contract."""

    model_config = ConfigDict(protected_namespaces=())

    checkpoint_path: Path
    checkpoint_id: str = "tokyo_eyes_v6_candidate"
    model_id: str = "gospc_v6"
    status: Literal["candidate", "production"] = "candidate"
    run_id: str | None = None

    def model_post_init(self, __context: object) -> None:
        self.checkpoint_path = Path(self.checkpoint_path)
