"""Pydantic training configuration for v6 GNN lifecycle runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

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
    pocket_bce_coeff: float = 0.0
    interface_bce_coeff: float = 0.0
    leak_bce_coeff: float = 0.0


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
    min_disc_sigma2_sigma1_save: float | None = None  # ineligible v6_best if disc streak
    min_disc_r_std_save: float | None = None  # ineligible if radial spread collapsed
    min_disc_line_thickness_save: float | None = None  # ineligible if rank-1 streak (visual)
    min_disc_effective_rank_save: float | None = None
    # P2 bridge: relaxed routing save ceiling ramp (saturated P1 → standard P2)
    p2_bridge: bool = False
    routing_save_ceiling_start: float | None = None
    routing_save_ceiling_final: float | None = None
    routing_save_ceiling_ramp_epochs: int = 0
    expert_dropout_ramp_epochs: int = 0  # hold dropout at 0, then ramp to expert_dropout_p
    angular_ramp_epochs: int = 0  # P2 bridge angular/domain ramp length (overrides default P2 warmup)
    path_alignment_train: bool = False  # only train disc projection + gate disc readout
    rec_ablation_train: bool = False  # only train radial_angular_fusion + hyp_proj_head_2d
    projection_recovery_train: bool = False  # train hyp_proj_head_2d + gate disc readout
    fusion_path_recovery_train: bool = False  # angular + fusion + disc proj + gate disc readout
    lift_path_recovery_train: bool = False  # radial + angular + fusion + disc (fix x_hyp wedge)
    epistemic_decoupling_ramp_epochs: int = 0
    epistemic_bf_align_coeff_final: float | None = None
    epistemic_sasa_pen_coeff_final: float | None = None
    epistemic_uncertainty_only_train: bool = False
    gate_only_train: bool = False
    # Option B staged decoupling: λ₁-only phase, then capped/log λ₂ ramp
    epistemic_staged_decoupling: bool = False
    epistemic_bf_only_epochs: int = 10
    epistemic_sasa_pen_cap: float = 0.05
    epistemic_sasa_pen_cap_epochs: int = 5
    max_probe_r_epi_ale_save: float | None = None
    max_probe_r_epi_sasa_save: float | None = None
    min_epistemic_std_save: float | None = None
    min_aleatoric_std_save: float | None = None


def routing_save_max_for_epoch(phase_cfg: PhaseConfig, epoch: int) -> float | None:
    """Interpolate routing_H save ceiling across a P2 bridge phase."""
    if phase_cfg.routing_save_ceiling_start is None:
        return None
    from science.training.checkpoint_score import ROUTING_ENTROPY_SAVE_MAX

    start = phase_cfg.routing_save_ceiling_start
    final = phase_cfg.routing_save_ceiling_final or ROUTING_ENTROPY_SAVE_MAX
    n = phase_cfg.routing_save_ceiling_ramp_epochs or phase_cfg.epochs
    if n <= 1:
        return final
    t = min(epoch, n - 1) / (n - 1)
    return start + t * (final - start)


class TrainingConfig(BaseModel):
    """Full v6 training run configuration."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str = "GOSPConeMapper-v6"
    device: str = "cpu"
    lr: float = 5e-4
    hidden: int = 128
    num_layers: int = 6
    num_experts: int = 4
    output_dir: Path = Path("checkpoints/v6/runs")
    pdb_dir: Path = Path("/tmp/dtie_pdb_cache")
    corpus_manifest: Path = Path("manifests/v6_corpus_120.json")
    phase: int | None = None  # None = full 3-phase curriculum
    resume: Path | None = None
    warm_start_v5: Path | None = None
    mlflow_experiment: str = "tokyo-eyes-v6"
    mlflow_tracking_uri: str = "file:/app/mlruns"
    max_proteins: int | None = None
    max_residues: int = STAGE_A_MAX_RESIDUES
    topology_only_gate: bool = False
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

    def model_post_init(self, __context: object) -> None:
        self.output_dir = Path(self.output_dir)
        self.pdb_dir = Path(self.pdb_dir)
        self.corpus_manifest = Path(self.corpus_manifest)
        if self.resume is not None:
            self.resume = Path(self.resume)
        if self.v2_teacher_checkpoint is not None:
            self.v2_teacher_checkpoint = Path(self.v2_teacher_checkpoint)

    def phase_preset_name(self) -> str | None:
        """Stable curriculum preset id for MLflow tags."""
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
            flat[key] = str(value) if isinstance(value, Path) else value
        return flat


def default_v6_phases(
    base_lr: float = 5e-4,
    *,
    gentle_phase2: bool = False,
    phase2_lr: float | None = None,
) -> list[PhaseConfig]:
    """Three-phase MoE specialization schedule per v6-moe spec."""
    p2_lr = phase2_lr if phase2_lr is not None else (base_lr * 0.2 if gentle_phase2 else base_lr)
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
        if phase_cfg.angular_coeff_final is not None and phase_cfg.coeffs.domain_sep_2d_coeff > 0:
            sep_scale = 0.5 + 0.5 * t
            out["domain_sep_2d_coeff"] = phase_cfg.coeffs.domain_sep_2d_coeff * sep_scale
            out["domain_sep_3d_coeff"] = phase_cfg.coeffs.domain_sep_3d_coeff * sep_scale
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
            out["epistemic_bf_align_coeff"] = t_epi * phase_cfg.epistemic_bf_align_coeff_final
        if phase_cfg.epistemic_sasa_pen_coeff_final is not None:
            out["epistemic_sasa_pen_coeff"] = t_epi * phase_cfg.epistemic_sasa_pen_coeff_final
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 10
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 15
    base = p2_bridge_phase_config(lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp)
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 15
    base = p2_hypmix_phase_config(lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp)
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 15
    base = p2_hypmix_phase_config(lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp)
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 15
    base = p2_hypmix_final_phase_config(lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp)
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 12
    base = p2_hypmix_final_phase_config(lr=lr, epochs=epochs, routing_save_ceiling_ramp_epochs=ramp)
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 8
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 10
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 10
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 10
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 10
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
    ramp = routing_save_ceiling_ramp_epochs if routing_save_ceiling_ramp_epochs is not None else 10
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
