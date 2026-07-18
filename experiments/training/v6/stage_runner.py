"""Stage-selective v6 training orchestration."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch

from science.training.topology_depth import topology_depth_lineage
from science.training.gnn_lineage import get_lineage
from science.training.checkpoint import CheckpointData, CheckpointManager
from science.training.checkpoint_score import score_checkpoint
from science.training.config import (
    PhaseConfig,
    TrainingConfig,
    apply_phase_coeff_ramp,
    apply_master_cold_dehydron_phases,
    apply_routing_load_floor_phase2,
    apply_slim_moe_structural_ssot_phases,
    default_v6_phases,
    dehydron_rim_recovery_phase_config,
    topology_routing_recovery_phase_config,
    topology_gate_disc_recovery_phase_config,
    topology_crescent_recovery_phase_config,
    p1b_phase_config,
    p1c_phase_config,
    p1d_extend_defaults,
    p1d_phase_config,
    p2_bridge_phase_config,
    p2_hypmix_phase_config,
    p2_hypmix3_phase_config,
    full_hyp_moe_theory_config,
    p2_hypmix_final_phase_config,
    p2_disc_occupancy_phase_config,
    p2_disc_occupancy_v2_phase_config,
    p2_disc_occupancy_v3_phase_config,
    p2_disc_occupancy_v4_phase_config,
    p2_disc_occupancy_v5_phase_config,
    p2_disc_gentle_arch_phase_config,
    p2_disc_path_align_phase_config,
    p2_disc_proj_recovery_phase_config,
    p2_disc_proj_recovery_v2_phase_config,
    p2_disc_proj_recovery_v3_phase_config,
    p2_disc_proj_recovery_v4_phase_config,
    p2_disc_proj_recovery_v5_phase_config,
    p2_rec_ablation_phase_config,
    p4_epistemic_decoupling_phase_config,
    p4_head_decouple_phase_config,
    p4_head_decouple_decorr_only_phase_config,
    p4_v3_aleatoric_shaping_phase_config,
    p4_g4_shaping_only_isolation_phase_config,
    p4_g4_ale_only_unshaped_phase_config,
    p4_corpus25_gate_phase_config,
    p4_corpus25_touchup_extended_phase_config,
    p4_gate_promotion_phase_config,
    p4_gate_uncertainty_touchup_phase_config,
    p4_uncertainty_calibration_phase_config,
    residue_stage1_phase_config,
    residue_stage2_phase_config,
    routing_save_max_for_epoch,
    routing_save_ceiling_for_display,
)
from science.training.expert_timeout import ExpertTimeoutController
from science.training.monitor import ConvergenceMonitor
from science.training.p3_entry_gate import p3_entry_gate_verdict
from science.training.dehydron_cone_gate import dehydron_cone_gate_verdict
from science.training.tracking import TrainingTracker
from experiments.training.v6.train_loop import (
    build_p4_optimizer,
    measure_geometry_health,
    set_expert_dropout,
    set_p4_aleatoric_only_freeze,
    set_p4_uncertainty_only_freeze,
    set_gate_only_freeze,
    set_routing_load_floor_min,
    set_uncertainty_from_backbone,
    train_epoch,
)

logger = logging.getLogger(__name__)


class StageRunner:
    """Executes v6 MoE three-phase curriculum with optional stage selection."""

    def __init__(
        self,
        model: torch.nn.Module,
        proteins: list[dict[str, Any]],
        config: TrainingConfig,
        tracker: TrainingTracker | None = None,
        resume_state: CheckpointData | None = None,
        v2_teacher: Any | None = None,
    ) -> None:
        self.model = model
        self.proteins = proteins
        self.config = config
        self.tracker = tracker
        self.v2_teacher = v2_teacher
        # P1 MoE-alive hard timeout (share>40% → ban 2 epochs); None outside slim P1.
        self._expert_timeout: ExpertTimeoutController | None = None
        self.checkpoint_mgr = CheckpointManager(
            config.output_dir,
            len(proteins),
            checkpoint_prefix=get_lineage(config.gnn_lineage).checkpoint_prefix,
            architecture_version=get_lineage(config.gnn_lineage).architecture_version,
        )
        self.monitor = ConvergenceMonitor()
        metrics_path = config.output_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                import json

                self.metrics_log = json.loads(metrics_path.read_text())
            except (json.JSONDecodeError, OSError):
                self.metrics_log = []
        else:
            self.metrics_log = []
        # Cross-run P3 resume: seed P2 routing history from the resume checkpoint's
        # sibling metrics.json so P3_ENTRY_GATE can see prior phase-2 entropy.
        if (
            not self.metrics_log
            and resume_state is not None
            and config.resume is not None
        ):
            sibling = Path(config.resume).resolve().parent / "metrics.json"
            if sibling.is_file() and sibling != metrics_path.resolve():
                try:
                    import json

                    seeded = json.loads(sibling.read_text())
                    if isinstance(seeded, list) and seeded:
                        self.metrics_log = seeded
                        logger.info(
                            "Seeded metrics_log (%d epochs) from resume sibling %s",
                            len(seeded),
                            sibling,
                        )
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Could not seed metrics from %s: %s", sibling, exc)
        self.global_epoch = resume_state.global_epoch if resume_state else 0
        self.best_score = resume_state.score if resume_state else -math.inf
        self._saved_eligible = resume_state is not None and resume_state.score > -math.inf
        self._resume_checkpoint_phase = resume_state.phase if resume_state else None
        self._shell_low_streak = 0
        self._last_focus_summary: dict[str, Any] | None = None
        self._best_disc_sigma = -1.0
        self._best_disc_visual_score = -1.0
        self._best_route_score = -math.inf
        holdout_raw = getattr(config, "epistemic_decoupling_holdouts", "") or ""
        self._epistemic_holdouts = frozenset(
            s.strip().upper() for s in holdout_raw.split(",") if s.strip()
        )
        self._topology_depth = topology_depth_lineage(
            master_cold=config.master_cold_lineage
            or config.slim_moe_structural_ssot
            or config.topology_routing_recovery
            or config.topology_gate_disc_recovery
            or config.topology_crescent_recovery
        )

    def _resolve_disc_proj_recovery_phase(self) -> PhaseConfig:
        epochs = self.config.epochs_override or 20
        thickness_floor = self.config.p2_disc_line_thickness_floor or 0.025
        if self.config.p2_disc_proj_recovery_v5:
            return p2_disc_proj_recovery_v5_phase_config(
                lr=self.config.p2_bridge_lr,
                epochs=epochs,
                disc_thickness_floor_min=thickness_floor,
            )
        if self.config.p2_disc_proj_recovery_v4:
            return p2_disc_proj_recovery_v4_phase_config(
                lr=self.config.p2_bridge_lr,
                epochs=epochs,
                disc_thickness_floor_min=thickness_floor,
            )
        if self.config.p2_disc_proj_recovery_v3:
            return p2_disc_proj_recovery_v3_phase_config(
                lr=self.config.p2_bridge_lr,
                epochs=epochs,
                disc_thickness_floor_min=thickness_floor,
            )
        if self.config.p2_disc_proj_recovery_v2:
            return p2_disc_proj_recovery_v2_phase_config(
                lr=self.config.p2_bridge_lr,
                epochs=epochs,
                disc_thickness_floor_min=thickness_floor,
            )
        return p2_disc_proj_recovery_phase_config(
            lr=self.config.p2_bridge_lr,
            epochs=epochs,
            disc_thickness_floor_min=thickness_floor,
        )

    def _resolve_rec_ablation_phase(self) -> PhaseConfig:
        epochs = self.config.epochs_override or 5
        return p2_rec_ablation_phase_config(
            lr=self.config.p2_bridge_lr,
            epochs=epochs,
        )

    def _resolve_disc_path_align_phase(self) -> PhaseConfig:
        epochs = self.config.epochs_override or 8
        coeff = self.config.p2_disc_path_align_coeff or 3.0
        thickness_floor = self.config.p2_disc_line_thickness_floor or 0.02
        return p2_disc_path_align_phase_config(
            lr=self.config.p2_bridge_lr,
            epochs=epochs,
            disc_path_align_coeff=coeff,
            min_disc_line_thickness_save=thickness_floor,
        )

    def _resolve_disc_gentle_arch_phase(self) -> PhaseConfig:
        epochs = self.config.epochs_override or 10
        ramp = self.config.p2_bridge_ramp_epochs
        return p2_disc_gentle_arch_phase_config(
            lr=self.config.p2_bridge_lr,
            epochs=epochs,
            routing_save_ceiling_ramp_epochs=ramp,
            disc_occupancy_coeff=self.config.p2_disc_occupancy_coeff or 1.2,
            freeze_radial_epochs=self.config.p2_radial_freeze_epochs or 4,
            min_disc_r_std_save=self.config.p2_disc_r_std_floor or 0.04,
            min_disc_line_thickness_save=self.config.p2_disc_line_thickness_floor or 0.02,
        )

    def _begin_phase_best_tracking(self, phase: int) -> None:
        """Reset v6_best promotion baseline when entering a new phase.

        Gate-only and touchup phases use different loss compositions, so a high
        gate score must not block touchup saves. Same-phase resume keeps the
        inherited best_score for mid-phase continuation.
        """
        if self._resume_checkpoint_phase == phase:
            self._resume_checkpoint_phase = None
            return
        if self.best_score > -math.inf:
            logger.info(
                "  v6_best baseline reset for phase %d (prior best score %.4f not comparable)",
                phase,
                self.best_score,
            )
        self.best_score = -math.inf
        self._saved_eligible = False
        self._best_disc_sigma = -1.0
        self._best_disc_visual_score = -1.0
        self._resume_checkpoint_phase = None

    def _maybe_export_disc_scatter(self, phase_cfg: PhaseConfig) -> None:
        interval = self.config.disc_scatter_interval_epochs
        if not interval or self.global_epoch % interval != 0:
            return
        spec = self.config.disc_scatter_structure.strip()
        if ":" in spec:
            pdb_id, chain = spec.split(":", 1)
        else:
            pdb_id, chain = spec, "A"
        from experiments.diagnostics.embedding_occupancy_audit import export_disc_scatter_from_model

        out_dir = self.checkpoint_mgr.output_dir / "disc_scatter"
        out_path = out_dir / f"epoch_{self.global_epoch:03d}_{pdb_id.lower()}.png"
        try:
            export_disc_scatter_from_model(
                self.model,
                (pdb_id.upper(), chain),
                self.config.pdb_dir,
                out_path,
                device=self.config.device,
                title_suffix=f"{phase_cfg.name} ep{self.global_epoch}",
                legacy_disc_projection=self.config.legacy_disc_projection,
            )
            logger.info("    disc scatter → %s", out_path.name)
        except Exception as exc:
            logger.warning("    disc scatter export failed: %s", exc)

    def _resolve_disc_occupancy_phase(self) -> PhaseConfig:
        epochs = self.config.epochs_override or (
            12
            if (
                self.config.p2_disc_occupancy_v2
                or self.config.p2_disc_occupancy_v3
                or self.config.p2_disc_occupancy_v4
                or self.config.p2_disc_occupancy_v5
            )
            else 15
        )
        ramp = self.config.p2_bridge_ramp_epochs
        lr = self.config.p2_bridge_lr
        coeff = self.config.p2_disc_occupancy_coeff
        freeze = self.config.p2_radial_freeze_epochs
        if self.config.p2_disc_occupancy_v5:
            kwargs: dict[str, Any] = {"lr": lr, "epochs": epochs, "routing_save_ceiling_ramp_epochs": ramp}
            if coeff is not None:
                kwargs["disc_occupancy_coeff"] = coeff
            if freeze is not None:
                kwargs["freeze_radial_epochs"] = freeze
            if self.config.p2_disc_r_std_floor is not None:
                kwargs["min_disc_r_std_save"] = self.config.p2_disc_r_std_floor
            if self.config.p2_disc_eff_rank_coeff is not None:
                kwargs["disc_eff_rank_coeff"] = self.config.p2_disc_eff_rank_coeff
            if self.config.p2_disc_batch_diversity_coeff is not None:
                kwargs["disc_batch_diversity_coeff"] = self.config.p2_disc_batch_diversity_coeff
            return p2_disc_occupancy_v5_phase_config(**kwargs)
        if self.config.p2_disc_occupancy_v4:
            kwargs: dict[str, Any] = {"lr": lr, "epochs": epochs, "routing_save_ceiling_ramp_epochs": ramp}
            if coeff is not None:
                kwargs["disc_occupancy_coeff"] = coeff
            if freeze is not None:
                kwargs["freeze_radial_epochs"] = freeze
            if self.config.p2_disc_r_std_floor is not None:
                kwargs["min_disc_r_std_save"] = self.config.p2_disc_r_std_floor
            if self.config.p2_disc_eff_rank_coeff is not None:
                kwargs["disc_eff_rank_coeff"] = self.config.p2_disc_eff_rank_coeff
            return p2_disc_occupancy_v4_phase_config(**kwargs)
        if self.config.p2_disc_occupancy_v3:
            kwargs: dict[str, Any] = {"lr": lr, "epochs": epochs, "routing_save_ceiling_ramp_epochs": ramp}
            if coeff is not None:
                kwargs["disc_occupancy_coeff"] = coeff
            if freeze is not None:
                kwargs["freeze_radial_epochs"] = freeze
            if self.config.p2_disc_r_std_floor is not None:
                kwargs["min_disc_r_std_save"] = self.config.p2_disc_r_std_floor
            return p2_disc_occupancy_v3_phase_config(**kwargs)
        if self.config.p2_disc_occupancy_v2:
            kwargs: dict[str, Any] = {"lr": lr, "epochs": epochs, "routing_save_ceiling_ramp_epochs": ramp}
            if coeff is not None:
                kwargs["disc_occupancy_coeff"] = coeff
            if freeze is not None:
                kwargs["freeze_radial_epochs"] = freeze
            if self.config.p2_disc_r_std_floor is not None:
                kwargs["min_disc_r_std_save"] = self.config.p2_disc_r_std_floor
            return p2_disc_occupancy_v2_phase_config(**kwargs)
        kwargs = {"lr": lr, "epochs": epochs, "routing_save_ceiling_ramp_epochs": ramp}
        if coeff is not None:
            kwargs["disc_occupancy_coeff"] = coeff
        if freeze is not None:
            kwargs["freeze_radial_epochs"] = freeze
        return p2_disc_occupancy_phase_config(**kwargs)

    def _resolve_p1d_phase(self) -> PhaseConfig:
        epochs = self.config.epochs_override or 5
        kwargs: dict[str, float | int | bool] = {}
        if self.config.p1d_extend:
            kwargs.update(p1d_extend_defaults(epochs=epochs))
        else:
            kwargs["epochs"] = epochs

        def _pick(key: str, cfg_attr: str, default: float | int) -> float | int:
            override = getattr(self.config, cfg_attr, None)
            if override is not None:
                return override
            return kwargs.get(key, default)

        return p1d_phase_config(
            lr=self.config.p1d_lr,
            epochs=int(kwargs.get("epochs", epochs)),
            disc_depth_scale_coeff=float(
                _pick("disc_depth_scale_coeff", "p1d_disc_depth_scale_coeff", 1.5)
            ),
            disc_target_start=float(
                _pick("disc_target_start", "p1d_disc_target_start", 0.20)
            ),
            disc_target_end=float(_pick("disc_target_end", "p1d_disc_target_end", 0.45)),
            freeze_radial_epochs=int(
                _pick("freeze_radial_epochs", "p1d_freeze_radial_epochs", 2)
            ),
            min_probe_r_depth_sasa=float(
                _pick("min_probe_r_depth_sasa", "p1d_min_probe_r_depth_sasa", 0.44)
            ),
            disc_spread_min_std=float(
                _pick("disc_spread_min_std", "p1d_disc_spread_min_std", 0.08)
            ),
            extend=bool(kwargs.get("extend", False)),
        )

    def _resolve_p4_epistemic_phase(self) -> PhaseConfig:
        epochs = self.config.epochs_override or 30
        ramp = self.config.p2_bridge_ramp_epochs
        thickness = self.config.p2_disc_line_thickness_floor or 0.18
        bf = self.config.epistemic_bf_align_coeff
        sasa = self.config.epistemic_sasa_pen_coeff
        shell_epi = self.config.shell_corr_epi_sasa_weight
        return p4_epistemic_decoupling_phase_config(
            lr=self.config.p4_epistemic_lr,
            epochs=epochs,
            routing_save_ceiling_ramp_epochs=ramp,
            disc_occupancy_coeff=self.config.p2_disc_occupancy_coeff or 3.5,
            disc_eff_rank_coeff=self.config.p2_disc_eff_rank_coeff or 1.0,
            disc_line_thickness_floor_coeff=0.5,
            disc_thickness_floor_min=thickness,
            epistemic_bf_align_coeff=bf if bf is not None else 0.22,
            epistemic_sasa_pen_coeff=sasa if sasa is not None else 0.246,
            shell_corr_epi_sasa_weight=shell_epi if shell_epi is not None else 0.5,
            staged_decoupling=bool(self.config.p4_epistemic_staged),
        )

    def _phases(self) -> list[PhaseConfig]:
        if self.config.residue_stage2:
            epochs = self.config.epochs_override or self.config.residue_stage2_epochs or 30
            return [
                residue_stage2_phase_config(
                    lr=self.config.residue_stage2_lr,
                    epochs=epochs,
                )
            ]
        if self.config.residue_stage1:
            epochs = self.config.epochs_override or self.config.residue_stage1_epochs or 30
            return [
                residue_stage1_phase_config(
                    lr=self.config.residue_stage1_lr,
                    epochs=epochs,
                )
            ]
        if self.config.p4_uncertainty_calibration:
            epochs = self.config.epochs_override or 25
            return [
                p4_uncertainty_calibration_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                )
            ]
        if self.config.p4_head_decouple_decorr_only:
            epochs = self.config.epochs_override or 20
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.78
            )
            return [
                p4_head_decouple_decorr_only_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                )
            ]
        if self.config.p4_v3_aleatoric_shaping:
            epochs = self.config.epochs_override or 20
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.78
            )
            return [
                p4_v3_aleatoric_shaping_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                    w_var_penalty=(
                        self.config.w_var_penalty
                        if self.config.w_var_penalty is not None
                        else 2.8
                    ),
                )
            ]
        if self.config.p4_g4_shaping_only_isolation:
            epochs = self.config.epochs_override or 12
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.78
            )
            return [
                p4_g4_shaping_only_isolation_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                    w_var_penalty=(
                        self.config.w_var_penalty
                        if self.config.w_var_penalty is not None
                        else 2.8
                    ),
                )
            ]
        if self.config.p4_g4_ale_only_unshaped:
            epochs = self.config.epochs_override or 12
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.78
            )
            return [
                p4_g4_ale_only_unshaped_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                )
            ]
        if self.config.p4_head_decouple:
            epochs = self.config.epochs_override or 20
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.78
            )
            return [
                p4_head_decouple_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                )
            ]
        if self.config.p4_corpus25_gate_promotion:
            epochs = self.config.epochs_override or 30
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.79
            )
            gate_kw: dict[str, float] = {}
            if self.config.p4_gate_balance_coeff is not None:
                gate_kw["balance_coeff"] = self.config.p4_gate_balance_coeff
            if self.config.p4_gate_load_floor_coeff is not None:
                gate_kw["routing_load_floor_coeff"] = self.config.p4_gate_load_floor_coeff
            if self.config.p4_gate_load_floor_min is not None:
                gate_kw["routing_load_floor_min"] = self.config.p4_gate_load_floor_min
            return [
                p4_corpus25_gate_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                    **gate_kw,
                )
            ]
        if self.config.p4_gate_promotion:
            epochs = self.config.epochs_override or 20
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.79
            )
            gate_kw = {}
            if self.config.p4_gate_balance_coeff is not None:
                gate_kw["balance_coeff"] = self.config.p4_gate_balance_coeff
            if self.config.p4_gate_load_floor_coeff is not None:
                gate_kw["routing_load_floor_coeff"] = self.config.p4_gate_load_floor_coeff
            if self.config.p4_gate_load_floor_min is not None:
                gate_kw["routing_load_floor_min"] = self.config.p4_gate_load_floor_min
            return [
                p4_gate_promotion_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                    **gate_kw,
                )
            ]
        if self.config.p4_corpus25_touchup_extended:
            epochs = self.config.epochs_override or 25
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.79
            )
            return [
                p4_corpus25_touchup_extended_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                )
            ]
        if self.config.p4_gate_uncertainty_touchup:
            epochs = self.config.epochs_override or 8
            sasa_save = (
                self.config.max_probe_r_epi_sasa_save
                if self.config.max_probe_r_epi_sasa_save is not None
                else 0.79
            )
            return [
                p4_gate_uncertainty_touchup_phase_config(
                    lr=self.config.p4_epistemic_lr,
                    epochs=epochs,
                    max_probe_r_epi_sasa_save=sasa_save,
                )
            ]
        if self.config.p4_epistemic_decoupling:
            return [self._resolve_p4_epistemic_phase()]
        if (
            self.config.p2_disc_proj_recovery
            or self.config.p2_disc_proj_recovery_v2
            or self.config.p2_disc_proj_recovery_v3
            or self.config.p2_disc_proj_recovery_v4
            or self.config.p2_disc_proj_recovery_v5
        ):
            return [self._resolve_disc_proj_recovery_phase()]
        if self.config.p2_rec_ablation:
            return [self._resolve_rec_ablation_phase()]
        if self.config.p2_disc_path_align:
            return [self._resolve_disc_path_align_phase()]
        if self.config.p2_disc_gentle_arch:
            return [self._resolve_disc_gentle_arch_phase()]
        if (
            self.config.p2_disc_occupancy_v5
            or self.config.p2_disc_occupancy_v4
            or self.config.p2_disc_occupancy_v3
            or self.config.p2_disc_occupancy_v2
            or self.config.p2_disc_occupancy
        ):
            return [self._resolve_disc_occupancy_phase()]
        if self.config.full_hyp_moe_test:
            epochs = self.config.epochs_override or 15
            ramp = self.config.p2_bridge_ramp_epochs
            return [
                full_hyp_moe_theory_config(
                    lr=self.config.theory_test_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                )
            ]
        if self.config.p2_hypmix_final:
            epochs = self.config.epochs_override or 10
            ramp = self.config.p2_bridge_ramp_epochs
            return [
                p2_hypmix_final_phase_config(
                    lr=self.config.p2_bridge_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                )
            ]
        if self.config.p2_hypmix3:
            epochs = self.config.epochs_override or 12
            ramp = self.config.p2_bridge_ramp_epochs
            return [
                p2_hypmix3_phase_config(
                    lr=self.config.p2_bridge_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                )
            ]
        if self.config.p2_hypmix:
            epochs = self.config.epochs_override or 15
            ramp = self.config.p2_bridge_ramp_epochs
            return [
                p2_hypmix_phase_config(
                    lr=self.config.p2_bridge_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                )
            ]
        if self.config.p2_bridge:
            epochs = self.config.epochs_override or 12
            ramp = self.config.p2_bridge_ramp_epochs
            return [
                p2_bridge_phase_config(
                    lr=self.config.p2_bridge_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                )
            ]
        if self.config.dehydron_rim_recovery:
            epochs = self.config.epochs_override or 12
            floor = self.config.p2_disc_line_thickness_floor or 0.025
            return [
                dehydron_rim_recovery_phase_config(
                    lr=self.config.dehydron_rim_recovery_lr,
                    epochs=epochs,
                    min_disc_line_thickness_save=floor,
                )
            ]
        if self.config.topology_crescent_recovery:
            epochs = self.config.epochs_override or 25
            ramp = self.config.p2_bridge_ramp_epochs
            floor = self.config.p2_disc_line_thickness_floor or 0.03
            return [
                topology_crescent_recovery_phase_config(
                    lr=self.config.topology_crescent_recovery_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                    disc_thickness_floor_min=floor,
                )
            ]
        if self.config.topology_gate_disc_recovery:
            epochs = self.config.epochs_override or 20
            ramp = self.config.p2_bridge_ramp_epochs
            return [
                topology_gate_disc_recovery_phase_config(
                    lr=self.config.topology_gate_disc_recovery_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                )
            ]
        if self.config.topology_routing_recovery:
            epochs = self.config.epochs_override or 40
            ramp = self.config.p2_bridge_ramp_epochs
            phases = [
                topology_routing_recovery_phase_config(
                    lr=self.config.topology_routing_recovery_lr,
                    epochs=epochs,
                    routing_save_ceiling_ramp_epochs=ramp,
                )
            ]
            if self.config.structural_disc_frozen:
                phases = apply_slim_moe_structural_ssot_phases(phases)
            return phases
        if self.config.p1d:
            return [self._resolve_p1d_phase()]
        if self.config.p1c:
            epochs = self.config.epochs_override or 5
            return [p1c_phase_config(lr=self.config.p1c_lr, epochs=epochs)]
        if self.config.p1b:
            epochs = self.config.epochs_override or 3
            return [p1b_phase_config(lr=self.config.p1b_lr, epochs=epochs)]
        phases = default_v6_phases(
            self.config.lr,
            gentle_phase2=self.config.gentle_phase2,
            phase2_lr=self.config.phase2_lr,
        )
        if self.config.routing_load_floor:
            phases = apply_routing_load_floor_phase2(
                phases,
                coeff=self.config.routing_load_floor_coeff,
                min_fraction=self.config.routing_load_floor_min,
            )
        if self.config.slim_moe_structural_ssot:
            phases = apply_slim_moe_structural_ssot_phases(phases)
        elif self.config.master_cold_lineage:
            phases = apply_master_cold_dehydron_phases(phases)
        if self.config.phase is not None:
            phases = [p for p in phases if p.phase == self.config.phase]
        if self.config.epochs_override is not None:
            override = self.config.epochs_override
            phases = [
                p.model_copy(update={"epochs": override}) if hasattr(p, "model_copy") else PhaseConfig(
                    phase=p.phase,
                    name=p.name,
                    epochs=override,
                    lr=p.lr,
                    freeze_radial=p.freeze_radial,
                    freeze_angular=p.freeze_angular,
                    freeze_backbone=p.freeze_backbone,
                    freeze_gate=p.freeze_gate,
                    expert_dropout_p=p.expert_dropout_p,
                    freeze_radial_epochs=p.freeze_radial_epochs,
                    coeffs=p.coeffs,
                )
                for p in phases
            ]
        return phases

    def run(self) -> dict[str, Any]:
        from science.dtie.v5.gnn.model import build_optimizer

        p3_entry_skipped = False
        p3_entry_reason = ""
        p2_dehydron_blocked = False
        p2_dehydron_reason = ""

        for phase_cfg in self._phases():
            if phase_cfg.phase in (2, 3) and p2_dehydron_blocked:
                logger.error(
                    "P_DEHYDRON_CONE_01 skipped Phase %d: %s",
                    phase_cfg.phase,
                    p2_dehydron_reason,
                )
                continue
            if phase_cfg.phase == 3:
                p2_route_h = [
                    float(e["losses"]["routing_entropy"])
                    for e in self.metrics_log
                    if e.get("phase") == 2 and "routing_entropy" in e.get("losses", {})
                ]
                from science.training.routing_gate_bounds import (
                    model_num_experts,
                    routing_entropy_save_ceiling,
                )

                _n_experts = model_num_experts(
                    self.model, default=self.config.num_experts
                )
                # Slim MoE uses a higher H band (~1.20–1.30); align entry gate with save ceiling.
                if self.config.slim_moe_structural_ssot:
                    _p3_ceiling = float(
                        phase_cfg.routing_save_ceiling_final
                        or phase_cfg.routing_save_ceiling_start
                        or 1.30
                    )
                else:
                    _p3_ceiling = routing_entropy_save_ceiling(num_experts=_n_experts)
                p3_gate = p3_entry_gate_verdict(p2_route_h, ceiling=_p3_ceiling)
                if not p3_gate.passed:
                    p3_entry_skipped = True
                    p3_entry_reason = p3_gate.reason
                    logger.error(
                        "P3_ENTRY_GATE blocked Phase 3: %s "
                        "(max_consecutive_below=%d, required=%d, ceiling=%.2f)",
                        p3_gate.reason,
                        p3_gate.max_consecutive_below_ceiling,
                        p3_gate.required_consecutive,
                        p3_gate.ceiling,
                    )
                    continue

            logger.info("=" * 70)
            logger.info("%s (%d epochs, lr=%.2e)", phase_cfg.name, phase_cfg.epochs, phase_cfg.lr)
            if phase_cfg.phase == 2 and phase_cfg.freeze_radial_epochs > 0:
                logger.info(
                    "  P2 gentle: freeze radial for first %d epochs",
                    phase_cfg.freeze_radial_epochs,
                )
            if phase_cfg.coeff_ramp_epochs > 0:
                logger.info(
                    "  Coeff ramp (%d ep): angular %.2f→%.2f | disc_spread_w %.1f→%.1f | disc_min_std %.2f→%.2f",
                    phase_cfg.coeff_ramp_epochs,
                    phase_cfg.coeffs.angular_coeff,
                    phase_cfg.angular_coeff_final or phase_cfg.coeffs.angular_coeff,
                    phase_cfg.coeffs.shell_corr_disc_spread_weight,
                    phase_cfg.shell_corr_disc_spread_weight_final
                    or phase_cfg.coeffs.shell_corr_disc_spread_weight,
                    phase_cfg.coeffs.disc_spread_min_std,
                    phase_cfg.disc_spread_min_std_final or phase_cfg.coeffs.disc_spread_min_std,
                )
            if phase_cfg.disc_depth_scale_target_final is not None:
                logger.info(
                    "  Disc-depth scale ramp: target %.2f→%.2f (coeff=%.2f)",
                    phase_cfg.coeffs.disc_depth_scale_target,
                    phase_cfg.disc_depth_scale_target_final,
                    phase_cfg.coeffs.disc_depth_scale_coeff,
                )
            if phase_cfg.freeze_radial_epochs > 0:
                freeze_note = (
                    "stabilize radial head"
                    if self.config.master_cold_lineage
                    else "protect depth×SASA"
                )
                logger.info(
                    "  Freeze radial for first %d epochs (%s)",
                    phase_cfg.freeze_radial_epochs,
                    freeze_note,
                )
            if phase_cfg.min_probe_r_depth_sasa is not None:
                logger.info(
                    "  Shell guard: abort if r(d,s) < %.2f for 2 consecutive epochs",
                    phase_cfg.min_probe_r_depth_sasa,
                )
            if phase_cfg.routing_save_ceiling_start is not None:
                logger.info(
                    "  Routing save ceiling: %.2f→%.2f over %d epochs",
                    phase_cfg.routing_save_ceiling_start,
                    phase_cfg.routing_save_ceiling_final or 1.21,
                    phase_cfg.routing_save_ceiling_ramp_epochs or phase_cfg.epochs,
                )
            if phase_cfg.min_probe_r_depth_sasa_save is not None:
                logger.info(
                    "  Shell save gate: r(d,s) >= %.2f required for v6_best",
                    phase_cfg.min_probe_r_depth_sasa_save,
                )
            if phase_cfg.p2_bridge and phase_cfg.expert_dropout_ramp_epochs > 0:
                logger.info(
                    "  Expert dropout: 0 for %d ep, then ramp to %.2f",
                    phase_cfg.expert_dropout_ramp_epochs,
                    phase_cfg.expert_dropout_p,
                )
            elif phase_cfg.p2_bridge and phase_cfg.expert_dropout_p > 0:
                logger.info("  Expert dropout: %.2f (constant)", phase_cfg.expert_dropout_p)
            if getattr(self.config, "deep_hyperbolic_gate", False):
                logger.info("  Hyperbolic gate: deep (3× Mobius trunk)")
            if getattr(self.config, "hyperbolic_expert_mix", False):
                logger.info("  Expert mix: Möbius weighted combine on ball")
            if getattr(self.config, "gate_disc_scale", 1.0) != 1.0:
                logger.info("  Gate disc feature scale: %.2f", self.config.gate_disc_scale)
            if getattr(self.config, "gate_gumbel", False):
                logger.info("  Gate routing: Gumbel-Softmax (hard)")
            if getattr(self.config, "expert_depth_decouple", False):
                logger.info("  Expert depth: per-expert radial offsets (decoupled)")
            if getattr(self.config, "structure_gate", False):
                logger.info("  Structure gate: pooled topo bias per protein")
            if getattr(self.config, "topology_crescent_recovery", False):
                logger.info(
                    "  Crescent recovery: angular + fusion + hyp_proj_2d; "
                    "radial + expert_depth_bias frozen; disc wedge floors on"
                )
            if getattr(self.config, "topology_gate_disc_recovery", False):
                logger.info(
                    "  Gate+disc recovery: gate + hyp_proj_2d trainable; expert_depth_bias frozen"
                )
            if getattr(self.config, "track_v6_best_route", False):
                logger.info("  Route checkpoint: v6_best_route.pt (routing-first saves)")
            self._begin_phase_best_tracking(phase_cfg.phase)
            if phase_cfg.coeffs.routing_load_floor_coeff > 0:
                logger.info(
                    "  Routing load floor: λ=%.2f min_share=%.2f (per-structure min-expert hinge)",
                    phase_cfg.coeffs.routing_load_floor_coeff,
                    phase_cfg.coeffs.routing_load_floor_min,
                )
            if phase_cfg.coeffs.routing_load_ceiling_coeff > 0:
                logger.info(
                    "  Routing load ceiling: λ=%.2f max_share=%.2f (anti-dominance hinge)",
                    phase_cfg.coeffs.routing_load_ceiling_coeff,
                    phase_cfg.coeffs.routing_load_ceiling_max,
                )
            if phase_cfg.slim_moe_structural_ssot_train and phase_cfg.freeze_experts:
                logger.info(
                    "  Frozen experts: %s (weights locked; gate may still route to them)",
                    ",".join(f"e{i}" for i in phase_cfg.freeze_experts),
                )
            # Soft expert timeout: slim MoE P1–P3. Default share>~50%; P3 may
            # tighten and restrict to the generalist (e2) only.
            if phase_cfg.slim_moe_structural_ssot_train and phase_cfg.phase in (1, 2, 3):
                from science.training.routing_gate_bounds import model_num_experts

                n_exp = model_num_experts(self.model, default=self.config.num_experts)
                max_share = (
                    float(phase_cfg.expert_timeout_max_share)
                    if phase_cfg.expert_timeout_max_share is not None
                    else 0.50
                )
                eligible = (
                    list(phase_cfg.expert_timeout_eligible_experts)
                    if phase_cfg.expert_timeout_eligible_experts is not None
                    else None
                )
                self._expert_timeout = ExpertTimeoutController(
                    num_experts=n_exp,
                    max_share=max_share,
                    ban_epochs=1,
                    cooldown_epochs=1,
                    eligible_experts=eligible,
                )
                eligible_txt = (
                    ",".join(f"e{i}" for i in eligible) if eligible is not None else "all"
                )
                logger.info(
                    "  Expert timeout: soft share>%.0f%% → ban %d epoch "
                    "(cooldown=%d, keep≥2 experts, eligible=%s, train-only mask%s)",
                    100.0 * self._expert_timeout.max_share,
                    self._expert_timeout.ban_epochs,
                    self._expert_timeout.cooldown_epochs,
                    eligible_txt,
                    "; gate frozen" if phase_cfg.freeze_gate else "",
                )
            else:
                if self._expert_timeout is not None:
                    self._expert_timeout.clear_gate(self.model)
                self._expert_timeout = None
            if phase_cfg.min_disc_r_std_save is not None:
                logger.info(
                    "  Disc save gate: disc_r_std >= %.3f required for v6_best",
                    phase_cfg.min_disc_r_std_save,
                )
            if phase_cfg.min_disc_line_thickness_save is not None:
                logger.info(
                    "  Disc visual gate: line_thickness_rms >= %.3f required for v6_best",
                    phase_cfg.min_disc_line_thickness_save,
                )
            if phase_cfg.aleatoric_only_train:
                set_p4_aleatoric_only_freeze(self.model)
                set_uncertainty_from_backbone(self.model, enabled=False)
                optimizer = build_p4_optimizer(self.model, lr=phase_cfg.lr)
                logger.info(
                    "  Phase 4 optimizer: AdamW on aleatoric uncertainty subpath only (lr=%.2e, AMP off)",
                    phase_cfg.lr,
                )
                logger.info("  Uncertainty probes: routed tangent (production inference path)")
            elif phase_cfg.epistemic_uncertainty_only_train:
                set_p4_uncertainty_only_freeze(self.model)
                set_uncertainty_from_backbone(self.model, enabled=False)
                optimizer = build_p4_optimizer(self.model, lr=phase_cfg.lr)
                logger.info(
                    "  Phase 4 optimizer: AdamW on uncertainty_head only (lr=%.2e, AMP off)",
                    phase_cfg.lr,
                )
                logger.info("  Uncertainty probes: routed tangent (production inference path)")
            elif phase_cfg.gate_only_train:
                set_gate_only_freeze(self.model)
                set_uncertainty_from_backbone(self.model, enabled=True)
                from science.dtie.v5.gnn.model import build_optimizer

                optimizer = build_optimizer(self.model, lr=phase_cfg.lr)
                logger.info(
                    "  Gate-only promotion: RiemannianAdam on gate+experts (lr=%.2e)",
                    phase_cfg.lr,
                )
                logger.info(
                    "  Uncertainty probes: backbone tangent (routing shifts decoupled from save gates)"
                )
            else:
                from science.dtie.v5.gnn.model import build_optimizer

                optimizer = build_optimizer(self.model, lr=phase_cfg.lr)
            phase_monitor = ConvergenceMonitor()
            self._shell_low_streak = 0
            stop_phase = False

            for epoch in range(phase_cfg.epochs):
                if stop_phase:
                    break
                self.global_epoch += 1
                t0 = time.time()
                coeffs = apply_phase_coeff_ramp(
                    phase_cfg.coeffs.model_dump(),
                    phase_cfg,
                    epoch,
                )

                # Phase 2 shock guard: ramp angular + dropout over first 10 epochs
                freeze_angular = phase_cfg.freeze_angular
                freeze_radial = phase_cfg.freeze_radial
                dropout_p = phase_cfg.expert_dropout_p
                if phase_cfg.p2_bridge:
                    n_ang = phase_cfg.angular_ramp_epochs or phase_cfg.epochs
                    ang_t = min(epoch + 1, n_ang) / max(n_ang, 1)
                    if phase_cfg.freeze_radial_epochs > 0 and epoch < phase_cfg.freeze_radial_epochs:
                        freeze_radial = True
                    if phase_cfg.expert_dropout_ramp_epochs > 0:
                        dr = phase_cfg.expert_dropout_ramp_epochs
                        if epoch < dr:
                            dropout_p = 0.0
                        else:
                            dt = min(epoch - dr + 1, dr) / dr
                            dropout_p = phase_cfg.expert_dropout_p * dt
                    else:
                        dropout_p = phase_cfg.expert_dropout_p
                    sep_scale = 0.5 + 0.5 * ang_t
                    coeffs["domain_sep_2d_coeff"] = phase_cfg.coeffs.domain_sep_2d_coeff * sep_scale
                    coeffs["domain_sep_3d_coeff"] = phase_cfg.coeffs.domain_sep_3d_coeff * sep_scale
                elif phase_cfg.phase == 2:
                    warmup = min(epoch + 1, 10) / 10.0
                    freeze_angular = phase_cfg.freeze_angular or epoch < 10
                    if phase_cfg.freeze_radial_epochs > 0 and epoch < phase_cfg.freeze_radial_epochs:
                        freeze_radial = True
                    dropout_p = phase_cfg.expert_dropout_p * warmup
                    coeffs["angular_coeff"] = phase_cfg.coeffs.angular_coeff * warmup
                    coeffs["domain_sep_2d_coeff"] = phase_cfg.coeffs.domain_sep_2d_coeff * warmup
                    coeffs["domain_sep_3d_coeff"] = phase_cfg.coeffs.domain_sep_3d_coeff * warmup
                elif phase_cfg.freeze_radial_epochs > 0 and epoch < phase_cfg.freeze_radial_epochs:
                    freeze_radial = True
                set_expert_dropout(self.model, dropout_p)
                if phase_cfg.coeffs.routing_load_floor_coeff > 0:
                    set_routing_load_floor_min(
                        self.model, phase_cfg.coeffs.routing_load_floor_min
                    )
                if self._expert_timeout is not None:
                    banned = self._expert_timeout.active_bans()
                    self._expert_timeout.apply_to_gate(self.model)
                    if banned:
                        logger.info(
                            "  Expert timeout active bans: %s",
                            ",".join(f"e{i}" for i in banned),
                        )

                losses = train_epoch(
                    self.model,
                    optimizer,
                    self.proteins,
                    coeffs,
                    freeze_radial=freeze_radial,
                    freeze_angular=freeze_angular,
                    freeze_backbone=phase_cfg.freeze_backbone,
                    freeze_gate=phase_cfg.freeze_gate,
                    freeze_experts=list(phase_cfg.freeze_experts),
                    path_alignment_train=phase_cfg.path_alignment_train,
                    rec_ablation_train=phase_cfg.rec_ablation_train,
                    projection_recovery_train=phase_cfg.projection_recovery_train,
                    fusion_path_recovery_train=phase_cfg.fusion_path_recovery_train,
                    lift_path_recovery_train=phase_cfg.lift_path_recovery_train,
                    device=self.config.device,
                    v2_teacher=(
                        None
                        if (
                            phase_cfg.epistemic_uncertainty_only_train
                            or phase_cfg.gate_only_train
                            or phase_cfg.topology_gate_disc_recovery_train
                            or phase_cfg.topology_crescent_recovery_train
                            or phase_cfg.slim_moe_structural_ssot_train
                            or phase_cfg.path_alignment_train
                            or phase_cfg.rec_ablation_train
                            or phase_cfg.projection_recovery_train
                            or phase_cfg.fusion_path_recovery_train
                            or phase_cfg.lift_path_recovery_train
                        )
                        else self.v2_teacher
                    ),
                    v2_teacher_depth_coeff=self.config.v2_teacher_depth_coeff,
                    v2_teacher_epistemic_coeff=self.config.v2_teacher_epistemic_coeff,
                    epistemic_decoupling_holdouts=(
                        self._epistemic_holdouts if phase_cfg.phase == 4 else None
                    ),
                    epistemic_uncertainty_only_train=phase_cfg.epistemic_uncertainty_only_train,
                    gate_only_train=phase_cfg.gate_only_train,
                    topology_gate_disc_recovery_train=phase_cfg.topology_gate_disc_recovery_train,
                    topology_crescent_recovery_train=phase_cfg.topology_crescent_recovery_train,
                    slim_moe_structural_ssot_train=phase_cfg.slim_moe_structural_ssot_train,
                    structural_disc_frozen=self.config.structural_disc_frozen,
                    topology_depth=self._topology_depth,
                )

                missing = ConvergenceMonitor.validate_epoch_metrics(losses)
                if missing:
                    logger.warning("Epoch metrics missing keys: %s", missing)

                health = measure_geometry_health(
                    self.model,
                    self.proteins,
                    self.config.device,
                    topology_depth=self._topology_depth,
                    structural_disc_frozen=self.config.structural_disc_frozen,
                )

                if phase_cfg.min_probe_r_depth_sasa is not None:
                    r_ds_guard = health.get("probe_r_depth_sasa")
                    if r_ds_guard is not None and r_ds_guard < phase_cfg.min_probe_r_depth_sasa:
                        self._shell_low_streak += 1
                    else:
                        self._shell_low_streak = 0
                    if self._shell_low_streak >= 2:
                        raise RuntimeError(
                            f"Training aborted: probe_r_depth_sasa below "
                            f"{phase_cfg.min_probe_r_depth_sasa} for 2 consecutive epochs"
                        )

                elapsed = time.time() - t0
                phase_monitor.record_epoch(losses, health)
                self.monitor.record_epoch(losses, health)

                abort, reason = phase_monitor.should_abort()
                if abort:
                    raise RuntimeError(f"Training aborted: {reason}")

                pf = health.get("proj_frac_mean")
                cr = health.get("cone_range_mean")
                r_dt = health.get("probe_r_depth_tau")
                r_dr = health.get("probe_r_depth_rho")
                r_ds = health.get("probe_r_depth_sasa")
                r_es = health.get("probe_r_epi_sasa")
                r_pd = health.get("probe_r_proj_depth")
                cd_std = health.get("cone_depth_std_mean")
                dr_std = health.get("disc_r_std_mean")
                disc_sigma = health.get("disc_sigma2_sigma1_mean")
                disc_eff = health.get("disc_effective_rank_mean")
                ang_c = coeffs.get("angular_coeff")
                disc_floor = coeffs.get("disc_spread_min_std")
                disc_target = coeffs.get("disc_depth_scale_target")
                from science.training.routing_gate_bounds import model_num_experts

                _n_experts = model_num_experts(
                    self.model, default=self.config.num_experts
                )
                route_ceiling = routing_save_max_for_epoch(
                    phase_cfg, epoch, num_experts=_n_experts
                )
                route_display, route_label = routing_save_ceiling_for_display(
                    phase_cfg, epoch, num_experts=_n_experts
                )
                skip_unc_gates = (
                    phase_cfg.gate_only_train
                    or phase_cfg.topology_gate_disc_recovery_train
                    or phase_cfg.topology_crescent_recovery_train
                )
                from science.training.mlflow_governance import (
                    governance_epoch_metrics,
                    stage_gate_passed,
                    telemetry_track_metrics,
                )
                from science.training.routing_metrics import inference_mode_routing_metrics

                # Eval / save gates must see unmasked routing (timeout is train-only).
                if self._expert_timeout is not None:
                    self._expert_timeout.clear_gate(self.model)

                infer_routing = inference_mode_routing_metrics(
                    self.model,
                    self.proteins,
                    self.config.device,
                    structural_disc_frozen=bool(self.config.structural_disc_frozen),
                )
                scored = score_checkpoint(
                    health,
                    losses,
                    phase=phase_cfg.phase,
                    routing_save_max=route_ceiling,
                    min_probe_r_depth_sasa_save=phase_cfg.min_probe_r_depth_sasa_save,
                    min_disc_sigma2_sigma1_save=phase_cfg.min_disc_sigma2_sigma1_save,
                    min_disc_r_std_save=phase_cfg.min_disc_r_std_save,
                    min_disc_effective_rank_save=phase_cfg.min_disc_effective_rank_save,
                    min_disc_line_thickness_save=phase_cfg.min_disc_line_thickness_save,
                    disc_radial_source=self.config.disc_radial_source,
                    max_probe_r_epi_ale_save=(
                        None if skip_unc_gates else phase_cfg.max_probe_r_epi_ale_save
                    ),
                    max_probe_r_epi_sasa_save=(
                        None if skip_unc_gates else phase_cfg.max_probe_r_epi_sasa_save
                    ),
                    min_probe_r_epi_sasa_save=(
                        None if skip_unc_gates else phase_cfg.min_probe_r_epi_sasa_save
                    ),
                    min_epistemic_std_save=(
                        None if skip_unc_gates else phase_cfg.min_epistemic_std_save
                    ),
                    min_aleatoric_std_save=(
                        None if skip_unc_gates else phase_cfg.min_aleatoric_std_save
                    ),
                    require_tau_ale_elevation_save=(
                        False
                        if skip_unc_gates
                        else phase_cfg.require_tau_ale_elevation_save
                    ),
                    topology_depth=self._topology_depth,
                    inference_routing=infer_routing,
                    max_expert_starvation_save=phase_cfg.max_expert_starvation_save,
                    min_eval_routing_fraction_save=phase_cfg.min_eval_routing_fraction_save,
                    max_eval_routing_fraction_save=phase_cfg.max_eval_routing_fraction_save,
                    routing_entropy_min_save=phase_cfg.routing_entropy_min_save,
                )
                score = scored.score

                log_metrics = {
                    **losses,
                    **health,
                    "score": score,
                    "checkpoint_eligible": scored.eligible,
                    "routing_save_max": route_ceiling,
                    "elapsed_s": elapsed,
                    "eval_min_routing_fraction": float(
                        infer_routing.get("min_routing_fraction", float("nan"))
                    ),
                    "eval_max_routing_fraction": float(
                        infer_routing.get("max_routing_fraction", float("nan"))
                    ),
                }
                # expert_load is a list for timeout observe; MLflow needs scalars only.
                log_metrics.pop("expert_load", None)
                from science.training.metric_focus import assess_training_focus, focus_mlflow_metrics

                focus = assess_training_focus(
                    health,
                    losses,
                    phase=phase_cfg.phase,
                    routing_save_max=route_ceiling,
                    min_probe_r_depth_sasa_save=phase_cfg.min_probe_r_depth_sasa_save,
                    checkpoint_eligible=scored.eligible,
                    topology_depth=self._topology_depth,
                )
                log_metrics.update(focus_mlflow_metrics(focus))
                gov = governance_epoch_metrics(
                    health,
                    losses,
                    self.model,
                    inference_routing=infer_routing,
                    master_cold_lineage=(
                        self.config.master_cold_lineage or self.config.slim_moe_structural_ssot
                    ),
                )
                log_metrics.update(gov)
                log_metrics.update(telemetry_track_metrics(health))
                if (
                    self.config.feature_liveness_probe
                    or self.config.use_dehydron_barcode
                    or self.config.dehydron_edge_barcode
                    or self.config.chem_edge_mp
                    or self.config.containment_edge_mp
                ):
                    from science.training.feature_liveness import run_feature_liveness_probes

                    live = run_feature_liveness_probes(
                        self.model,
                        self.proteins,
                        self.config.device,
                        use_dehydron_barcode=bool(self.config.use_dehydron_barcode),
                        dehydron_edge_barcode=bool(self.config.dehydron_edge_barcode),
                        chem_edge_mp=bool(self.config.chem_edge_mp),
                        containment_edge_mp=bool(self.config.containment_edge_mp),
                        structural_disc_frozen=bool(self.config.structural_disc_frozen),
                        fail_if_dead=bool(self.config.feature_liveness_fail_if_dead),
                        min_epochs_before_fail=int(self.config.feature_liveness_min_epochs),
                        epoch_in_phase=int(epoch),
                    )
                    if live.get("barcode"):
                        bc = live["barcode"]
                        for k, v in bc.items():
                            if isinstance(v, (int, float)) and k.startswith("delta_"):
                                log_metrics[f"liveness_barcode_{k}"] = float(v)
                        log_metrics["liveness_barcode_alive"] = (
                            1.0 if bc.get("alive") else 0.0
                        )
                    if live.get("barcode_edge"):
                        bec = live["barcode_edge"]
                        for k, v in bec.items():
                            if isinstance(v, (int, float)) and k.startswith("delta_"):
                                log_metrics[f"liveness_barcode_edge_{k}"] = float(v)
                        log_metrics["liveness_barcode_edge_alive"] = (
                            1.0 if bec.get("alive") else 0.0
                        )
                    if live.get("chem"):
                        chem = live["chem"]
                        for k, v in chem.items():
                            if isinstance(v, (int, float)) and (
                                k.startswith("delta_")
                                or k.startswith("radial_var_")
                                or k.startswith("n_")
                            ):
                                log_metrics[f"liveness_chem_{k}"] = float(v)
                        if chem.get("skipped"):
                            log_metrics["liveness_chem_skipped"] = 1.0
                        else:
                            log_metrics["liveness_chem_alive"] = (
                                1.0 if chem.get("alive") else 0.0
                            )
                    if live.get("containment"):
                        contain = live["containment"]
                        for k, v in contain.items():
                            if isinstance(v, (int, float)) and (
                                k.startswith("delta_")
                                or k.startswith("radial_var_")
                                or k.startswith("n_")
                            ):
                                log_metrics[f"liveness_containment_{k}"] = float(v)
                        if contain.get("skipped"):
                            log_metrics["liveness_containment_skipped"] = 1.0
                        else:
                            log_metrics["liveness_containment_alive"] = (
                                1.0 if contain.get("alive") else 0.0
                            )
                    if live.get("mp"):
                        mp = live["mp"]
                        log_metrics["liveness_mp_alive"] = 1.0 if mp.get("alive") else 0.0
                    if not live.get("ok", True) and self.config.feature_liveness_fail_if_dead:
                        raise RuntimeError(
                            "Feature liveness probe failed — barcode/MP/chem/containment "
                            "channel is a no-op. "
                            f"Report: {live}"
                        )
                from science.training.stage_a_stop import (
                    check_stage_a_inference_stop,
                    format_stop_message,
                    should_enforce_stage_a_stop,
                )

                stop_verdict = None
                if should_enforce_stage_a_stop(self.config, phase_cfg.phase):
                    stop_verdict = check_stage_a_inference_stop(infer_routing)
                self._last_focus_summary = focus
                if self.tracker:
                    self.tracker.log_metrics(log_metrics, step=self.global_epoch)

                elig_tag = "ok" if scored.eligible else "skip"
                if self._topology_depth:
                    logger.info(
                        "  Ep %3d | loss=%.4f route_H=%.3f starve=%.0f | "
                        "proj=%.3f cone_rng=%.4f score=%.4f [%s] | "
                        "topo r(d,τ)=%.3f r(d,ρ)=%.3f r(|p|,d)=%.3f | "
                        "depth_std=%.4f disc_r_std=%.4f σ₂/σ₁=%.3f eff_rank=%.3f ang=%.3f disc_tgt=%.3f | "
                        "%s=%.3f | %.1fs",
                        self.global_epoch,
                        losses["total"],
                        losses.get("routing_entropy", 0.0),
                        losses.get("expert_starvation_count", 0.0),
                        pf if pf is not None else -1.0,
                        cr if cr is not None else -1.0,
                        score,
                        elig_tag,
                        r_dt if r_dt is not None else float("nan"),
                        r_dr if r_dr is not None else float("nan"),
                        r_pd if r_pd is not None else float("nan"),
                        cd_std if cd_std is not None else float("nan"),
                        dr_std if dr_std is not None else float("nan"),
                        disc_sigma if disc_sigma is not None else float("nan"),
                        disc_eff if disc_eff is not None else float("nan"),
                        ang_c if ang_c is not None else float("nan"),
                        disc_target if disc_target is not None else float("nan"),
                        route_label,
                        route_display,
                        elapsed,
                    )
                else:
                    logger.info(
                        "  Ep %3d | loss=%.4f route_H=%.3f starve=%.0f | "
                        "proj=%.3f cone_rng=%.4f score=%.4f [%s] | "
                        "shell r(d,s)=%.3f r(e,s)=%.3f r(|p|,d)=%.3f | "
                        "depth_std=%.4f disc_r_std=%.4f σ₂/σ₁=%.3f eff_rank=%.3f ang=%.3f disc_tgt=%.3f | "
                        "%s=%.3f | %.1fs",
                        self.global_epoch,
                        losses["total"],
                        losses.get("routing_entropy", 0.0),
                        losses.get("expert_starvation_count", 0.0),
                        pf if pf is not None else -1.0,
                        cr if cr is not None else -1.0,
                        score,
                        elig_tag,
                        r_ds if r_ds is not None else float("nan"),
                        r_es if r_es is not None else float("nan"),
                        r_pd if r_pd is not None else float("nan"),
                        cd_std if cd_std is not None else float("nan"),
                        dr_std if dr_std is not None else float("nan"),
                        disc_sigma if disc_sigma is not None else float("nan"),
                        disc_eff if disc_eff is not None else float("nan"),
                        ang_c if ang_c is not None else float("nan"),
                        disc_target if disc_target is not None else float("nan"),
                        route_label,
                        route_display,
                        elapsed,
                    )
                logger.info(
                    "    eval hard routing: min_frac=%.3f max_frac=%.3f eff_experts=%.2f",
                    float(infer_routing.get("min_routing_fraction", float("nan"))),
                    float(infer_routing.get("max_routing_fraction", float("nan"))),
                    float(infer_routing.get("effective_experts", float("nan"))),
                )
                if phase_cfg.phase == 4 or float(losses.get("epistemic_decoupling", 0.0)) > 0.0:
                    logger.info(
                        "    decouple r(epi,bf_resid)=%.3f partial(epi,sasa|depth)=%.3f "
                        "r(epi,ale)=%.3f epi_std=%.4f ale_std=%.4f | λ_bf=%.3f λ_sasa=%.3f",
                        float(losses.get("r_epi_bf_resid", float("nan"))),
                        float(losses.get("partial_epi_sasa_given_depth", float("nan"))),
                        float(health.get("probe_r_epi_ale", float("nan"))),
                        float(health.get("epistemic_std_mean", float("nan"))),
                        float(health.get("aleatoric_std_mean", float("nan"))),
                        float(coeffs.get("epistemic_bf_align_coeff", 0.0)),
                        float(coeffs.get("epistemic_sasa_pen_coeff", 0.0)),
                    )
                loads = losses.get("expert_load")
                soft_load_list: list[float] | None = None
                if isinstance(loads, (list, tuple)) and len(loads) >= 1:
                    soft_load_list = [float(x) for x in loads]
                elif loads is not None and hasattr(loads, "numel") and loads.numel() >= 1:
                    soft_load_list = [float(loads[i]) for i in range(loads.numel())]
                elif any(f"expert_load_{i}" in losses for i in range(4)):
                    soft_load_list = [
                        float(losses.get(f"expert_load_{i}", 0.0)) for i in range(4)
                    ]
                if soft_load_list is not None:
                    logger.info(
                        "    expert_load=%s",
                        ", ".join(f"{x:.3f}" for x in soft_load_list[:4]),
                    )
                if self._expert_timeout is not None and soft_load_list is not None:
                    # Decrement bans that just trained under the mask, then maybe ban.
                    still = self._expert_timeout.tick_end_of_epoch()
                    newly = self._expert_timeout.observe_loads(
                        soft_load_list,
                        epoch=self.global_epoch,
                    )
                    if newly or still:
                        logger.info(
                            "    expert_timeout: newly_banned=%s still_banned=%s",
                            newly,
                            self._expert_timeout.active_bans(),
                        )
                if self.config.full_hyp_moe_test or self._topology_depth:
                    expert_bits = []
                    disc_std_bits = []
                    for e in range(len(self.model.experts)):
                        d = health.get(f"expert_{e}_depth_mean")
                        if d is None:
                            continue
                        r = health.get(f"expert_{e}_disc_r_mean", float("nan"))
                        rs = health.get(f"expert_{e}_disc_r_std", float("nan"))
                        rt = health.get(f"expert_{e}_r_depth_tau", float("nan"))
                        tau_m = health.get(f"expert_{e}_tau_mean", float("nan"))
                        rho_m = health.get(f"expert_{e}_rho_mean", float("nan"))
                        ss_h = health.get(f"expert_{e}_ss_helix_frac", float("nan"))
                        ss_e = health.get(f"expert_{e}_ss_sheet_frac", float("nan"))
                        ss_c = health.get(f"expert_{e}_ss_coil_frac", float("nan"))
                        expert_bits.append(
                            f"e{e}: depth={d:.3f} disc_r={r:.3f} r(d,τ)={rt:.3f} "
                            f"τ={tau_m:.3f} ρ={rho_m:.1f} ss=H{ss_h:.2f}/E{ss_e:.2f}/C{ss_c:.2f}"
                        )
                        if rs == rs:
                            disc_std_bits.append(f"e{e}={rs:.3f}")
                    if expert_bits:
                        logger.info("    per_expert: %s", " | ".join(expert_bits))
                    if disc_std_bits:
                        logger.info("    expert_disc_r_std: %s", " ".join(disc_std_bits))
                    e0_rdt = health.get("expert_0_r_depth_tau")
                    if e0_rdt is not None and e0_rdt == e0_rdt:
                        logger.info("    e0_r_depth_tau=%.3f", float(e0_rdt))
                thick_pre = health.get("disc_line_thickness_pre_mean")
                thick_post = health.get("disc_line_thickness_post_mean")
                x_hyp_thick = health.get("x_hyp_line_thickness_mean")
                if x_hyp_thick is not None and x_hyp_thick == x_hyp_thick:
                    logger.info("    x_hyp thick=%.4f", float(x_hyp_thick))
                if thick_pre is not None and thick_post is not None and thick_pre == thick_pre:
                    logger.info(
                        "    disc thick pre=%.4f post=%.4f (model output pre-routing)",
                        float(thick_pre),
                        float(thick_post),
                    )
                if losses.get("disc_occupancy") is not None and float(losses["disc_occupancy"]) > 0:
                    logger.info(
                        "    disc_occupancy_loss=%.4f batch_σ₂/σ₁=%.3f",
                        float(losses["disc_occupancy"]),
                        float(losses.get("disc_sigma2_sigma1", 0.0)),
                    )
                if losses.get("disc_pc_repulsion") is not None and float(losses["disc_pc_repulsion"]) > 0:
                    logger.info(
                        "    disc_pc_repulsion=%.4f pc2_std=%.4f",
                        float(losses["disc_pc_repulsion"]),
                        float(losses.get("disc_pc2_std", 0.0)),
                    )
                if losses.get("shell_floor") is not None and float(losses["shell_floor"]) > 0:
                    logger.info("    shell_floor_loss=%.4f", float(losses["shell_floor"]))
                if losses.get("disc_depth_scale") is not None:
                    logger.info(
                        "    disc_scale_loss=%.4f",
                        float(losses["disc_depth_scale"]),
                    )
                if losses.get("x_hyp_thickness_floor") is not None and float(losses["x_hyp_thickness_floor"]) > 0:
                    logger.info(
                        "    x_hyp_thickness_floor=%.4f thick=%.4f",
                        float(losses["x_hyp_thickness_floor"]),
                        float(losses.get("x_hyp_line_thickness_rms", 0.0)),
                    )
                if losses.get("disc_path_align") is not None:
                    logger.info(
                        "    disc_path_align_loss=%.6f",
                        float(losses["disc_path_align"]),
                    )
                if not scored.eligible and scored.reasons:
                    logger.info("    ineligible: %s", ", ".join(scored.reasons))
                if focus.get("primary_focus_str") and focus["primary_focus_str"] != "none":
                    logger.info("    focus: %s — %s", focus["primary_focus_str"], focus.get("recommendation", ""))

                disc_thickness_pre = health.get("disc_line_thickness_pre_mean")
                if disc_thickness_pre is None:
                    disc_thickness_pre = health.get("disc_line_thickness_rms_mean")
                thickness_ok = (
                    disc_thickness_pre is not None
                    and phase_cfg.min_disc_line_thickness_save is not None
                    and float(disc_thickness_pre) >= float(phase_cfg.min_disc_line_thickness_save)
                )
                if phase_cfg.rec_ablation_train:
                    should_save = score > self.best_score
                elif (
                    phase_cfg.path_alignment_train
                    or phase_cfg.projection_recovery_train
                    or phase_cfg.fusion_path_recovery_train
                    or phase_cfg.lift_path_recovery_train
                ):
                    should_save = thickness_ok and score > self.best_score
                else:
                    should_save = scored.eligible and score > self.best_score
                if should_save:
                    self.best_score = score
                    self._saved_eligible = True
                    ckpt_path = self.checkpoint_mgr.save_best(
                        self.model,
                        optimizer,
                        global_epoch=self.global_epoch,
                        phase=phase_cfg.phase,
                        phase_name=phase_cfg.name,
                        metrics=log_metrics,
                        training_config=self.config.model_dump(mode="json"),
                        score=score,
                    )
                    if self.tracker:
                        from science.contracts.model_registry import compute_checkpoint_sha256

                        sha = compute_checkpoint_sha256(str(ckpt_path)) or ""
                        self.tracker.log_best_checkpoint(
                            ckpt_path,
                            score=score,
                            global_epoch=self.global_epoch,
                            sha256=sha or None,
                        )
                    logger.info("    ★ New best checkpoint (score=%.4f, eligible)", score)

                if self.config.track_v6_best_route:
                    from science.training.checkpoint_score import score_route_checkpoint

                    route_scored = score_route_checkpoint(
                        health,
                        losses,
                        routing_save_max=route_ceiling,
                        topology_depth=self._topology_depth,
                        inference_routing=infer_routing,
                    )
                    if route_scored.eligible and route_scored.score > self._best_route_score:
                        self._best_route_score = route_scored.score
                        route_path = self.checkpoint_mgr.save_best_route(
                            self.model,
                            optimizer,
                            global_epoch=self.global_epoch,
                            phase=phase_cfg.phase,
                            phase_name=phase_cfg.name,
                            metrics=log_metrics,
                            training_config=self.config.model_dump(mode="json"),
                            score=route_scored.score,
                            routing_entropy=float(losses.get("routing_entropy", 0.0)),
                        )
                        logger.info(
                            "    ★ New best route checkpoint (score=%.4f, H=%.3f) → %s",
                            route_scored.score,
                            float(losses.get("routing_entropy", 0.0)),
                            route_path.name,
                        )
                    elif not route_scored.eligible and route_scored.reasons:
                        logger.debug(
                            "    route ineligible: %s",
                            ", ".join(route_scored.reasons),
                        )

                disc_sigma = health.get("disc_sigma2_sigma1_mean")
                disc_r_std = health.get("disc_r_std_mean")
                disc_thickness = health.get("disc_line_thickness_pre_mean")
                if disc_thickness is None:
                    disc_thickness = health.get("disc_line_thickness_rms_mean")
                min_r_std = phase_cfg.min_disc_r_std_save
                min_thickness = phase_cfg.min_disc_line_thickness_save
                use_visual_score = min_thickness is not None
                passes_disc_gates = (
                    disc_sigma is not None
                    and disc_r_std is not None
                    and (min_r_std is None or float(disc_r_std) >= float(min_r_std))
                    and (
                        min_thickness is None
                        or (
                            disc_thickness is not None
                            and float(disc_thickness) >= float(min_thickness)
                        )
                    )
                )
                if use_visual_score:
                    t_norm = min(float(disc_thickness or 0.0) / 0.03, 1.0)
                    s_norm = min(float(disc_sigma or 0.0) / 0.5, 1.0)
                    visual_score = 0.6 * t_norm + 0.4 * s_norm
                    should_save_disc = (
                        passes_disc_gates and visual_score > self._best_disc_visual_score
                    )
                    if should_save_disc:
                        self._best_disc_visual_score = visual_score
                        self._best_disc_sigma = float(disc_sigma)
                elif (
                    disc_sigma is not None
                    and float(disc_sigma) > self._best_disc_sigma
                    and passes_disc_gates
                ):
                    should_save_disc = True
                    self._best_disc_sigma = float(disc_sigma)
                else:
                    should_save_disc = False

                if should_save_disc:
                    disc_path = self.checkpoint_mgr.save_best_disc(
                        self.model,
                        global_epoch=self.global_epoch,
                        phase=phase_cfg.phase,
                        phase_name=phase_cfg.name,
                        disc_sigma2_sigma1=float(disc_sigma),
                        metrics=log_metrics,
                        training_config=self.config.model_dump(mode="json"),
                    )
                    logger.info(
                        "    ★ New best disc occupancy (σ₂/σ₁=%.3f, thick=%.4f) → %s",
                        float(disc_sigma or 0),
                        float(disc_thickness or 0),
                        disc_path.name,
                    )

                if (
                    self.config.p2_disc_path_align
                    and thickness_ok
                ):
                    logger.info(
                        "    Path align: line_thickness gate passed — early stop"
                    )
                    stop_phase = True
                elif (
                    self.config.p2_disc_gentle_arch
                    and scored.eligible
                    and thickness_ok
                ):
                    logger.info(
                        "    Gentle disc arch: eligible visual checkpoint — early stop"
                    )
                    stop_phase = True

                entry = {
                    "global_epoch": self.global_epoch,
                    "phase": phase_cfg.phase,
                    "phase_name": phase_cfg.name,
                    "losses": losses,
                    "health": health,
                    "score": score,
                    "checkpoint_eligible": scored.eligible,
                    "elapsed": elapsed,
                    "inference_routing": infer_routing,
                    "stage_gate_passed": stage_gate_passed(
                        health,
                        losses,
                        inference_routing=infer_routing,
                        num_experts=len(self.model.experts),
                        master_cold_lineage=(
                        self.config.master_cold_lineage or self.config.slim_moe_structural_ssot
                    ),
                    ),
                }
                if stop_verdict is not None:
                    entry["stop_enforced"] = stop_verdict.tripped
                    if stop_verdict.reasons:
                        entry["stop_reasons"] = list(stop_verdict.reasons)
                self.metrics_log.append(entry)
                self.checkpoint_mgr.write_metrics_log(self.metrics_log)

                if self.config.save_epoch_snapshots:
                    self.checkpoint_mgr.save_epoch(
                        self.model,
                        global_epoch=self.global_epoch,
                        phase=phase_cfg.phase,
                        training_config=self.config.model_dump(mode="json"),
                    )

                self._maybe_export_disc_scatter(phase_cfg)

                if stop_verdict is not None and stop_verdict.tripped:
                    logger.error(
                        "STAGE_A_STOP at global epoch %d: %s",
                        self.global_epoch,
                        stop_verdict.summary,
                    )
                    raise RuntimeError(format_stop_message(stop_verdict))

            self.checkpoint_mgr.save_phase(
                self.model,
                phase=phase_cfg.phase,
                phase_name=phase_cfg.name,
                global_epoch=self.global_epoch,
            )
            if phase_cfg.phase == 1 and self.config.master_cold_lineage:
                p1_entries = [e for e in self.metrics_log if e.get("phase") == 1]
                if p1_entries:
                    last_health = p1_entries[-1].get("health", {})
                    p2_gate = dehydron_cone_gate_verdict(last_health)
                    if not p2_gate.passed:
                        p2_dehydron_blocked = True
                        p2_dehydron_reason = p2_gate.reason
                        logger.error(
                            "P_DEHYDRON_CONE_01 blocked Phase 2 entry: %s",
                            p2_gate.reason,
                        )
            if not self._saved_eligible:
                logger.warning(
                    "Phase %d finished with no eligible v6_best.pt — "
                    "relax gates or extend training; use v6_phase%d_%dprot.pt for latest weights.",
                    phase_cfg.phase,
                    phase_cfg.phase,
                    len(self.proteins),
                )

        return {
            "global_epoch": self.global_epoch,
            "best_score": self.best_score,
            "summary": self.monitor.stage_summary(),
            "output_dir": str(self.config.output_dir),
            "focus_summary": self._last_focus_summary,
            "p3_entry_gate_passed": not p3_entry_skipped,
            "p3_entry_gate_skipped": p3_entry_skipped,
            "p3_entry_gate_reason": p3_entry_reason or None,
            "p2_dehydron_gate_passed": not p2_dehydron_blocked,
            "p2_dehydron_gate_blocked": p2_dehydron_blocked,
            "p2_dehydron_gate_reason": p2_dehydron_reason or None,
        }
