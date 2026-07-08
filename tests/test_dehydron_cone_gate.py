"""Tests for P_DEHYDRON_CONE_01 and tau_dehydron_rim cone supervision."""

from __future__ import annotations

import torch

from science.dtie.v6.loss import cone_alignment_loss
from science.training.config import (
    apply_master_cold_dehydron_config,
    apply_master_cold_dehydron_lineage,
    apply_master_cold_dehydron_phases,
    default_v6_phases,
)
from science.training.dehydron_cone_gate import (
    GATE_NAME,
    MIN_R_CONE_DEPTH_TAU,
    dehydron_cone_gate_passed,
    dehydron_cone_gate_verdict,
)
from science.training.mlflow_governance import master_cold_stage_gate_passed, stage_gate_passed


def test_dehydron_cone_gate_passes_when_tau_aligned() -> None:
    health = {
        "probe_r_depth_tau": 0.22,
        "probe_r_depth_rho": -0.05,
    }
    verdict = dehydron_cone_gate_verdict(health)
    assert verdict.passed is True
    assert dehydron_cone_gate_passed(health) is True


def test_dehydron_cone_gate_passes_despite_high_sasa_correlation() -> None:
    """SASA is not part of P_DEHYDRON_CONE_01 — τ alignment alone decides pass."""
    health = {
        "probe_r_depth_tau": 0.641,
        "probe_r_depth_rho": -0.586,
    }
    verdict = dehydron_cone_gate_verdict(health)
    assert verdict.passed is True


def test_dehydron_cone_gate_fails_low_tau_correlation() -> None:
    health = {
        "probe_r_depth_tau": 0.03,
    }
    verdict = dehydron_cone_gate_verdict(health)
    assert verdict.passed is False
    assert MIN_R_CONE_DEPTH_TAU == 0.15
    assert "dehydron rim" in verdict.reason.lower() or "τ" in verdict.reason


def test_cone_alignment_loss_tau_mode_prefers_dehydron_depth() -> None:
  tau = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
  good = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.85, 0.15]).unsqueeze(-1)
  bad = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.15, 0.85]).unsqueeze(-1)
  loss_good = cone_alignment_loss(
      good, target_dehydron=tau, mode="tau_dehydron_rim"
  )
  loss_bad = cone_alignment_loss(
      bad, target_dehydron=tau, mode="tau_dehydron_rim"
  )
  assert float(loss_good) < float(loss_bad)


def test_apply_master_cold_dehydron_phases() -> None:
    phases = apply_master_cold_dehydron_phases(default_v6_phases())
    assert all(p.coeffs.cone_target_mode == "tau_dehydron_rim" for p in phases)
    assert phases[0].coeffs.shell_corr_depth_sasa_weight == 0.0
    assert phases[0].coeffs.shell_corr_disc_sasa_weight == 0.0
    assert phases[0].coeffs.shell_corr_coeff == 0.12
    assert phases[0].min_probe_r_depth_sasa is None
    assert phases[0].min_probe_r_depth_sasa_save is None
    p2 = next(p for p in phases if p.phase == 2)
    assert p2.routing_save_ceiling_start == 1.35
    assert p2.routing_save_ceiling_final == 1.21


def test_routing_save_ceiling_for_display_no_nan_on_p1() -> None:
    from science.training.config import routing_save_ceiling_for_display

    p1 = default_v6_phases()[0]
    val, label = routing_save_ceiling_for_display(p1, epoch=0)
    assert val == val  # not NaN
    assert label == "route_ref"


def test_topology_routing_recovery_phase_config() -> None:
    from science.training.config import topology_routing_recovery_phase_config

    phase = topology_routing_recovery_phase_config()
    assert phase.phase == 2
    assert phase.epochs == 40
    assert phase.coeffs.balance_coeff == 0.001
    assert phase.coeffs.cone_target_mode == "tau_dehydron_rim"
    assert phase.routing_save_ceiling_final == 1.14


def test_topology_gate_disc_recovery_phase_config() -> None:
    from science.training.config import topology_gate_disc_recovery_phase_config

    phase = topology_gate_disc_recovery_phase_config()
    assert phase.topology_gate_disc_recovery_train is True
    assert phase.epochs == 20
    assert phase.coeffs.cone_depth_anticollapse_coeff == 0.0
    assert phase.coeffs.disc_occupancy_coeff == 0.35
    assert phase.freeze_backbone is True


def test_topology_gate_disc_recovery_freeze() -> None:
    from experiments.training.v6.train_loop import set_topology_gate_disc_recovery_freeze
    from science.dtie.v6.gnn.model import GOSPConeMapperV6

    model = GOSPConeMapperV6(
        node_dim=3,
        hidden=32,
        num_layers=2,
        num_experts=4,
        expert_depth_decouple=True,
        structure_gate=True,
        topology_only_gate=True,
    )
    set_topology_gate_disc_recovery_freeze(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("gate.") for n in trainable)
    assert any(n.startswith("hyp_proj_head_2d.") for n in trainable)
    assert not any(n.startswith("experts.") for n in trainable)
    assert not any(n.startswith("radial_head.") for n in trainable)
    assert model.expert_depth_bias is not None
    assert not model.expert_depth_bias.requires_grad


def test_score_route_checkpoint_prefers_low_entropy() -> None:
    from science.training.checkpoint_score import score_route_checkpoint

    good = score_route_checkpoint(
        {"probe_r_depth_tau": 0.9},
        {
            "routing_entropy": 1.15,
            "expert_load_0": 0.1,
            "expert_load_1": 0.35,
            "expert_load_2": 0.35,
            "expert_load_3": 0.2,
            "min_routing_fraction": 0.10,
        },
        routing_save_max=1.20,
        topology_depth=True,
    )
    bad = score_route_checkpoint(
        {"probe_r_depth_tau": 0.9},
        {
            "routing_entropy": 1.32,
            "expert_load_0": 0.25,
            "expert_load_1": 0.25,
            "expert_load_2": 0.25,
            "expert_load_3": 0.25,
            "min_routing_fraction": 0.10,
        },
        routing_save_max=1.20,
        topology_depth=True,
    )
    assert good.eligible
    assert not bad.eligible
    assert good.score > bad.score


def test_dehydron_rim_recovery_phase_config() -> None:
    from science.training.config import dehydron_rim_recovery_phase_config

    phase = dehydron_rim_recovery_phase_config()
    assert phase.coeffs.cone_target_mode == "tau_dehydron_rim"
    assert phase.freeze_backbone is True
    assert phase.freeze_gate is True
    assert phase.coeffs.shell_corr_depth_sasa_weight == 0.0


def test_apply_master_cold_dehydron_lineage_topology_only_no_v2() -> None:
    from pathlib import Path

    from science.training.config import TrainingConfig

    cfg = TrainingConfig(
        corpus_manifest=Path("manifests/v6_corpus_stage_a_small_v1.json"),
        master_cold_lineage=True,
        v2_teacher_checkpoint=Path("/app/science/dtie/v3/checkpoints/v2_bridge_epoch_014.pt"),
    )
    phases = default_v6_phases()
    cfg2, phases2 = apply_master_cold_dehydron_lineage(cfg, phases)
    assert cfg2.topology_only_gate is True
    assert cfg2.v2_teacher_checkpoint is None
    assert cfg2.v2_teacher_depth_coeff == 0.0
    assert all(p.coeffs.cone_target_mode == "tau_dehydron_rim" for p in phases2)


def test_topology_only_hyperbolic_gate_has_no_mobius_trunk() -> None:
    from science.dtie.v6.gnn.hyperbolic_moe import HyperbolicPrototypeGate

    gate = HyperbolicPrototypeGate(
        hidden_dim=32, num_experts=6, topology_only=True, use_disc_position=True
    )
    assert gate.topology_only is True
    assert not hasattr(gate, "mobius1")
    assert gate.topo_encoder[0].in_features == HyperbolicPrototypeGate.TOPO_DIM + HyperbolicPrototypeGate.DISC_DIM


def test_master_cold_stage_gate_ignores_sasa_target() -> None:
    health = {
        "disc_sigma2_sigma1_mean": 0.665,
        "probe_r_depth_sasa": 0.20,
        "probe_r_depth_tau": 0.22,
    }
    losses = {
        "effective_experts": 3.9,
        "effective_experts_min": 3.5,
        "min_routing_fraction": 0.18,
    }
    assert master_cold_stage_gate_passed(health, losses) == 1
    assert stage_gate_passed(health, losses, master_cold_lineage=True) == 1


def test_master_cold_stage_gate_fails_without_tau_alignment() -> None:
    health = {
        "disc_sigma2_sigma1_mean": 0.665,
        "probe_r_depth_sasa": 0.73,
        "probe_r_depth_tau": 0.03,
    }
    losses = {
        "effective_experts": 3.9,
        "effective_experts_min": 3.5,
        "min_routing_fraction": 0.18,
    }
    assert master_cold_stage_gate_passed(health, losses) == 0


def test_gate_name_is_preregistered() -> None:
    assert GATE_NAME == "P_DEHYDRON_CONE_01"
