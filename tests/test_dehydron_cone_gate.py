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
        "probe_r_depth_sasa": 0.10,
        "probe_r_depth_rho": -0.05,
    }
    verdict = dehydron_cone_gate_verdict(health)
    assert verdict.passed is True
    assert dehydron_cone_gate_passed(health) is True


def test_dehydron_cone_gate_fails_low_tau_correlation() -> None:
    health = {
        "probe_r_depth_tau": 0.03,
        "probe_r_depth_sasa": 0.10,
    }
    verdict = dehydron_cone_gate_verdict(health)
    assert verdict.passed is False
    assert MIN_R_CONE_DEPTH_TAU == 0.15
    assert "dehydron rim" in verdict.reason.lower() or "τ" in verdict.reason


def test_dehydron_cone_gate_fails_sasa_dominance() -> None:
    health = {
        "probe_r_depth_tau": 0.20,
        "probe_r_depth_sasa": 0.62,
    }
    verdict = dehydron_cone_gate_verdict(health)
    assert verdict.passed is False
    assert verdict.sasa_dominates_tau is True


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
