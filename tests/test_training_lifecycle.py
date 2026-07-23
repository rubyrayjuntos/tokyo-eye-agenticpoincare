"""Tests for v6 GNN training lifecycle (config, monitor, corpus, promotion)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from science.training.checkpoint import CheckpointManager
from science.training.config import TrainingConfig, default_v6_phases
from science.training.monitor import ConvergenceMonitor
from science.training.promote import promote_checkpoint
from science.training.config import PromotionConfig
from experiments.training.v6.corpus import iter_corpus_entries, load_corpus_manifest


def test_training_config_serializes_paths() -> None:
    cfg = TrainingConfig(output_dir="checkpoints/v6/test", corpus_manifest="manifests/v6_corpus_120.json")
    params = cfg.to_mlflow_params()
    assert "checkpoints/v6/test" in params["output_dir"]
    assert params["model_version"] == "TokyoEye-v8"
    assert params["num_experts"] == 4


def test_training_config_num_experts_override() -> None:
    cfg = TrainingConfig(num_experts=6)
    assert cfg.num_experts == 6
    assert cfg.to_mlflow_params()["num_experts"] == 6


def test_training_config_mlflow_params_lowercase_bools() -> None:
    from science.training.config import apply_master_cold_dehydron_config

    cfg = TrainingConfig(master_cold_lineage=True)
    cfg = apply_master_cold_dehydron_config(cfg)
    params = cfg.to_mlflow_params()
    assert params["topology_only_gate"] == "true"
    assert params["master_cold_lineage"] == "true"


def test_default_v6_phases_schedule() -> None:
    phases = default_v6_phases()
    assert len(phases) == 3
    assert phases[0].phase == 1
    assert phases[0].expert_dropout_p == 0.0
    assert phases[1].expert_dropout_p == 0.15
    assert phases[2].freeze_gate is True
    assert phases[0].coeffs.balance_coeff == 0.1
    assert phases[1].coeffs.balance_coeff == 0.001
    assert phases[1].coeffs.cone_coeff == 0.25
    assert phases[1].coeffs.domain_sep_2d_coeff == 0.15


def test_gentle_phase2_lowers_lr_and_freezes_radial() -> None:
    from science.training.config import default_v6_phases

    normal = default_v6_phases(base_lr=5e-4)
    gentle = default_v6_phases(base_lr=5e-4, gentle_phase2=True, phase2_lr=1e-4)
    assert normal[1].lr == 5e-4
    assert gentle[1].lr == 1e-4
    assert gentle[1].freeze_radial_epochs == 5
    assert normal[1].freeze_radial_epochs == 0


def test_p1b_phase_config_angular_unfreeze() -> None:
    from science.training.config import p1b_phase_config

    p1b = p1b_phase_config(lr=1e-4, epochs=3)
    assert p1b.phase == 1
    assert p1b.epochs == 3
    assert p1b.lr == 1e-4
    assert p1b.freeze_angular is False
    assert p1b.freeze_radial is False
    assert p1b.coeffs.angular_coeff == 0.05
    assert p1b.coeffs.shell_corr_coeff == 0.5
    assert p1b.coeffs.cone_depth_anticollapse_coeff == 1.5
    assert p1b.coeffs.cone_depth_min_std == 0.08
    assert p1b.coeffs.proj_violation_coeff == 0.5


def test_p1c_phase_config_disc_expansion() -> None:
    from science.training.config import p1c_phase_config

    p1c = p1c_phase_config(lr=1e-4, epochs=5)
    assert p1c.name == "Phase 1c: Disc expansion"
    assert p1c.coeff_ramp_epochs == 5
    assert p1c.angular_coeff_final == 0.15
    assert p1c.shell_corr_disc_spread_weight_final == 1.0
    assert p1c.disc_spread_min_std_final == 0.15
    assert p1c.min_probe_r_depth_sasa == 0.38
    assert p1c.coeffs.angular_coeff == 0.05
    assert p1c.coeffs.shell_corr_coeff == 0.6
    assert p1c.coeffs.disc_spread_min_std == 0.05
    assert p1c.coeffs.shell_corr_disc_spread_weight == 0.5


def test_apply_phase_coeff_ramp_interpolates() -> None:
    from science.training.config import apply_phase_coeff_ramp, p1c_phase_config

    phase = p1c_phase_config(epochs=5)
    base = phase.coeffs.model_dump()
    ep0 = apply_phase_coeff_ramp(base, phase, epoch=0)
    ep4 = apply_phase_coeff_ramp(base, phase, epoch=4)
    assert ep0["angular_coeff"] == pytest.approx(0.07, abs=0.01)
    assert ep4["angular_coeff"] == pytest.approx(0.15, abs=0.01)
    assert ep0["disc_spread_min_std"] == pytest.approx(0.07, abs=0.01)
    assert ep4["disc_spread_min_std"] == pytest.approx(0.15, abs=0.01)
    assert ep0["shell_corr_disc_spread_weight"] == pytest.approx(0.6, abs=0.01)
    assert ep4["shell_corr_disc_spread_weight"] == pytest.approx(1.0, abs=0.01)


def test_p1d_phase_config_disc_depth_scale() -> None:
    from science.training.config import apply_phase_coeff_ramp, p1d_phase_config

    p1d = p1d_phase_config(epochs=5)
    assert p1d.name == "Phase 1d: Disc-depth scale"
    assert p1d.freeze_radial_epochs == 2
    assert p1d.coeffs.disc_depth_scale_coeff == 1.5
    assert p1d.disc_depth_scale_target_final == 0.45
    assert p1d.min_probe_r_depth_sasa == 0.44
    ramped = apply_phase_coeff_ramp(p1d.coeffs.model_dump(), p1d, epoch=4)
    assert ramped["disc_depth_scale_target"] == pytest.approx(0.45, abs=0.01)


def test_p2_bridge_phase_config() -> None:
    from science.training.config import p2_bridge_phase_config

    p2 = p2_bridge_phase_config(epochs=12)
    assert p2.phase == 2
    assert p2.p2_bridge is True
    assert p2.routing_save_ceiling_start == 1.40
    assert p2.routing_save_ceiling_final == 1.21
    assert p2.min_probe_r_depth_sasa == 0.60
    assert p2.min_probe_r_depth_sasa_save == 0.60
    assert p2.coeffs.disc_depth_scale_coeff == 2.2
    assert p2.freeze_radial_epochs == 3
    assert p2.expert_dropout_ramp_epochs == 5


def test_p2_hypmix_final_phase_config() -> None:
    from science.training.config import p2_hypmix_final_phase_config

    pf = p2_hypmix_final_phase_config(epochs=10, routing_save_ceiling_ramp_epochs=15)
    assert pf.expert_dropout_p == 0.18
    assert pf.coeffs.balance_coeff == 0.01
    assert pf.min_probe_r_depth_sasa == 0.60
    assert pf.min_probe_r_depth_sasa_save == 0.65


def test_full_hyp_moe_theory_config() -> None:
    from science.training.config import full_hyp_moe_theory_config, routing_save_max_for_epoch

    tt = full_hyp_moe_theory_config(epochs=15, routing_save_ceiling_ramp_epochs=15)
    assert tt.expert_dropout_p == 0.15
    assert tt.expert_dropout_ramp_epochs == 0
    assert tt.coeffs.balance_coeff == 0.008
    assert tt.routing_save_ceiling_start == 1.32
    assert tt.routing_save_ceiling_final == 1.18
    assert tt.lr == pytest.approx(3e-5)
    assert routing_save_max_for_epoch(tt, 0) == pytest.approx(1.32, abs=0.01)
    assert routing_save_max_for_epoch(tt, 14) == pytest.approx(1.18, abs=0.01)


def test_infer_v6_model_kwargs_reads_gate_disc_scale_from_training_config() -> None:
    from science.dtie.v6.gnn.model import infer_v6_model_kwargs

    kwargs = infer_v6_model_kwargs(
        {},
        architecture={"deep_hyperbolic_gate": True, "gate_gumbel": True},
        training_config={"gate_disc_scale": 2.5},
    )
    assert kwargs["gate_disc_scale"] == pytest.approx(2.5)
    assert kwargs["gate_gumbel"] is True
    assert kwargs["deep_hyperbolic_gate"] is True


def test_infer_v6_model_kwargs_reads_node_dim_from_state_dict() -> None:
    import torch
    from science.dtie.v6.gnn.model import infer_v6_model_kwargs

    state = {"node_emb.weight": torch.zeros(128, 3)}
    kwargs = infer_v6_model_kwargs(state, architecture={}, training_config={})
    assert kwargs["node_dim"] == 3

    kwargs4 = infer_v6_model_kwargs(
        {"node_emb.weight": torch.zeros(128, 4)},
        architecture={"node_dim": 4},
        training_config={},
    )
    assert kwargs4["node_dim"] == 4


def test_p2_hypmix3_phase_config() -> None:
    from science.training.config import p2_hypmix3_phase_config

    p3 = p2_hypmix3_phase_config(epochs=12, routing_save_ceiling_ramp_epochs=15)
    assert p3.expert_dropout_p == 0.15
    assert p3.coeffs.balance_coeff == 0.008
    assert p3.min_probe_r_depth_sasa_save == 0.65


def test_p2_hypmix_phase_config() -> None:
    from science.training.config import p2_hypmix_phase_config

    p2 = p2_hypmix_phase_config(epochs=15, routing_save_ceiling_ramp_epochs=15)
    assert p2.expert_dropout_p == 0.12
    assert p2.min_probe_r_depth_sasa == 0.60
    assert p2.min_probe_r_depth_sasa_save == 0.65
    assert p2.coeffs.balance_coeff == 0.005


def test_p2_bridge_ramp_epochs_override() -> None:
    from science.training.config import p2_bridge_phase_config, routing_save_max_for_epoch

    phase = p2_bridge_phase_config(epochs=15, routing_save_ceiling_ramp_epochs=15)
    assert phase.routing_save_ceiling_ramp_epochs == 15
    assert phase.angular_ramp_epochs == 15
    # Mid-run ceiling stays high longer than default 10-ep ramp
    assert routing_save_max_for_epoch(phase, 7) == pytest.approx(1.305, abs=0.01)
    assert routing_save_max_for_epoch(phase, 4) == pytest.approx(1.346, abs=0.01)


def test_routing_save_max_for_epoch_ramps() -> None:
    from science.training.config import p2_bridge_phase_config, routing_save_max_for_epoch

    phase = p2_bridge_phase_config(epochs=10)
    assert routing_save_max_for_epoch(phase, 0) == pytest.approx(1.40, abs=0.01)
    assert routing_save_max_for_epoch(phase, 9) == pytest.approx(1.21, abs=0.01)
    assert routing_save_max_for_epoch(phase, 4) == pytest.approx(1.316, abs=0.01)


def test_score_checkpoint_p2_bridge_accepts_ep34_routing() -> None:
    from science.training.checkpoint_score import score_checkpoint

    shell_ok = {
        "proj_frac_mean": 0.0,
        "cone_range_mean": 0.95,
        "probe_r_depth_sasa": 0.649,
        "probe_r_epi_sasa": 0.85,
        "probe_r_proj_depth": 0.98,
        "disc_sigma2_sigma1_mean": 0.40,
        "disc_r_std_mean": 0.05,
    }
    result = score_checkpoint(
        shell_ok,
        {"routing_entropy": 1.344, "expert_starvation_count": 0, "total": 9.0},
        phase=2,
        routing_save_max=1.40,
        min_probe_r_depth_sasa_save=0.60,
    )
    assert result.eligible is True

    eroded = score_checkpoint(
        {**shell_ok, "probe_r_depth_sasa": 0.573},
        {"routing_entropy": 1.385, "expert_starvation_count": 0, "total": 9.0},
        phase=2,
        routing_save_max=1.40,
        min_probe_r_depth_sasa_save=0.60,
    )
    assert eroded.eligible is False
    assert any("shell save gate" in r for r in eroded.reasons)


def test_p1d_extend_defaults() -> None:
    from science.training.config import p1d_extend_defaults, p1d_phase_config

    ext = p1d_extend_defaults(epochs=8)
    phase = p1d_phase_config(epochs=8, **{k: v for k, v in ext.items() if k != "epochs"})
    assert phase.name.endswith("(extend)")
    assert phase.coeffs.disc_depth_scale_coeff == 2.2
    assert phase.disc_depth_scale_target_final == 0.35
    assert phase.freeze_radial_epochs == 3
    assert phase.min_probe_r_depth_sasa == 0.48
    assert phase.coeffs.disc_spread_min_std == 0.10
    assert phase.coeff_ramp_epochs == 8


def test_convergence_monitor_abort_on_nan() -> None:
    monitor = ConvergenceMonitor(collapse_window=3)
    for _ in range(3):
        monitor.record_epoch({"cone_consistency": float("nan")}, {})
    abort, reason = monitor.should_abort()
    assert abort is True
    assert "nonfinite" in reason or "NaN" in reason


def test_convergence_monitor_ignores_missing_health() -> None:
    monitor = ConvergenceMonitor(collapse_window=3)
    for _ in range(5):
        monitor.record_epoch({"cone_consistency": 0.1}, {})
    abort, _ = monitor.should_abort()
    assert abort is False


def test_convergence_monitor_validate_metrics() -> None:
    missing = ConvergenceMonitor.validate_epoch_metrics({"total": 1.0})
    assert "cone_consistency" in missing
    assert "grad_radial" in missing


def test_corpus_manifest_loads() -> None:
    manifest = load_corpus_manifest(Path("manifests/v6_corpus_120.json"))
    assert manifest["version"] == "1.0"
    entries = iter_corpus_entries(Path("manifests/v6_corpus_120.json"), max_proteins=5)
    assert len(entries) == 5
    assert entries[0]["pdb_id"] == "4OBE"


def test_checkpoint_manager_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    mgr = CheckpointManager(tmp_path, protein_count=3)
    path = mgr.save_best(
        model,
        opt,
        global_epoch=1,
        phase=1,
        phase_name="Phase 1",
        metrics={"total": 0.5},
        training_config={"lr": 1e-3},
        score=0.9,
    )
    loaded = mgr.load(path)
    assert loaded.global_epoch == 1
    assert loaded.phase == 1
    restored = torch.nn.Linear(4, 2)
    restored.load_state_dict(loaded.model_state_dict)
    for p1, p2 in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(p1, p2)


def test_promotion_gate_logic() -> None:
    from experiments.training.v6.assess_checkpoint import promotion_gate

    shell_ok = {
        "probe_r_depth_sasa": 0.32,
        "probe_r_epi_sasa": 0.55,
        "probe_r_proj_depth": 0.85,
    }

    passed = promotion_gate(
        {
            "moe": {"routing_entropy_mean": 0.8, "expert_starvation_count": 0},
            "geometry": {"proj_frac_mean": 0.5, **shell_ok},
        }
    )
    assert passed["passed"] is True

    failed = promotion_gate(
        {
            "moe": {"routing_entropy_mean": 1.3, "expert_starvation_count": 1},
            "geometry": {"proj_frac_mean": 0.99, **shell_ok},
        }
    )
    assert failed["passed"] is False
    assert len(failed["failures"]) >= 2

    failed_shell = promotion_gate(
        {
            "moe": {"routing_entropy_mean": 0.8, "expert_starvation_count": 0},
            "geometry": {
                "proj_frac_mean": 0.0,
                "probe_r_depth_sasa": -0.42,
                "probe_r_epi_sasa": 0.68,
                "probe_r_proj_depth": 1.0,
            },
        }
    )
    assert failed_shell["passed"] is False
    assert any("inverted shell" in f for f in failed_shell["failures"])


def test_checkpoint_score_prefers_geometry_and_routing() -> None:
    from science.training.checkpoint_score import score_checkpoint

    shell_ok = {
        "probe_r_depth_sasa": 0.35,
        "probe_r_epi_sasa": 0.55,
        "probe_r_proj_depth": 0.85,
        "disc_sigma2_sigma1_mean": 0.40,
        "disc_r_std_mean": 0.05,
    }
    good = score_checkpoint(
        {"proj_frac_mean": 0.0, "cone_range_mean": 0.12, **shell_ok},
        {"routing_entropy": 1.05, "expert_starvation_count": 0},
        phase=2,
    )
    bad_geom = score_checkpoint(
        {"proj_frac_mean": 0.0, "cone_range_mean": 0.01, **shell_ok},
        {"routing_entropy": 1.05, "expert_starvation_count": 0},
        phase=2,
    )
    bad_route = score_checkpoint(
        {"proj_frac_mean": 0.0, "cone_range_mean": 0.12, **shell_ok},
        {"routing_entropy": 1.35, "expert_starvation_count": 0},
        phase=2,
    )

    assert good.eligible is True
    assert bad_geom.eligible is False
    assert bad_route.eligible is False
    assert good.score > bad_geom.score
    assert good.score > bad_route.score


def test_checkpoint_score_slim_moe_rejects_starvation_and_eval_collapse() -> None:
    from science.training.checkpoint_score import score_checkpoint

    shell_ok = {
        "probe_r_depth_sasa": 0.70,
        "probe_r_epi_sasa": 0.20,
        "probe_r_proj_depth": 0.85,
        "probe_r_epi_ale": 0.40,
        "disc_sigma2_sigma1_mean": 0.40,
        "disc_r_std_mean": 0.05,
        "epistemic_std_mean": 0.05,
        "aleatoric_std_mean": 0.20,
    }
    health = {"proj_frac_mean": 0.0, "cone_range_mean": 0.12, **shell_ok}
    base_losses = {
        "routing_entropy": 1.05,
        "expert_starvation_count": 0,
        "total": 8.0,
        "expert_load_0": 0.28,
        "expert_load_1": 0.24,
        "expert_load_2": 0.24,
        "expert_load_3": 0.24,
    }
    moe_kwargs = {
        "phase": 1,
        "routing_save_max": 1.25,
        "max_expert_starvation_save": 0,
        "min_eval_routing_fraction_save": 0.08,
        "max_eval_routing_fraction_save": 0.50,
        "routing_entropy_min_save": 0.90,
        "min_probe_r_epi_sasa_save": 0.0,
        "max_probe_r_epi_ale_save": 0.95,
    }
    good = score_checkpoint(
        health,
        base_losses,
        inference_routing={"min_routing_fraction": 0.20, "max_routing_fraction": 0.35},
        **moe_kwargs,
    )
    starved = score_checkpoint(
        health,
        {**base_losses, "expert_starvation_count": 1},
        inference_routing={"min_routing_fraction": 0.20, "max_routing_fraction": 0.35},
        **moe_kwargs,
    )
    collapsed = score_checkpoint(
        health,
        base_losses,
        inference_routing={"min_routing_fraction": 0.02, "max_routing_fraction": 0.80},
        **moe_kwargs,
    )
    inverted_epi = score_checkpoint(
        {**health, "probe_r_epi_sasa": -0.3},
        base_losses,
        inference_routing={"min_routing_fraction": 0.20, "max_routing_fraction": 0.35},
        **moe_kwargs,
    )
    assert good.eligible is True
    assert starved.eligible is False
    assert any("starvation" in r for r in starved.reasons)
    assert collapsed.eligible is False
    assert any("eval_min_r" in r or "eval_max_r" in r for r in collapsed.reasons)
    assert inverted_epi.eligible is False
    assert any("inverted epi/sasa" in r for r in inverted_epi.reasons)


def test_checkpoint_score_rejects_collapsed_disc_r_std() -> None:
    from science.training.checkpoint_score import score_checkpoint

    result = score_checkpoint(
        {
            "proj_frac_mean": 0.0,
            "cone_range_mean": 0.12,
            "probe_r_depth_sasa": 0.45,
            "disc_sigma2_sigma1_mean": 0.55,
            "disc_r_std_mean": 0.0006,
        },
        {"routing_entropy": 1.05, "expert_starvation_count": 0},
        phase=2,
    )
    assert result.eligible is False
    assert any("disc_r_std" in r for r in result.reasons)


def test_disc_occupancy_loss_skips_when_origin_collapsed() -> None:
    from science.training.disc_occupancy import disc_occupancy_loss
    import torch

    origin = torch.randn(32, 2) * 0.001
    spread = torch.randn(32, 2) * 0.25
    origin_losses = disc_occupancy_loss(origin, min_sigma_ratio=0.40)
    spread_losses = disc_occupancy_loss(spread, min_sigma_ratio=0.40)
    assert float(origin_losses["disc_r_collapse"]) > 0.0
    assert float(spread_losses["disc_r_collapse"]) == 0.0


def test_checkpoint_score_rejects_collapsed_disc() -> None:
    from science.training.checkpoint_score import score_checkpoint

    healthy_disc = {
        "proj_frac_mean": 0.0,
        "cone_range_mean": 0.12,
        "probe_r_depth_sasa": 0.45,
        "probe_r_epi_sasa": 0.55,
        "probe_r_proj_depth": 0.85,
        "disc_sigma2_sigma1_mean": 0.14,
        "disc_r_std_mean": 0.04,
    }
    result = score_checkpoint(
        healthy_disc,
        {"routing_entropy": 1.05, "expert_starvation_count": 0},
        phase=2,
    )
    assert result.eligible is False
    assert any("disc_sigma2_sigma1" in r for r in result.reasons)


def test_p2_disc_occupancy_phase_config() -> None:
    from science.training.config import (
        p2_disc_occupancy_phase_config,
        p2_disc_occupancy_v2_phase_config,
        p2_disc_occupancy_v3_phase_config,
        p2_disc_occupancy_v4_phase_config,
        p2_disc_occupancy_v5_phase_config,
    )

    phase = p2_disc_occupancy_phase_config(epochs=12)
    assert phase.coeffs.disc_occupancy_coeff == 2.5
    assert phase.min_disc_sigma2_sigma1_save == 0.35
    assert phase.min_probe_r_depth_sasa is None
    assert phase.min_probe_r_depth_sasa_save == 0.55

    v2 = p2_disc_occupancy_v2_phase_config(epochs=12)
    assert v2.coeffs.disc_occupancy_coeff == 2.8
    assert v2.coeffs.disc_min_r_mean == 0.05
    assert v2.min_disc_r_std_save == 0.02
    assert v2.coeffs.disc_pc_repulsion_coeff == 1.5
    assert v2.coeffs.shell_floor_coeff == 0.50
    assert v2.freeze_radial_epochs == 4
    assert v2.lr == 2e-5

    v3 = p2_disc_occupancy_v3_phase_config(epochs=12)
    assert v3.coeffs.disc_occupancy_coeff == 3.5
    assert v3.coeffs.disc_pc_repulsion_coeff == 2.5
    assert v3.min_disc_r_std_save == 0.045
    assert v3.lr == 1e-5

    v4 = p2_disc_occupancy_v4_phase_config(epochs=12)
    assert v4.coeffs.disc_eff_rank_coeff == 1.0
    assert v4.min_disc_effective_rank_save == 1.5
    assert v4.min_disc_sigma2_sigma1_save == 0.45

    v5 = p2_disc_occupancy_v5_phase_config(epochs=12)
    assert v5.coeffs.disc_batch_diversity_coeff == 1.25
    assert v5.coeffs.disc_eff_rank_coeff == 1.0
    assert v5.min_disc_effective_rank_save == 1.5
    assert phase.coeffs.disc_depth_scale_coeff == 0.0


def test_checkpoint_score_rejects_inverted_shell_probes() -> None:
    """High cone_range must not win v6_best when depth×SASA is inverted (b2 pathology)."""
    from science.training.checkpoint_score import score_checkpoint

    b1_like = score_checkpoint(
        {
            "proj_frac_mean": 0.0,
            "cone_range_mean": 0.54,
            "probe_r_depth_sasa": 0.32,
            "probe_r_epi_sasa": 0.55,
            "probe_r_proj_depth": 0.99,
        },
        {"routing_entropy": 1.386, "expert_starvation_count": 0},
        phase=1,
    )
    b2_like = score_checkpoint(
        {
            "proj_frac_mean": 0.0,
            "cone_range_mean": 1.51,
            "probe_r_depth_sasa": -0.42,
            "probe_r_epi_sasa": 0.68,
            "probe_r_proj_depth": 1.0,
        },
        {"routing_entropy": 1.384, "expert_starvation_count": 0},
        phase=1,
    )

    assert b1_like.eligible is True
    assert b2_like.eligible is False
    assert any("inverted shell" in r for r in b2_like.reasons)
    # b2 raw score can exceed b1 on cone_range alone; only eligible epochs save v6_best
    assert b2_like.score > b1_like.score


def test_checkpoint_score_radial_depth_biological_proxy() -> None:
    """Lever A: eff_rank / routing_H must not block tier-1 when shell + thickness pass."""
    from science.training.checkpoint_score import score_checkpoint

    clean_slate = {
        "proj_frac_mean": 0.0,
        "cone_range_mean": 1.0131,
        "probe_r_depth_sasa": 0.730,
        "probe_r_epi_sasa": 0.780,
        "probe_r_proj_depth": 1.0,
        "disc_sigma2_sigma1_mean": 0.665,
        "disc_effective_rank_mean": 1.443,
        "disc_r_std_mean": 0.1207,
        "disc_line_thickness_pre_mean": 0.2192,
        "disc_line_thickness_rms_mean": 0.2192,
    }
    result = score_checkpoint(
        clean_slate,
        {"routing_entropy": 1.379, "expert_starvation_count": 0, "total": 9.1},
        phase=2,
        min_probe_r_depth_sasa_save=0.58,
        min_disc_effective_rank_save=1.5,
        min_disc_sigma2_sigma1_save=0.45,
        min_disc_line_thickness_save=0.025,
        disc_radial_source="radial_depth",
    )
    assert result.eligible is True
    assert not any("effective_rank" in r for r in result.reasons)
    assert not any("routing_H" in r for r in result.reasons)


def test_checkpoint_score_mobius_still_rejects_low_eff_rank() -> None:
    from science.training.checkpoint_score import score_checkpoint

    health = {
        "proj_frac_mean": 0.0,
        "cone_range_mean": 1.01,
        "probe_r_depth_sasa": 0.73,
        "probe_r_epi_sasa": 0.78,
        "probe_r_proj_depth": 1.0,
        "disc_sigma2_sigma1_mean": 0.665,
        "disc_effective_rank_mean": 1.443,
        "disc_r_std_mean": 0.12,
        "disc_line_thickness_rms_mean": 0.22,
    }
    result = score_checkpoint(
        health,
        {"routing_entropy": 1.05, "expert_starvation_count": 0, "total": 9.0},
        phase=2,
        min_disc_effective_rank_save=1.5,
        disc_radial_source="mobius",
    )
    assert result.eligible is False
    assert any("effective_rank" in r for r in result.reasons)


def test_promote_checkpoint_updates_contract(tmp_path: Path) -> None:
    ckpt = tmp_path / "test.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        yaml.dump(
            {
                "gnn_models": {
                    "production_model_id": "gospc_v6",
                    "models": {
                        "gospc_v6": {
                            "model_version": "GOSPConeMapper-v6",
                            "api_alias": "v6",
                            "status": "production",
                            "runner_module": "science.dtie.v6.gnn.runner",
                            "runner_class": "V6GNNRunner",
                            "production_checkpoint_id": "tokyo_eyes_v6",
                        }
                    },
                    "checkpoints": {
                        "tokyo_eyes_v6": {
                            "model_id": "gospc_v6",
                            "path": "checkpoints/v6/tokyo_eyes_v6.pt",
                            "status": "production",
                        }
                    },
                }
            }
        )
    )

    dest_dir = tmp_path / "checkpoints" / "v6"
    dest_dir.mkdir(parents=True)

    # Patch repo root for promote destination
    import science.training.promote as promote_mod

    original_root = promote_mod._REPO_ROOT
    promote_mod._REPO_ROOT = tmp_path
    try:
        result = promote_checkpoint(
            PromotionConfig(checkpoint_path=ckpt, checkpoint_id="tokyo_eyes_v6_candidate"),
            contract_path=contract_path,
        )
    finally:
        promote_mod._REPO_ROOT = original_root

    assert result["checkpoint_id"] == "tokyo_eyes_v6_candidate"
    assert result["status"] == "candidate"
    updated = yaml.safe_load(contract_path.read_text())
    assert "tokyo_eyes_v6_candidate" in updated["gnn_models"]["checkpoints"]


def test_phase_preset_name() -> None:
    from science.training.config import TrainingConfig

    assert TrainingConfig(full_hyp_moe_test=True).phase_preset_name() == "full_hyp_moe_test"
    assert TrainingConfig(p2_hypmix_final=True).phase_preset_name() == "p2_hypmix_final"
    assert TrainingConfig(p2_bridge=True).phase_preset_name() == "p2_bridge"
    assert TrainingConfig(phase=2).phase_preset_name() == "phase_2"
    assert (
        TrainingConfig(p4_head_decouple_decorr_only=True).phase_preset_name()
        == "p4_head_decouple_decorr_only"
    )


def test_mlflow_resolve_run_checkpoint_from_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    from science.training.mlflow_run import find_run_id_by_checkpoint_path, resolve_run_checkpoint_path

    ckpt = tmp_path / "v6_best.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-v6-glue")
    with mlflow.start_run(run_name="glue_tag_test") as run:
        mlflow.set_tag("checkpoint_path", str(ckpt))
        run_id = run.info.run_id

    assert find_run_id_by_checkpoint_path(ckpt, tracking_uri=tracking_uri, experiment_name="test-v6-glue") == run_id
    resolved = resolve_run_checkpoint_path(run_id, tracking_uri=tracking_uri)
    assert resolved == ckpt.resolve()


def test_mlflow_resolve_run_checkpoint_from_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    from science.training.mlflow_run import resolve_run_checkpoint_path

    ckpt = tmp_path / "v6_best.pt"
    torch.save({"model_state_dict": {}, "score": 2.1}, ckpt)
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-v6-artifact")
    with mlflow.start_run(run_name="glue_artifact_test") as run:
        mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
        run_id = run.info.run_id
    ckpt.unlink()

    resolved = resolve_run_checkpoint_path(run_id, tracking_uri=tracking_uri)
    assert resolved.is_file()
    loaded = torch.load(resolved, map_location="cpu", weights_only=False)
    assert loaded.get("score") == pytest.approx(2.1)


def test_promote_from_mlflow_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    from science.training.mlflow_run import resolve_run_checkpoint_path
    from science.training.promote import promote_checkpoint

    ckpt = tmp_path / "runs" / "test_run" / "v6_best.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {}}, ckpt)
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-v6-promote")
    with mlflow.start_run(run_name="promote_test") as run:
        mlflow.set_tag("checkpoint_path", str(ckpt))
        run_id = run.info.run_id

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        yaml.dump({"gnn_models": {"models": {}, "checkpoints": {}}})
    )

    import science.training.promote as promote_mod

    original_root = promote_mod._REPO_ROOT
    promote_mod._REPO_ROOT = tmp_path
    try:
        resolved = resolve_run_checkpoint_path(run_id, tracking_uri=tracking_uri)
        result = promote_checkpoint(
            PromotionConfig(
                checkpoint_path=resolved,
                checkpoint_id="tokyo_eyes_v6_candidate",
                run_id=run_id,
            ),
            contract_path=contract_path,
        )
    finally:
        promote_mod._REPO_ROOT = original_root

    assert result["mlflow_run_id"] == run_id
    assert (tmp_path / "checkpoints" / "v6" / "v6_best.pt").is_file()


def test_metric_focus_ep100_routing_blocks_save() -> None:
    """Ep100-like snapshot: shell excellent, routing above save ceiling."""
    from science.training.metric_focus import assess_training_focus, focus_mlflow_metrics

    health = {
        "proj_frac_mean": 0.0,
        "cone_range_mean": 0.95,
        "probe_r_depth_sasa": 0.80,
        "probe_r_epi_sasa": 0.86,
        "probe_r_proj_depth": 0.99,
        "probe_r_disc_sasa": 0.85,
        "disc_r_std_mean": 0.12,
    }
    losses = {
        "routing_entropy": 1.306,
        "expert_starvation_count": 0,
        "expert_load_0": 0.24,
        "expert_load_1": 0.26,
        "expert_load_2": 0.25,
        "expert_load_3": 0.25,
        "total": 8.28,
    }
    summary = assess_training_focus(
        health,
        losses,
        phase=2,
        routing_save_max=1.18,
        min_probe_r_depth_sasa_save=0.65,
        checkpoint_eligible=False,
    )
    assert summary["primary_focus"][0] == "routing_entropy"
    assert "routing_entropy" in summary["primary_focus"]
    assert summary["counts"]["critical"] >= 1
    assert summary["counts"]["matters_needs_work"] >= 1
    assert "routing_entropy" in summary["recommendation"]
    healthy_names = {i["metric"] for i in summary["healthy"]}
    assert "probe_r_depth_sasa" in healthy_names
    assert "probe_r_proj_depth" in healthy_names

    mlflow = focus_mlflow_metrics(summary)
    assert mlflow["focus_critical_count"] >= 1.0
    assert mlflow["focus_flag_routing_entropy"] == 1.0
    assert mlflow["focus_healthy_count"] >= 2.0


def test_metric_focus_ep85_production_winner() -> None:
    """Ep85-like snapshot: eligible, routing in stretch band."""
    from science.training.metric_focus import assess_training_focus

    health = {
        "proj_frac_mean": 0.0,
        "cone_range_mean": 0.95,
        "probe_r_depth_sasa": 0.808,
        "probe_r_epi_sasa": 0.85,
        "probe_r_proj_depth": 0.99,
    }
    losses = {"routing_entropy": 1.238, "expert_starvation_count": 0, "total": 8.0}
    summary = assess_training_focus(
        health,
        losses,
        phase=2,
        routing_save_max=1.25,  # ceiling when ep85 saved (later theory-test ramp is tighter)
        min_probe_r_depth_sasa_save=0.65,
        checkpoint_eligible=True,
    )
    assert summary["counts"]["critical"] == 0
    assert summary["checkpoint_eligible"] is True
    assert "Eligible epoch" in summary["recommendation"]
    route_items = [i for i in summary["watch"] + summary["healthy"] if i["metric"] == "routing_entropy"]
    assert route_items
    assert route_items[0]["status"] in {"watch", "healthy"}


def test_stage_runner_resets_best_score_on_cross_phase_resume(tmp_path: Path) -> None:
    """Touchup from gate v6_best must not inherit gate score as save baseline."""
    import math

    from experiments.training.v6.stage_runner import StageRunner
    from science.training.checkpoint import CheckpointData

    model = torch.nn.Linear(2, 1)
    cfg = TrainingConfig(output_dir=str(tmp_path), corpus_manifest="manifests/v6_corpus_120.json")
    resume = CheckpointData(
        model_state_dict={},
        optimizer_state_dict=None,
        global_epoch=30,
        phase=2,
        phase_name="Gate promotion",
        metrics={},
        training_config={},
        architecture={},
        score=3.32,
    )
    runner = StageRunner(model, [], cfg, resume_state=resume)
    assert runner.best_score == pytest.approx(3.32)

    runner._begin_phase_best_tracking(4)
    assert runner.best_score == -math.inf
    assert runner._saved_eligible is False

    runner2 = StageRunner(model, [], cfg, resume_state=resume)
    runner2._begin_phase_best_tracking(2)
    assert runner2.best_score == pytest.approx(3.32)
    assert runner2._saved_eligible is True

