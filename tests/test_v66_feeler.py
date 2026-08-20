"""v6.6 feeler cold-start: phase whitelist + expert timeout enablement."""

from __future__ import annotations

import pytest

from science.training.config import (
    TrainingConfig,
    apply_v66_feeler_config,
    apply_v66_feeler_phases,
)


def test_v66_feeler_phases_p1_through_p4() -> None:
    phases = apply_v66_feeler_phases(base_lr=5e-4, epochs=20)
    assert len(phases) == 4
    p1, p2, p3, p4 = phases
    assert p1.phase == 1
    assert p1.epochs == 20
    assert p1.freeze_angular is True
    assert p1.freeze_backbone is False
    assert p1.slim_moe_structural_ssot_train is False
    assert p1.expert_timeout_max_share == 0.45
    assert p1.expert_timeout_eligible_experts is None

    assert p2.phase == 2
    assert p2.epochs == 20
    assert p2.freeze_angular is False
    assert p2.lr == 2.5e-4
    assert p2.expert_timeout_max_share == 0.45
    assert p2.coeffs.angular_coeff == 0.10
    assert p2.coeffs.balance_coeff == 0.02
    assert p2.coeffs.disc_occupancy_coeff == 0.45

    assert p3.phase == 3
    assert p3.epochs == 20
    assert p3.freeze_angular is False
    assert p3.lr == 1.25e-4
    assert p3.coeffs.angular_coeff == 0.10
    assert p3.coeffs.disc_occupancy_coeff == 0.0
    assert p3.coeffs.disc_pc_repulsion_coeff == 0.0
    assert p3.coeffs.disc_thickness_floor_coeff == 0.0
    assert p3.coeffs.disc_origin_span_floor_coeff == 0.0
    assert p3.coeffs.disc_eff_rank_coeff == 0.0
    assert p3.coeffs.cone_coeff == 0.15
    assert p3.coeffs.rim_angular_repulsion_coeff == 0.0

    assert p4.phase == 4
    assert p4.epochs == 10
    assert p4.freeze_angular is False
    assert p4.lr == 1.0e-4
    assert p4.coeffs.disc_occupancy_coeff == 0.0
    assert p4.coeffs.disc_pc_repulsion_coeff == 0.0
    assert p4.coeffs.rim_angular_repulsion_coeff == 0.25
    assert p4.coeffs.rim_pc2_floor_coeff == 0.15
    assert p4.coeffs.rim_angular_min_r == 0.35
    assert p4.coeffs.rim_pc2_min_std == 0.06


def test_v66_feeler_loss_whitelist() -> None:
    coeffs = apply_v66_feeler_phases()[0].coeffs
    assert coeffs.balance_coeff == 0.05
    assert coeffs.cone_coeff == 0.30
    assert coeffs.neighborhood_coeff == 0.10
    assert coeffs.cone_depth_anticollapse_coeff == 0.50
    assert coeffs.disc_occupancy_coeff == 0.35
    assert coeffs.disc_thickness_floor_coeff == 0.80
    assert coeffs.disc_pc_repulsion_coeff == 0.25
    assert coeffs.proj_violation_coeff == 2.0
    assert coeffs.cone_target_mode == "rho_rim"

    assert coeffs.evidential_coeff == 0.0
    assert coeffs.angular_coeff == 0.0
    assert coeffs.domain_sep_2d_coeff == 0.0
    assert coeffs.shell_corr_coeff == 0.0
    assert coeffs.epistemic_sasa_pen_coeff == 0.0
    assert coeffs.epi_ale_decorrelation_coeff == 0.0
    assert coeffs.routing_load_floor_coeff == 0.0
    assert coeffs.routing_load_ceiling_coeff == 0.0
    assert coeffs.pocket_bce_coeff == 0.0


def test_v66_feeler_config_learned_path() -> None:
    cfg = apply_v66_feeler_config(TrainingConfig())
    assert cfg.v66_feeler_lineage is True
    assert cfg.master_cold_lineage is False
    assert cfg.slim_moe_structural_ssot is False
    assert cfg.structural_disc_frozen is False
    assert cfg.disc_layout_source == "gnn_learned"
    assert cfg.gnn_lineage == "v6.6"
    assert cfg.topology_only_gate is True
    assert cfg.thermo_edge_features is False
    assert cfg.multi_rel_edge_mp is True
    assert cfg.role_edge_mp is True
    assert cfg.phase_preset_name() == "v66_feeler_p1"


def test_v66_feeler_epochs_override_for_expand() -> None:
    """Expand runs pass --epochs 50 with --phase 2; both phases honor override."""
    phases = apply_v66_feeler_phases(base_lr=5e-4, epochs=20, p2_epochs=50)
    assert phases[0].epochs == 20
    assert phases[1].epochs == 50
    # stage_runner also copies epochs_override onto the selected phase
    p2 = phases[1].model_copy(update={"epochs": 50})
    assert p2.epochs == 50
    assert p2.coeffs.angular_coeff == 0.10
    assert p2.expert_timeout_max_share == 0.45


def test_v66_feeler_p4_epochs_default() -> None:
    phases = apply_v66_feeler_phases(base_lr=5e-4, epochs=20, p4_epochs=12)
    assert phases[3].epochs == 12


def test_v66_feeler_angular_lift_phase() -> None:
    from science.training.config import v66_feeler_angular_lift_phase_config

    phase = v66_feeler_angular_lift_phase_config(lr=1.25e-4, epochs=15)
    assert phase.phase == 5
    assert phase.lift_path_recovery_train is True
    assert phase.coeffs.disc_occupancy_coeff == 0.0
    assert phase.coeffs.disc_pc_repulsion_coeff == 0.0
    assert phase.coeffs.disc_thickness_floor_coeff == 0.0
    assert phase.coeffs.x_hyp_thickness_floor_coeff == 0.0
    assert phase.coeffs.angular_coeff == 0.10
    assert phase.expert_timeout_max_share == 0.45


def test_v66_feeler_coupling_phase() -> None:
    from science.training.config import v66_feeler_coupling_phase_config

    phase = v66_feeler_coupling_phase_config(lr=1.0e-4, epochs=12)
    assert phase.phase == 6
    assert phase.coeffs.disc_occupancy_coeff == 0.0


def test_v66_feeler_no_exclusivity_and_angular_phases() -> None:
    from science.training.config import (
        v66_feeler_dehydron_angular_phase_config,
        v66_feeler_no_exclusivity_phase_config,
    )

    p7 = v66_feeler_no_exclusivity_phase_config()
    p8 = v66_feeler_dehydron_angular_phase_config()
    assert p7.phase == 7
    assert p8.phase == 8
    assert p7.coeffs.rim_angular_repulsion_coeff == 0.0


def test_v66_feeler_p3_geom_half_stack_coeffs() -> None:
    from science.training.config import _v66_feeler_p3_geom_coeffs

    full = _v66_feeler_p3_geom_coeffs(stack_scale=1.0)
    half = _v66_feeler_p3_geom_coeffs(stack_scale=0.5)
    assert half.disc_occupancy_coeff == full.disc_occupancy_coeff * 0.5
    assert half.disc_pc_repulsion_coeff == full.disc_pc_repulsion_coeff * 0.5
    assert half.disc_thickness_floor_coeff == full.disc_thickness_floor_coeff * 0.5
    assert half.disc_origin_span_floor_coeff == full.disc_origin_span_floor_coeff * 0.5
    assert half.disc_eff_rank_coeff == full.disc_eff_rank_coeff * 0.5
    assert half.disc_occupancy_min_sigma_ratio == full.disc_occupancy_min_sigma_ratio


def test_v66_feeler_p3_geom_phase() -> None:
    from science.training.config import v66_feeler_p3_geom_phase_config

    phase = v66_feeler_p3_geom_phase_config(lr=1.25e-4, epochs=20)
    assert phase.phase == 11
    assert phase.freeze_radial_epochs == 3
    assert phase.coeffs.disc_occupancy_coeff == 0.65
    assert phase.coeffs.disc_origin_span_floor_coeff == 2.5


def test_v66_feeler_p3_geom_edges_phase() -> None:
    from science.training.config import v66_feeler_p3_geom_edges_phase_config

    phase = v66_feeler_p3_geom_edges_phase_config(lr=1.0e-4, epochs=15)
    assert phase.phase == 10
    assert phase.freeze_radial_epochs == 3
    assert phase.coeffs.disc_occupancy_coeff == 0.65
    assert phase.coeffs.disc_pc_repulsion_coeff == 0.50
    assert phase.coeffs.disc_origin_span_floor_coeff == 2.5
    assert phase.coeffs.disc_thickness_floor_coeff == 1.20


def test_v66_feeler_rim_fanout_model_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_model_phase_config

    phase = v66_feeler_rim_fanout_model_phase_config(lr=1.0e-4, epochs=12)
    assert phase.phase == 12
    assert phase.coeffs.disc_occupancy_coeff == 0.65
    assert phase.coeffs.disc_thickness_floor_coeff == 1.20
    assert phase.coeffs.rim_angular_repulsion_coeff == 0.0
    assert phase.coeffs.rim_pc2_floor_coeff == 0.0
    assert phase.coeffs.angular_coeff == 0.15


def test_v66_feeler_rim_fanout_cold_curriculum() -> None:
    from science.training.config import apply_v66_feeler_rim_fanout_cold_phases

    phases = apply_v66_feeler_rim_fanout_cold_phases(p12_epochs=10)
    assert [p.phase for p in phases] == [1, 2, 12]
    assert phases[0].freeze_angular is True
    assert phases[1].coeffs.disc_occupancy_coeff == 0.45
    assert phases[1].coeffs.angular_coeff == 0.10
    assert phases[2].coeffs.disc_occupancy_coeff == 0.65


def test_v66_feeler_rim_fanout_warm_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_warm_phase_config

    phase = v66_feeler_rim_fanout_warm_phase_config(lr=1.0e-4, epochs=15)
    assert phase.phase == 12
    assert phase.coeffs.disc_occupancy_coeff == 0.325
    assert phase.coeffs.rim_angular_repulsion_coeff == 0.25
    assert phase.coeffs.rim_pc2_floor_coeff == 0.15


def test_v66_feeler_rim_fanout_polish_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_polish_phase_config

    phase = v66_feeler_rim_fanout_polish_phase_config(lr=5.0e-5, epochs=10)
    assert phase.phase == 12
    assert phase.coeffs.disc_occupancy_coeff == 0.1625
    assert phase.coeffs.disc_depth_scale_coeff == 1.0
    assert phase.coeffs.rim_angular_repulsion_coeff == 0.15
    assert phase.coeffs.rim_pc2_floor_coeff == 0.10


def test_v66_feeler_rim_fanout_angular_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_angular_phase_config

    phase = v66_feeler_rim_fanout_angular_phase_config(lr=5.0e-5, epochs=8)
    assert phase.phase == 12
    assert phase.coeffs.disc_occupancy_coeff == pytest.approx(0.4875)
    assert phase.coeffs.disc_depth_scale_coeff == 1.2
    assert phase.coeffs.rim_angular_min_r == 0.20
    assert phase.coeffs.rim_angular_repulsion_coeff == 0.35
    assert phase.coeffs.angular_coeff == 0.20


def test_v66_feeler_rim_fanout_angular_v2_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_angular_v2_phase_config

    phase = v66_feeler_rim_fanout_angular_v2_phase_config(lr=5.0e-5, epochs=8)
    assert phase.phase == 12
    assert phase.coeffs.rim_angular_min_r == 0.12
    assert phase.coeffs.rim_angular_repulsion_coeff == 0.40
    assert phase.coeffs.rim_pc2_floor_coeff == 0.30
    assert phase.coeffs.disc_depth_scale_coeff == 1.25
    assert phase.coeffs.angular_coeff == 0.22


def test_v66_feeler_rim_fanout_radius_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_radius_phase_config

    phase = v66_feeler_rim_fanout_radius_phase_config(lr=5.0e-5, epochs=8)
    assert phase.phase == 12
    assert phase.coeffs.disc_depth_scale_target == 0.55
    assert phase.coeffs.disc_depth_scale_coeff == 1.25
    assert phase.coeffs.disc_occupancy_coeff == pytest.approx(0.4875)
    assert phase.coeffs.rim_angular_min_r == 0.12
    assert phase.coeffs.rim_angular_repulsion_coeff == 0.40


def test_v66_feeler_rim_fanout_coverage_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_coverage_phase_config

    phase = v66_feeler_rim_fanout_coverage_phase_config(lr=5.0e-5, epochs=8)
    assert phase.phase == 12
    assert phase.coeffs.disc_angular_coverage_coeff == 0.45
    assert phase.coeffs.disc_angular_coverage_min_r == 0.12
    assert phase.coeffs.disc_angular_coverage_n_bins == 12
    assert phase.coeffs.disc_angular_coverage_min_bin_frac == 0.40
    assert phase.coeffs.disc_depth_scale_target == 0.55
    assert phase.coeffs.disc_depth_scale_coeff == 1.25


def test_v66_feeler_rim_fanout_antibarrier_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_antibarrier_phase_config

    phase = v66_feeler_rim_fanout_antibarrier_phase_config(lr=5.0e-5, epochs=25)
    assert phase.phase == 12
    assert phase.coeffs.expert_angular_diversity_coeff == 0.40
    assert phase.coeffs.expert_sector_recruit_coeff == 0.35
    assert phase.coeffs.disc_angular_coverage_coeff == 0.30
    assert phase.coeffs.expert_angular_max_R == 0.55
    assert phase.coeffs.disc_depth_scale_target == 0.55


def test_v66_feeler_rim_fanout_expert_arc_phase() -> None:
    from science.training.config import v66_feeler_rim_fanout_expert_arc_phase_config

    phase = v66_feeler_rim_fanout_expert_arc_phase_config(lr=5.0e-5, epochs=20)
    assert phase.phase == 12
    assert phase.coeffs.expert_angular_diversity_coeff == 0.18
    assert phase.coeffs.expert_angular_max_R == 0.60
    assert phase.coeffs.expert_angular_min_mean_sep == 0.40
    assert phase.coeffs.expert_sector_recruit_coeff == 0.0
    assert phase.coeffs.disc_angular_coverage_coeff == 0.40
    assert phase.coeffs.disc_depth_scale_target == 0.55


def test_order_proteins_with_anchors_puts_1f88_first() -> None:
    from experiments.training.v6.train_loop import order_proteins_with_anchors

    proteins = [
        {"pdb_id": "4OBE", "n": 1},
        {"pdb_id": "1F88", "n": 2},
        {"pdb_id": "3OMV", "n": 3},
    ]
    ordered = order_proteins_with_anchors(proteins, ["1F88"])
    assert [p["pdb_id"] for p in ordered] == ["1F88", "4OBE", "3OMV"]


def test_order_proteins_with_anchors_missing_raises() -> None:
    import pytest

    from experiments.training.v6.train_loop import order_proteins_with_anchors

    with pytest.raises(ValueError, match="missing from loaded corpus"):
        order_proteins_with_anchors([{"pdb_id": "4OBE"}], ["1F88"])


def test_v66_model_rim_fanout_forward_flag() -> None:
    from science.dtie.v66.gnn.model import GOSPConeMapperV66

    m = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=1,
        num_experts=2,
        role_edge_mp=True,
        rim_fanout_forward=True,
        spoke_edge_scale=1.5,
        ribbon_edge_scale=1.35,
    )
    assert m.rim_fanout is not None
    assert m.rim_fanout_forward is True
    assert m.spoke_edge_scale == 1.5
    assert m.convs[0].spoke_edge_scale == 1.5
    assert m.convs[0].ribbon_edge_scale == 1.35


def test_expert_timeout_enable_predicate() -> None:
    """Mirror stage_runner enable rule: max_share set OR slim P1–P3."""
    feeler = apply_v66_feeler_phases()[0]
    enable = feeler.expert_timeout_max_share is not None or (
        feeler.slim_moe_structural_ssot_train and feeler.phase in (1, 2, 3)
    )
    assert enable is True

    from science.training.config import default_v6_phases

    default_p1 = default_v6_phases(5e-4)[0]
    enable_default = default_p1.expert_timeout_max_share is not None or (
        default_p1.slim_moe_structural_ssot_train and default_p1.phase in (1, 2, 3)
    )
    assert enable_default is False
