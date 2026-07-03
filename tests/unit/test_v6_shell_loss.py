"""Tests for v6 dist0 anti-collapse and shell correlation losses."""

from __future__ import annotations

import torch

from science.dtie.v6.loss import (
    cone_depth_anticollapse_loss,
    gosp_loss_v6,
    shell_correlation_loss,
)


def test_cone_depth_anticollapse_penalizes_flat() -> None:
    flat = torch.ones(20, 1) * 1.5
    spread = torch.linspace(0.5, 2.0, 20).unsqueeze(1)
    assert cone_depth_anticollapse_loss(flat).item() > 0.0
    assert cone_depth_anticollapse_loss(spread).item() == 0.0


def test_shell_correlation_loss_prefers_aligned_signals() -> None:
    n = 24
    sasa = torch.linspace(0.0, 1.0, n)
    depth = sasa.unsqueeze(1) * 1.5 + 0.2
    epi = sasa * 0.8 + 0.1
    hyp = torch.stack([sasa * 0.9, sasa * 0.1], dim=1)

    good = shell_correlation_loss(depth, epi.unsqueeze(1), sasa, hyp)
    bad_depth = torch.ones_like(depth) * depth.mean()
    bad = shell_correlation_loss(bad_depth, epi.unsqueeze(1), sasa, hyp)

    assert good["shell_corr_total"].item() < bad["shell_corr_total"].item()
    assert good["shell_r_depth_sasa"].item() > 0.9
    assert good["disc_r_std"].item() > 0.1


def test_shell_correlation_loss_penalizes_central_disc() -> None:
    n = 32
    sasa = torch.linspace(0.0, 1.0, n)
    depth = sasa.unsqueeze(1) * 1.2 + 0.3
    epi = sasa * 0.5 + 0.2
    spread_hyp = torch.stack([sasa * 0.8, sasa * 0.2], dim=1)
    central_hyp = torch.randn(n, 2) * 0.01

    spread = shell_correlation_loss(depth, epi.unsqueeze(1), sasa, spread_hyp)
    central = shell_correlation_loss(depth, epi.unsqueeze(1), sasa, central_hyp)

    assert spread["shell_corr_disc_spread"].item() < central["shell_corr_disc_spread"].item()
    assert spread["shell_corr_total"].item() < central["shell_corr_total"].item()


def test_disc_depth_scale_loss_pulls_disc_toward_depth() -> None:
    from science.dtie.v6.loss import disc_depth_scale_loss

    depth = torch.linspace(0.2, 1.0, 16).unsqueeze(1)
    # disc too small vs depth
    hyp_small = torch.stack([depth.squeeze() * 0.02, torch.zeros(16)], dim=1)
    # disc matched to normalized depth * 0.45
    depth_norm = (depth.squeeze() - depth.min()) / (depth.max() - depth.min())
    hyp_good = torch.stack([depth_norm * 0.45, torch.zeros(16)], dim=1)

    assert disc_depth_scale_loss(depth, hyp_good, target_radius=0.45).item() < 1e-4
    assert disc_depth_scale_loss(depth, hyp_small, target_radius=0.45).item() > 0.01


def test_gosp_loss_v6_includes_shell_terms() -> None:
    n = 16
    device = torch.device("cpu")
    output = {
        "x_hyp": torch.randn(n, 2, device=device),
        "x_routed_hyp": torch.randn(n, 2, device=device),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n, device=device).unsqueeze(1),
        "hyp_projections_2d": torch.randn(n, 2, device=device) * 0.3,
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.0, device=device),
        "expert_load": torch.ones(4, device=device) * 0.25,
        "evidence": {
            "mu": torch.zeros(n, device=device),
            "nu": torch.ones(n, device=device),
            "alpha": torch.ones(n, device=device) * 2.0,
            "beta": torch.ones(n, device=device),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n, device=device).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
    }
    target_rho = torch.linspace(1.0, 20.0, n, device=device)
    ca_coords = torch.randn(n, 3, device=device)
    sasa = torch.linspace(0.0, 1.0, n, device=device)

    losses = gosp_loss_v6(
        output,
        target_rho,
        ca_coords,
        sasa=sasa,
        cone_depth_anticollapse_coeff=0.5,
        shell_corr_coeff=0.25,
    )
    assert losses["total"].ndim == 0
    assert "cone_depth_anticollapse" in losses
    assert "shell_correlation" in losses
    assert losses["shell_r_depth_sasa"].item() > 0.0
    assert "shell_corr_disc_spread" in losses
    assert "disc_r_std" in losses


def test_gosp_loss_v6_disc_depth_scale() -> None:
    n = 16
    device = torch.device("cpu")
    output = {
        "x_hyp": torch.randn(n, 2, device=device),
        "x_routed_hyp": torch.randn(n, 2, device=device),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n, device=device).unsqueeze(1),
        "hyp_projections_2d": torch.randn(n, 2, device=device) * 0.02,
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.0, device=device),
        "expert_load": torch.ones(4, device=device) * 0.25,
        "evidence": {
            "mu": torch.zeros(n, device=device),
            "nu": torch.ones(n, device=device),
            "alpha": torch.ones(n, device=device) * 2.0,
            "beta": torch.ones(n, device=device),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n, device=device).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
    }
    target_rho = torch.linspace(1.0, 20.0, n, device=device)
    ca_coords = torch.randn(n, 3, device=device)

    with_scale = gosp_loss_v6(
        output,
        target_rho,
        ca_coords,
        disc_depth_scale_coeff=1.5,
        disc_depth_scale_target=0.45,
    )
    without = gosp_loss_v6(
        output,
        target_rho,
        ca_coords,
        disc_depth_scale_coeff=0.0,
    )
    assert "disc_depth_scale" in with_scale
    assert with_scale["disc_depth_scale"].item() > 0.0
def test_gosp_loss_v6_disc_occupancy() -> None:
    n = 32
    device = torch.device("cpu")
    streak = torch.stack(
        [torch.linspace(-0.4, 0.4, n, device=device), torch.linspace(-0.4, 0.4, n, device=device)],
        dim=1,
    )
    cloud = torch.randn(n, 2, device=device) * 0.25
    base = {
        "x_hyp": torch.randn(n, 2, device=device),
        "x_routed_hyp": torch.randn(n, 2, device=device),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n, device=device).unsqueeze(1),
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.0, device=device),
        "expert_load": torch.ones(4, device=device) * 0.25,
        "evidence": {
            "mu": torch.zeros(n, device=device),
            "nu": torch.ones(n, device=device),
            "alpha": torch.ones(n, device=device) * 2.0,
            "beta": torch.ones(n, device=device),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n, device=device).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
    }
    target_rho = torch.linspace(1.0, 20.0, n, device=device)
    ca_coords = torch.randn(n, 3, device=device)

    streak_losses = gosp_loss_v6(
        {**base, "hyp_projections_2d": streak},
        target_rho,
        ca_coords,
        disc_occupancy_coeff=2.5,
        disc_occupancy_min_sigma_ratio=0.35,
    )
    cloud_losses = gosp_loss_v6(
        {**base, "hyp_projections_2d": cloud},
        target_rho,
        ca_coords,
        disc_occupancy_coeff=2.5,
        disc_occupancy_min_sigma_ratio=0.35,
    )
    assert streak_losses["disc_occupancy"].item() > cloud_losses["disc_occupancy"].item()
    assert cloud_losses["disc_occupancy"].item() == 0.0
    assert streak_losses["disc_sigma2_sigma1"].item() < 0.2


def test_gosp_loss_v6_disc_eff_rank() -> None:
    n = 32
    device = torch.device("cpu")
    streak = torch.stack(
        [torch.linspace(-0.4, 0.4, n, device=device), torch.linspace(-0.4, 0.4, n, device=device)],
        dim=1,
    )
    cloud = torch.randn(n, 2, device=device) * 0.25
    base = {
        "x_hyp": torch.randn(n, 2, device=device),
        "x_routed_hyp": torch.randn(n, 2, device=device),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n, device=device).unsqueeze(1),
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.0, device=device),
        "expert_load": torch.ones(4, device=device) * 0.25,
        "evidence": {
            "mu": torch.zeros(n, device=device),
            "nu": torch.ones(n, device=device),
            "alpha": torch.ones(n, device=device) * 2.0,
            "beta": torch.ones(n, device=device),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n, device=device).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
    }
    target_rho = torch.linspace(1.0, 20.0, n, device=device)
    ca_coords = torch.randn(n, 3, device=device)

    streak_losses = gosp_loss_v6(
        {**base, "hyp_projections_2d": streak},
        target_rho,
        ca_coords,
        disc_eff_rank_coeff=1.0,
        disc_eff_rank_min=1.6,
        disc_min_r_mean=0.05,
    )
    cloud_losses = gosp_loss_v6(
        {**base, "hyp_projections_2d": cloud},
        target_rho,
        ca_coords,
        disc_eff_rank_coeff=1.0,
        disc_eff_rank_min=1.6,
        disc_min_r_mean=0.05,
    )
    assert streak_losses["disc_eff_rank"].item() > cloud_losses["disc_eff_rank"].item()
    assert cloud_losses["disc_eff_rank"].item() == 0.0


def test_gosp_loss_v6_disc_batch_diversity() -> None:
    n = 32
    device = torch.device("cpu")
    streak = torch.stack(
        [torch.linspace(-0.4, 0.4, n, device=device), torch.linspace(-0.4, 0.4, n, device=device)],
        dim=1,
    )
    cloud = torch.randn(n, 2, device=device) * 0.25
    base = {
        "x_hyp": torch.randn(n, 2, device=device),
        "x_routed_hyp": torch.randn(n, 2, device=device),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n, device=device).unsqueeze(1),
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.0, device=device),
        "expert_load": torch.ones(4, device=device) * 0.25,
        "evidence": {
            "mu": torch.zeros(n, device=device),
            "nu": torch.ones(n, device=device),
            "alpha": torch.ones(n, device=device) * 2.0,
            "beta": torch.ones(n, device=device),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n, device=device).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
    }
    target_rho = torch.linspace(1.0, 20.0, n, device=device)
    ca_coords = torch.randn(n, 3, device=device)

    streak_losses = gosp_loss_v6(
        {**base, "hyp_projections_2d": streak},
        target_rho,
        ca_coords,
        disc_batch_diversity_coeff=1.25,
        disc_min_r_mean=0.05,
    )
    cloud_losses = gosp_loss_v6(
        {**base, "hyp_projections_2d": cloud},
        target_rho,
        ca_coords,
        disc_batch_diversity_coeff=1.25,
        disc_min_r_mean=0.05,
    )
    assert streak_losses["disc_batch_diversity"].item() > cloud_losses["disc_batch_diversity"].item()
