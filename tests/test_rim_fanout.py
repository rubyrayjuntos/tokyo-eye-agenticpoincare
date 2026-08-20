"""Rim fan-out losses — angular spread at high disc radius without global occupancy."""

from __future__ import annotations

import torch

from science.dtie.v6.loss import gosp_loss_v6
from science.training.rim_fanout import (
    RimFanoutSpread,
    apply_rim_fanout_spread,
    rim_angular_repulsion_loss,
    rim_pc2_floor_loss,
)


def test_rim_angular_repulsion_penalizes_collapsed_ray() -> None:
    xy = torch.tensor([[0.40, 0.0], [0.41, 0.0], [0.10, 0.05], [0.12, -0.03]])
    ca = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    collapsed = rim_angular_repulsion_loss(xy, ca, min_r_rim=0.35, min_angular_sep=0.12)
    spread = rim_angular_repulsion_loss(
        torch.tensor([[0.40, 0.0], [0.0, 0.41], [0.10, 0.05], [0.12, -0.03]]),
        ca,
        min_r_rim=0.35,
        min_angular_sep=0.12,
    )
    assert collapsed["rim_angular_repulsion"] > spread["rim_angular_repulsion"]
    assert collapsed["rim_angular_pairs"] >= torch.tensor(1.0)


def test_rim_pc2_floor_penalizes_rank1_rim_band() -> None:
    line = torch.tensor([[0.40, 0.0], [0.41, 0.01], [0.42, 0.02], [0.05, 0.0]])
    cloud = torch.tensor([[0.40, 0.0], [0.0, 0.40], [-0.38, 0.05], [0.05, 0.0]])
    line_loss = rim_pc2_floor_loss(line, min_r_rim=0.35, min_pc2_std=0.06)
    cloud_loss = rim_pc2_floor_loss(cloud, min_r_rim=0.35, min_pc2_std=0.06)
    assert line_loss["rim_pc2_floor"] > cloud_loss["rim_pc2_floor"]
    assert cloud_loss["rim_pc2_std"] >= torch.tensor(0.06)


def test_gosp_loss_v6_rim_fanout_terms() -> None:
    n = 8
    device = torch.device("cpu")
    xy = torch.randn(n, 2, device=device) * 0.15 + torch.tensor([0.38, 0.0], device=device)
    xy = xy / xy.norm(dim=-1, keepdim=True).clamp(min=1e-6) * 0.40
    ca = torch.randn(n, 3, device=device) * 15.0
    out = {
        "x_hyp": xy.clone(),
        "x_routed_hyp": xy.clone(),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n, device=device).unsqueeze(1),
        "hyp_projections_2d": xy,
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.2, device=device),
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
    target = torch.linspace(1.0, 20.0, n, device=device)
    zero = dict(
        balance_coeff=0.0,
        cone_coeff=0.0,
        neighborhood_coeff=0.0,
        angular_coeff=0.0,
        evidential_coeff=0.0,
        cone_depth_anticollapse_coeff=0.0,
        proj_violation_coeff=0.0,
    )
    with_rim = gosp_loss_v6(
        output=out,
        target_rho=target,
        ca_coords=ca,
        rim_angular_repulsion_coeff=0.25,
        rim_pc2_floor_coeff=0.15,
        **zero,
    )
    without = gosp_loss_v6(
        output=out,
        target_rho=target,
        ca_coords=ca,
        **zero,
    )
    assert with_rim["rim_angular_repulsion"] >= 0.0
    assert with_rim["rim_pc2_floor"] >= 0.0
    assert with_rim["total"] > without["total"]


def test_apply_rim_fanout_spread_increases_angular_separation() -> None:
    xy = torch.tensor([[0.40, 0.0], [0.41, 0.0], [0.10, 0.05], [0.12, -0.03]])
    ca = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    out = apply_rim_fanout_spread(xy, ca, spread_strength=0.2)
    dirs = out[:2] / out[:2].norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sep_after = (dirs[0] - dirs[1]).norm()
    sep_before = (xy[0] / xy[0].norm() - xy[1] / xy[1].norm()).norm()
    assert sep_after > sep_before
    assert torch.allclose(out[:2].norm(dim=-1), xy[:2].norm(dim=-1), atol=1e-4)


def test_rim_fanout_spread_module_is_differentiable() -> None:
    mod = RimFanoutSpread(spread_strength=0.15, learnable_strength=True)
    xy = torch.tensor([[0.40, 0.0], [0.41, 0.0], [0.05, 0.0]], requires_grad=True)
    out = mod(xy)
    out.pow(2).sum().backward()
    assert xy.grad is not None
    assert mod.log_strength.grad is not None


def test_disc_angular_coverage_penalizes_empty_wedge() -> None:
    from science.training.disc_occupancy import disc_angular_coverage_loss

    # Two occupied arcs opposite a blank sector at mid/rim radius.
    occupied = torch.tensor(
        [
            [0.35, 0.05],
            [0.36, -0.04],
            [0.34, 0.08],
            [-0.35, 0.04],
            [-0.36, -0.05],
            [-0.33, 0.06],
        ]
    )
    filled = torch.tensor(
        [
            [0.35, 0.0],
            [0.25, 0.25],
            [0.0, 0.35],
            [-0.25, 0.25],
            [-0.35, 0.0],
            [-0.25, -0.25],
            [0.0, -0.35],
            [0.25, -0.25],
        ]
    )
    wedge = disc_angular_coverage_loss(occupied, min_r=0.12, n_bins=8, min_bin_frac=0.5)
    ring = disc_angular_coverage_loss(filled, min_r=0.12, n_bins=8, min_bin_frac=0.5)
    assert wedge["disc_angular_coverage"] > ring["disc_angular_coverage"]
    assert wedge["disc_angular_coverage_empty_bins"] > ring["disc_angular_coverage_empty_bins"]

    xy = occupied.clone().requires_grad_(True)
    loss = disc_angular_coverage_loss(xy, min_r=0.12, n_bins=8, min_bin_frac=0.5)[
        "disc_angular_coverage"
    ]
    loss.backward()
    assert xy.grad is not None
    assert xy.grad.abs().sum() > 0


def test_gosp_loss_v6_angular_coverage_term() -> None:
    n = 8
    device = torch.device("cpu")
    # Crescent / wedge — mass only on +x hemisphere.
    angles = torch.linspace(-0.4, 0.4, n, device=device)
    xy = torch.stack([0.40 * torch.cos(angles), 0.40 * torch.sin(angles)], dim=-1)
    ca = torch.randn(n, 3, device=device) * 15.0
    out = {
        "x_hyp": xy.clone(),
        "x_routed_hyp": xy.clone(),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n, device=device).unsqueeze(1),
        "hyp_projections_2d": xy,
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.2, device=device),
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
    target = torch.linspace(1.0, 20.0, n, device=device)
    zero = dict(
        balance_coeff=0.0,
        cone_coeff=0.0,
        neighborhood_coeff=0.0,
        angular_coeff=0.0,
        evidential_coeff=0.0,
        cone_depth_anticollapse_coeff=0.0,
        proj_violation_coeff=0.0,
    )
    with_cov = gosp_loss_v6(
        output=out,
        target_rho=target,
        ca_coords=ca,
        disc_angular_coverage_coeff=0.45,
        **zero,
    )
    without = gosp_loss_v6(output=out, target_rho=target, ca_coords=ca, **zero)
    assert with_cov["disc_angular_coverage"] > 0.0
    assert with_cov["total"] > without["total"]
