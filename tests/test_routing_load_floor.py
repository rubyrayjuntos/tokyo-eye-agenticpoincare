"""Property tests for Stage A routing load floor (contested 1PGB fix)."""

from __future__ import annotations

import pytest
import torch

from science.dtie.v6.loss import gosp_loss_v6
from science.training.config import apply_routing_load_floor_phase2, default_v6_phases
from science.training.routing_metrics import routing_load_floor_penalty


def test_routing_load_floor_zero_when_all_experts_above_threshold() -> None:
    load = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True)
    penalty = routing_load_floor_penalty(load, min_fraction=0.05)
    assert float(penalty.item()) == pytest.approx(0.0)


def test_routing_load_floor_positive_when_min_below_threshold() -> None:
    load = torch.tensor([0.03, 0.30, 0.32, 0.35])
    penalty = routing_load_floor_penalty(load, min_fraction=0.05)
    assert float(penalty.item()) == pytest.approx(0.02**2, rel=1e-5)


def test_routing_load_floor_gradient_flows_to_soft_load() -> None:
    logits = torch.zeros(8, 4, requires_grad=True)
    logits.data[:, 0] = 3.0
    batch_soft = torch.softmax(logits, dim=-1)
    expert_frac = batch_soft.mean(dim=0)
    assert float(expert_frac.min()) < 0.05
    penalty = routing_load_floor_penalty(expert_frac, min_fraction=0.05)
    penalty.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_gosp_loss_includes_weighted_floor_when_coeff_positive() -> None:
    device = "cpu"
    n = 12
    evidence = {
        "mu": torch.rand(n, device=device),
        "nu": torch.rand(n, device=device).abs() + 0.1,
        "alpha": torch.rand(n, device=device).abs() + 1.0,
        "beta": torch.rand(n, device=device).abs() + 0.1,
    }
    # Soft load below phase floor (0.10) — loss recomputes hinge from expert_load.
    collapsed_load = torch.tensor([0.70, 0.20, 0.07, 0.03], device=device)
    output = {
        "x_hyp": torch.randn(n, 3, device=device),
        "x_routed_hyp": torch.randn(n, 3, device=device),
        "hyp_projections_2d": 0.1 * torch.randn(n, 2, device=device),
        "hyp_projections_3d": 0.1 * torch.randn(n, 3, device=device),
        "radial_features": torch.randn(n, 1, device=device),
        "cone_depth": torch.rand(n, 1, device=device),
        "evidence": evidence,
        "uncertainty": {"epistemic": torch.rand(n, device=device)},
        "capacity_loss": torch.tensor(0.0, device=device),
        "routing_load_floor": torch.tensor(0.0, device=device),  # gate raw ignored
        "routing_entropy": torch.tensor(1.3, device=device),
        "expert_load": collapsed_load,
        "audit_trail": {"curvature_value": torch.tensor(-1.0, device=device)},
    }
    target = torch.rand(n, device=device)
    coords = torch.randn(n, 3, device=device)
    zeroed = dict(
        evidential_coeff=0.0,
        balance_coeff=0.0,
        cone_coeff=0.0,
        neighborhood_coeff=0.0,
        angular_coeff=0.0,
        domain_sep_2d_coeff=0.0,
        domain_sep_3d_coeff=0.0,
        cone_depth_anticollapse_coeff=0.0,
        proj_violation_coeff=0.0,
    )

    without = gosp_loss_v6(
        output, target, coords, routing_load_floor_coeff=0.0, **zeroed,
    )
    with_floor = gosp_loss_v6(
        output,
        target,
        coords,
        routing_load_floor_coeff=10.0,
        routing_load_floor_min=0.10,
        **zeroed,
    )
    # hinge = (0.10 - 0.03)^2 = 0.0049; λ=10 → 0.049
    assert float(with_floor["routing_load_floor"]) == pytest.approx(0.049, rel=1e-5)
    assert float(with_floor["total"]) - float(without["total"]) == pytest.approx(0.049, rel=1e-5)


def test_gosp_loss_floor_fires_before_gate_min_usage() -> None:
    """Phase floor min=0.15 must bite even when gate.min_usage hinge would be 0."""
    device = "cpu"
    n = 8
    evidence = {
        "mu": torch.rand(n, device=device),
        "nu": torch.rand(n, device=device).abs() + 0.1,
        "alpha": torch.rand(n, device=device).abs() + 1.0,
        "beta": torch.rand(n, device=device).abs() + 0.1,
    }
    # All experts above gate default 0.05, but one below phase floor 0.15.
    load = torch.tensor([0.40, 0.30, 0.20, 0.10], device=device)
    output = {
        "x_hyp": torch.randn(n, 3, device=device),
        "x_routed_hyp": torch.randn(n, 3, device=device),
        "hyp_projections_2d": 0.1 * torch.randn(n, 2, device=device),
        "hyp_projections_3d": 0.1 * torch.randn(n, 3, device=device),
        "radial_features": torch.randn(n, 1, device=device),
        "cone_depth": torch.rand(n, 1, device=device),
        "evidence": evidence,
        "uncertainty": {"epistemic": torch.rand(n, device=device)},
        "capacity_loss": torch.tensor(0.0, device=device),
        "routing_load_floor": torch.tensor(0.0, device=device),
        "routing_entropy": torch.tensor(1.2, device=device),
        "expert_load": load,
        "audit_trail": {"curvature_value": torch.tensor(-1.0, device=device)},
    }
    target = torch.rand(n, device=device)
    coords = torch.randn(n, 3, device=device)
    zeroed = dict(
        evidential_coeff=0.0,
        balance_coeff=0.0,
        cone_coeff=0.0,
        neighborhood_coeff=0.0,
        angular_coeff=0.0,
        domain_sep_2d_coeff=0.0,
        domain_sep_3d_coeff=0.0,
        cone_depth_anticollapse_coeff=0.0,
        proj_violation_coeff=0.0,
    )
    out = gosp_loss_v6(
        output,
        target,
        coords,
        routing_load_floor_coeff=12.0,
        routing_load_floor_min=0.15,
        **zeroed,
    )
    # (0.15 - 0.10)^2 * 12 = 0.03
    assert float(out["routing_load_floor"]) == pytest.approx(0.03, rel=1e-5)


def test_apply_routing_load_floor_only_phase_two() -> None:
    phases = default_v6_phases()
    patched = apply_routing_load_floor_phase2(phases, coeff=10.0, min_fraction=0.05)
    assert patched[0].coeffs.routing_load_floor_coeff == 0.0
    assert patched[1].coeffs.routing_load_floor_coeff == 10.0
    assert patched[1].coeffs.routing_load_floor_min == 0.05
    assert patched[2].coeffs.routing_load_floor_coeff == 0.0


def test_floor_two_sided_healthy_loads_near_zero_penalty() -> None:
    """Contested 1PGB (~0.049) pays; healthy lever_a-like (~0.21 min) does not."""
    contested = torch.tensor([0.049, 0.30, 0.31, 0.341])
    healthy = torch.tensor([0.21, 0.27, 0.26, 0.26])
    p_contested = float(routing_load_floor_penalty(contested, 0.05).item())
    p_healthy = float(routing_load_floor_penalty(healthy, 0.05).item())
    assert p_contested > 0.0
    assert p_healthy == pytest.approx(0.0)


def test_routing_load_ceiling_positive_when_max_above_threshold() -> None:
    from science.training.routing_metrics import routing_load_ceiling_penalty

    dominant = torch.tensor([0.55, 0.15, 0.15, 0.15])
    balanced = torch.tensor([0.28, 0.24, 0.24, 0.24])
    assert float(routing_load_ceiling_penalty(dominant, 0.40).item()) == pytest.approx(
        0.15**2, rel=1e-5
    )
    assert float(routing_load_ceiling_penalty(balanced, 0.40).item()) == pytest.approx(0.0)


def test_gosp_loss_ceiling_fires_on_dominance() -> None:
    device = "cpu"
    n = 8
    evidence = {
        "mu": torch.rand(n, device=device),
        "nu": torch.rand(n, device=device).abs() + 0.1,
        "alpha": torch.rand(n, device=device).abs() + 1.0,
        "beta": torch.rand(n, device=device).abs() + 0.1,
    }
    load = torch.tensor([0.50, 0.17, 0.17, 0.16], device=device)
    output = {
        "x_hyp": torch.randn(n, 3, device=device),
        "x_routed_hyp": torch.randn(n, 3, device=device),
        "hyp_projections_2d": 0.1 * torch.randn(n, 2, device=device),
        "hyp_projections_3d": 0.1 * torch.randn(n, 3, device=device),
        "radial_features": torch.randn(n, 1, device=device),
        "cone_depth": torch.rand(n, 1, device=device),
        "evidence": evidence,
        "uncertainty": {"epistemic": torch.rand(n, device=device)},
        "capacity_loss": torch.tensor(0.0, device=device),
        "routing_load_floor": torch.tensor(0.0, device=device),
        "routing_entropy": torch.tensor(1.2, device=device),
        "expert_load": load,
        "audit_trail": {"curvature_value": torch.tensor(-1.0, device=device)},
    }
    target = torch.rand(n, device=device)
    coords = torch.randn(n, 3, device=device)
    zeroed = dict(
        evidential_coeff=0.0,
        balance_coeff=0.0,
        cone_coeff=0.0,
        neighborhood_coeff=0.0,
        angular_coeff=0.0,
        domain_sep_2d_coeff=0.0,
        domain_sep_3d_coeff=0.0,
        cone_depth_anticollapse_coeff=0.0,
        proj_violation_coeff=0.0,
        routing_load_floor_coeff=0.0,
    )
    out = gosp_loss_v6(
        output,
        target,
        coords,
        routing_load_ceiling_coeff=12.0,
        routing_load_ceiling_max=0.40,
        **zeroed,
    )
    # (0.50 - 0.40)^2 * 12 = 0.12
    assert float(out["routing_load_ceiling"]) == pytest.approx(0.12, rel=1e-5)
