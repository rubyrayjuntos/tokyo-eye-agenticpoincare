"""Sprint 3: topology-aware hard-commitment MoE (v8 isolated)."""

from __future__ import annotations

import torch

from science.tokyo_eye.v8.moe import (
    NUM_EXPERTS,
    TopologyAwareHardMoE,
    cv_load_balance_loss,
    topology_gate_features,
)


def test_num_experts_is_four() -> None:
    assert NUM_EXPERTS == 4


def test_topology_gate_features_shapes() -> None:
    n, d = 5, 8
    z = torch.randn(n, d) * 0.05
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    chem = torch.randn(n, 7)
    feat = topology_gate_features(z, edge_index, chem=chem)
    # ρ,τ,SS×3,deg,SASA + Poincaré radius + non-ball marker
    assert feat.shape == (n, 9)
    assert torch.isfinite(feat).all()


def test_train_hard_gumbel_one_hot() -> None:
    torch.manual_seed(0)
    moe = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=1.0)
    moe.train()
    z = torch.randn(7, 6) * 0.04
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 0], [1, 0, 3, 2, 2, 3]], dtype=torch.long
    )
    out, aux = moe(z, edge_index)
    assert out.shape == z.shape
    assert aux["routing"].shape == (7, 4)
    # STE hard path: rows are one-hot (within float tol)
    rows = aux["routing"]
    assert torch.allclose(rows.sum(dim=-1), torch.ones(7), atol=1e-5)
    assert ((rows - rows.round()).abs() < 1e-5).all()
    assert "cv_loss" in aux
    assert float(aux["cv_loss"].detach()) >= 0.0


def test_eval_uses_argmax() -> None:
    torch.manual_seed(1)
    moe = TopologyAwareHardMoE(dim=4, gate_hidden=4, temperature=0.5)
    moe.eval()
    z = torch.randn(4, 4) * 0.03
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    out, aux = moe(z, edge_index)
    assert out.shape == z.shape
    r = aux["routing"]
    assert torch.allclose(r.sum(dim=-1), torch.ones(4), atol=1e-5)
    # Exactly one expert per node
    assert (r.max(dim=-1).values > 0.99).all()


def test_cv_loss_penalizes_monopoly() -> None:
    # All mass on expert 0 → high CV; uniform → ~0
    monopoly = torch.zeros(10, 4)
    monopoly[:, 0] = 1.0
    uniform = torch.full((10, 4), 0.25)
    assert float(cv_load_balance_loss(monopoly)) > float(cv_load_balance_loss(uniform))


def test_empty_graph_isolates_still_route() -> None:
    moe = TopologyAwareHardMoE(dim=5, gate_hidden=3)
    moe.train()
    z = torch.randn(3, 5) * 0.02
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    out, aux = moe(z, edge_index)
    assert torch.isfinite(out).all()
    assert aux["routing"].shape == (3, 4)


def test_gyro_expert_damped_residual_preserves_radius_spread() -> None:
    """Undamped z⊕R(z) pinned every node to τ; damped expert must not."""
    from science.tokyo_eye.v8.attention import clamp_ball_radius
    from science.tokyo_eye.v8.moe import GyroExpert

    torch.manual_seed(0)
    tau = 0.7
    n, d = 24, 16
    radii = torch.linspace(0.30, 0.60, n)
    z = torch.zeros(n, d)
    z[:, 0] = radii
    expert = GyroExpert(d)
    expert.eval()
    with torch.no_grad():
        out = expert(z, c=1.0, eps=1e-5, tau_ceiling=tau)
        out = clamp_ball_radius(out, max_r=tau, eps=1e-5)
    r = torch.linalg.vector_norm(out, dim=-1)
    spread = float((r.max() - r.min()).detach())
    n_at_tau = int(((r - tau).abs() < 1e-5).sum())
    assert spread > 0.05, f"expert spread collapsed: {spread}"
    assert n_at_tau < n, f"all {n} nodes pinned to τ"


def test_eval_proxy_default_t_has_gate_gradient_under_monopoly_margin() -> None:
    """T=0.05 saturates when mean max−2nd margin≈0.8; default must not."""
    from science.tokyo_eye.v8.moe import eval_proxy_routing, switch_load_balance_loss

    # Reproduce smoke-scale logit separation: E0 ahead by ~0.8.
    logits = torch.tensor(
        [[0.68, -0.39, -0.68, -0.15]] * 24,
        dtype=torch.float32,
        requires_grad=True,
    )
    ep = eval_proxy_routing(logits)  # module default
    loss = switch_load_balance_loss(ep)
    loss.backward()
    g = float(logits.grad.norm())
    load0 = float(ep.detach().mean(0)[0])
    assert load0 > 0.85, f"proxy no longer tracks monopoly: load0={load0}"
    assert g > 1e-2, f"eval_proxy gradient starved: ||g||={g:.2e}"


def test_tau_relative_step_shrinks_above_ref() -> None:
    from science.tokyo_eye.v8.attention import tau_relative_residual_step

    logit = torch.tensor(-1.0986122886681098)  # → 0.25 at τ_ref
    t70 = float(tau_relative_residual_step(logit, tau_ceiling=0.70))
    t95 = float(tau_relative_residual_step(logit, tau_ceiling=0.95))
    assert abs(t70 - 0.25) < 1e-5
    assert t95 < t70
    assert abs(t95 - 0.25 * (0.70 / 0.95) ** 2) < 1e-5
    # Headroom: node at ceiling gets ~0 step
    r = torch.tensor([0.70, 0.35])
    t_h = tau_relative_residual_step(logit, tau_ceiling=0.70, radius=r)
    assert t_h.shape == (2, 1)
    assert float(t_h[0]) < 1e-5
    assert float(t_h[1]) > float(t_h[0])
