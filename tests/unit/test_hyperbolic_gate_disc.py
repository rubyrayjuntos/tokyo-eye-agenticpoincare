"""Hyperbolic gate with pre-routing disc-position inputs."""

from __future__ import annotations

import torch

from science.dtie.v6.gnn.hyperbolic_moe import HyperbolicPrototypeGate


def test_hyperbolic_gate_disc_topo_dim() -> None:
    gate = HyperbolicPrototypeGate(hidden_dim=32, num_experts=4, use_disc_position=True)
    assert gate.topo_encoder[0].in_features == HyperbolicPrototypeGate.TOPO_DIM + HyperbolicPrototypeGate.DISC_DIM

    gate_no_disc = HyperbolicPrototypeGate(hidden_dim=32, num_experts=4, use_disc_position=False)
    assert gate_no_disc.topo_encoder[0].in_features == HyperbolicPrototypeGate.TOPO_DIM


def test_hyperbolic_gate_forward_with_disc() -> None:
    gate = HyperbolicPrototypeGate(hidden_dim=32, num_experts=4, use_disc_position=True)
    gate.train(mode=False)
    n = 16
    k = torch.tensor(-1.0)
    x_hyp = torch.randn(n, 32) * 0.01
    scores, cap_loss, audit = gate(
        x_hyp=x_hyp,
        k=k,
        clustering=torch.rand(n),
        cone_depth=torch.rand(n, 1),
        degree=torch.randint(1, 8, (n,)).float(),
        rho=torch.rand(n),
        ss_onehot=torch.zeros(n, 3).scatter_(1, torch.randint(0, 3, (n, 1)), 1.0),
    )
    assert scores.shape == (n, 4)
    assert torch.isfinite(scores).all()
    assert audit["gate_disc_input"] is True
    assert "gate_disc_r_mean" in audit
    assert cap_loss.ndim == 0


def test_hyperbolic_gate_forward_with_external_disc() -> None:
    gate = HyperbolicPrototypeGate(hidden_dim=32, num_experts=4, use_disc_position=True)
    gate.train(mode=False)
    n = 16
    k = torch.tensor(-1.0)
    x_hyp = torch.randn(n, 32) * 0.01
    disc_xy = torch.randn(n, 2) * 0.1
    disc_r = disc_xy.norm(dim=-1, keepdim=True)
    scores, cap_loss, audit = gate(
        x_hyp=x_hyp,
        k=k,
        clustering=torch.rand(n),
        cone_depth=torch.rand(n, 1),
        degree=torch.randint(1, 8, (n,)).float(),
        rho=torch.rand(n),
        ss_onehot=torch.zeros(n, 3).scatter_(1, torch.randint(0, 3, (n, 1)), 1.0),
        disc_xy=disc_xy,
        disc_r=disc_r,
    )
    assert scores.shape == (n, 4)
    assert audit.get("gate_disc_external") is True
