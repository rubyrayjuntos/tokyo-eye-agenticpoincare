"""Two-sided property tests for subsystem gradient telemetry."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.v6.gnn.model import GOSPConeMapperV6
from science.dtie.v6.loss import gosp_loss_v6
from science.training.grad_probe import collect_subsystem_grad_norms, grad_probe_is_live


def _tiny_graph(n: int = 12) -> Data:
    x = torch.randn(n, 4)
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.shape[1], 4),
        clustering=torch.rand(n),
        degree=torch.ones(n),
        ss_onehot=torch.zeros(n, 3),
        rho=x[:, 0],
    )


def _backward_step(model: GOSPConeMapperV6, data: Data) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    out = model(data)
    losses = gosp_loss_v6(
        out,
        target_rho=data.x[:, 0],
        ca_coords=torch.randn(data.num_nodes, 3),
        sasa=data.x[:, 3] if data.x.shape[1] > 3 else torch.rand(data.num_nodes),
        angular_coeff=0.2,
        neighborhood_coeff=0.1,
    )
    losses["total"].backward()
    return collect_subsystem_grad_norms(model)


def test_grad_probe_live_when_subsystems_unfrozen() -> None:
    """Positive: unfrozen heads receive non-zero gradients after backward."""
    model = GOSPConeMapperV6(
        node_dim=4,
        hidden=32,
        num_layers=2,
        num_experts=4,
        hyperbolic_gate=False,
    )
    model.train()
    metrics = _backward_step(model, _tiny_graph())

    assert grad_probe_is_live(metrics, subsystem="radial")
    assert grad_probe_is_live(metrics, subsystem="angular")
    assert grad_probe_is_live(metrics, subsystem="backbone")
    for key in ("grad_radial", "grad_angular", "grad_backbone"):
        assert torch.isfinite(torch.tensor(metrics[key]))
        assert metrics[key] > 0.0


def test_grad_probe_zero_when_angular_frozen() -> None:
    """Negative: frozen angular head reads zero — distinguishes real-zero from broken NaN."""
    model = GOSPConeMapperV6(
        node_dim=4,
        hidden=32,
        num_layers=2,
        num_experts=4,
        hyperbolic_gate=False,
    )
    for p in model.angular_head.parameters():
        p.requires_grad = False
    model.train()
    metrics = _backward_step(model, _tiny_graph())

    assert metrics["grad_angular"] == 0.0
    assert not grad_probe_is_live(metrics, subsystem="angular")
    assert grad_probe_is_live(metrics, subsystem="radial")
    assert grad_probe_is_live(metrics, subsystem="backbone")


def test_grad_probe_ignores_nonfinite_param_grads() -> None:
    """One poisoned parameter must not NaN the subsystem mean."""
    model = GOSPConeMapperV6(node_dim=4, hidden=16, num_layers=1, num_experts=2, hyperbolic_gate=False)
    model.train()
    metrics = _backward_step(model, _tiny_graph())
    for p in model.radial_head.parameters():
        if p.grad is not None:
            p.grad.fill_(float("nan"))
            break
    # Re-collect after poisoning one tensor — other radial params still finite.
    from science.training.grad_probe import collect_subsystem_grad_norms

    poisoned = collect_subsystem_grad_norms(model)
    assert np.isfinite(poisoned["grad_radial"])
    assert poisoned["grad_radial"] > 0.0
    assert metrics["grad_angular"] >= 0.0


def test_grad_probe_not_nan_after_backward() -> None:
    """Regression: post-step collection used to yield NaN; pre-step must be finite."""
    model = GOSPConeMapperV6(node_dim=4, hidden=16, num_layers=1, num_experts=2, hyperbolic_gate=False)
    model.train()
    metrics = _backward_step(model, _tiny_graph())
    assert all(torch.isfinite(torch.tensor(metrics[k])) for k in metrics)
