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
    h_proj = torch.randn(n, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    feat = topology_gate_features(z, h_proj, edge_index)
    # r, density, deg + projected h (4)
    assert feat.shape == (n, 3 + 4)
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
