"""GNN forward with frozen structural disc SSOT (Step 1 wiring)."""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from science.dtie.v6.gnn.hyperbolic_moe import project_disc_2d
from science.dtie.v6.gnn.model import GOSPConeMapperV6


def _tiny_graph(n: int = 16) -> Data:
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


def _ring_disc(n: int) -> torch.Tensor:
    angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r = torch.linspace(0.15, 0.65, n)
    return torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1)


def test_forward_emits_structural_disc_not_learned_layout() -> None:
    """Frozen structural_z_disc drives hyp_projections_2d; audit marks SSOT source."""
    model = GOSPConeMapperV6(
        node_dim=4,
        hidden=32,
        num_layers=2,
        num_experts=4,
        hyperbolic_gate=False,
    )
    data_learned = _tiny_graph()
    data_frozen = _tiny_graph()
    structural = _ring_disc(data_frozen.num_nodes)
    data_frozen.structural_z_disc = structural
    data_frozen.structural_z_disc_frozen = True

    with torch.no_grad():
        learned_out = model(data_learned)
        frozen_out = model(data_frozen)

    k = -model.curvature
    expected, _ = project_disc_2d(
        data_frozen.structural_z_disc[:, :2],
        k=k,
        softness=model.disc_proj_softness,
    )
    assert torch.allclose(frozen_out["hyp_projections_2d"], expected, atol=1e-4)
    assert learned_out["audit_trail"]["disc_projection_source"] != "structural_ssot_frozen"
    assert frozen_out["audit_trail"]["disc_projection_source"] == "structural_ssot_frozen"
    assert frozen_out["audit_trail"]["structural_disc_frozen"] is True
    assert not torch.allclose(
        learned_out["hyp_projections_2d"],
        frozen_out["hyp_projections_2d"],
        atol=0.05,
    )


def test_forward_moe_and_uncertainty_still_run_when_disc_frozen() -> None:
    """MoE routing + uncertainty heads remain active under frozen geometry."""
    model = GOSPConeMapperV6(
        node_dim=4,
        hidden=32,
        num_layers=1,
        num_experts=3,
        hyperbolic_gate=False,
    )
    data = _tiny_graph(12)
    data.structural_z_disc = _ring_disc(data.num_nodes)
    data.structural_z_disc_frozen = True

    out = model(data)
    assert out["expert_weights"].shape == (data.num_nodes, 3)
    assert out["x_routed_hyp"].shape == (data.num_nodes, model.hidden)
    assert "capacity_loss" in out
    assert out["uncertainty"] is not None
