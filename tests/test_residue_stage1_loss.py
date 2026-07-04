"""ResidueStage1 binding-head loss wiring."""

from __future__ import annotations

import torch

from science.dtie.v6.gnn.model import GOSPConeMapperV6
from science.dtie.v6.loss import gosp_loss_v6


def test_gosp_loss_v6_binding_bce() -> None:
    model = GOSPConeMapperV6(node_dim=4, hidden=32, num_layers=2, num_experts=4)
    n = 12
    x = torch.randn(n, 4)
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    edge_attr = torch.randn(edge_index.shape[1], 4)
    clustering = torch.rand(n)
    from torch_geometric.data import Data

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        clustering=clustering,
        degree=torch.ones(n),
        ss_onehot=torch.zeros(n, 3),
        rho=x[:, 0],
    )
    out = model(data)
    target_pocket = torch.zeros(n, 1)
    target_pocket[:2] = 1.0
    losses = gosp_loss_v6(
        out,
        target_rho=x[:, 0],
        ca_coords=torch.randn(n, 3),
        target_pocket=target_pocket,
        pocket_bce_coeff=1.0,
    )
    assert float(losses["pocket_bce"]) > 0.0
    assert float(losses["total"]) > 0.0
