"""ResidueStage2 leak BCE wiring."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from science.dtie.v6.gnn.model import GOSPConeMapperV6
from science.dtie.v6.loss import gosp_loss_v6


def test_leak_bce_contributes_to_total() -> None:
    model = GOSPConeMapperV6(node_dim=4, hidden=32, num_layers=2, num_experts=4)
    n = 12
    x = torch.randn(n, 4)
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    edge_attr = torch.randn(edge_index.shape[1], 4)
    clustering = torch.rand(n)
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
    target_leak = torch.zeros(n, 1)
    target_leak[:3] = 1.0
    losses = gosp_loss_v6(
        out,
        target_rho=x[:, 0],
        ca_coords=torch.randn(n, 3),
        target_leak=target_leak,
        leak_label_mask=torch.ones(n, dtype=torch.bool),
        leak_bce_coeff=1.0,
    )
    assert float(losses["leak_bce"]) > 0.0
    assert float(losses["total"]) > 0.0
