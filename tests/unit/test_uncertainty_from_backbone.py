"""Tests for backbone vs routed uncertainty input mode."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from experiments.training.v6.train_loop import set_gate_only_freeze, set_uncertainty_from_backbone
from science.dtie.v5.gnn.model import precompute_clustering
from science.dtie.v6.gnn.model import GOSPConeMapperV6


def test_uncertainty_from_backbone_changes_forward_input() -> None:
    model = GOSPConeMapperV6(
        hidden=32,
        num_layers=2,
        num_experts=2,
        hyperbolic_gate=False,
        topology_only_gate=True,
        decoupled_uncertainty_heads=True,
        uncertainty_from_backbone=False,
    )
    n = 12
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    data = Data(
        x=torch.randn(n, 4),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), 4).abs(),
    )
    data.clustering = torch.rand(n)

    batch = precompute_clustering(data)
    model.eval()
    with torch.no_grad():
        model.uncertainty_from_backbone = False
        routed = model(batch)
        model.uncertainty_from_backbone = True
        backbone = model(batch)
    assert routed["audit_trail"]["uncertainty_input"] == "routed"
    assert backbone["audit_trail"]["uncertainty_input"] == "backbone"


def test_gate_only_freeze_enables_backbone_uncertainty_mode() -> None:
    model = GOSPConeMapperV6(hidden=16, num_layers=2, num_experts=2, hyperbolic_gate=False)
    set_gate_only_freeze(model)
    set_uncertainty_from_backbone(model, enabled=True)
    assert model.uncertainty_from_backbone is True
    assert all(not p.requires_grad for n, p in model.named_parameters() if "uncertainty_head" in n)
    assert any(p.requires_grad for n, p in model.named_parameters() if n.startswith("gate."))
