"""Focused tests for barcode-aware v6 training launch wiring."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from experiments.training.v6.launch_training import build_model, _node_dim_from_loaded_graphs
from science.training.config import TrainingConfig


def test_build_model_uses_provided_node_dim() -> None:
    model = build_model(TrainingConfig(use_dehydron_barcode=True), node_dim=17)

    assert model.node_emb.in_features == 17


def test_node_dim_from_loaded_graphs_rejects_mismatch() -> None:
    proteins = [
        {"pdb_id": "4OBE", "chain": "A", "data": Data(x=torch.zeros(2, 15))},
        {"pdb_id": "1IVO", "chain": "A", "data": Data(x=torch.zeros(2, 55))},
    ]

    with pytest.raises(ValueError, match="node feature dim mismatch"):
        _node_dim_from_loaded_graphs(proteins)
