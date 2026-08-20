"""Focused tests for barcode-aware v6 training launch wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from experiments.training.v6.launch_training import build_model, _node_dim_from_loaded_graphs
from science.training.config import TrainingConfig


def test_build_model_uses_provided_node_dim() -> None:
    model = build_model(TrainingConfig(use_dehydron_barcode=True), node_dim=17)

    assert model.node_emb.in_features == 17


def test_barcode_cold_models_isolate_gate_init_across_node_width() -> None:
    baseline = build_model(TrainingConfig(init_seed=17), node_dim=3)
    barcode = build_model(
        TrainingConfig(use_dehydron_barcode=True, init_seed=17),
        node_dim=7,
    )

    assert torch.equal(
        baseline.gate.prototype_bank.prototype_tangent,
        barcode.gate.prototype_bank.prototype_tangent,
    )
    assert torch.equal(
        baseline.gate.topo_encoder[0].weight,
        barcode.gate.topo_encoder[0].weight,
    )


def test_v66_lineage_isolates_gate_init_across_node_width() -> None:
    """Active feeler model class must carry the same isolation as the gate unit tests."""
    baseline = build_model(
        TrainingConfig(gnn_lineage="v6.6", init_seed=17),
        node_dim=3,
    )
    barcode = build_model(
        TrainingConfig(gnn_lineage="v6.6", use_dehydron_barcode=True, init_seed=17),
        node_dim=7,
    )
    assert baseline.__class__.__name__ == "GOSPConeMapperV66"
    assert torch.equal(
        baseline.gate.prototype_bank.prototype_tangent,
        barcode.gate.prototype_bank.prototype_tangent,
    )


def test_matched_cold_make_targets_always_pass_same_init_seed() -> None:
    text = Path("Makefile").read_text()
    baseline_target = text.split("train-v65-master-cold:", 1)[1].split("\n\n", 1)[0]
    barcode_target = text.split("train-v65-dbh-scalars-cold:", 1)[1].split(
        "\n\n", 1
    )[0]
    v66_scalars = text.split("train-v66-feeler-dbh-scalars:", 1)[1].split(
        "\n\n", 1
    )[0]

    expected = "--seed $(or $(SEED),1)"
    assert expected in baseline_target
    assert expected in barcode_target
    assert expected in v66_scalars


def test_node_dim_from_loaded_graphs_rejects_mismatch() -> None:
    proteins = [
        {"pdb_id": "4OBE", "chain": "A", "data": Data(x=torch.zeros(2, 15))},
        {"pdb_id": "1IVO", "chain": "A", "data": Data(x=torch.zeros(2, 55))},
    ]

    with pytest.raises(ValueError, match="node feature dim mismatch"):
        _node_dim_from_loaded_graphs(proteins)


def test_resolve_training_node_dim_accepts_barcode_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.training.v6.launch_training import _resolve_training_node_dim

    monkeypatch.setenv("GNN_INPUT_MODE", "topology_three_vector")
    proteins = [{"pdb_id": "4OBE", "chain": "A", "data": Data(x=torch.zeros(5, 7))}]
    assert (
        _resolve_training_node_dim(
            proteins,
            use_dehydron_barcode=True,
            use_binned_dehydron=False,
        )
        == 7
    )


def test_resolve_training_node_dim_rejects_barcode_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.training.v6.launch_training import _resolve_training_node_dim

    monkeypatch.setenv("GNN_INPUT_MODE", "topology_three_vector")
    proteins = [{"pdb_id": "4OBE", "chain": "A", "data": Data(x=torch.zeros(5, 7))}]
    with pytest.raises(ValueError, match="node_dim=7.*expects 3"):
        _resolve_training_node_dim(proteins)
