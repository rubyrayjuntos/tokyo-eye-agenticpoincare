"""GnnInputMode — SASA stripped from data.x, persisted on side-channel."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from science.dtie.common import residue_features as rf


def test_stack_topology_three_vector_strips_sasa() -> None:
    rho = np.array([10.0, 20.0])
    tau = np.array([1.0, 0.0])
    ss = np.array([0.0, 0.5])
    sasa = np.array([45.0, 12.0])
    x = rf.stack_gnn_node_features(
        rho, tau, ss, sasa, mode=rf.GnnInputMode.TOPOLOGY_THREE_VECTOR
    )
    assert x.shape == (2, 3)
    np.testing.assert_allclose(x[:, 0], rho)
    np.testing.assert_allclose(x[:, 2], ss)


def test_stack_legacy_four_vector_keeps_sasa() -> None:
    rho = np.array([10.0])
    tau = np.array([1.0])
    ss = np.array([0.0])
    sasa = np.array([45.0])
    x = rf.stack_gnn_node_features(
        rho, tau, ss, sasa, mode=rf.GnnInputMode.LEGACY_FOUR_VECTOR
    )
    assert x.shape == (1, 4)
    assert float(x[0, 3]) == 45.0


def test_resolve_gnn_input_mode_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNN_INPUT_MODE", "topology_three_vector")
    assert rf.resolve_gnn_input_mode() == rf.GnnInputMode.TOPOLOGY_THREE_VECTOR
    monkeypatch.delenv("GNN_INPUT_MODE")
    assert rf.resolve_gnn_input_mode() == rf.GnnInputMode.LEGACY_FOUR_VECTOR


def test_residue_sasa_from_data_side_channel() -> None:
    data = Data(x=torch.zeros(4, 3))
    data.sasa = torch.tensor([1.0, 2.0, 3.0, 4.0])
    s = rf.residue_sasa_from_data(data)
    assert s.shape == (4, 1)


def test_residue_sasa_from_data_legacy_x_fallback() -> None:
    data = Data(x=torch.tensor([[1.0, 0.0, 0.5, 99.0]]))
    s = rf.residue_sasa_from_data(data)
    assert float(s.squeeze()) == 99.0


def test_gnn_input_dim() -> None:
    assert rf.gnn_input_dim(rf.GnnInputMode.TOPOLOGY_THREE_VECTOR) == 3
    assert rf.gnn_input_dim(rf.GnnInputMode.LEGACY_FOUR_VECTOR) == 4


def test_corpus_cache_key_includes_feature_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing GNN_INPUT_MODE must not reuse the other mode's graphs_*.pt."""
    import hashlib
    from pathlib import Path

    monkeypatch.setenv("GNN_INPUT_MODE", "topology_three_vector")
    tag3 = rf.gnn_feature_set_id()
    monkeypatch.setenv("GNN_INPUT_MODE", "legacy_four_vector")
    tag4 = rf.gnn_feature_set_id()
    assert tag3 != tag4
    manifest = Path("/tmp/fake_manifest.json")
    base = f"{manifest.resolve()}|12|1200|bf_v1|rs0"
    k3 = hashlib.sha256(f"{base}|{tag3}".encode()).hexdigest()[:16]
    k4 = hashlib.sha256(f"{base}|{tag4}".encode()).hexdigest()[:16]
    assert k3 != k4


def test_resolve_training_node_dim_rejects_stale_four_vector_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch_geometric.data import Data

    from experiments.training.v6.launch_training import _resolve_training_node_dim

    monkeypatch.setenv("GNN_INPUT_MODE", "topology_three_vector")
    proteins = [{"pdb_id": "1MBN", "chain": "A", "data": Data(x=torch.zeros(5, 4))}]
    with pytest.raises(ValueError, match="node_dim=4.*expects 3"):
        _resolve_training_node_dim(proteins)


def test_resolve_training_node_dim_accepts_three_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch_geometric.data import Data

    from experiments.training.v6.launch_training import _resolve_training_node_dim

    monkeypatch.setenv("GNN_INPUT_MODE", "topology_three_vector")
    proteins = [{"pdb_id": "1MBN", "chain": "A", "data": Data(x=torch.zeros(5, 3))}]
    assert _resolve_training_node_dim(proteins) == 3
