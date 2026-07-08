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
