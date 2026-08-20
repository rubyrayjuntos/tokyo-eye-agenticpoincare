"""Unit tests for Top-K euclidean_shortcut construction (v66)."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.v66.chem_edge_graph import EDGE_ATTR_CHEM_DIM, attach_chem_edge_graph
from science.dtie.v66.euclidean_shortcut_graph import (
    EDGE_ATTR_EUC_DIM,
    EUC_SPATIAL_CUTOFF_A,
    EUC_TOP_K,
    attach_euclidean_shortcut_graph,
    compute_euclidean_shortcut_pairs,
    pad_chem_edge_attr_for_euc_shortcut,
)
from science.dtie.v66.role_edge_graph import attach_role_edge_graph


def _chem_data(coords: np.ndarray, rids: list[str]) -> Data:
    n = coords.shape[0]
    rho = torch.full((n,), 15.0)
    tau = (rho < 12.0).float()
    x = torch.stack([rho, tau, torch.zeros(n)], dim=-1)
    data = Data(
        x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 4),
    )
    data.rho = rho
    data = attach_role_edge_graph(data, coords, residue_ids=rids)
    data = attach_chem_edge_graph(
        data,
        coords,
        [],
        residue_ids=rids,
        structure_id="toy",
        chain_label="A",
    )
    return data


def test_pad_chem_edge_attr_inserts_one_col() -> None:
    ea = torch.zeros(3, EDGE_ATTR_CHEM_DIM)
    padded = pad_chem_edge_attr_for_euc_shortcut(ea)
    assert padded.shape[-1] == EDGE_ATTR_EUC_DIM


def test_shortcut_requires_chem_first() -> None:
    coords = np.zeros((12, 3))
    coords[:, 0] = np.arange(12) * 3.8
    rids = [f"A:{i + 1}:" for i in range(12)]
    data = Data(
        x=torch.zeros(12, 3),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 4),
    )
    import pytest

    with pytest.raises(ValueError, match="chem_edge_graph"):
        attach_euclidean_shortcut_graph(data, coords, residue_ids=rids)


def test_topk_bound_and_distant_contact() -> None:
    """With empty baseline (all hops=inf), Top-K keeps ≤2N nearest within 20Å."""
    n = 30
    coords = np.random.default_rng(0).normal(size=(n, 3)) * 5.0
    coords[0] = np.array([0.0, 0.0, 0.0])
    coords[1] = np.array([5.0, 0.0, 0.0])
    coords[2] = np.array([0.0, 5.0, 0.0])
    rids = [f"A:{i + 1}:" for i in range(n)]
    empty = torch.zeros(2, 0, dtype=torch.long)
    src, dst, stats = compute_euclidean_shortcut_pairs(
        coords, empty, residue_ids=rids
    )
    assert int(stats["n_shortcut_undirected"]) == len(src)
    assert int(stats["n_shortcut_undirected"]) > 0
    assert int(stats["n_shortcut_undirected"]) <= n * EUC_TOP_K
    assert EUC_SPATIAL_CUTOFF_A == 20.0
    assert EUC_TOP_K == 2


def test_cross_chain_shortcuts_allowed() -> None:
    """Different chains must not be blocked by seq mask (IGPS interface)."""
    n_a, n_b = 12, 12
    n = n_a + n_b
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:n_a, 0] = np.arange(n_a) * 3.8
    coords[n_a:, 0] = np.arange(n_b) * 3.8
    coords[n_a:, 1] = 8.0  # parallel chain ~8Å away
    rids = [f"A:{i + 1}:" for i in range(n_a)] + [f"B:{i + 1}:" for i in range(n_b)]
    empty = torch.zeros(2, 0, dtype=torch.long)
    src, dst, stats = compute_euclidean_shortcut_pairs(
        coords, empty, residue_ids=rids
    )
    assert int(stats["n_shortcut_undirected"]) > 0
    cross = False
    for i, j in zip(src.tolist(), dst.tolist()):
        if rids[i][0] != rids[j][0]:
            cross = True
            break
    assert cross, "expected at least one A–B shortcut under Top-K"


def test_attach_pads_and_increments() -> None:
    n = 24
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    coords[0, 2] = 0.0
    coords[-1, 2] = 10.0
    rids = [f"A:{i + 1}:" for i in range(n)]
    chem = _chem_data(coords, rids)
    e0 = chem.edge_index.size(1)
    euc = attach_euclidean_shortcut_graph(chem, coords, residue_ids=rids)
    assert euc.edge_attr.size(-1) == EDGE_ATTR_EUC_DIM
    assert euc.edge_index.size(1) >= e0
