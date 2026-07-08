"""Hyperbolic graph wiring — explicit fields, curvature provenance, v6 MP path."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from science.dtie.common.curvature_loader import CANONICAL_V6_CURVATURE
from science.dtie.common.graph_builder import ProteinGraph, ResidueFeatures
from science.dtie.common.hyperbolic_disc_graph import cone_depth_from_disc
from science.dtie.common.structural_disc_compose import (
    attach_structural_disc_to_pyg,
    compose_from_protein_graph,
)
from science.dtie.v6.gnn.model import resolve_message_passing_edges

C = CANONICAL_V6_CURVATURE


def _mini_graph() -> ProteinGraph:
    return ProteinGraph(
        structure_id="1crn",
        residues=[
            ResidueFeatures(
                residue_id=f"A:{i}:",
                residue_index=i,
                chain_label="A",
                rho=12.0 + i,
                tau_flag=0.0 if i > 3 else 1.0,
                ss_type=0.5,
                sasa=0.3,
                ca_x=float(i),
                ca_y=0.0,
                ca_z=0.0,
            )
            for i in range(1, 9)
        ],
        edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        edge_attr=np.ones((3, 4), dtype=np.float32),
        chain_ids=["A"] * 8,
        residue_indices=list(range(1, 9)),
        residue_ids=[f"A:{i}:" for i in range(1, 9)],
    )


def test_hyperbolic_edges_are_explicit_c_alpha_preserved() -> None:
    graph = _mini_graph()
    ca_ei = graph.edge_index.copy()
    art = compose_from_protein_graph(graph, C)
    data = Data(x=torch.zeros(8, 4), edge_index=torch.tensor(ca_ei, dtype=torch.long))
    attach_structural_disc_to_pyg(data, art, residue_ids=graph.residue_ids)
    assert data.hyperbolic_graph is True
    assert data.hyperbolic_edge_index.shape[1] > ca_ei.shape[1]
    assert torch.equal(data.edge_index, torch.tensor(ca_ei, dtype=torch.long))


def test_v6_resolve_message_passing_edges_prefers_hyperbolic() -> None:
    graph = _mini_graph()
    art = compose_from_protein_graph(graph, C)
    data = Data(x=torch.zeros(8, 4), edge_index=torch.zeros(2, 0, dtype=torch.long))
    attach_structural_disc_to_pyg(data, art, residue_ids=graph.residue_ids)
    mp_ei, mp_ea = resolve_message_passing_edges(data)
    assert mp_ei.shape == data.hyperbolic_edge_index.shape
    assert mp_ea.shape == data.hyperbolic_edge_attr.shape


def test_compose_stores_passed_curvature_pin() -> None:
    graph = _mini_graph()
    art = compose_from_protein_graph(graph, C)
    assert art.curvature_c == pytest.approx(C)
    depth = cone_depth_from_disc(art.z_disc_matrix, art.curvature_c)
    assert depth.shape[0] == len(graph.residues)
    assert np.all(depth >= 0.0)


def test_compose_rejects_missing_curvature() -> None:
    graph = _mini_graph()
    with pytest.raises(Exception):
        compose_from_protein_graph(graph, 0.0)
