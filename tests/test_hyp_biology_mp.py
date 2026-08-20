"""Unit tests for hyp_biology_mp audit + fail-closed resolve + biology extract."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import AtomRecord, ResidueRecord
from science.tokyo_eye.biology_graph import (
    BIO_SALT_BRIDGE,
    attach_biology_mp_graph,
    compute_biology_mp_graph,
    detect_pi_stacks,
    detect_salt_bridges,
)
from science.tokyo_eye.biology_mp_audit import (
    assert_biology_mp_ontology,
    build_biology_mp_audit,
    edge_type_counts,
)
from science.tokyo_eye.hyperbolic_mp import resolve_hyp_mp_edges


def test_resolve_hyp_mp_edges_ca_fallback_default() -> None:
    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    ei = resolve_hyp_mp_edges(data)
    assert torch.equal(ei, data.edge_index)


def test_resolve_hyp_mp_edges_biology_fail_closed() -> None:
    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    data.hyp_biology_mp = True
    data.allow_ca_fallback = False
    with pytest.raises(ValueError, match="Cα edge_index fallback forbidden"):
        resolve_hyp_mp_edges(data)


def test_resolve_hyp_mp_edges_biology_uses_hyperbolic() -> None:
    hei = torch.tensor([[0], [1]], dtype=torch.long)
    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    data.hyperbolic_graph = True
    data.hyperbolic_edge_index = hei
    data.hyp_biology_mp = True
    data.allow_ca_fallback = False
    ei = resolve_hyp_mp_edges(data)
    assert torch.equal(ei, hei)


def test_biology_mp_audit_rejects_ca() -> None:
    with pytest.raises(AssertionError, match="ca_in_mp"):
        build_biology_mp_audit(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_types=torch.zeros((0,), dtype=torch.long),
            num_nodes=3,
            ca_in_mp=True,
            allow_ca_fallback=False,
        )


def test_biology_mp_audit_degree_zero() -> None:
    audit = build_biology_mp_audit(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_types=torch.zeros((0,), dtype=torch.long),
        num_nodes=4,
        ca_in_mp=False,
        allow_ca_fallback=False,
    )
    assert audit["degree_zero_count"] == 4
    assert audit["n_biology_edges_undirected"] == 0
    assert_biology_mp_ontology(None, ca_in_mp=False)


def test_edge_type_counts_directed() -> None:
    types = torch.tensor([0, 0, 1, 1, 3, 3], dtype=torch.long)
    counts = edge_type_counts(types, directed=True)
    assert counts["hbond"] == 1
    assert counts["dehydron"] == 1
    assert counts["salt_bridge"] == 1
    assert counts["pi_stack"] == 0


def _atom(name: str, xyz: tuple[float, float, float], res: str) -> AtomRecord:
    return AtomRecord(
        atom_name=name,
        element=name[0],
        coord=np.array(xyz, dtype=np.float64),
        parent_residue_name=res,
    )


def test_detect_salt_bridge_arg_asp() -> None:
    arg = ResidueRecord(
        chain_label="A",
        residue_index=10,
        residue_name="ARG",
        atoms=(
            _atom("CA", (0.0, 0.0, 0.0), "ARG"),
            _atom("NH1", (1.0, 0.0, 0.0), "ARG"),
            _atom("NH2", (1.2, 0.5, 0.0), "ARG"),
        ),
    )
    asp = ResidueRecord(
        chain_label="A",
        residue_index=20,
        residue_name="ASP",
        atoms=(
            _atom("CA", (4.0, 0.0, 0.0), "ASP"),
            _atom("OD1", (2.5, 0.0, 0.0), "ASP"),
            _atom("OD2", (2.7, 0.3, 0.0), "ASP"),
        ),
    )
    pairs = detect_salt_bridges(
        [arg, asp], index_by_auth={10: 0, 20: 1}, max_dist=4.0
    )
    assert len(pairs) == 1
    i, j, cp, d = pairs[0]
    assert {i, j} == {0, 1}
    assert cp == -1.0
    assert d <= 4.0


def test_detect_pi_stack_face() -> None:
    def phe(idx: int, z: float) -> ResidueRecord:
        ring = [
            ("CG", (0.0, 0.0, z)),
            ("CD1", (1.4, 0.0, z)),
            ("CD2", (-1.4, 0.0, z)),
            ("CE1", (1.4, 1.4, z)),
            ("CE2", (-1.4, 1.4, z)),
            ("CZ", (0.0, 2.0, z)),
        ]
        return ResidueRecord(
            chain_label="A",
            residue_index=idx,
            residue_name="PHE",
            atoms=tuple(_atom(n, xyz, "PHE") for n, xyz in ring),
        )

    pairs = detect_pi_stacks(
        [phe(1, 0.0), phe(2, 5.0)],
        index_by_auth={1: 0, 2: 1},
    )
    assert len(pairs) == 1
    assert pairs[0][4] == 0  # face-to-face


def test_attach_biology_mp_sets_fail_closed_flags() -> None:
    coords = np.zeros((3, 3), dtype=np.float64)
    coords[1, 0] = 5.0
    coords[2, 0] = 10.0
    rho = np.array([20.0, 20.0, 5.0], dtype=np.float64)
    data = Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    attach_biology_mp_graph(data, coords=coords, rho=rho, residue_records=[])
    assert data.hyp_biology_mp is True
    assert data.allow_ca_fallback is False
    assert data.hyperbolic_graph is True
    ei = resolve_hyp_mp_edges(data)
    assert ei.shape[0] == 2
    assert data.biology_mp_audit["ca_in_mp"] is False


def test_compute_graph_includes_salt_type() -> None:
    arg = ResidueRecord(
        chain_label="A",
        residue_index=1,
        residue_name="ARG",
        atoms=(
            _atom("CA", (0.0, 0.0, 0.0), "ARG"),
            _atom("NH1", (0.0, 0.0, 0.0), "ARG"),
        ),
    )
    asp = ResidueRecord(
        chain_label="A",
        residue_index=2,
        residue_name="ASP",
        atoms=(
            _atom("CA", (3.0, 0.0, 0.0), "ASP"),
            _atom("OD1", (3.0, 0.0, 0.0), "ASP"),
        ),
    )
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    rho = np.array([20.0, 20.0])
    ei, _ea, et, meta = compute_biology_mp_graph(
        coords,
        rho,
        residue_ids=["A:1:", "A:2:"],
        residue_records=[arg, asp],
    )
    assert meta["salt_bridge"] >= 1
    assert BIO_SALT_BRIDGE in set(et.tolist())
    assert ei.shape[1] >= 2
