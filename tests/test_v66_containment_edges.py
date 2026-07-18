"""Path B containment: SSE parent nodes + contain_up/down directed edges."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from science.dtie.common.isolated_init import isolated_torch_seed
from science.dtie.v66.chem_edge_graph import EDGE_ATTR_CHEM_DIM
from science.dtie.v66.gnn.model import GOSPConeMapperV66
from science.dtie.v66.containment_edge_graph import (
    EDGE_ATTR_CONTAIN_DIM,
    NUM_ROLE_RELATIONS_WITH_CONTAINMENT,
    ROLE_CONTAIN_DOWN,
    ROLE_CONTAIN_UP,
    attach_containment_edge_graph,
    pad_chem_edge_attr_for_containment,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM


def _toy_chem_ready_data(n: int = 6, feat_dim: int = 3) -> Data:
    x = torch.randn(n, feat_dim)
    ei = torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(0, EDGE_ATTR_CHEM_DIM)
    data = Data(x=x, edge_index=ei, edge_attr=ea)
    data.role_edge_graph = True
    data.chem_edge_graph = True
    return data


def test_pad_chem_edge_attr_inserts_containment_onehot_slots() -> None:
    ea = torch.zeros(2, EDGE_ATTR_CHEM_DIM)
    ea[:, GEO_DIM + 3] = 1.0  # ribbon slot
    aux_col = GEO_DIM + 7
    ea[:, aux_col] = 0.42
    padded = pad_chem_edge_attr_for_containment(ea)
    assert padded.shape == (2, EDGE_ATTR_CONTAIN_DIM)
    assert torch.allclose(
        padded[:, GEO_DIM : GEO_DIM + 7],
        ea[:, GEO_DIM : GEO_DIM + 7],
    )
    assert torch.allclose(
        padded[:, GEO_DIM + 7 : GEO_DIM + 9],
        torch.zeros(2, 2),
    )
    assert torch.allclose(padded[:, GEO_DIM + 9], ea[:, aux_col])


def test_attach_appends_parent_and_directed_relations() -> None:
    data = _toy_chem_ready_data()
    ca = torch.randn(6, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    pdb = (
        "HELIX    1   1 ALA A   10  ALA A   12  1                                   3\n"
        "END\n"
    )
    out = attach_containment_edge_graph(
        data, ca, pdb, residue_ids=residue_ids, force=True
    )
    assert out.n_residue_nodes == 6
    assert out.n_parent_nodes == 1
    assert out.containment_edge_graph is True
    assert out.x.shape[0] == 7
    assert torch.allclose(out.x[6], data.x[:3].mean(0), atol=1e-5)
    oh = out.edge_attr[:, GEO_DIM:]
    assert (oh[:, ROLE_CONTAIN_DOWN] > 0).any()
    assert (oh[:, ROLE_CONTAIN_UP] > 0).any()
    assert out.containment_edge_counts["contain_down"] == 3
    assert out.containment_edge_counts["contain_up"] == 3


def test_empty_sse_pads_attr_but_does_not_grow_n() -> None:
    data = _toy_chem_ready_data()
    ca = torch.randn(6, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    out = attach_containment_edge_graph(
        data, ca, "END\n", residue_ids=residue_ids, force=True
    )
    assert out.x.shape[0] == 6
    assert out.n_parent_nodes == 0
    assert out.edge_attr.size(-1) == EDGE_ATTR_CONTAIN_DIM
    assert out.containment_edge_graph is True


def test_containment_radial_mlp_count_is_nine() -> None:
    with isolated_torch_seed(123):
        m = GOSPConeMapperV66(
            node_dim=3,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=True,
            init_seed=123,
        )
    assert len(m.convs[0].radial_mlps) == 9


def test_containment_edge_mp_requires_chem_edge_mp() -> None:
    with pytest.raises(ValueError, match="containment_edge_mp requires chem_edge_mp"):
        GOSPConeMapperV66(
            node_dim=3,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=False,
            containment_edge_mp=True,
        )


def test_containment_radial_mlp_expansion_preserves_gate_init_under_isolated_seed() -> None:
    """New contain MLP slots must not scramble gate/prototype under init_seed discipline.

    Hard prerequisite before any cold containment run — same as Chem-MVP.
    Required: max |Δ| = 0 on gate/prototype tensors (torch.equal).
    """

    def _gate_proto_snapshot(model: GOSPConeMapperV66) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if name.startswith("gate.") or "prototype" in name.lower():
                out[name] = tensor.detach().clone()
        return out

    with isolated_torch_seed(123):
        baseline = GOSPConeMapperV66(
            node_dim=4,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=False,
            init_seed=123,
        )
        base_snap = _gate_proto_snapshot(baseline)

    with isolated_torch_seed(123):
        contain = GOSPConeMapperV66(
            node_dim=4,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=True,
            init_seed=123,
        )
        contain_snap = _gate_proto_snapshot(contain)

    assert base_snap.keys() == contain_snap.keys()
    for key in base_snap:
        assert torch.equal(base_snap[key], contain_snap[key]), (
            f"{key} diverged containment-off vs on "
            f"(max |Δ|={(base_snap[key] - contain_snap[key]).abs().max().item()})"
        )
    assert len(contain.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CONTAINMENT
    assert len(baseline.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CONTAINMENT - 2


def test_empty_containment_forward_no_nan() -> None:
    n = 6
    data = _toy_chem_ready_data(n=n, feat_dim=4)
    ca = torch.randn(n, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 10 + n)]
    data = attach_containment_edge_graph(
        data, ca, "END\n", residue_ids=residue_ids, force=True
    )
    assert data.n_parent_nodes == 0
    assert data.edge_attr.size(-1) == EDGE_ATTR_CONTAIN_DIM

    data.clustering = torch.rand(n)
    data.degree = torch.ones(n)
    data.ss_onehot = torch.zeros(n, 3)
    data.rho = data.x[:, 0].clone()

    model = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=1,
        num_experts=2,
        role_edge_mp=True,
        chem_edge_mp=True,
        containment_edge_mp=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(data)

    for key in ("x_hyp", "cone_depth", "hyp_projections_2d"):
        assert torch.isfinite(out[key]).all(), key
    for ukey, utensor in out["uncertainty"].items():
        assert torch.isfinite(utensor).all(), f"uncertainty.{ukey}"
