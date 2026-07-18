"""Chem-MVP: fact_covalent_bond → role multi-rel edge rows (train-side only)."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.isolated_init import isolated_torch_seed
from science.dtie.v66.chem_edge_graph import (
    CHEM_BOND_TYPES,
    EDGE_ATTR_CHEM_DIM,
    NUM_ROLE_RELATIONS_WITH_CHEM,
    ROLE_COVALE,
    ROLE_DISULF,
    attach_chem_edge_graph,
    map_bond_endpoints_to_nodes,
    pad_role_edge_attr_for_chem,
)
from science.dtie.v66.gnn.equivariant_conv_multirel import EquivariantConvMultiRel
from science.dtie.v66.gnn.model import GOSPConeMapperV66
from science.dtie.v66.role_edge_graph import (
    EDGE_ATTR_ROLE_DIM,
    NUM_ROLE_RELATIONS,
    ROLE_PACKING,
    attach_role_edge_graph,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM


def _toy_role_graph(n: int = 8) -> tuple[Data, np.ndarray, list[str]]:
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n) * 3.8
    x = torch.zeros(n, 4)
    x[:, 0] = torch.linspace(20, 8, n)
    x[:, 1] = (x[:, 0] < 13).float()
    data = Data(
        x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 4),
    )
    data.rho = x[:, 0].clone()
    residue_ids = [f"A:{i + 1}:" for i in range(n)]
    data = attach_role_edge_graph(data, coords, residue_ids=residue_ids)
    return data, coords, residue_ids


def test_pad_role_edge_attr_inserts_chem_onehot_slots() -> None:
    ea = torch.zeros(3, EDGE_ATTR_ROLE_DIM)
    ea[:, GEO_DIM + ROLE_PACKING] = 1.0
    ea[:, GEO_DIM + NUM_ROLE_RELATIONS] = 0.7  # spoke_rho / aux slot
    padded = pad_role_edge_attr_for_chem(ea)
    assert padded.shape == (3, EDGE_ATTR_CHEM_DIM)
    assert torch.allclose(
        padded[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS],
        ea[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS],
    )
    assert torch.allclose(
        padded[:, GEO_DIM + NUM_ROLE_RELATIONS : GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM],
        torch.zeros(3, 2),
    )
    # Aux columns shifted by +2
    assert torch.allclose(
        padded[:, GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM],
        ea[:, GEO_DIM + NUM_ROLE_RELATIONS],
    )


def test_map_bond_endpoints_resolves_canonical_and_skips_missing() -> None:
    residue_ids = [f"A:{i}:" for i in (1, 6, 30, 64)]
    bonds = [
        {
            "residue_id_1": "1lyz:A:6",
            "residue_id_2": "1lyz:A:30",
            "bond_type": "disulf",
        },
        {
            "residue_id_1": "1lyz:A:6",
            "residue_id_2": "1lyz:A:999",
            "bond_type": "disulf",
        },
        {
            "residue_id_1": "1lyz:B:6",
            "residue_id_2": "1lyz:A:30",
            "bond_type": "covale",
        },
    ]
    mapped, skipped = map_bond_endpoints_to_nodes(
        bonds, residue_ids, structure_id="1lyz", chain_label="A"
    )
    assert mapped == [(1, 2, "disulf")]
    assert skipped == 2


def test_attach_chem_appends_disulf_and_covale_rows() -> None:
    data, coords, residue_ids = _toy_role_graph(n=10)
    e_before = int(data.edge_index.size(1))
    bonds = [
        {
            "residue_id_1": "toy:A:2",
            "residue_id_2": "toy:A:8",
            "bond_type": "disulf",
        },
        {
            "residue_id_1": "toy:A:3",
            "residue_id_2": "toy:A:4",
            "bond_type": "covale",
        },
    ]
    out = attach_chem_edge_graph(
        data,
        coords,
        bonds,
        residue_ids=residue_ids,
        structure_id="toy",
        chain_label="A",
    )
    assert out.chem_edge_graph is True
    assert out.edge_attr.size(-1) == EDGE_ATTR_CHEM_DIM
    assert out.edge_index.size(1) == e_before + 4  # two undirected → 4 directed
    oh = out.edge_attr[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM]
    assert torch.allclose(oh.sum(dim=-1), torch.ones(oh.size(0)))
    assert int((oh[:, ROLE_DISULF] > 0.5).sum().item()) == 2
    assert int((oh[:, ROLE_COVALE] > 0.5).sum().item()) == 2
    assert out.chem_edge_counts["disulf"] == 1
    assert out.chem_edge_counts["covale"] == 1
    assert set(CHEM_BOND_TYPES) == {"disulf", "covale"}


def test_attach_chem_empty_bonds_is_noop_layout_compatible() -> None:
    data, coords, residue_ids = _toy_role_graph()
    e_before = int(data.edge_index.size(1))
    attr_before = data.edge_attr.clone()
    out = attach_chem_edge_graph(
        data,
        coords,
        [],
        residue_ids=residue_ids,
        structure_id="4obe",
        chain_label="A",
    )
    # Empty set still pads layout so chem-enabled models see fixed one-hot width.
    assert out.edge_index.size(1) == e_before
    assert out.edge_attr.size(-1) == EDGE_ATTR_CHEM_DIM
    assert out.chem_edge_counts == {"disulf": 0, "covale": 0}
    # Role one-hots preserved
    assert torch.allclose(
        out.edge_attr[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS],
        attr_before[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS],
    )


def test_empty_chem_relation_skipped_in_multirel_forward() -> None:
    torch.manual_seed(0)
    conv = EquivariantConvMultiRel(
        32,
        num_relations=NUM_ROLE_RELATIONS_WITH_CHEM,
        onehot_offset=GEO_DIM,
    )
    n, e = 6, 8
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    ea = torch.zeros(e, EDGE_ATTR_CHEM_DIM)
    ea[:, :3] = torch.randn(e, 3)
    ea[:, 3] = ea[:, 3].abs() + 0.5
    ea[:, GEO_DIM + ROLE_PACKING] = 1.0
    y = conv(x, edge_index, ea)
    assert torch.isfinite(y).all()
    # No chem edges → chem MLP never required for finite forward
    assert not bool((ea[:, GEO_DIM + ROLE_DISULF] > 0.5).any())
    assert not bool((ea[:, GEO_DIM + ROLE_COVALE] > 0.5).any())


def test_model_chem_edge_selects_seven_rel_conv() -> None:
    model = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=1,
        num_experts=2,
        role_edge_mp=True,
        chem_edge_mp=True,
    )
    assert model.chem_edge_mp is True
    assert len(model.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CHEM


def test_chem_radial_mlp_expansion_preserves_gate_init_under_isolated_seed() -> None:
    """New chem MLP slots must not scramble gate/prototype under init_seed discipline."""

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
            chem_edge_mp=False,
            init_seed=123,
        )
        base_snap = _gate_proto_snapshot(baseline)

    with isolated_torch_seed(123):
        chem = GOSPConeMapperV66(
            node_dim=4,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            init_seed=123,
        )
        chem_snap = _gate_proto_snapshot(chem)

    assert base_snap.keys() == chem_snap.keys()
    for key in base_snap:
        assert torch.equal(base_snap[key], chem_snap[key]), key
    assert len(chem.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CHEM
    assert len(baseline.convs[0].radial_mlps) < NUM_ROLE_RELATIONS_WITH_CHEM
