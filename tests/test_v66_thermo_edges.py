"""v6.6 bond wrapping SSOT + thermodynamic edge / EquivariantConvThermo tests."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import (
    AtomRecord,
    ResidueRecord,
    compute_bond_wrapping_count,
    compute_dehydron_wrapping_count,
    wrapping_carbon_coords,
)
from science.dtie.v66.gnn.equivariant_conv_thermo import EquivariantConvThermo
from science.dtie.v66.thermo_edge_features import (
    EDGE_ATTR_THERMO_DIM,
    EDGE_TYPE_DEHYDRON,
    EDGE_TYPE_GENERIC,
    EDGE_TYPE_WRAPPED_HBOND,
    GEO_DIM,
    THERMO_DIM,
    attach_thermo_edge_features,
    compute_thermo_edge_attr,
)
from science.training.config import TrainingConfig, apply_v66_feeler_config


def _toy_residue(index: int, n_coord, o_coord, carbons: list[np.ndarray]) -> ResidueRecord:
    atoms = [
        AtomRecord("N", "N", np.asarray(n_coord, dtype=np.float64), "ALA"),
        AtomRecord("O", "O", np.asarray(o_coord, dtype=np.float64), "ALA"),
        AtomRecord("CA", "C", (np.asarray(n_coord) + np.asarray(o_coord)) / 2.0, "ALA"),
    ]
    for k, c in enumerate(carbons):
        atoms.append(AtomRecord(f"CB{k}", "C", np.asarray(c, dtype=np.float64), "ALA"))
    return ResidueRecord("A", index, "ALA", tuple(atoms), residue_id=f"A:{index}:")


def test_bond_wrapping_matches_node_rho_on_same_residue_no() -> None:
    """Donor=acceptor same residue → bond mid == node mid → same count."""
    carbons = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 10.0]),  # outside 6.5Å of mid
    ]
    res = _toy_residue(1, n_coord=[0.0, 0.0, 1.0], o_coord=[0.0, 0.0, -1.0], carbons=carbons)
    all_atoms = list(res.atoms)
    node = compute_dehydron_wrapping_count(res, all_atoms)
    bond = compute_bond_wrapping_count(res, res, all_atoms)
    assert node == bond
    assert node >= 2.0  # CBs (+ CA if within radius)


def test_bond_wrapping_uses_inter_residue_midpoint() -> None:
    donor = _toy_residue(1, n_coord=[0.0, 0.0, 0.0], o_coord=[5.0, 0.0, 0.0], carbons=[])
    acceptor = _toy_residue(5, n_coord=[20.0, 0.0, 0.0], o_coord=[2.0, 0.0, 0.0], carbons=[])
    # Carbon at bond midpoint (donor N + acceptor O) / 2 = (1, 0, 0)
    wrapper = AtomRecord("CB", "C", np.array([1.0, 0.0, 0.0]), "ALA")
    all_atoms = list(donor.atoms) + list(acceptor.atoms) + [wrapper]
    bond = compute_bond_wrapping_count(donor, acceptor, all_atoms)
    assert bond >= 1.0
    # Without the wrapper, count drops
    bond_no = compute_bond_wrapping_count(
        donor, acceptor, list(donor.atoms) + list(acceptor.atoms)
    )
    assert bond == bond_no + 1.0


def test_compute_thermo_prefers_bond_wrapping_when_records_present() -> None:
    n = 6
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n) * 3.8
    src, dst, attrs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[j] - coords[i]))
            if d < 8.0:
                diff = coords[j] - coords[i]
                src.extend([i, j])
                dst.extend([j, i])
                attrs.append([*diff.tolist(), d])
                attrs.append([*(-diff).tolist(), d])
    rho = np.full(n, 20.0, dtype=np.float32)  # high node ρ → mean would say wrapped
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    rids = [f"A:{i + 1}:" for i in range(n)]

    # Build records with low bond wrapping for close pairs (few carbons near N–O)
    records = []
    for i in range(n):
        records.append(
            _toy_residue(
                i + 1,
                n_coord=coords[i] + np.array([0.0, 0.5, 0.0]),
                o_coord=coords[i] + np.array([0.0, -0.5, 0.0]),
                carbons=[],  # no wrappers → bond ρ=0 → dehydron
            )
        )
    thermo = compute_thermo_edge_attr(
        edge_index, edge_attr, rho, residue_ids=rids, residue_records=records
    )
    assert thermo.shape[1] == THERMO_DIM
    # At least one candidate H-bond should use bond path (fallback=0) and dehydron type
    hbond_rows = thermo[:, 5] < 0.5
    assert hbond_rows.any()
    assert (thermo[hbond_rows, 2 + EDGE_TYPE_DEHYDRON] == 1.0).any()


def _toy_graph(n: int = 8) -> Data:
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n, dtype=np.float32) * 3.8
    src, dst, attrs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[j] - coords[i]))
            if d < 8.0:
                diff = coords[j] - coords[i]
                src.extend([i, j])
                dst.extend([j, i])
                attrs.append([diff[0], diff[1], diff[2], d])
                attrs.append([-diff[0], -diff[1], -diff[2], d])
    rho = np.linspace(5.0, 25.0, n).astype(np.float32)
    tau = (rho < 13.0).astype(np.float32)
    x = np.stack([rho, tau, np.zeros(n, np.float32), np.ones(n, np.float32)], axis=1)
    data = Data(
        x=torch.tensor(x),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
    )
    data.rho = data.x[:, 0].clone()
    return data


def test_attach_thermo_edge_features_width() -> None:
    data = _toy_graph()
    rids = [f"A:{i + 1}:" for i in range(data.x.size(0))]
    out = attach_thermo_edge_features(data, residue_ids=rids)
    assert out.edge_attr.size(-1) == EDGE_ATTR_THERMO_DIM


def test_equivariant_conv_thermo_additive_differs_from_geo_only() -> None:
    torch.manual_seed(0)
    conv = EquivariantConvThermo(32)
    n, e = 6, 12
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    geo = torch.randn(e, GEO_DIM)
    geo[:, 3] = geo[:, 3].abs() + 0.5
    thermo = torch.zeros(e, THERMO_DIM)
    thermo[:, 0] = torch.linspace(0.1, 0.9, e)
    thermo[:, 2 + EDGE_TYPE_DEHYDRON] = 1.0
    y_t = conv(x, edge_index, torch.cat([geo, thermo], dim=-1))
    y_g = conv(x, edge_index, geo)
    assert not torch.allclose(y_t, y_g)


def test_v66_feeler_enables_thermo_edge_features() -> None:
    cfg = apply_v66_feeler_config(TrainingConfig())
    # Feeler moved to role-typed edges (packing/dehydron/spoke/ribbon).
    assert cfg.role_edge_mp is True
    assert cfg.multi_rel_edge_mp is True
    assert cfg.thermo_edge_features is False


def test_equivariant_conv_multirel_type_isolation() -> None:
    """Zeroing one relation's radial MLP must change output when that type is present."""
    from science.dtie.v66.gnn.equivariant_conv_multirel import EquivariantConvMultiRel

    torch.manual_seed(1)
    conv = EquivariantConvMultiRel(32)  # thermo layout defaults
    n, e = 6, 12
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    geo = torch.randn(e, GEO_DIM)
    geo[:, 3] = geo[:, 3].abs() + 0.5
    thermo = torch.zeros(e, THERMO_DIM)
    # Half dehydron, half generic
    thermo[: e // 2, 2 + EDGE_TYPE_DEHYDRON] = 1.0
    thermo[e // 2 :, 2 + EDGE_TYPE_GENERIC] = 1.0
    ea = torch.cat([geo, thermo], dim=-1)
    y0 = conv(x, edge_index, ea)
    # Null dehydron radial path
    with torch.no_grad():
        for p in conv.radial_mlps[EDGE_TYPE_DEHYDRON].parameters():
            p.zero_()
    y1 = conv(x, edge_index, ea)
    assert not torch.allclose(y0, y1)


def test_model_multi_rel_selects_conv() -> None:
    from science.dtie.v66.gnn.equivariant_conv_multirel import EquivariantConvMultiRel
    from science.dtie.v66.gnn.model import GOSPConeMapperV66

    m = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=2,
        num_experts=2,
        multi_rel_edge_mp=True,
        thermo_edge_message_gate=True,  # should be suppressed
    )
    assert m.multi_rel_edge_mp is True
    assert m.thermo_edge_message_gate is False
    assert isinstance(m.convs[0], EquivariantConvMultiRel)
    assert m.convs[0].num_relations == 3
