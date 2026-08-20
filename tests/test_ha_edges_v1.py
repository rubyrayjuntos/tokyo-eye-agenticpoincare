"""Unit tests for ha_edges_v1 construction (graph communication ablation)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import AtomRecord, ResidueRecord, TAU
from science.dtie.v66.chem_edge_graph import (
    EDGE_ATTR_CHEM_DIM,
    NUM_ROLE_RELATIONS_WITH_CHEM,
    attach_chem_edge_graph,
    pad_role_edge_attr_for_chem,
)
from science.dtie.v66.ha_edge_graph import (
    EDGE_ATTR_CHEM_HA_DIM,
    EDGE_ATTR_ROLE_HA_DIM,
    HA_MIN_DIST_COL_CHEM,
    HA_PACKING_CA_MAX_A,
    HA_PACKING_MIN_DIST_A,
    HA_STRENGTH_COL_CHEM,
    HA_STRENGTH_COL_ROLE,
    dehydron_strength,
    ha_column_map,
    packing_contact_ha,
    packing_strength,
)
from science.dtie.v66.role_edge_graph import (
    EDGE_ATTR_ROLE_DIM,
    NUM_ROLE_RELATIONS,
    ROLE_DEHYDRON,
    ROLE_PACKING,
    ROLE_RIBBON,
    ROLE_SPOKE,
    attach_role_edge_graph,
    compute_role_edge_graph,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM


def _toy_records(n: int, coords: np.ndarray) -> list[ResidueRecord]:
    """Minimal residue atoms: N/CA/C/O plus a sidechain heavy near CA."""
    out: list[ResidueRecord] = []
    for i in range(n):
        ca = coords[i]
        atoms = (
            AtomRecord("N", "N", ca + np.array([-1.2, 0.0, 0.0])),
            AtomRecord("CA", "C", ca.copy()),
            AtomRecord("C", "C", ca + np.array([1.2, 0.0, 0.0])),
            AtomRecord("O", "O", ca + np.array([1.5, 1.0, 0.0])),
            AtomRecord("CB", "C", ca + np.array([0.0, 1.5, 0.0])),
        )
        out.append(
            ResidueRecord(
                chain_label="A",
                residue_index=i + 1,
                residue_name="ALA",
                atoms=atoms,
            )
        )
    return out


def test_packing_contact_ha_frozen_rule() -> None:
    assert packing_contact_ha(7.5, ha_min_dist=10.0) is True  # Cα band
    assert packing_contact_ha(9.0, ha_min_dist=4.0) is True  # HA rescue
    assert packing_contact_ha(9.0, ha_min_dist=5.0) is False  # HA too far
    assert packing_contact_ha(HA_PACKING_CA_MAX_A + 0.1, ha_min_dist=3.0) is False
    assert packing_contact_ha(0.05, ha_min_dist=1.0) is False


def test_dehydron_and_packing_strength_bounds() -> None:
    s = dehydron_strength(5.0, 3.0, tau=TAU)
    assert 0.0 <= s <= 1.0
    assert packing_strength(0.0) == pytest.approx(1.0)
    assert packing_strength(8.0) == pytest.approx(0.0)


def test_ha_edges_same_relation_ids_wider_attr() -> None:
    n = 10
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    rho = np.array([20.0] * 5 + [8.0] * 5, dtype=np.float64)
    rids = [f"A:{i + 1}:" for i in range(n)]
    records = _toy_records(n, coords)

    ei0, ea0 = compute_role_edge_graph(
        coords, rho, residue_ids=rids, residue_records=records, ha_edges=False
    )
    ei1, ea1 = compute_role_edge_graph(
        coords, rho, residue_ids=rids, residue_records=records, ha_edges=True
    )
    assert ea0.shape[1] == EDGE_ATTR_ROLE_DIM
    assert ea1.shape[1] == EDGE_ATTR_ROLE_HA_DIM
    # Same relation vocabulary (first 5 one-hots after GEO)
    assert ea1.shape[1] == EDGE_ATTR_ROLE_DIM + 2
    assert np.allclose(
        ea1[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS].sum(axis=1), 1.0
    )
    # No new relation one-hots beyond packing..coupling
    assert ea1[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS].shape[1] == NUM_ROLE_RELATIONS
    # Strength only on packing/dehydron
    pack = ea1[:, GEO_DIM + ROLE_PACKING] > 0.5
    dehy = ea1[:, GEO_DIM + ROLE_DEHYDRON] > 0.5
    other = ~(pack | dehy)
    assert np.all(ea1[other, HA_STRENGTH_COL_ROLE] == 0.0)
    if pack.any():
        assert np.all(ea1[pack, HA_STRENGTH_COL_ROLE] >= 0.0)


def test_ha_edges_no_new_relation_ids_with_chem() -> None:
    n = 8
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n) * 3.8
    x = torch.zeros(n, 3)
    x[:, 0] = torch.linspace(20, 8, n)
    x[:, 1] = (x[:, 0] < 13).float()
    data = Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 4))
    data.rho = x[:, 0].clone()
    records = _toy_records(n, coords.astype(np.float64))
    rids = [f"A:{i + 1}:" for i in range(n)]
    out = attach_role_edge_graph(
        data, coords, residue_ids=rids, residue_records=records, ha_edges=True
    )
    assert out.ha_edge_graph is True
    assert out.edge_attr.size(-1) == EDGE_ATTR_ROLE_HA_DIM
    out = attach_chem_edge_graph(
        out,
        coords,
        [],  # zero disulfides — empty chem path
        residue_ids=rids,
        structure_id="1lyz",
        chain_label="A",
    )
    assert out.edge_attr.size(-1) == EDGE_ATTR_CHEM_HA_DIM
    assert torch.allclose(
        out.edge_attr[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM].sum(dim=-1),
        torch.ones(out.edge_attr.size(0)),
    )
    # Column map chem indices
    cmap = ha_column_map(chem=True)
    assert cmap["HA_STRENGTH"] == HA_STRENGTH_COL_CHEM
    assert cmap["HA_MIN_DIST_NORM"] == HA_MIN_DIST_COL_CHEM


def test_pad_role_ha_for_chem_preserves_ha_tail() -> None:
    ea = torch.zeros(3, EDGE_ATTR_ROLE_HA_DIM)
    ea[:, GEO_DIM + ROLE_PACKING] = 1.0
    ea[:, HA_STRENGTH_COL_ROLE] = 0.4
    padded = pad_role_edge_attr_for_chem(ea, ha_edges=True)
    assert padded.shape[-1] == EDGE_ATTR_CHEM_HA_DIM
    assert float(padded[0, HA_STRENGTH_COL_CHEM]) == pytest.approx(0.4)


def test_ha_packing_rescue_can_add_contacts() -> None:
    """Two ordered residues with Cα > 8 but HA min ≤ 4.5 become packing under HA."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],  # Cα = 9 > 8
        ],
        dtype=np.float64,
    )
    rho = np.array([20.0, 20.0], dtype=np.float64)
    # Place CB atoms close across the pair so HA_min ≤ 4.5
    rec0 = ResidueRecord(
        "A",
        1,
        "ALA",
        (
            AtomRecord("N", "N", np.array([-1.0, 0.0, 0.0])),
            AtomRecord("CA", "C", coords[0].copy()),
            AtomRecord("C", "C", np.array([1.0, 0.0, 0.0])),
            AtomRecord("O", "O", np.array([1.2, 1.0, 0.0])),
            AtomRecord("CB", "C", np.array([4.5, 0.0, 0.0])),
        ),
    )
    rec1 = ResidueRecord(
        "A",
        2,
        "ALA",
        (
            AtomRecord("N", "N", np.array([8.0, 0.0, 0.0])),
            AtomRecord("CA", "C", coords[1].copy()),
            AtomRecord("C", "C", np.array([10.0, 0.0, 0.0])),
            AtomRecord("O", "O", np.array([10.2, 1.0, 0.0])),
            AtomRecord("CB", "C", np.array([5.0, 0.0, 0.0])),  # CB–CB = 0.5 Å
        ),
    )
    rids = ["A:1:", "A:2:"]
    _, ea0 = compute_role_edge_graph(
        coords, rho, residue_ids=rids, residue_records=[rec0, rec1], ha_edges=False
    )
    _, ea1 = compute_role_edge_graph(
        coords, rho, residue_ids=rids, residue_records=[rec0, rec1], ha_edges=True
    )
    n_pack0 = int((ea0[:, GEO_DIM + ROLE_PACKING] > 0.5).sum() // 2)
    n_pack1 = int((ea1[:, GEO_DIM + ROLE_PACKING] > 0.5).sum() // 2)
    assert n_pack0 == 0
    assert n_pack1 >= 1
    assert HA_PACKING_MIN_DIST_A >= 4.5


def test_spoke_ribbon_membership_unchanged_without_ha_rescue() -> None:
    n = 12
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    rho = np.array([20.0] * 6 + [8.0] * 6, dtype=np.float64)
    rids = [f"A:{i + 1}:" for i in range(n)]
    # No residue_records → HA rescue inactive; spoke/ribbon/packing match baseline
    ei0, ea0 = compute_role_edge_graph(coords, rho, residue_ids=rids, ha_edges=False)
    ei1, ea1 = compute_role_edge_graph(coords, rho, residue_ids=rids, ha_edges=True)
    assert ea1.shape[1] == EDGE_ATTR_ROLE_HA_DIM
    for role in (ROLE_SPOKE, ROLE_RIBBON, ROLE_PACKING, ROLE_DEHYDRON):
        c0 = int((ea0[:, GEO_DIM + role] > 0.5).sum() // 2)
        c1 = int((ea1[:, GEO_DIM + role] > 0.5).sum() // 2)
        assert c0 == c1
    # Edge endpoints match for baseline relations
    assert ei0.shape == ei1.shape
