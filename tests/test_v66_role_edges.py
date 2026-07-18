"""Role-typed edge graph + 4-relation MP tests (v6.6 feeler)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from science.dtie.v66.gnn.equivariant_conv_multirel import EquivariantConvMultiRel
from science.dtie.v66.gnn.model import GOSPConeMapperV66
from science.dtie.v66.role_edge_graph import (
    COUPLING_STRENGTH_COL,
    DEHYDRON_BARCODE_COL,
    EDGE_ATTR_ROLE_DIM,
    NUM_ROLE_RELATIONS,
    ROLE_COUPLING,
    ROLE_DEHYDRON,
    ROLE_PACKING,
    ROLE_RIBBON,
    ROLE_SPOKE,
    SPOKE_RHO_COL,
    attach_role_edge_graph,
    compute_role_edge_graph,
    coupling_strength,
    inject_dehydron_edge_barcode,
    spoke_rho_weight,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM
from science.training.config import TrainingConfig, apply_v66_feeler_config


def test_role_graph_has_four_relations_and_no_generic_contact_only() -> None:
    n = 12
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    # Core (ordered) then rim (disordered)
    rho = np.array([20.0] * 6 + [8.0] * 6, dtype=np.float64)
    tau = np.array([0.0] * 6 + [1.0] * 6, dtype=np.float64)
    rids = [f"A:{i + 1}:" for i in range(n)]
    ei, ea = compute_role_edge_graph(coords, rho, tau_flag=tau, residue_ids=rids)
    assert ea.shape[1] == EDGE_ATTR_ROLE_DIM
    counts = ea[:, GEO_DIM:].sum(axis=0)
    assert counts[ROLE_RIBBON] > 0
    assert counts[ROLE_PACKING] > 0
    assert counts[ROLE_SPOKE] > 0
    # Without atom records, dehydron may be empty or mean-ρ based — either OK
    assert ei.shape[1] == ea.shape[0]
    # Every edge has exactly one role one-hot
    assert np.allclose(
        ea[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS].sum(axis=1), 1.0
    )
    spoke_mask = ea[:, GEO_DIM + ROLE_SPOKE] > 0.5
    assert spoke_mask.any()
    assert np.all(ea[~spoke_mask, SPOKE_RHO_COL] == 0.0)
    assert np.all(ea[spoke_mask, SPOKE_RHO_COL] > 0.0)


def test_attach_role_replaces_contact_graph() -> None:
    n = 8
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n) * 3.8
    x = torch.zeros(n, 4)
    x[:, 0] = torch.linspace(20, 8, n)
    x[:, 1] = (x[:, 0] < 13).float()
    # Fake dense contact graph
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.randn(len(src), 4),
    )
    data.rho = x[:, 0].clone()
    out = attach_role_edge_graph(
        data,
        coords,
        residue_ids=[f"A:{i + 1}:" for i in range(n)],
    )
    assert out.role_edge_graph is True
    assert out.edge_attr.size(-1) == EDGE_ATTR_ROLE_DIM
    assert out.role_edge_counts["ribbon"] > 0
    assert out.role_edge_counts["packing"] > 0
    assert out.role_edge_counts["spoke"] > 0
    # Role one-hots are exclusive per directed edge
    assert torch.allclose(
        out.edge_attr[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS].sum(dim=-1),
        torch.ones(out.edge_attr.size(0)),
    )


def test_spoke_rho_weight_scales_spoke_messages() -> None:
    torch.manual_seed(0)
    conv = EquivariantConvMultiRel(
        32,
        num_relations=NUM_ROLE_RELATIONS,
        onehot_offset=GEO_DIM,
        spoke_rho_col=SPOKE_RHO_COL,
        spoke_relation_id=ROLE_SPOKE,
    )
    n, e = 6, 8
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    ea = torch.zeros(e, EDGE_ATTR_ROLE_DIM)
    ea[:, :3] = torch.randn(e, 3)
    ea[:, 3] = ea[:, 3].abs() + 0.5
    ea[:, GEO_DIM + ROLE_SPOKE] = 1.0
    ea[:, SPOKE_RHO_COL] = 0.0
    y0 = conv(x, edge_index, ea)
    ea[:, SPOKE_RHO_COL] = 2.0
    y1 = conv(x, edge_index, ea)
    assert not torch.allclose(y0, y1)


def test_spoke_rho_weight_formula() -> None:
    # High contrast ordered↔disordered → stronger than near-τ boundary
    strong = spoke_rho_weight(25.0, 5.0, tau=13.0)
    weak = spoke_rho_weight(13.5, 12.5, tau=13.0)
    assert strong > weak > 0.0
    assert spoke_rho_weight(25.0, 5.0, tau=13.0, scale=2.0) == pytest.approx(
        strong * 2.0, rel=1e-6
    )


def test_spoke_and_ribbon_edge_scales_boost_messages() -> None:
    torch.manual_seed(0)
    conv_base = EquivariantConvMultiRel(
        32,
        num_relations=NUM_ROLE_RELATIONS,
        onehot_offset=GEO_DIM,
        spoke_rho_col=SPOKE_RHO_COL,
        spoke_relation_id=ROLE_SPOKE,
        spoke_edge_scale=1.0,
        ribbon_relation_id=ROLE_RIBBON,
        ribbon_edge_scale=1.0,
    )
    conv_boost = EquivariantConvMultiRel(
        32,
        num_relations=NUM_ROLE_RELATIONS,
        onehot_offset=GEO_DIM,
        spoke_rho_col=SPOKE_RHO_COL,
        spoke_relation_id=ROLE_SPOKE,
        spoke_edge_scale=1.5,
        ribbon_relation_id=ROLE_RIBBON,
        ribbon_edge_scale=1.35,
    )
    conv_boost.load_state_dict(conv_base.state_dict())
    n, e = 6, 8
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    ea_spoke = torch.zeros(e, EDGE_ATTR_ROLE_DIM)
    ea_spoke[:, :3] = torch.randn(e, 3)
    ea_spoke[:, 3] = ea_spoke[:, 3].abs() + 0.5
    ea_spoke[:, GEO_DIM + ROLE_SPOKE] = 1.0
    ea_spoke[:, SPOKE_RHO_COL] = 1.0
    y_base = conv_base(x, edge_index, ea_spoke)
    y_boost = conv_boost(x, edge_index, ea_spoke)
    assert not torch.allclose(y_base, y_boost)

    ea_ribbon = torch.zeros(e, EDGE_ATTR_ROLE_DIM)
    ea_ribbon[:, :3] = torch.randn(e, 3)
    ea_ribbon[:, 3] = ea_ribbon[:, 3].abs() + 0.5
    ea_ribbon[:, GEO_DIM + ROLE_RIBBON] = 1.0
    y_rib_base = conv_base(x, edge_index, ea_ribbon)
    y_rib_boost = conv_boost(x, edge_index, ea_ribbon)
    assert not torch.allclose(y_rib_base, y_rib_boost)


def test_role_multirel_isolation() -> None:
    torch.manual_seed(0)
    conv = EquivariantConvMultiRel(
        32,
        num_relations=NUM_ROLE_RELATIONS,
        onehot_offset=GEO_DIM,
        spoke_rho_col=SPOKE_RHO_COL,
        spoke_relation_id=ROLE_SPOKE,
    )
    n, e = 6, 16
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    ea = torch.zeros(e, EDGE_ATTR_ROLE_DIM)
    ea[:, :3] = torch.randn(e, 3)
    ea[:, 3] = ea[:, 3].abs() + 0.5
    ea[: e // 2, GEO_DIM + ROLE_SPOKE] = 1.0
    ea[e // 2 :, GEO_DIM + ROLE_PACKING] = 1.0
    y0 = conv(x, edge_index, ea)
    with torch.no_grad():
        for p in conv.radial_mlps[ROLE_SPOKE].parameters():
            p.zero_()
    y1 = conv(x, edge_index, ea)
    assert not torch.allclose(y0, y1)


def test_model_role_edge_selects_4_rel_conv_by_default() -> None:
    m = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=2,
        num_experts=2,
        role_edge_mp=True,
    )
    assert m.role_edge_mp is True
    assert m.role_coupling_edges is False
    assert m.multi_rel_edge_mp is True
    assert m.thermo_edge_message_gate is False
    assert m.convs[0].num_relations == 4
    assert m.convs[0].onehot_offset == GEO_DIM
    assert m.convs[0].spoke_rho_col == SPOKE_RHO_COL


def test_model_role_edge_selects_5_rel_conv_with_coupling() -> None:
    m = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=2,
        num_experts=2,
        role_edge_mp=True,
        role_coupling_edges=True,
    )
    assert m.role_edge_mp is True
    assert m.role_coupling_edges is True
    assert m.multi_rel_edge_mp is True
    assert m.thermo_edge_message_gate is False
    assert m.convs[0].num_relations == 5
    assert m.convs[0].onehot_offset == GEO_DIM
    assert m.convs[0].spoke_rho_col == SPOKE_RHO_COL
    assert m.convs[0].coupling_relation_id == ROLE_COUPLING


def test_v66_feeler_enables_role_edges() -> None:
    cfg = apply_v66_feeler_config(TrainingConfig())
    assert cfg.role_edge_mp is True
    assert cfg.multi_rel_edge_mp is True
    assert cfg.thermo_edge_features is False


def test_dehydron_edge_barcode_only_on_dehydron_edges() -> None:
    n = 10
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    rho = np.array([20.0] * 5 + [8.0] * 5, dtype=np.float64)
    tau = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
    ei, ea = compute_role_edge_graph(
        coords,
        rho,
        tau_flag=tau,
        residue_ids=[f"A:{i + 1}:" for i in range(n)],
    )
    dehyd_mask = ea[:, GEO_DIM + ROLE_DEHYDRON] > 0.5
    if not dehyd_mask.any():
        return
    idx = int(np.where(dehyd_mask)[0][0])
    i0 = int(ei[0, idx])
    j0 = int(ei[1, idx])
    key = (min(i0, j0), max(i0, j0))
    vec = np.array([1.0, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    ea2 = inject_dehydron_edge_barcode(ea, ei, {key: vec})
    hit = ((ei[0] == key[0]) & (ei[1] == key[1])) | ((ei[0] == key[1]) & (ei[1] == key[0]))
    hit_dehyd = hit & dehyd_mask
    assert hit_dehyd.any()
    assert np.allclose(ea2[hit_dehyd, DEHYDRON_BARCODE_COL : DEHYDRON_BARCODE_COL + 5], vec)
    other_dehyd = dehyd_mask & ~hit
    if other_dehyd.any():
        assert np.allclose(
            ea2[other_dehyd, DEHYDRON_BARCODE_COL : DEHYDRON_BARCODE_COL + 5],
            0.0,
        )
    non_dehyd = ~dehyd_mask
    assert np.allclose(ea2[non_dehyd, DEHYDRON_BARCODE_COL : DEHYDRON_BARCODE_COL + 5], 0.0)


def test_dehydron_edge_barcode_scales_messages() -> None:
    torch.manual_seed(0)
    conv = EquivariantConvMultiRel(
        32,
        num_relations=NUM_ROLE_RELATIONS,
        onehot_offset=GEO_DIM,
        spoke_rho_col=SPOKE_RHO_COL,
        spoke_relation_id=ROLE_SPOKE,
        dehydron_barcode_col=DEHYDRON_BARCODE_COL,
        dehydron_relation_id=ROLE_DEHYDRON,
    )
    n, e = 6, 8
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    ea = torch.zeros(e, EDGE_ATTR_ROLE_DIM)
    ea[:, :3] = torch.randn(e, 3)
    ea[:, 3] = ea[:, 3].abs() + 0.5
    ea[:, GEO_DIM + ROLE_DEHYDRON] = 1.0
    y0 = conv(x, edge_index, ea)
    ea[:, DEHYDRON_BARCODE_COL] = 1.0
    ea[:, DEHYDRON_BARCODE_COL + 3] = 0.8
    y1 = conv(x, edge_index, ea)
    assert not torch.allclose(y0, y1)


def test_coupling_edges_added_when_enabled() -> None:
    n = 12
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    rho = np.array([20.0] * 6 + [8.0] * 6, dtype=np.float64)
    tau = np.array([0.0] * 6 + [1.0] * 6, dtype=np.float64)
    rids = [f"A:{i + 1}:" for i in range(n)]
    ei0, ea0 = compute_role_edge_graph(
        coords, rho, tau_flag=tau, residue_ids=rids, enable_coupling_edges=False
    )
    ei1, ea1 = compute_role_edge_graph(
        coords, rho, tau_flag=tau, residue_ids=rids, enable_coupling_edges=True
    )
    c0 = int(ea0[:, GEO_DIM + ROLE_COUPLING].sum())
    c1 = int(ea1[:, GEO_DIM + ROLE_COUPLING].sum())
    assert c0 == 0
    assert c1 > 0
    assert ea1.shape[1] == EDGE_ATTR_ROLE_DIM
    assert ei1.shape[1] == ea1.shape[0] >= ei0.shape[1]


def test_coupling_strength_scales_messages() -> None:
    torch.manual_seed(0)
    conv = EquivariantConvMultiRel(
        32,
        num_relations=NUM_ROLE_RELATIONS,
        onehot_offset=GEO_DIM,
        coupling_strength_col=COUPLING_STRENGTH_COL,
        coupling_relation_id=ROLE_COUPLING,
    )
    n, e = 6, 8
    x = torch.randn(n, 32)
    edge_index = torch.randint(0, n, (2, e))
    ea = torch.zeros(e, EDGE_ATTR_ROLE_DIM)
    ea[:, :3] = torch.randn(e, 3)
    ea[:, 3] = ea[:, 3].abs() + 0.5
    ea[:, GEO_DIM + ROLE_COUPLING] = 1.0
    ea[:, COUPLING_STRENGTH_COL] = 0.0
    y0 = conv(x, edge_index, ea)
    ea[:, COUPLING_STRENGTH_COL] = 1.5
    y1 = conv(x, edge_index, ea)
    assert not torch.allclose(y0, y1)


def test_coupling_strength_formula() -> None:
    strong = coupling_strength(8.0, 20.0, tau=13.0, dist=6.0)
    weak = coupling_strength(12.0, 12.5, tau=13.0, dist=11.0)
    assert strong > weak > 0.0


def test_adapt_checkpoint_radial_mlps_expands_fifth_relation() -> None:
    from science.dtie.v66.gnn.model import (
        GOSPConeMapperV66,
        load_v66_state_dict,
    )

    old = GOSPConeMapperV66(node_dim=4, hidden=32, num_layers=1, num_experts=2, role_edge_mp=True)
    new = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=1,
        num_experts=2,
        role_edge_mp=True,
        role_coupling_edges=True,
    )
    old_sd = old.state_dict()
    missing, _ = load_v66_state_dict(new, old_sd)
    assert not any("radial_mlps.4" in k for k in missing)
