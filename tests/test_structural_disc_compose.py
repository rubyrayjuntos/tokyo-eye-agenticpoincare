"""Tests for deterministic structural disc composition (macro ⊕ micro SSOT)."""

import json

import numpy as np
import pytest

from science.dtie.common.graph_builder import ProteinGraph, ResidueFeatures
from science.dtie.common.hyperbolic_lorentz_ops import (
    ball_radius,
    expmap0_tangent_at_origin,
    hyperbolic_distance_from_origin,
    logmap0_at_origin,
    mobius_add,
    poincare_distance,
)
from science.dtie.common.structural_disc_compose import (
    LAYOUT_METHOD,
    StructuralResidueInput,
    attach_structural_disc_for_forward,
    attach_structural_disc_to_pyg,
    clear_structural_disc_compose_cache,
    compose_from_protein_graph,
    compose_from_training_prot,
    compose_structural_disc,
    export_viewer_json,
    residue_inputs_from_protein_graph,
    verify_mobius_composition,
    z_disc_matrix_for_residue_ids,
)
from experiments.training.v6.train_loop import attach_v6_features

C = 0.6054342985153198


def _synthetic_residues(n: int = 48) -> list[StructuralResidueInput]:
    rng = np.random.default_rng(7)
    residues: list[StructuralResidueInput] = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        ca = np.array([10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 1.5])
        rho = float(rng.uniform(8.0, 22.0))
        tau_flag = 1.0 if rho < 13.0 else 0.0
        residues.append(
            StructuralResidueInput(
                residue_id=f"A:{i + 1}:",
                residue_index=i + 1,
                chain_label="A",
                rho=rho,
                tau_flag=tau_flag,
                ss_type=0.0 if i % 5 == 0 else 1.0,
                ca_xyz=ca,
            )
        )
    return residues


def test_compose_produces_macros_and_nodes():
    art = compose_structural_disc(_synthetic_residues(60), C, structure_id="test")
    assert len(art.macros) >= 2
    assert len(art.nodes) == 60
    assert art.structure_id == "test"


def test_mobius_composition_identity_holds():
    art = compose_structural_disc(_synthetic_residues(40), C)
    assert verify_mobius_composition(art)


def test_dehydrons_tend_to_larger_depth_than_wrapped():
    residues = _synthetic_residues(50)
    art = compose_structural_disc(residues, C)
    dehyd_depth = [n.depth_norm for n in art.nodes if n.kind == "dehydron"]
    wrapped_depth = [n.depth_norm for n in art.nodes if n.kind == "residue"]
    assert dehyd_depth
    assert wrapped_depth
    assert np.median(dehyd_depth) > np.median(wrapped_depth)


def test_all_disc_positions_inside_ball():
    art = compose_structural_disc(_synthetic_residues(80), C)
    r_max = ball_radius(C) * 0.999
    for node in art.nodes:
        assert np.linalg.norm(node.z_disc) < r_max
    for macro in art.macros:
        assert np.linalg.norm(macro.z_disc) < r_max


def test_export_viewer_json_schema():
    art = compose_structural_disc(_synthetic_residues(20), C, structure_id="4obe")
    payload = export_viewer_json(art)
    assert payload["metadata"]["layout"] == LAYOUT_METHOD
    assert payload["metadata"]["structure_id"] == "4obe"
    assert len(payload["macros"]) >= 1
    assert len(payload["nodes"]) == 20
    json.dumps(payload)


def test_compose_from_protein_graph():
    residues = [
        ResidueFeatures(
            residue_id=f"A:{i}:",
            residue_index=i,
            chain_label="A",
            rho=10.0 + i * 0.1,
            tau_flag=1.0 if i % 4 == 0 else 0.0,
            ss_type=0.5,
            sasa=0.3,
            ca_x=float(np.cos(i)),
            ca_y=float(np.sin(i)),
            ca_z=float(i),
        )
        for i in range(1, 31)
    ]
    graph = ProteinGraph(
        structure_id="1crn",
        residues=residues,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_attr=np.zeros((0, 4), dtype=np.float32),
        chain_ids=["A"] * 30,
        residue_indices=list(range(1, 31)),
        residue_ids=[r.residue_id for r in residues],
    )
    inputs = residue_inputs_from_protein_graph(graph)
    assert len(inputs) == 30
    art = compose_from_protein_graph(graph, C)
    assert verify_mobius_composition(art)
    macro_by_id = {m.macro_id: m.z_disc for m in art.macros}
    for node in art.nodes:
        recomposed = mobius_add(macro_by_id[node.parent_macro_id], node.z_micro, C)
        assert np.max(np.abs(recomposed - node.z_disc)) < 1e-8


def test_requires_positive_curvature():
    with pytest.raises(ValueError, match="curvature"):
        compose_structural_disc(_synthetic_residues(5), 0.0)


def test_attach_structural_disc_for_training_prot() -> None:
    import torch
    from torch_geometric.data import Data

    n = 12
    x = torch.randn(n, 4)
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    prot = {
        "pdb_id": "1crn",
        "chain": "A",
        "data": Data(x=x, edge_index=edge_index),
        "ca_coords": torch.randn(n, 3),
        "residue_ids": [f"A:{i}:" for i in range(1, n + 1)],
    }
    data = attach_v6_features(prot["data"])
    attached = attach_structural_disc_for_forward(data, prot, C)
    assert attached.structural_z_disc_frozen is True
    assert attached.structural_z_disc.shape == (n, 2)


def test_disc_xy_from_model_output_prefers_ssot_when_frozen() -> None:
    import torch

    from science.dtie.v6.visualization.interactive_viewer import disc_xy_from_model_output

    pre = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    post = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    out = {
        "hyp_projections_2d_pre": pre,
        "hyp_projections_2d": post,
        "audit_trail": {
            "structural_disc_frozen": True,
            "disc_projection_source": "structural_ssot_frozen",
        },
    }
    assert torch.allclose(disc_xy_from_model_output(out), post)


def test_attach_structural_disc_to_pyg() -> None:
    import torch
    from torch_geometric.data import Data

    graph = ProteinGraph(
        structure_id="1crn",
        residues=[
            ResidueFeatures(
                residue_id=f"A:{i}:",
                residue_index=i,
                chain_label="A",
                rho=12.0,
                tau_flag=0.0,
                ss_type=0.5,
                sasa=0.3,
                ca_x=float(i),
                ca_y=0.0,
                ca_z=0.0,
            )
            for i in range(1, 11)
        ],
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_attr=np.zeros((0, 4), dtype=np.float32),
        chain_ids=["A"] * 10,
        residue_indices=list(range(1, 11)),
        residue_ids=[f"A:{i}:" for i in range(1, 11)],
    )
    art = compose_from_protein_graph(graph, C)
    data = Data(
        x=torch.zeros(10, 4),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_attr, dtype=torch.float32),
    )
    attach_structural_disc_to_pyg(data, art, residue_ids=graph.residue_ids)
    z = z_disc_matrix_for_residue_ids(art, graph.residue_ids)
    assert data.structural_z_disc_frozen is True
    assert data.structural_z_disc.shape == (10, 2)
    assert np.allclose(data.structural_z_disc.numpy(), z, atol=1e-6)
    assert data.hyperbolic_graph is True
    assert data.hyperbolic_edge_index.shape[1] > 0
    assert data.structural_z_disc.device == data.x.device
    assert data.hyperbolic_edge_index.device == data.x.device
    assert data.edge_index.shape == graph.edge_index.shape  # Cα preserved
    assert data.structural_cone_depth.shape == (10,)
    assert np.allclose(
        data.structural_cone_depth.numpy(),
        np.array([n.hyperbolic_r for n in art.nodes]),
        atol=1e-6,
    )


def test_placed_points_are_expmap0_of_tangent():
    """Disc coords must lie on expmap₀ rays, not arbitrary Euclidean polar radii."""
    art = compose_structural_disc(_synthetic_residues(60), C)
    for node in art.nodes:
        z = node.z_disc
        v = logmap0_at_origin(z, C)
        z2 = expmap0_tangent_at_origin(v, C)
        assert np.max(np.abs(z - z2)) < 1e-9
        assert abs(np.linalg.norm(v) - node.hyperbolic_r) < 1e-8


def test_depth_norm_monotone_in_hyperbolic_radius():
    art = compose_structural_disc(_synthetic_residues(80), C)
    depths = np.array([n.depth_norm for n in art.nodes])
    hyp = np.array([n.hyperbolic_r for n in art.nodes])
    assert np.corrcoef(depths, hyp)[0, 1] > 0.95


def test_wrapped_high_rho_closer_to_origin_than_dehydrons():
    residues = _synthetic_residues(50)
    art = compose_structural_disc(residues, C)
    dehyd_hyp = [n.hyperbolic_r for n in art.nodes if n.kind == "dehydron"]
    wrapped_hyp = [n.hyperbolic_r for n in art.nodes if n.kind == "residue"]
    assert dehyd_hyp and wrapped_hyp
    assert np.median(wrapped_hyp) < np.median(dehyd_hyp)


def test_depth_norm_continuous_at_tau_boundary():
    """Depth must not jump when tau_flag flips at ρ=TAU (ring artifact class)."""
    from science.dtie.common.residue_features import TAU
    from science.dtie.common.structural_disc_compose import _structural_depth_norm

    eps_values = [1e-3, 1e-2, 0.05]
    for eps in eps_values:
        below = _structural_depth_norm(
            np.array([TAU - eps]), np.array([1.0])
        )[0]
        above = _structural_depth_norm(
            np.array([TAU + eps]), np.array([0.0])
        )[0]
        assert abs(below - above) < max(0.02, 3.0 * eps)


def test_no_empty_center_hole_on_high_rho_corpus():
    """Wrapped residues with ρ ≫ TAU should reach the hyperbolic core, not a shared ring."""
    residues: list[StructuralResidueInput] = []
    for i in range(40):
        residues.append(
            StructuralResidueInput(
                residue_id=f"A:{i + 1}:",
                residue_index=i + 1,
                chain_label="A",
                rho=float(20.0 + i * 0.5),
                tau_flag=0.0,
                ss_type=1.0,
                ca_xyz=np.array([np.cos(i), np.sin(i), float(i)], dtype=np.float64),
            )
        )
    art = compose_structural_disc(residues, C)
    hyp = np.array([n.hyperbolic_r for n in art.nodes])
    assert hyp.min() < 0.15
    assert len(np.unique(np.round(hyp, 3))) > 10


def test_compose_from_training_prot_caches_by_structure_and_rounded_c(monkeypatch):
    """Identical (structure, round(c,4)) must not re-run full compose."""
    import torch
    from torch_geometric.data import Data

    clear_structural_disc_compose_cache()
    n = 12
    x = torch.randn(n, 4)
    x[:, 0] = torch.linspace(8.0, 20.0, n)
    x[:, 1] = (x[:, 0] < 13.0).float()
    prot = {
        "pdb_id": "1TST",
        "chain": "A",
        "data": Data(x=x, edge_index=torch.tensor([[0], [1]], dtype=torch.long)),
        "ca_coords": torch.randn(n, 3),
        "residue_ids": [f"A:{i}:" for i in range(1, n + 1)],
    }
    calls = {"n": 0}
    real = compose_structural_disc

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "science.dtie.common.structural_disc_compose.compose_structural_disc",
        _counting,
    )
    a1 = compose_from_training_prot(prot, C)
    a2 = compose_from_training_prot(prot, C + 1e-6)  # same round(c, 4)
    a3 = compose_from_training_prot(prot, C + 0.01)  # new bucket
    assert calls["n"] == 2
    assert a1 is a2
    assert a3 is not a1
    clear_structural_disc_compose_cache()

