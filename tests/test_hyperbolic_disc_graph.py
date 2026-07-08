"""Hyperbolic disc graph construction and cone-depth alignment tests."""

import numpy as np

from science.dtie.common.hyperbolic_disc_graph import (
    assert_hyperbolic_ops_consistent,
    build_hyperbolic_disc_graph,
    cone_depth_from_disc,
)
from science.dtie.common.hyperbolic_lorentz_ops import poincare_distance
from science.dtie.common.hyperbolic_utils import hyperbolic_dist0, hyperbolic_pairwise_distance
from science.dtie.common.poincare_properties import run_property_suite
from science.dtie.common.structural_disc_compose import (
    StructuralResidueInput,
    compose_structural_disc,
    export_viewer_json,
)

C = 0.6054342985153198


def _synthetic_residues(n: int = 40) -> list[StructuralResidueInput]:
    rng = np.random.default_rng(3)
    out: list[StructuralResidueInput] = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        out.append(
            StructuralResidueInput(
                residue_id=f"A:{i + 1}:",
                residue_index=i + 1,
                chain_label="A",
                rho=float(rng.uniform(8.0, 24.0)),
                tau_flag=1.0 if rng.random() < 0.2 else 0.0,
                ss_type=1.0,
                ca_xyz=np.array([np.cos(angle), np.sin(angle), float(i)], dtype=np.float64),
            )
        )
    return out


def test_cone_depth_matches_hyperbolic_dist0() -> None:
    art = compose_structural_disc(_synthetic_residues(30), C)
    z = art.z_disc_matrix
    depth = cone_depth_from_disc(z, C)
    expected = hyperbolic_dist0(z, c=C)
    assert np.allclose(depth, expected, atol=1e-10)


def test_hyperbolic_ops_consistent() -> None:
    art = compose_structural_disc(_synthetic_residues(25), C)
    assert_hyperbolic_ops_consistent(art.z_disc_matrix, C)


def test_build_hyperbolic_disc_graph_edges() -> None:
    art = compose_structural_disc(_synthetic_residues(20), C)
    z = art.z_disc_matrix
    ei, ea = build_hyperbolic_disc_graph(z, C, k_neighbors=4)
    assert ei.shape[0] == 2
    assert ea.shape[1] == 4
    assert ea.shape[0] == ei.shape[1]
    assert ei.shape[1] > 0
    for k in range(min(10, ea.shape[0])):
        i, j = int(ei[0, k]), int(ei[1, k])
        d = poincare_distance(z[i], z[j], C)
        assert abs(ea[k, 3] - d) < 1e-8


def test_viewer_hyperbolic_r_matches_cone_depth() -> None:
    art = compose_structural_disc(_synthetic_residues(24), C, structure_id="test")
    export_viewer_json(art)
    z = art.z_disc_matrix
    depth = cone_depth_from_disc(z, C)
    hyp_nodes = np.array([n.hyperbolic_r for n in art.nodes])
    assert np.allclose(hyp_nodes, depth, atol=1e-8)


def test_property_suite_on_composed_disc() -> None:
    art = compose_structural_disc(_synthetic_residues(50), C, structure_id="synthetic")
    z = art.z_disc_matrix
    depth = cone_depth_from_disc(z, C)
    report = run_property_suite(
        structure_id="synthetic",
        curvature=C,
        source="structural_disc_compose",
        disc_2d=z,
        cone_depth=depth,
    )
    assert report.physics is not None
    assert report.physics.cone_depth_std > 0.0


def test_hyperbolic_pairwise_symmetric() -> None:
    art = compose_structural_disc(_synthetic_residues(15), C)
    d = hyperbolic_pairwise_distance(art.z_disc_matrix, c=C)
    assert np.allclose(d, d.T, atol=1e-10)
    assert np.allclose(np.diag(d), 0.0, atol=1e-12)
