"""Unit tests for dehydron barcode features (Tasks 1–4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    BINNED_DIM,
    SCALAR_DIM,
    DehydronMidpoint,
    StructureAtom,
    extract_dehydron_midpoints,
    residue_map_key,
)


def _make_backbone_pair(
    *,
    donor_residue_index: int = 3,
    acceptor_residue_index: int = 2,
    n_coord: np.ndarray | None = None,
    o_coord: np.ndarray | None = None,
    wrapping_carbons: list[np.ndarray] | None = None,
    chain: str = "A",
) -> tuple[list[StructureAtom], dict[tuple[str, int], int]]:
    """Two-residue backbone H-bond with optional apolar wrapping carbons."""
    n_coord = np.asarray(n_coord if n_coord is not None else [5.5, 0.0, 0.0], dtype=np.float64)
    o_coord = np.asarray(o_coord if o_coord is not None else [3.2, 0.0, 0.0], dtype=np.float64)

    atoms: list[StructureAtom] = [
        StructureAtom("N", "N", n_coord, "ALA", chain, donor_residue_index),
        StructureAtom("O", "O", o_coord, "ALA", chain, acceptor_residue_index),
        StructureAtom("CA", "C", n_coord + [0.0, 1.0, 0.0], "ALA", chain, donor_residue_index),
        StructureAtom("CA", "C", o_coord + [0.0, 1.0, 0.0], "ALA", chain, acceptor_residue_index),
    ]
    for i, coord in enumerate(wrapping_carbons or []):
        atoms.append(
            StructureAtom(
                f"CB{i}",
                "C",
                np.asarray(coord, dtype=np.float64),
                "LEU",
                chain,
                99 + i,
            )
        )

    residue_index_map = {
        residue_map_key(chain, acceptor_residue_index): 0,
        residue_map_key(chain, donor_residue_index): 1,
    }
    return atoms, residue_index_map


def test_constants_defined():
    assert BARCODE_FEATURE_VERSION == "dehydron_barcode_v1"
    assert SCALAR_DIM == 11
    assert BINNED_DIM == 40


def test_extract_midpoints_requires_inter_residue_hbond():
    atoms, residue_index_map = _make_backbone_pair(wrapping_carbons=[])
    midpoints = extract_dehydron_midpoints(atoms, residue_index_map)

    assert isinstance(midpoints, list)
    assert len(midpoints) == 1
    mp = midpoints[0]
    assert isinstance(mp, DehydronMidpoint)
    assert mp.donor_idx != mp.acceptor_idx
    assert mp.donor_idx == 1
    assert mp.acceptor_idx == 0
    assert mp.wrapping_count < 13.0
    expected_mid = (np.array([5.5, 0.0, 0.0]) + np.array([3.2, 0.0, 0.0])) / 2.0
    np.testing.assert_allclose(mp.coord, expected_mid)


def test_extract_midpoints_empty_when_overwrapped():
    midpoint = (np.array([5.5, 0.0, 0.0]) + np.array([3.2, 0.0, 0.0])) / 2.0
    # Pack 13+ apolar side-chain carbons within 6.5 Å of the H-bond midpoint.
    wrapping = [midpoint + np.array([1.0, 0.0, 0.0]) * (k % 3) + np.array([0.0, (k % 4) * 0.5, 0.0]) for k in range(14)]
    atoms, residue_index_map = _make_backbone_pair(wrapping_carbons=wrapping)

    midpoints = extract_dehydron_midpoints(atoms, residue_index_map)
    assert midpoints == []


def test_extract_midpoints_skips_same_residue_n_o():
    coord = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    atoms = [
        StructureAtom("N", "N", coord, "ALA", "A", 1),
        StructureAtom("O", "O", coord + [1.0, 0.0, 0.0], "ALA", "A", 1),
    ]
    residue_index_map = {residue_map_key("A", 1): 0}

    midpoints = extract_dehydron_midpoints(atoms, residue_index_map)
    assert midpoints == []


@pytest.fixture
def simple_midpoints() -> list[DehydronMidpoint]:
    """Five spread 3D midpoints — enough for k-means landmarks and witness complex."""
    coords = [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        np.array([2.5, 4.0, 0.0]),
        np.array([2.5, 1.5, 3.0]),
        np.array([8.0, 2.0, 1.0]),
    ]
    return [
        DehydronMidpoint(
            coord=c,
            donor_idx=i,
            acceptor_idx=(i + 1) % len(coords),
            wrapping_count=5.0,
        )
        for i, c in enumerate(coords)
    ]


@pytest.fixture
def simple_bars(simple_midpoints):
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    return compute_witness_persistence(simple_midpoints, max_alpha_angstrom=20.0)


def test_witness_persistence_empty_midpoints():
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    assert compute_witness_persistence([]) == []


def test_witness_persistence_single_midpoint_returns_empty():
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    one = DehydronMidpoint(
        coord=np.array([0.0, 0.0, 0.0]),
        donor_idx=0,
        acceptor_idx=1,
        wrapping_count=5.0,
    )
    assert compute_witness_persistence([one]) == []


def test_witness_persistence_returns_nonneg_persistence(simple_midpoints):
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    bars = compute_witness_persistence(simple_midpoints, max_alpha_angstrom=20.0)
    assert len(bars) > 0
    assert all(b.persistence >= 0.0 for b in bars)
    assert all(b.dim in (0, 1) for b in bars)


def test_aggregate_missing_mask_when_no_midpoints():
    from science.dtie.common.dehydron_barcode_features import aggregate_residue_barcode_features

    out = aggregate_residue_barcode_features(5, [], [])
    assert out["scalars"].shape == (5, 11)
    assert out["binned"] is None
    assert out["missing"].shape == (5, 1)
    assert np.allclose(out["missing"], 1.0)


def test_aggregate_binned_shape_when_enabled(simple_midpoints, simple_bars):
    from science.dtie.common.dehydron_barcode_features import aggregate_residue_barcode_features

    out = aggregate_residue_barcode_features(10, simple_midpoints, simple_bars, use_binned=True)
    assert out["binned"].shape == (10, 40)


def test_aggregate_structure_stats_not_kfold_duplicated(simple_bars):
    """Residue touching K midpoints must not K-fold duplicate structure-level bar stats."""
    from science.dtie.common.dehydron_barcode_features import aggregate_residue_barcode_features

    midpoints = [
        DehydronMidpoint(
            coord=np.array([0.0, 0.0, 0.0]),
            donor_idx=0,
            acceptor_idx=1,
            wrapping_count=5.0,
        ),
        DehydronMidpoint(
            coord=np.array([5.0, 0.0, 0.0]),
            donor_idx=0,
            acceptor_idx=2,
            wrapping_count=5.0,
        ),
    ]
    out = aggregate_residue_barcode_features(5, midpoints, simple_bars)
    scalars = out["scalars"]

    assert scalars[0, 0] == pytest.approx(np.log1p(len(simple_bars)))
    assert scalars[0, 0] != pytest.approx(np.log1p(2 * len(simple_bars)))
    assert scalars[0, 10] == pytest.approx(np.log1p(2))


def test_stack_dims_scalars_only():
    from science.dtie.common.dehydron_barcode_features import stack_node_features_with_barcode

    x = np.zeros((8, 3), np.float32)
    barcode = {
        "scalars": np.zeros((8, 11), np.float32),
        "binned": None,
        "missing": np.ones((8, 1), np.float32),
    }
    y = stack_node_features_with_barcode(x, barcode, use_binned=False)
    assert y.shape == (8, 15)


def test_stack_dims_full():
    from science.dtie.common.dehydron_barcode_features import stack_node_features_with_barcode

    x = np.zeros((8, 3), np.float32)
    barcode = {
        "scalars": np.zeros((8, 11), np.float32),
        "binned": np.zeros((8, 40), np.float32),
        "missing": np.zeros((8, 1), np.float32),
    }
    y = stack_node_features_with_barcode(x, barcode, use_binned=True)
    assert y.shape == (8, 55)


FOUROBE_PDB = Path(
    "/home/rswan/Documents/tokyo-eyes-consolidation/tokyo-eye-agenticpoincare/"
    "checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/viewers/4obe/4obe_gosp_native.pdb"
)


@pytest.mark.skipif(not FOUROBE_PDB.is_file(), reason="4obe PDB not available")
def test_featurize_4obe_chain_a_smoke():
    from science.dtie.common.dehydron_barcode_features import (
        BARCODE_FEATURE_VERSION,
        featurize_chain_dehydron_barcode,
    )

    out = featurize_chain_dehydron_barcode(FOUROBE_PDB, "A")
    assert out["metadata"]["version"] == BARCODE_FEATURE_VERSION
    assert out["metadata"]["n_midpoints"] >= 1
    assert out["metadata"]["n_bars"] >= 0
    n = out["scalars"].shape[0]
    assert out["scalars"].shape == (n, 11)
    assert out["missing"].shape == (n, 1)
    assert np.all(np.isfinite(out["scalars"]))
    assert np.all(np.isfinite(out["missing"]))
