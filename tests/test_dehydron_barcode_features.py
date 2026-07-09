"""Unit tests for dehydron barcode midpoint extraction (Task 1)."""

from __future__ import annotations

import numpy as np

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
