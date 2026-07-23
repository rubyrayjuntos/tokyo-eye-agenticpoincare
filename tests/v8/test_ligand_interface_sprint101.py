"""Sprint 10.1.0 — ligand HETATM sanitize, 10-ch features, R6 off-cache."""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import numpy as np

import science.tokyo_eye.v8.ligand_interface as li
from science.tokyo_eye.v8.ligand_interface import (
    HALOGENS,
    LIGAND_FEAT_DIM,
    R6_DISTANCE_A,
    LigandAtoms,
    build_r6_edges,
    extract_ligand_hetatm,
    ligand_feature_matrix,
    residue_proxy_coords,
)
from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord


def _write_pdb(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "toy.pdb"
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def test_ligand_feat_dim_is_ten() -> None:
    assert LIGAND_FEAT_DIM == 10
    assert R6_DISTANCE_A == 4.5
    assert "F" in HALOGENS and "CL" in HALOGENS


def test_hetatm_drops_water_and_ions_selects_largest(tmp_path: Path) -> None:
    pdb = _write_pdb(
        tmp_path,
        """
        HETATM    1  O   HOH A   1      0.000   0.000   0.000  1.00  0.00           O
        HETATM    2 NA    NA A   2      1.000   0.000   0.000  1.00  0.00          NA
        HETATM    3  C1  LIG A  10      5.000   0.000   0.000  1.00  0.00           C
        HETATM    4  N1  LIG A  10      6.000   0.000   0.000  1.00  0.00           N
        HETATM    5  O1  LIG A  10      7.000   0.000   0.000  1.00  0.00           O
        HETATM    6  C1  SM  A  20      9.000   0.000   0.000  1.00  0.00           C
        HETATM    7  C2  SM  A  20     10.000   0.000   0.000  1.00  0.00           C
        END
        """,
    )
    lig = extract_ligand_hetatm(pdb)
    assert lig is not None
    assert lig.resname == "LIG"
    assert lig.n_atoms == 3
    assert lig.coords.shape == (3, 3)


def test_ten_channel_neutral_default_and_halogen() -> None:
    atoms = LigandAtoms(
        coords=np.zeros((3, 3), dtype=np.float32),
        elements=("C", "CL", "FE"),
        charges=(None, 0.0, -1.0),
        resname="LIG",
        chain="A",
        resseq=1,
        icode="",
    )
    feats = ligand_feature_matrix(atoms)
    assert feats.shape == (3, 10)
    assert feats.dtype == np.float32
    assert feats[0, 0] == 1.0 and feats[0, 8] == 1.0  # C + neutral
    assert feats[1, 5] == 1.0 and feats[1, 8] == 1.0  # halogen + neutral
    assert feats[2, 6] == 1.0 and feats[2, 7] == 1.0  # other + negative


def test_r6_bidirectional_inclusive_cutoff() -> None:
    """Undirected contact → two directed columns (res→lig and lig→res marker).

    Convention: ``edge_index`` stores residue index in row 0 and ligand index in
    row 1 for **both** directed emissions of an undirected contact (affinity head
    filters to unique undirected or uses res→lig only). Bidirectional = duplicate
    undirected listing with ``meta['bidirectional']=True`` and ``n_edges == 2 *
    n_contacts``.
    """
    res_xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    lig_xyz = np.array([[4.5, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
    ei, meta = build_r6_edges(res_xyz, lig_xyz, cutoff=4.5)
    assert ei.shape[0] == 2
    assert int(meta["n_contacts"]) == 1
    assert int(meta["n_edges"]) == 2  # both directions
    assert bool(meta["bidirectional"]) is True
    assert int(meta["r6_empty"]) == 0
    # Both columns encode the same (res=0, lig=0) contact
    assert set(zip(ei[0].tolist(), ei[1].tolist())) == {(0, 0)}


def test_r6_empty_flag() -> None:
    res_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    lig_xyz = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    ei, meta = build_r6_edges(res_xyz, lig_xyz, cutoff=4.5)
    assert ei.shape == (2, 0)
    assert int(meta["r6_empty"]) == 1
    assert int(meta["n_edges"]) == 0
    assert int(meta["n_contacts"]) == 0


def test_r6_builder_never_touches_graph_cache() -> None:
    src = inspect.getsource(li)
    forbidden = (
        "v8_graph_cache",
        "graph_cache_path",
        "_save_graph_cache",
        "torch.save",
        "BIOPHYS_CACHE_VERSION",
    )
    for token in forbidden:
        assert token not in src, f"ligand_interface must not reference {token}"
    r6_src = inspect.getsource(build_r6_edges)
    assert "torch.save" not in r6_src
    assert "graph_cache" not in r6_src


def test_residue_proxy_cb_else_ca() -> None:
    gly = ResidueRecord(
        chain_label="A",
        residue_index=1,
        residue_name="GLY",
        atoms=(
            AtomRecord("CA", "C", np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        ),
    )
    ala = ResidueRecord(
        chain_label="A",
        residue_index=2,
        residue_name="ALA",
        atoms=(
            AtomRecord("CA", "C", np.array([0.0, 0.0, 0.0], dtype=np.float32)),
            AtomRecord("CB", "C", np.array([4.0, 5.0, 6.0], dtype=np.float32)),
        ),
    )
    xyz = residue_proxy_coords([gly, ala])
    assert xyz.shape == (2, 3)
    np.testing.assert_allclose(xyz[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(xyz[1], [4.0, 5.0, 6.0])


def test_no_ligand_returns_none(tmp_path: Path) -> None:
    pdb = _write_pdb(
        tmp_path,
        """
        HETATM    1  O   HOH A   1      0.000   0.000   0.000  1.00  0.00           O
        HETATM    2 NA    NA A   2      1.000   0.000   0.000  1.00  0.00          NA
        END
        """,
    )
    assert extract_ligand_hetatm(pdb) is None


def test_frozen_split_manifest_untouched() -> None:
    path = Path("manifests/v8_pdbbind_refined_cluster30_v1.json")
    assert path.is_file()
    src = inspect.getsource(li)
    assert "v8_pdbbind_refined_cluster30" not in src
