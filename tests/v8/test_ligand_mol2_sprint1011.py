"""Sprint 10.1.1 — native MOL2/SDF 12-channel ligand parse."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from science.tokyo_eye.v8.ligand_interface import (
    LIGAND_FEAT_DIM,
    extract_ligand_auto,
    extract_ligand_mol2,
    extract_ligand_sdf,
    ligand_feature_matrix,
    resolve_ligand_path,
)


def test_feat_dim_twelve() -> None:
    assert LIGAND_FEAT_DIM == 12


def test_resolve_mol2_before_sdf(tmp_path: Path) -> None:
    root = tmp_path / "ligands"
    root.mkdir()
    (root / "1abc.sdf").write_text("junk")
    (root / "1abc.mol2").write_text("ok")
    p = resolve_ligand_path("1ABC", root=root)
    assert p is not None and p.suffix.lower() == ".mol2"


def test_mol2_charge_aromatic_degree(tmp_path: Path) -> None:
    mol2 = textwrap.dedent(
        """
        @<TRIPOS>MOLECULE
        toy
        3 2 0 0 0
        @<TRIPOS>ATOM
              1 C1         0.0000    0.0000    0.0000 C.ar      1 LIG       0.0000
              2 C2         1.4000    0.0000    0.0000 C.ar      1 LIG       0.0000
              3 O1         2.4000    0.0000    0.0000 O.3       1 LIG      -0.5000
        @<TRIPOS>BOND
             1    1    2 ar
             2    2    3 1
        """
    ).lstrip()
    path = tmp_path / "toy.mol2"
    path.write_text(mol2)
    atoms = extract_ligand_mol2(path)
    assert atoms is not None
    assert atoms.source == "mol2"
    assert atoms.n_atoms == 3
    assert atoms.aromatic[0] and atoms.aromatic[1]
    assert not atoms.aromatic[2]
    assert atoms.degrees[0] == 1 and atoms.degrees[1] == 2 and atoms.degrees[2] == 1
    assert atoms.charges[2] is not None and atoms.charges[2] < 0
    feats = ligand_feature_matrix(atoms)
    assert feats.shape == (3, 12)
    assert feats[0, 10] == 1.0  # aromatic
    assert feats[2, 7] == 1.0  # negative charge bin
    assert feats[1, 11] == 2.0 / 4.0  # degree


def test_mol2_strips_h_and_neutralizes_gasteiger(tmp_path: Path) -> None:
    """PDBbind mol2 is protonated + GAST_HUCK; do not treat |q|<0.5 as formal."""
    mol2 = textwrap.dedent(
        """
        @<TRIPOS>MOLECULE
        toy
        4 3 0 0 0
        @<TRIPOS>ATOM
              1 C1         0.0000    0.0000    0.0000 C.ar      1 LIG      -0.0428
              2 C2         1.4000    0.0000    0.0000 C.ar      1 LIG      -0.0603
              3 O1         2.4000    0.0000    0.0000 O.co2     1 LIG      -0.6653
              4 H1         0.0000    1.0000    0.0000 H         1 LIG       0.0557
        @<TRIPOS>BOND
             1    1    2 ar
             2    2    3 1
             3    1    4 1
        """
    ).lstrip()
    path = tmp_path / "prot.mol2"
    path.write_text(mol2)
    atoms = extract_ligand_mol2(path)
    assert atoms is not None
    assert atoms.n_atoms == 3
    assert "H" not in atoms.elements
    assert atoms.charges[0] is None and atoms.charges[1] is None
    assert atoms.charges[2] is not None and atoms.charges[2] < 0
    assert atoms.degrees[0] == 1  # H neighbor dropped
    feats = ligand_feature_matrix(atoms)
    assert feats[0, 8] == 1.0  # Neutral
    assert feats[2, 7] == 1.0  # Negative (O.co2)


def test_sdf_aromatic_bond_type4_and_chg(tmp_path: Path) -> None:
    # Minimal V2000: 2 atoms, 1 aromatic bond, M CHG on atom 2
    sdf = (
        "toy\n"
        "  -OEChem-\n"
        "\n"
        "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.4000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  4  0  0  0  0\n"
        "M  CHG  1   2   1\n"
        "M  END\n"
        "$$$$\n"
    )
    path = tmp_path / "toy.sdf"
    path.write_text(sdf)
    atoms = extract_ligand_sdf(path)
    assert atoms is not None
    assert atoms.source == "sdf"
    assert atoms.aromatic[0] and atoms.aromatic[1]
    assert atoms.charges[1] == 1.0
    feats = ligand_feature_matrix(atoms)
    assert feats[1, 9] == 1.0  # positive
    assert feats[0, 10] == 1.0


def test_auto_falls_back_hetatm_with_pad(tmp_path: Path) -> None:
    pdb = textwrap.dedent(
        """
        HETATM    1  C1  LIG A  10      0.000   0.000   0.000  1.00  0.00           C
        HETATM    2  N1  LIG A  10      1.000   0.000   0.000  1.00  0.00           N
        HETATM    3  O1  LIG A  10      2.000   0.000   0.000  1.00  0.00           O
        END
        """
    ).lstrip()
    pdb_path = tmp_path / "1xyz.pdb"
    pdb_path.write_text(pdb)
    atoms, source, meta = extract_ligand_auto(
        "1xyz", pdb_path=pdb_path, ligand_root=tmp_path / "empty_ligands"
    )
    assert source == "hetatm"
    assert meta.get("hetatm_feature_pad") == 1
    assert atoms is not None
    feats = ligand_feature_matrix(atoms)
    assert feats.shape[1] == 12
    assert np.allclose(feats[:, 10], 0.0)
    assert np.allclose(feats[:, 11], 0.0)


def test_no_rdkit_import_in_module() -> None:
    import ast
    from pathlib import Path

    path = Path("science/tokyo_eye/v8/ligand_interface.py")
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0].lower())
    assert "rdkit" not in imported
    assert "openbabel" not in imported
    assert "babel" not in imported