"""Ligand extract + R6 contacts (Sprint 10.1.0 HETATM + 10.1.1 MOL2/SDF).

On-the-fly only — never writes the frozen protein biophysics graph cache.
Native string parsers only (no heavy cheminformatics toolkits).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from science.tokyo_eye.v8.types import ResidueRecord

LIGAND_FEAT_DIM = 12
R6_DISTANCE_A = 4.5
R6_PROTEIN_LIGAND = 6  # affinity-path constant; not MoE num_relations
DEFAULT_LIGAND_ROOT = Path("data/pdbbind/ligands")

ELEMENT_TO_IDX: dict[str, int] = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4}
HALOGENS = frozenset({"F", "CL", "BR", "I"})
# 0–6 element; 7–9 charge; 10 aromatic; 11 degree/4

WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "TIP"})
ION_RESNAMES = frozenset({"CL", "NA", "K", "MG", "ZN", "CA", "SO4", "PO4"})

ATOMIC_NUM_TO_ELEM = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "CL",
    35: "BR",
    53: "I",
}


@dataclass(frozen=True)
class LigandAtoms:
    coords: np.ndarray  # [N, 3] float32
    elements: tuple[str, ...]
    charges: tuple[float | None, ...]
    resname: str
    chain: str
    resseq: int
    icode: str
    aromatic: tuple[bool, ...] = ()
    degrees: tuple[int, ...] = ()
    source: str = "hetatm"  # mol2 | sdf | hetatm

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])


def _norm_element(raw: str, atom_name: str = "") -> str:
    e = (raw or "").strip().upper()
    if not e:
        an = atom_name.strip().upper()
        e = "".join(c for c in an if c.isalpha())[:2]
        if len(e) >= 2 and e[0] in "CNOSPHF" and e[1].isalpha():
            if e not in HALOGENS and e not in {"NA", "MG", "ZN", "FE", "CU", "MN"}:
                e = e[0]
        elif e:
            e = e[0]
    if e == "BR":
        return "BR"
    if e == "CL":
        return "CL"
    return e


def _is_aromatic_type(atom_type: str) -> bool:
    t = (atom_type or "").strip().lower()
    if not t:
        return False
    if "ar" in t or "@" in t:
        return True
    # common sybyl: c.ar, n.ar, o.ar, ...
    if ".ar" in t:
        return True
    return False


def resolve_ligand_path(
    pdb_id: str,
    *,
    root: Path | str = DEFAULT_LIGAND_ROOT,
) -> Path | None:
    """Option C: ``{id}.mol2`` → ``{id}.sdf`` under ligand root."""
    root = Path(root)
    pid = str(pdb_id).strip().lower()
    for ext in (".mol2", ".sdf", ".MOL2", ".SDF"):
        p = root / f"{pid}{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _parse_hetatm_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("HETATM"):
        return None
    if len(line) < 54:
        return None
    atom_name = line[12:16].strip()
    resname = line[17:20].strip().upper()
    chain = line[21].strip() if len(line) > 21 else ""
    try:
        resseq = int(line[22:26].strip())
    except ValueError:
        return None
    icode = line[26].strip() if len(line) > 26 else ""
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    elem_col = line[76:78].strip() if len(line) >= 78 else ""
    element = _norm_element(elem_col, atom_name)
    charge: float | None = None
    if len(line) >= 80:
        ch = line[78:80].strip()
        if ch:
            sign = -1.0 if "-" in ch else 1.0
            digits = "".join(c for c in ch if c.isdigit())
            if digits:
                charge = sign * float(digits)
    return {
        "atom_name": atom_name,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
        "coord": np.array([x, y, z], dtype=np.float32),
        "element": element,
        "charge": charge,
    }


def extract_ligand_hetatm(pdb_path: Path | str) -> LigandAtoms | None:
    """Option A bootstrap: largest multi-atom hetero after water/ion purge."""
    path = Path(pdb_path)
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            rec = _parse_hetatm_line(line)
            if rec is None:
                continue
            rn = rec["resname"]
            if rn in WATER_RESNAMES or rn in ION_RESNAMES:
                continue
            key = (rec["chain"], rec["resseq"], rec["icode"], rn)
            groups[key].append(rec)
    candidates = [(k, v) for k, v in groups.items() if len(v) >= 2]
    if not candidates:
        return None
    key, rows = max(candidates, key=lambda kv: len(kv[1]))
    chain, resseq, icode, resname = key
    n = len(rows)
    return LigandAtoms(
        coords=np.stack([r["coord"] for r in rows], axis=0).astype(np.float32),
        elements=tuple(str(r["element"]) for r in rows),
        charges=tuple(r["charge"] for r in rows),
        resname=resname,
        chain=chain,
        resseq=int(resseq),
        icode=str(icode),
        aromatic=tuple(False for _ in range(n)),
        degrees=tuple(0 for _ in range(n)),
        source="hetatm",
    )


def extract_ligand_mol2(path: Path | str) -> LigandAtoms | None:
    """Native ``@<TRIPOS>ATOM`` / ``@<TRIPOS>BOND`` scan (no RDKit)."""
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    # Find ATOM block
    atom_start = None
    bond_start = None
    for i, line in enumerate(lines):
        u = line.strip().upper()
        if u.startswith("@<TRIPOS>ATOM"):
            atom_start = i + 1
        elif u.startswith("@<TRIPOS>BOND"):
            bond_start = i + 1
            break
        elif atom_start is not None and u.startswith("@<TRIPOS>"):
            # next section without BOND
            break
    if atom_start is None:
        return None

    atoms: list[dict[str, Any]] = []
    for line in lines[atom_start:]:
        s = line.strip()
        if not s or s.upper().startswith("@<TRIPOS>"):
            break
        parts = s.split()
        if len(parts) < 6:
            continue
        try:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        atom_type = parts[5] if len(parts) > 5 else ""
        elem = _norm_element(atom_type.split(".")[0], parts[1] if len(parts) > 1 else "")
        charge: float | None = None
        if len(parts) >= 9:
            try:
                charge = float(parts[8])
            except ValueError:
                charge = None
        atoms.append(
            {
                "coord": np.array([x, y, z], dtype=np.float32),
                "element": elem,
                "charge": charge,
                "aromatic": _is_aromatic_type(atom_type),
            }
        )
    if len(atoms) < 1:
        return None

    n = len(atoms)
    deg = [0] * n
    if bond_start is not None:
        for line in lines[bond_start:]:
            s = line.strip()
            if not s or s.upper().startswith("@<TRIPOS>"):
                break
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                a = int(parts[1]) - 1
                b = int(parts[2]) - 1
            except ValueError:
                continue
            if 0 <= a < n and 0 <= b < n:
                deg[a] += 1
                deg[b] += 1

    return LigandAtoms(
        coords=np.stack([a["coord"] for a in atoms], axis=0),
        elements=tuple(a["element"] for a in atoms),
        charges=tuple(a["charge"] for a in atoms),
        resname="LIG",
        chain="",
        resseq=1,
        icode="",
        aromatic=tuple(bool(a["aromatic"]) for a in atoms),
        degrees=tuple(int(d) for d in deg),
        source="mol2",
    )


def extract_ligand_sdf(path: Path | str) -> LigandAtoms | None:
    """Native SDF V2000 atom/bond blocks (aromatic bond type 4)."""
    text = Path(path).read_text(errors="replace")
    # First molecule only (through $$$$ or EOF)
    mol = text.split("$$$$")[0]
    lines = mol.splitlines()
    if len(lines) < 4:
        return None
    counts = lines[3]
    try:
        n_atoms = int(counts[0:3])
        n_bonds = int(counts[3:6])
    except ValueError:
        parts = counts.split()
        if len(parts) < 2:
            return None
        n_atoms, n_bonds = int(parts[0]), int(parts[1])

    atoms_lines = lines[4 : 4 + n_atoms]
    bonds_lines = lines[4 + n_atoms : 4 + n_atoms + n_bonds]
    if len(atoms_lines) < n_atoms:
        return None

    atoms: list[dict[str, Any]] = []
    for line in atoms_lines:
        try:
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
        except ValueError:
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            elem = _norm_element(parts[3])
            charge = None
        else:
            elem_raw = line[31:34].strip() if len(line) >= 34 else ""
            elem = _norm_element(elem_raw)
            charge = None
            # V2000 charge code cols 36-39 (rare); prefer M  CHG below
        atoms.append(
            {
                "coord": np.array([x, y, z], dtype=np.float32),
                "element": elem,
                "charge": charge,
                "aromatic": False,
            }
        )
    n = len(atoms)
    if n < 1:
        return None
    deg = [0] * n
    for line in bonds_lines:
        try:
            a = int(line[0:3]) - 1
            b = int(line[3:6]) - 1
            btype = int(line[6:9])
        except ValueError:
            parts = line.split()
            if len(parts) < 3:
                continue
            a, b, btype = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
        if 0 <= a < n and 0 <= b < n:
            deg[a] += 1
            deg[b] += 1
            if btype == 4:
                atoms[a]["aromatic"] = True
                atoms[b]["aromatic"] = True

    # M  CHG lines
    for line in lines[4 + n_atoms + n_bonds :]:
        if line.startswith("M  END"):
            break
        if line.startswith("M  CHG"):
            parts = line.split()
            # M  CHG nn idx charge idx charge ...
            try:
                nn = int(parts[2])
            except (IndexError, ValueError):
                continue
            for k in range(nn):
                try:
                    idx = int(parts[3 + 2 * k]) - 1
                    chg = float(parts[4 + 2 * k])
                except (IndexError, ValueError):
                    break
                if 0 <= idx < n:
                    atoms[idx]["charge"] = chg

    return LigandAtoms(
        coords=np.stack([a["coord"] for a in atoms], axis=0),
        elements=tuple(a["element"] for a in atoms),
        charges=tuple(a["charge"] for a in atoms),
        resname="LIG",
        chain="",
        resseq=1,
        icode="",
        aromatic=tuple(bool(a["aromatic"]) for a in atoms),
        degrees=tuple(int(d) for d in deg),
        source="sdf",
    )


def extract_ligand_auto(
    pdb_id: str,
    *,
    pdb_path: Path | str | None = None,
    ligand_root: Path | str = DEFAULT_LIGAND_ROOT,
) -> tuple[LigandAtoms | None, str, dict[str, Any]]:
    """Resolve mol2 → sdf → HETATM. Returns ``(atoms, source, meta)``."""
    meta: dict[str, Any] = {"hetatm_feature_pad": 0}
    lig_path = resolve_ligand_path(pdb_id, root=ligand_root)
    if lig_path is not None:
        suf = lig_path.suffix.lower()
        if suf == ".mol2":
            atoms = extract_ligand_mol2(lig_path)
            if atoms is not None and atoms.n_atoms >= 1:
                return atoms, "mol2", meta
        elif suf == ".sdf":
            atoms = extract_ligand_sdf(lig_path)
            if atoms is not None and atoms.n_atoms >= 1:
                return atoms, "sdf", meta
    if pdb_path is None:
        return None, "hetatm", {**meta, "hetatm_feature_pad": 1}
    atoms = extract_ligand_hetatm(pdb_path)
    if atoms is None:
        return None, "hetatm", {**meta, "hetatm_feature_pad": 1}
    return atoms, "hetatm", {**meta, "hetatm_feature_pad": 1}


def _element_channel(element: str) -> int:
    e = element.strip().upper()
    if e in ELEMENT_TO_IDX:
        return ELEMENT_TO_IDX[e]
    if e in HALOGENS:
        return 5
    return 6


def _charge_channels(charge: float | None) -> tuple[float, float, float]:
    if charge is None:
        return (0.0, 1.0, 0.0)
    if charge < 0:
        return (1.0, 0.0, 0.0)
    if charge > 0:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


def ligand_feature_matrix(atoms: LigandAtoms) -> np.ndarray:
    """Return ``float32 [N_lig, 12]`` element + charge + aromatic + degree."""
    n = atoms.n_atoms
    out = np.zeros((n, LIGAND_FEAT_DIM), dtype=np.float32)
    arom = atoms.aromatic if len(atoms.aromatic) == n else tuple(False for _ in range(n))
    degs = atoms.degrees if len(atoms.degrees) == n else tuple(0 for _ in range(n))
    for i, (elem, ch) in enumerate(zip(atoms.elements, atoms.charges)):
        out[i, _element_channel(elem)] = 1.0
        neg, neu, pos = _charge_channels(ch)
        out[i, 7] = neg
        out[i, 8] = neu
        out[i, 9] = pos
        out[i, 10] = 1.0 if arom[i] else 0.0
        out[i, 11] = float(min(int(degs[i]), 4)) / 4.0
    return out


def residue_proxy_coords(records: Sequence[ResidueRecord]) -> np.ndarray:
    """Cβ if present else Cα — shape ``[N_res, 3]``."""
    rows: list[np.ndarray] = []
    for r in records:
        cb = r.get_atom("CB")
        ca = r.get_atom("CA")
        atom = cb if cb is not None else ca
        if atom is None:
            raise ValueError(
                f"residue {r.residue_name}{r.residue_index} missing CB/CA"
            )
        rows.append(np.asarray(atom.coord, dtype=np.float32).reshape(3))
    return np.stack(rows, axis=0)


def build_r6_edges(
    res_proxy_xyz: np.ndarray,
    lig_xyz: np.ndarray,
    *,
    cutoff: float = R6_DISTANCE_A,
) -> tuple[np.ndarray, dict[str, Any]]:
    """On-the-fly R6 contacts; never touches protein graph cache."""
    res = np.asarray(res_proxy_xyz, dtype=np.float64)
    lig = np.asarray(lig_xyz, dtype=np.float64)
    if res.ndim != 2 or res.shape[1] != 3:
        raise ValueError("res_proxy_xyz must be [N, 3]")
    if lig.ndim != 2 or lig.shape[1] != 3:
        raise ValueError("lig_xyz must be [L, 3]")
    n_res, n_lig = res.shape[0], lig.shape[0]
    empty_meta = {
        "r6_empty": 1,
        "n_edges": 0,
        "n_contacts": 0,
        "bidirectional": True,
        "cutoff": float(cutoff),
    }
    if n_res == 0 or n_lig == 0:
        return np.zeros((2, 0), dtype=np.int64), empty_meta

    d2 = (
        (res[:, None, 0] - lig[None, :, 0]) ** 2
        + (res[:, None, 1] - lig[None, :, 1]) ** 2
        + (res[:, None, 2] - lig[None, :, 2]) ** 2
    )
    hits = np.argwhere(d2 <= float(cutoff) ** 2)
    if hits.size == 0:
        return np.zeros((2, 0), dtype=np.int64), empty_meta

    cols: list[tuple[int, int]] = []
    for ri, lj in hits:
        cols.append((int(ri), int(lj)))
        cols.append((int(ri), int(lj)))
    edge_index = np.asarray(cols, dtype=np.int64).T
    return edge_index, {
        "r6_empty": 0,
        "n_edges": int(edge_index.shape[1]),
        "n_contacts": int(hits.shape[0]),
        "bidirectional": True,
        "cutoff": float(cutoff),
    }


def unique_r6_contacts(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    pairs = list({(int(a), int(b)) for a, b in zip(edge_index[0], edge_index[1])})
    pairs.sort()
    return np.asarray(pairs, dtype=np.int64).T


__all__ = [
    "DEFAULT_LIGAND_ROOT",
    "ELEMENT_TO_IDX",
    "HALOGENS",
    "ION_RESNAMES",
    "LIGAND_FEAT_DIM",
    "LigandAtoms",
    "R6_DISTANCE_A",
    "R6_PROTEIN_LIGAND",
    "WATER_RESNAMES",
    "build_r6_edges",
    "extract_ligand_auto",
    "extract_ligand_hetatm",
    "extract_ligand_mol2",
    "extract_ligand_sdf",
    "ligand_feature_matrix",
    "residue_proxy_coords",
    "resolve_ligand_path",
    "unique_r6_contacts",
]
