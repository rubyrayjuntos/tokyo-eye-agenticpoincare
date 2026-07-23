"""Ligand HETATM sanitize + R6 protein–ligand contacts (Sprint 10.1.0).

On-the-fly only — never writes the frozen protein biophysics graph cache.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from science.tokyo_eye.v8.types import ResidueRecord

LIGAND_FEAT_DIM = 10
R6_DISTANCE_A = 4.5
R6_PROTEIN_LIGAND = 6  # affinity-path constant; not MoE num_relations

ELEMENT_TO_IDX: dict[str, int] = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4}
HALOGENS = frozenset({"F", "CL", "BR", "I"})
# channel 5 = halogen, 6 = other
# channels 7–9 = negative / neutral / positive

WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "TIP"})
ION_RESNAMES = frozenset({"CL", "NA", "K", "MG", "ZN", "CA", "SO4", "PO4"})


@dataclass(frozen=True)
class LigandAtoms:
    coords: np.ndarray  # [N, 3] float32
    elements: tuple[str, ...]
    charges: tuple[float | None, ...]
    resname: str
    chain: str
    resseq: int
    icode: str

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])


def _norm_element(raw: str, atom_name: str = "") -> str:
    e = (raw or "").strip().upper()
    if not e:
        # PDB column 77-78 missing → guess from atom name
        an = atom_name.strip().upper()
        e = "".join(c for c in an if c.isalpha())[:2]
        if len(e) == 2 and e[1].islower():
            pass
        elif len(e) >= 2 and e[0] in "CNOSPHF" and e[1].isalpha():
            # e.g. CA → C if not Cl; keep two-letter only for known
            if e not in HALOGENS and e not in {"NA", "MG", "ZN", "FE", "CU", "MN"}:
                e = e[0]
        elif e:
            e = e[0]
    if e == "BR":
        return "BR"
    if e == "CL":
        return "CL"
    return e


def _parse_hetatm_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("HETATM"):
        return None
    # PDB fixed columns (1-based): 13-16 atom, 18-20 resname, 22 chain,
    # 23-26 resseq, 27 icode, 31-38 x, 39-46 y, 47-54 z, 77-78 element
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
            # PDB charge is like "1+" / "2-"
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
    # multi-atom only
    candidates = [(k, v) for k, v in groups.items() if len(v) >= 2]
    if not candidates:
        return None
    key, rows = max(candidates, key=lambda kv: len(kv[1]))
    chain, resseq, icode, resname = key
    coords = np.stack([r["coord"] for r in rows], axis=0).astype(np.float32)
    elements = tuple(str(r["element"]) for r in rows)
    charges = tuple(r["charge"] for r in rows)
    return LigandAtoms(
        coords=coords,
        elements=elements,
        charges=charges,
        resname=resname,
        chain=chain,
        resseq=int(resseq),
        icode=str(icode),
    )


def _element_channel(element: str) -> int:
    e = element.strip().upper()
    if e in ELEMENT_TO_IDX:
        return ELEMENT_TO_IDX[e]
    if e in HALOGENS:
        return 5
    return 6


def _charge_channels(charge: float | None) -> tuple[float, float, float]:
    if charge is None:
        return (0.0, 1.0, 0.0)  # neutral default (10.1.0 HETATM)
    if charge < 0:
        return (1.0, 0.0, 0.0)
    if charge > 0:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


def ligand_feature_matrix(atoms: LigandAtoms) -> np.ndarray:
    """Return ``float32 [N_lig, 10]`` one-hot element + charge bins."""
    n = atoms.n_atoms
    out = np.zeros((n, LIGAND_FEAT_DIM), dtype=np.float32)
    for i, (elem, ch) in enumerate(zip(atoms.elements, atoms.charges)):
        out[i, _element_channel(elem)] = 1.0
        neg, neu, pos = _charge_channels(ch)
        out[i, 7] = neg
        out[i, 8] = neu
        out[i, 9] = pos
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
    """On-the-fly R6 contacts; never touches protein graph cache.

    ``edge_index`` shape ``[2, E]`` with row0=residue idx, row1=ligand idx.
    Each undirected contact is emitted **twice** (bidirectional marker) so
    ``n_edges == 2 * n_contacts``. Affinity head should unique on (res, lig).
    """
    res = np.asarray(res_proxy_xyz, dtype=np.float64)
    lig = np.asarray(lig_xyz, dtype=np.float64)
    if res.ndim != 2 or res.shape[1] != 3:
        raise ValueError("res_proxy_xyz must be [N, 3]")
    if lig.ndim != 2 or lig.shape[1] != 3:
        raise ValueError("lig_xyz must be [L, 3]")
    n_res, n_lig = res.shape[0], lig.shape[0]
    if n_res == 0 or n_lig == 0:
        empty = np.zeros((2, 0), dtype=np.int64)
        return empty, {
            "r6_empty": 1,
            "n_edges": 0,
            "n_contacts": 0,
            "bidirectional": True,
            "cutoff": float(cutoff),
        }

    # Pairwise distances [N, L]
    d2 = (
        (res[:, None, 0] - lig[None, :, 0]) ** 2
        + (res[:, None, 1] - lig[None, :, 1]) ** 2
        + (res[:, None, 2] - lig[None, :, 2]) ** 2
    )
    hits = np.argwhere(d2 <= float(cutoff) ** 2)
    if hits.size == 0:
        empty = np.zeros((2, 0), dtype=np.int64)
        return empty, {
            "r6_empty": 1,
            "n_edges": 0,
            "n_contacts": 0,
            "bidirectional": True,
            "cutoff": float(cutoff),
        }

    # Duplicate each undirected contact for bidirectional emission
    cols: list[tuple[int, int]] = []
    for ri, lj in hits:
        cols.append((int(ri), int(lj)))
        cols.append((int(ri), int(lj)))  # second directed emission (same endpoints)
    edge_index = np.asarray(cols, dtype=np.int64).T  # [2, E]
    n_contacts = int(hits.shape[0])
    return edge_index, {
        "r6_empty": 0,
        "n_edges": int(edge_index.shape[1]),
        "n_contacts": n_contacts,
        "bidirectional": True,
        "cutoff": float(cutoff),
    }


def unique_r6_contacts(edge_index: np.ndarray) -> np.ndarray:
    """Deduplicate bidirectional emissions → unique (res, lig) columns."""
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    pairs = list({(int(a), int(b)) for a, b in zip(edge_index[0], edge_index[1])})
    pairs.sort()
    return np.asarray(pairs, dtype=np.int64).T


__all__ = [
    "ELEMENT_TO_IDX",
    "HALOGENS",
    "ION_RESNAMES",
    "LIGAND_FEAT_DIM",
    "LigandAtoms",
    "R6_DISTANCE_A",
    "R6_PROTEIN_LIGAND",
    "WATER_RESNAMES",
    "build_r6_edges",
    "extract_ligand_hetatm",
    "ligand_feature_matrix",
    "residue_proxy_coords",
    "unique_r6_contacts",
]
