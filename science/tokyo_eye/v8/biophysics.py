"""v8-vendored biophysics helpers (wrap / salt / π / PDB parse).

Copied for lineage isolation — intentionally does **not** import
``science.tokyo_eye.biology_graph``, ``thermo_edge_features``, or
``science.dtie.common.residue_features``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord

# H-bond candidate gates (v8-owned constants)
HBOND_SEQ_CUTOFF = 5
HBOND_SPATIAL_CUTOFF = 5.5

WRAPPING_RADIUS = 6.5
# Sprint 8: Kabsch–Sander + double-cone wrapping
# Classic DSSP: E = f * q1 * q2 * (1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN)
DSSP_ENERGY_CUTOFF = -0.5  # kcal/mol; admit iff E <= cutoff
DSSP_ENERGY_SCALE = 332.0
DSSP_Q_NH = 0.42
DSSP_Q_CO = 0.20
DSSP_FQQ = DSSP_ENERGY_SCALE * DSSP_Q_NH * DSSP_Q_CO  # ≈27.888
AMIDE_H_BOND_LENGTH_A = 1.01
WRAP_CONE_HALF_ANGLE_DEG = 45.0
WRAP_CONE_COS = float(np.cos(np.deg2rad(WRAP_CONE_HALF_ANGLE_DEG)))  # ≈0.7071
BIOPHYS_CACHE_VERSION = "v8_biophys_freeze_r19"

POLAR_SIDECHAINS = frozenset(
    {"ARG", "ASN", "ASP", "GLN", "GLU", "HIS", "LYS", "SER", "THR", "TYR", "TRP"}
)

SALT_POS_ATOMS: Mapping[str, tuple[str, ...]] = {
    "ARG": ("NH1", "NH2"),
    "LYS": ("NZ",),
    "HIS": ("ND1", "NE2"),
}
SALT_NEG_ATOMS: Mapping[str, tuple[str, ...]] = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}
SALT_MAX_DIST = 4.0

PI_RESIDUES = frozenset({"PHE", "TYR", "TRP", "HIS"})
PI_RING_ATOMS: Mapping[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
}
PI_MAX_CENTROID_DIST = 5.5
PI_FACE_DIHEDRAL_MAX_DEG = 30.0
PI_EDGE_DIHEDRAL_MIN_DEG = 60.0
PI_ALIGN_FACE = 0
PI_ALIGN_EDGE = 1


def counts_as_wrapping_carbon(atom: AtomRecord) -> bool:
    if atom.element != "C":
        return False
    parent = atom.parent_residue_name.strip().upper()
    if parent in POLAR_SIDECHAINS:
        return False
    if atom.atom_name == "C":
        return False
    return True


def wrapping_carbon_coords(all_atoms: Sequence[AtomRecord]) -> np.ndarray:
    coords = [atom.coord for atom in all_atoms if counts_as_wrapping_carbon(atom)]
    if not coords:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def place_backbone_amide_h(
    donor: ResidueRecord,
    prev: ResidueRecord | None,
) -> np.ndarray | None:
    """Reconstruct amide H at 1.01 Å along ∠C(prev)–N–Cα planar bisector.

    Returns ``None`` if N/CA missing or previous carbonyl C unavailable.
    """
    n_atom = donor.get_atom("N")
    ca_atom = donor.get_atom("CA")
    if n_atom is None or ca_atom is None:
        return None
    # Prefer explicit H if present
    h_atom = donor.get_atom("H") or donor.get_atom("HN")
    if h_atom is not None:
        return np.asarray(h_atom.coord, dtype=np.float64).reshape(3)

    c_prev = None
    if prev is not None:
        c_atom = prev.get_atom("C")
        if c_atom is not None:
            c_prev = np.asarray(c_atom.coord, dtype=np.float64).reshape(3)
    if c_prev is None:
        return None

    n = np.asarray(n_atom.coord, dtype=np.float64).reshape(3)
    ca = np.asarray(ca_atom.coord, dtype=np.float64).reshape(3)
    a = ca - n
    b = c_prev - n
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None
    a = a / na
    b = b / nb
    # H lies opposite the bisector of C–N–CA
    bis = -(a + b)
    bn = float(np.linalg.norm(bis))
    if bn < 1e-8:
        # Collinear fallback: perpendicular in plane of a,b
        bis = np.cross(a, np.cross(a, b))
        bn = float(np.linalg.norm(bis))
        if bn < 1e-8:
            return None
    bis = bis / bn
    return n + AMIDE_H_BOND_LENGTH_A * bis


def kabsch_sander_energy(
    donor: ResidueRecord,
    acceptor: ResidueRecord,
    *,
    h_coord: np.ndarray | None = None,
    prev_donor: ResidueRecord | None = None,
) -> float | None:
    """Kabsch–Sander electrostatic H-bond energy (kcal/mol).

    Classic DSSP:
    ``E = 332 * 0.42 * 0.20 * (1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN)``.
    Returns ``None`` if required atoms cannot be resolved.
    """
    n_atom = donor.get_atom("N")
    o_atom = acceptor.get_atom("O")
    c_atom = acceptor.get_atom("C")
    if n_atom is None or o_atom is None or c_atom is None:
        return None
    if h_coord is None:
        h_coord = place_backbone_amide_h(donor, prev_donor)
    if h_coord is None:
        return None
    n = np.asarray(n_atom.coord, dtype=np.float64).reshape(3)
    o = np.asarray(o_atom.coord, dtype=np.float64).reshape(3)
    c = np.asarray(c_atom.coord, dtype=np.float64).reshape(3)
    h = np.asarray(h_coord, dtype=np.float64).reshape(3)

    def _inv(a: np.ndarray, b: np.ndarray) -> float | None:
        d = float(np.linalg.norm(a - b))
        if d < 0.5:  # clash / missing
            return None
        return 1.0 / d

    r_on = _inv(o, n)
    r_ch = _inv(c, h)
    r_oh = _inv(o, h)
    r_cn = _inv(c, n)
    if None in (r_on, r_ch, r_oh, r_cn):
        return None
    return float(DSSP_FQQ * (r_on + r_ch - r_oh - r_cn))


def _count_wrapping_double_cone(
    midpoint: np.ndarray,
    axis_hat: np.ndarray,
    carbon_coords: np.ndarray,
    *,
    wrapping_radius: float,
    cone_cos: float = WRAP_CONE_COS,
) -> float:
    """Count nonpolar carbons in radius AND double-cone about ``axis_hat``.

    Vectorized in PyTorch (CPU) for cache-build throughput.
    Strict gate: ``abs(dot(v_hat, u_hat)) >= cos(45°)``.
    """
    import torch

    if carbon_coords.size == 0:
        return 0.0
    mid = torch.as_tensor(midpoint, dtype=torch.float64).reshape(3)
    u = torch.as_tensor(axis_hat, dtype=torch.float64).reshape(3)
    un = float(torch.linalg.norm(u))
    if un < 1e-8:
        return 0.0
    u_hat = u / un
    carbons = torch.as_tensor(carbon_coords, dtype=torch.float64)
    delta = carbons - mid
    dist = torch.linalg.norm(delta, dim=1)
    in_ball = dist <= float(wrapping_radius)
    if not bool(torch.any(in_ball)):
        return 0.0
    safe = torch.clamp(dist, min=1e-8)
    v_hat = delta / safe.unsqueeze(1)
    cosang = torch.abs(v_hat @ u_hat)
    in_cone = cosang >= float(cone_cos)
    return float(torch.count_nonzero(in_ball & in_cone).item())


def compute_bond_wrapping_count(
    donor: ResidueRecord,
    acceptor: ResidueRecord,
    all_atoms: Sequence[AtomRecord],
    *,
    wrapping_radius: float = WRAPPING_RADIUS,
    carbon_coords: np.ndarray | None = None,
    h_coord: np.ndarray | None = None,
    prev_donor: ResidueRecord | None = None,
    use_double_cone: bool = True,
) -> float:
    """Wrapping count for backbone H-bond donor(N) → acceptor(O).

    Sprint 8 default: double-cone gate about H→O (fallback N→O).
    Returns ``-1.0`` if donor/acceptor atoms missing.
    """
    n_atom = donor.get_atom("N")
    o_atom = acceptor.get_atom("O")
    if n_atom is None or o_atom is None:
        return -1.0
    n = np.asarray(n_atom.coord, dtype=np.float64).reshape(3)
    o = np.asarray(o_atom.coord, dtype=np.float64).reshape(3)
    if h_coord is None:
        h_coord = place_backbone_amide_h(donor, prev_donor)
    if h_coord is not None:
        h = np.asarray(h_coord, dtype=np.float64).reshape(3)
        mid = 0.5 * (h + o)
        axis = o - h
    else:
        mid = 0.5 * (n + o)
        axis = o - n
    carbons = (
        carbon_coords
        if carbon_coords is not None
        else wrapping_carbon_coords(all_atoms)
    )
    if use_double_cone:
        return _count_wrapping_double_cone(
            mid, axis, carbons, wrapping_radius=wrapping_radius
        )
    return _count_wrapping_from_coords(mid, carbons, wrapping_radius=wrapping_radius)


# Back-compat alias
compute_double_cone_wrapping_count = compute_bond_wrapping_count


def _count_wrapping_from_coords(
    midpoint: np.ndarray,
    carbon_coords: np.ndarray,
    *,
    wrapping_radius: float,
) -> float:
    if carbon_coords.size == 0:
        return 0.0
    mid = np.asarray(midpoint, dtype=np.float64).reshape(3)
    if carbon_coords.shape[0] >= 64:
        from scipy.spatial import cKDTree

        return float(len(cKDTree(carbon_coords).query_ball_point(mid, r=wrapping_radius)))
    d2 = np.sum((carbon_coords - mid) ** 2, axis=1)
    return float(np.count_nonzero(d2 < wrapping_radius * wrapping_radius))


def _atom_coords(rec: ResidueRecord, names: Sequence[str]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for name in names:
        a = rec.get_atom(name)
        if a is not None:
            out.append(np.asarray(a.coord, dtype=np.float64).reshape(3))
    return out


def _min_pair_dist(
    pos_a: Sequence[np.ndarray],
    pos_b: Sequence[np.ndarray],
) -> float:
    best = float("inf")
    for a in pos_a:
        for b in pos_b:
            d = float(np.linalg.norm(a - b))
            if d < best:
                best = d
    return best


def detect_salt_bridges(
    records: Sequence[ResidueRecord],
    *,
    index_by_auth: Mapping[int, int],
    max_dist: float = SALT_MAX_DIST,
) -> list[tuple[int, int, float, float]]:
    pos: list[tuple[int, list[np.ndarray]]] = []
    neg: list[tuple[int, list[np.ndarray]]] = []
    for rec in records:
        name = (rec.residue_name or "").strip().upper()
        idx = index_by_auth.get(int(rec.residue_index))
        if idx is None:
            continue
        if name in SALT_POS_ATOMS:
            coords = _atom_coords(rec, SALT_POS_ATOMS[name])
            if coords:
                pos.append((idx, coords))
        if name in SALT_NEG_ATOMS:
            coords = _atom_coords(rec, SALT_NEG_ATOMS[name])
            if coords:
                neg.append((idx, coords))
    out: list[tuple[int, int, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for i, pc in pos:
        for j, nc in neg:
            if i == j:
                continue
            d = _min_pair_dist(pc, nc)
            if d <= float(max_dist):
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                out.append((key[0], key[1], -1.0, float(d)))
    return out


def _ring_centroid_and_normal(
    rec: ResidueRecord,
) -> tuple[np.ndarray, np.ndarray] | None:
    name = (rec.residue_name or "").strip().upper()
    atoms = PI_RING_ATOMS.get(name)
    if not atoms:
        return None
    coords = _atom_coords(rec, atoms)
    if len(coords) < 3:
        return None
    pts = np.stack(coords, axis=0)
    centroid = pts.mean(axis=0)
    v1 = pts[0] - centroid
    v2 = pts[1] - centroid
    n = np.cross(v1, v2)
    nn = float(np.linalg.norm(n))
    if nn < 1e-8:
        v2 = pts[min(2, len(pts) - 1)] - centroid
        n = np.cross(v1, v2)
        nn = float(np.linalg.norm(n))
    if nn < 1e-8:
        return None
    return centroid, n / nn


def _dihedral_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    c = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(c))))


def detect_pi_stacks(
    records: Sequence[ResidueRecord],
    *,
    index_by_auth: Mapping[int, int],
    max_centroid_dist: float = PI_MAX_CENTROID_DIST,
    face_max_deg: float = PI_FACE_DIHEDRAL_MAX_DEG,
    edge_min_deg: float = PI_EDGE_DIHEDRAL_MIN_DEG,
) -> list[tuple[int, int, float, float, int]]:
    rings: list[tuple[int, np.ndarray, np.ndarray]] = []
    for rec in records:
        name = (rec.residue_name or "").strip().upper()
        if name not in PI_RESIDUES:
            continue
        idx = index_by_auth.get(int(rec.residue_index))
        if idx is None:
            continue
        rn = _ring_centroid_and_normal(rec)
        if rn is None:
            continue
        rings.append((idx, rn[0], rn[1]))
    out: list[tuple[int, int, float, float, int]] = []
    seen: set[tuple[int, int]] = set()
    for a in range(len(rings)):
        i, ci, ni = rings[a]
        for b in range(a + 1, len(rings)):
            j, cj, nj = rings[b]
            if i == j:
                continue
            d = float(np.linalg.norm(ci - cj))
            if d > float(max_centroid_dist):
                continue
            dih = _dihedral_deg(ni, nj)
            if dih < float(face_max_deg):
                align = PI_ALIGN_FACE
            elif dih > float(edge_min_deg):
                align = PI_ALIGN_EDGE
            else:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            out.append((key[0], key[1], d, dih, align))
    return out


def parse_residue_records_from_pdb_chain(
    pdb_path: str | Path | Any,
    chain_id: str,
) -> list[ResidueRecord]:
    """Parse standard residues for one chain (Bio.PDB; v8-local)."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("v8", str(pdb_path))
    # NMR / multi-MODEL PDBs: only model 0 — else duplicate residue_index blows the graph.
    model = next(structure.get_models())
    residues_raw = [
        r
        for r in model.get_residues()
        if r.get_id()[0] == " " and r.parent.id == chain_id
    ]
    records: list[ResidueRecord] = []
    for res in residues_raw:
        res_name = res.get_resname().strip().upper()
        atoms = tuple(
            AtomRecord(
                atom_name=atom.name,
                element=(atom.element or "").upper(),
                coord=np.asarray(atom.coord, dtype=np.float64),
                parent_residue_name=res_name,
            )
            for atom in res.get_atoms()
        )
        records.append(
            ResidueRecord(
                chain_label=chain_id,
                residue_index=int(res.get_id()[1]),
                residue_name=res_name,
                atoms=atoms,
                residue_id=f"{chain_id}:{int(res.get_id()[1])}:",
            )
        )
    return records


__all__ = [
    "AMIDE_H_BOND_LENGTH_A",
    "BIOPHYS_CACHE_VERSION",
    "DSSP_ENERGY_CUTOFF",
    "DSSP_ENERGY_SCALE",
    "DSSP_FQQ",
    "DSSP_Q_CO",
    "DSSP_Q_NH",
    "HBOND_SEQ_CUTOFF",
    "HBOND_SPATIAL_CUTOFF",
    "WRAP_CONE_COS",
    "WRAP_CONE_HALF_ANGLE_DEG",
    "WRAPPING_RADIUS",
    "compute_bond_wrapping_count",
    "compute_double_cone_wrapping_count",
    "detect_pi_stacks",
    "detect_salt_bridges",
    "kabsch_sander_energy",
    "parse_residue_records_from_pdb_chain",
    "place_backbone_amide_h",
    "wrapping_carbon_coords",
]
