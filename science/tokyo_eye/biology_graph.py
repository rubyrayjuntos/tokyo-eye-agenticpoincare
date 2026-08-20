"""Structured biology graph for hyp_biology_mp (H-bond, dehydron, π-stack, salt).

Cα contact edges are never emitted. Degree-0 nodes are allowed (Option A).
Does not write Normalizer / fact_graph_edge.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import (
    TAU,
    AtomRecord,
    ResidueRecord,
    compute_bond_wrapping_count,
    wrapping_carbon_coords,
)
from science.tokyo_eye.biology_mp_audit import build_biology_mp_audit
from science.tokyo_eye.thermo_edge_features import (
    HBOND_SEQ_CUTOFF,
    HBOND_SPATIAL_CUTOFF,
    parse_auth_seq_ids,
)

BIO_HBOND = 0
BIO_DEHYDRON = 1
BIO_PI_STACK = 2
BIO_SALT_BRIDGE = 3

BIO_TYPE_TO_NAME = {
    BIO_HBOND: "hbond",
    BIO_DEHYDRON: "dehydron",
    BIO_PI_STACK: "pi_stack",
    BIO_SALT_BRIDGE: "salt_bridge",
}

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

# Attr layout (padded to common width for stacking):
# [0]=charge_product or 0, [1]=min_dist or centroid_dist,
# [2]=dihedral_deg or 0, [3]=alignment_type_idx or 0, [4]=type_code
BIO_EDGE_ATTR_DIM = 5


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
    # Plane normal from first two edges relative to centroid.
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
    ang = float(np.degrees(np.arccos(abs(c))))  # acute angle between planes
    return ang


def detect_salt_bridges(
    records: Sequence[ResidueRecord],
    *,
    index_by_auth: Mapping[int, int],
    max_dist: float = SALT_MAX_DIST,
) -> list[tuple[int, int, float, float]]:
    """Return (i, j, charge_product, min_dist) undirected pairs."""
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


def detect_pi_stacks(
    records: Sequence[ResidueRecord],
    *,
    index_by_auth: Mapping[int, int],
    max_centroid_dist: float = PI_MAX_CENTROID_DIST,
    face_max_deg: float = PI_FACE_DIHEDRAL_MAX_DEG,
    edge_min_deg: float = PI_EDGE_DIHEDRAL_MIN_DEG,
) -> list[tuple[int, int, float, float, int]]:
    """Return (i, j, centroid_dist, dihedral_deg, alignment_type_idx)."""
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


def detect_hbond_dehydron_pairs(
    coords: np.ndarray,
    rho: np.ndarray,
    *,
    residue_ids: Sequence[str] | None,
    residue_records: Sequence[ResidueRecord] | None,
    tau: float = TAU,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (hbond_pairs, dehydron_pairs) as undirected index pairs."""
    coords = np.asarray(coords, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    n = int(coords.shape[0])
    seq = parse_auth_seq_ids(residue_ids, n)

    by_seq: dict[int, ResidueRecord] | None = None
    all_atoms: list[AtomRecord] | None = None
    carbon_coords = None
    if residue_records:
        by_seq = {int(r.residue_index): r for r in residue_records}
        all_atoms = [a for r in residue_records for a in r.atoms]
        carbon_coords = wrapping_carbon_coords(all_atoms)

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    hbond: list[tuple[int, int]] = []
    dehydron: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dists[i, j])
            seq_sep = abs(int(seq[i]) - int(seq[j]))
            if seq_sep > HBOND_SEQ_CUTOFF or d >= HBOND_SPATIAL_CUTOFF:
                continue
            rho_bond = -1.0
            if by_seq is not None and all_atoms is not None:
                donor = by_seq.get(int(seq[i]))
                acceptor = by_seq.get(int(seq[j]))
                if donor is not None and acceptor is not None:
                    rho_bond = compute_bond_wrapping_count(
                        donor,
                        acceptor,
                        all_atoms,
                        carbon_coords=carbon_coords,
                    )
                if rho_bond < 0:
                    donor_r = by_seq.get(int(seq[j]))
                    acceptor_r = by_seq.get(int(seq[i]))
                    if donor_r is not None and acceptor_r is not None:
                        rho_bond = compute_bond_wrapping_count(
                            donor_r,
                            acceptor_r,
                            all_atoms,
                            carbon_coords=carbon_coords,
                        )
            if rho_bond < 0:
                # No atom-validated bond — skip (do not invent from Cα alone).
                continue
            key = (i, j)
            if rho_bond < float(tau):
                dehydron.append(key)
            else:
                hbond.append(key)
    return hbond, dehydron


def _append_bidir(
    src: list[int],
    dst: list[int],
    types: list[int],
    attrs: list[list[float]],
    i: int,
    j: int,
    type_code: int,
    attr: Sequence[float],
) -> None:
    row = [float(x) for x in attr]
    while len(row) < BIO_EDGE_ATTR_DIM - 1:
        row.append(0.0)
    row = row[: BIO_EDGE_ATTR_DIM - 1] + [float(type_code)]
    src.extend([i, j])
    dst.extend([j, i])
    types.extend([type_code, type_code])
    attrs.append(row)
    attrs.append(row)


def compute_biology_mp_graph(
    coords: np.ndarray,
    rho: np.ndarray,
    *,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build biology-only MP graph.

    Returns:
        edge_index [2, E], edge_attr [E, BIO_EDGE_ATTR_DIM], edge_types [E],
        meta dict with undirected counts.
    """
    coords = np.asarray(coords, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    n = int(coords.shape[0])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N,3], got {coords.shape}")
    if rho.shape[0] != n:
        raise ValueError("rho length must match coords")

    records = list(residue_records or ())
    index_by_auth: dict[int, int] = {}
    if residue_ids is not None:
        seq = parse_auth_seq_ids(residue_ids, n)
        for i, s in enumerate(seq):
            index_by_auth[int(s)] = i
    else:
        for i, rec in enumerate(records):
            index_by_auth[int(rec.residue_index)] = i

    hbond_pairs, dehydron_pairs = detect_hbond_dehydron_pairs(
        coords, rho, residue_ids=residue_ids, residue_records=records, tau=tau
    )
    salt_pairs = (
        detect_salt_bridges(records, index_by_auth=index_by_auth)
        if records
        else []
    )
    pi_pairs = (
        detect_pi_stacks(records, index_by_auth=index_by_auth) if records else []
    )

    src: list[int] = []
    dst: list[int] = []
    types: list[int] = []
    attrs: list[list[float]] = []

    for i, j in hbond_pairs:
        _append_bidir(src, dst, types, attrs, i, j, BIO_HBOND, [0.0, 0.0, 0.0, 0.0])
    for i, j in dehydron_pairs:
        _append_bidir(src, dst, types, attrs, i, j, BIO_DEHYDRON, [0.0, 0.0, 0.0, 0.0])
    for i, j, cp, d in salt_pairs:
        _append_bidir(
            src, dst, types, attrs, i, j, BIO_SALT_BRIDGE, [cp, d, 0.0, 0.0]
        )
    for i, j, d, dih, align in pi_pairs:
        _append_bidir(
            src, dst, types, attrs, i, j, BIO_PI_STACK, [0.0, d, dih, float(align)]
        )

    if src:
        edge_index = np.stack(
            [np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)]
        )
        edge_attr = np.asarray(attrs, dtype=np.float32)
        edge_types = np.asarray(types, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BIO_EDGE_ATTR_DIM), dtype=np.float32)
        edge_types = np.zeros((0,), dtype=np.int64)

    meta = {
        "hbond": len(hbond_pairs),
        "dehydron": len(dehydron_pairs),
        "salt_bridge": len(salt_pairs),
        "pi_stack": len(pi_pairs),
        "num_nodes": n,
    }
    return edge_index, edge_attr, edge_types, meta


def attach_biology_mp_graph(
    data: Data,
    *,
    coords: np.ndarray | torch.Tensor,
    rho: np.ndarray | torch.Tensor,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
    device: torch.device | str | None = None,
) -> Data:
    """Attach biology hyperbolic MP graph; forbid Cα fallback on this Data."""
    if torch.is_tensor(coords):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)
    if torch.is_tensor(rho):
        rho_np = rho.detach().cpu().numpy()
    else:
        rho_np = np.asarray(rho)

    ei, ea, et, meta = compute_biology_mp_graph(
        coords_np,
        rho_np,
        residue_ids=residue_ids,
        residue_records=residue_records,
        tau=tau,
    )
    if device is None:
        # Prefer existing graph tensors' device when present.
        if torch.is_tensor(getattr(data, "x", None)):
            device = data.x.device
        elif torch.is_tensor(getattr(data, "edge_index", None)):
            device = data.edge_index.device
        else:
            device = "cpu"
    data.hyperbolic_graph = True
    data.hyperbolic_edge_index = torch.tensor(ei, dtype=torch.long, device=device)
    data.hyperbolic_edge_attr = torch.tensor(ea, dtype=torch.float32, device=device)
    data.biology_edge_types = torch.tensor(et, dtype=torch.long, device=device)
    data.hyp_biology_mp = True
    data.allow_ca_fallback = False
    n = int(coords_np.shape[0])
    data.biology_mp_audit = build_biology_mp_audit(
        edge_index=data.hyperbolic_edge_index,
        edge_types=data.biology_edge_types,
        num_nodes=n,
        ca_in_mp=False,
        allow_ca_fallback=False,
    )
    data.biology_mp_meta = meta
    return data


__all__ = [
    "BIO_DEHYDRON",
    "BIO_EDGE_ATTR_DIM",
    "BIO_HBOND",
    "BIO_PI_STACK",
    "BIO_SALT_BRIDGE",
    "BIO_TYPE_TO_NAME",
    "attach_biology_mp_graph",
    "compute_biology_mp_graph",
    "detect_hbond_dehydron_pairs",
    "detect_pi_stacks",
    "detect_salt_bridges",
]
