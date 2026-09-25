"""TokyoEye-v8 R0–R5 dual-layer biophysical communication graph (FROZEN).

v8-isolated loader (``science.tokyo_eye.v8``). Does **not** rewrite PDB ingest / Normalizer /
``fact_graph_edge``. Spec:
``docs/superpowers/specs/2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md`` §5.

Layers
------
* **Layer A (R0):** always emit peptide ±1 directed pairs.
* **Layer B (R1–R5):** descending exclusivity R1/R2 → R3 → R4 → R5;
  one primary type per undirected pair; secondary chemistry → attr flags.
* R0 **coexists** with Layer B (dual-layer multiplexing).
* Every undirected relation emits ``(i→j)`` and ``(j→i)`` with identical type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from science.tokyo_eye.v8.biophysics import (
    DSSP_ENERGY_CUTOFF,
    HBOND_SEQ_CUTOFF,
    HBOND_SPATIAL_CUTOFF,
    WRAP_CONE_COS,
    WRAP_CONE_HALF_ANGLE_DEG,
    compute_bond_wrapping_count,
    detect_pi_stacks,
    detect_salt_bridges,
    kabsch_sander_energy,
    parse_residue_records_from_pdb_chain,
    place_backbone_amide_h,
    wrapping_carbon_coords,
)
from science.tokyo_eye.v8.types import ResidueRecord

# --- Frozen relation IDs -------------------------------------------------------
R0_COVALENT = 0
R1_HBOND = 1
R2_DEHYDRON = 2
R3_HYDROPHOBIC_PI = 3
R4_SALT_BRIDGE = 4
R5_LOCAL_NEIGHBORHOOD = 5

R_TYPE_TO_NAME: dict[int, str] = {
    R0_COVALENT: "covalent",
    R1_HBOND: "hbond",
    R2_DEHYDRON: "dehydron",
    R3_HYDROPHOBIC_PI: "hydrophobic_pi",
    R4_SALT_BRIDGE: "salt_bridge",
    R5_LOCAL_NEIGHBORHOOD: "local_neighborhood",
}

# Dehydron wrap gate (edge classification) — not node τ TAU=13.
# AMEND 2026-09-17 (tokyo_eye_equ_wrap_threshold): Stage-A-12 median-then-descend
# under double-cone wraps → 1. Classical Fernández 19 was sphere-calibrated.
DEHYDRON_WRAP_MAX = 1
WRAPPING_RADIUS_A = 6.5
CB_NEIGHBORHOOD_A = 8.0
PI_CENTROID_MAX_A = 5.0
HYDROPHOBIC_CENTROID_MAX_A = 5.0
# Model-facing (decoupled input) R3 reach. Label typing and the SDRP target keep the
# frozen 5.0 A above: raising the global constant re-labels ~9.5% of residues' SDRP
# target (322/3393 over the 12 LOSO structures, measured 2026-09-25).
HYDROPHOBIC_CENTROID_INPUT_MAX_A = 6.5

# Frozen wrap (addendum §2.7 AMEND). Runtime retune is forbidden.
_ACTIVE_DEHYDRON_WRAP_MAX = DEHYDRON_WRAP_MAX

HYDROPHOBIC_RESIDUES = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO"}
)
_BACKBONE = frozenset({"N", "CA", "C", "O", "H", "HA"})

# edge_attr: [wrap_or_0, dist, secondary_flags, aux, type_code]
EDGE_ATTR_DIM = 5


@dataclass(frozen=True)
class R0R5GraphResult:
    """Sparse typed graph tensors (numpy) + audit meta."""

    edge_index: np.ndarray  # [2, E] int64
    edge_type: np.ndarray  # [E] int64
    edge_attr: np.ndarray  # [E, EDGE_ATTR_DIM] float32
    num_nodes: int
    meta: dict[str, Any]
    # Supervision-side typing (R2 by wrap gate). ``None`` => same as the input
    # graph (legacy). Set only when ``decouple_r2_input=True`` -- see builder.
    label_edge_index: np.ndarray | None = None
    label_edge_type: np.ndarray | None = None


def get_dehydron_wrap_max() -> int:
    return int(_ACTIVE_DEHYDRON_WRAP_MAX)


def set_dehydron_wrap_max(value: int) -> None:
    """Freeze §5 wrap is DEHYDRON_WRAP_MAX. Any other value is forbidden retune."""
    global _ACTIVE_DEHYDRON_WRAP_MAX
    v = int(value)
    if v != int(DEHYDRON_WRAP_MAX):
        raise ValueError(
            f"dehydron wrap is frozen at {DEHYDRON_WRAP_MAX}; "
            f"runtime retune to {v} is forbidden "
            "(tokyo_eye_equ_wrap_threshold / addendum §2.7)"
        )
    _ACTIVE_DEHYDRON_WRAP_MAX = v


def classify_hbond_vs_dehydron(
    wrap_count: float,
    *,
    wrap_max: int | None = None,
) -> int:
    """≤ wrap_max wrapping non-polar carbons → R2; else stable R1."""
    if wrap_count < 0:
        raise ValueError("wrap_count must be ≥0 for an atom-validated H-bond")
    thr = float(get_dehydron_wrap_max() if wrap_max is None else wrap_max)
    if float(wrap_count) <= thr:
        return R2_DEHYDRON
    return R1_HBOND

def resolve_layer_b_primary(
    *,
    has_hbond: bool,
    is_dehydron: bool,
    has_pi_or_hydrophobic: bool,
    has_salt: bool,
    has_r5_neighborhood: bool,
    salt_first: bool = False,
) -> tuple[int | None, int]:
    """Descending R1/R2 → R3 → R4 → R5. Returns (primary_or_None, secondary_flags).

    ``secondary_flags`` bit ``(1 << R)`` is set for every Layer-B relation that
    qualified, including the primary.
    """
    flags = 0
    if has_hbond:
        flags |= 1 << (R2_DEHYDRON if is_dehydron else R1_HBOND)
    if has_pi_or_hydrophobic:
        flags |= 1 << R3_HYDROPHOBIC_PI
    if has_salt:
        flags |= 1 << R4_SALT_BRIDGE
    if has_r5_neighborhood:
        flags |= 1 << R5_LOCAL_NEIGHBORHOOD

    if salt_first and has_salt:
        # Input-side only (decoupled graph): a salt bridge is not swallowed by a
        # coincident H-bond. Label typing never sets this, so R1/R2 labels and
        # the SDRP target keep the frozen descending exclusivity.
        return R4_SALT_BRIDGE, flags
    if has_hbond:
        primary = R2_DEHYDRON if is_dehydron else R1_HBOND
        return primary, flags
    if has_pi_or_hydrophobic:
        return R3_HYDROPHOBIC_PI, flags
    if has_salt:
        return R4_SALT_BRIDGE, flags
    if has_r5_neighborhood:
        return R5_LOCAL_NEIGHBORHOOD, flags
    return None, 0


def assert_layer_b_exclusivity(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
) -> None:
    """Property: at most one Layer-B type per undirected pair; R0 may coexist."""
    ei = np.asarray(edge_index, dtype=np.int64)
    et = np.asarray(edge_type, dtype=np.int64).reshape(-1)
    layer_b: dict[tuple[int, int], set[int]] = {}
    for k in range(et.shape[0]):
        i, j, r = int(ei[0, k]), int(ei[1, k]), int(et[k])
        if r == R0_COVALENT:
            continue
        if r < R1_HBOND or r > R5_LOCAL_NEIGHBORHOOD:
            raise AssertionError(f"edge_type {r} outside R0–R5")
        key = (i, j) if i < j else (j, i)
        layer_b.setdefault(key, set()).add(r)
    for key, types in layer_b.items():
        if len(types) != 1:
            raise AssertionError(
                f"Layer-B exclusivity violated for {key}: types={sorted(types)}"
            )


def edge_type_directed_fractions(
    edge_type: np.ndarray, *, num_relations: int = 6
) -> dict[str, float]:
    """Directed edge-type fractions for R0–R5 (logged every run)."""
    et = np.asarray(edge_type, dtype=np.int64).reshape(-1)
    out: dict[str, float] = {}
    for r in range(int(num_relations)):
        out[f"edge_frac_r{r}"] = float(np.mean(et == r)) if et.size else 0.0
    out["n_edges_directed"] = float(et.shape[0])
    return out


def chemistry_gate_features(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Per-node ``[ρ, τ, ss_H, ss_E, ss_C, degree, sasa]`` for the MoE gate.

    ρ: incident R1 (wrapped H-bond) fraction of degree.
    τ: incident R2 (dehydron) fraction of degree.
    ss_*: crude CA-geometry helix/sheet/coil one-hot (not DSSP assign).
    sasa: inverse CA-neighborhood density within 8 Å, scaled to (0,1].
    """
    n = int(num_nodes)
    ei = np.asarray(edge_index, dtype=np.int64)
    et = np.asarray(edge_type, dtype=np.int64).reshape(-1)
    deg = np.zeros(n, dtype=np.float64)
    n_r1 = np.zeros(n, dtype=np.float64)
    n_r2 = np.zeros(n, dtype=np.float64)
    if et.size:
        dst = ei[1]
        for k in range(et.shape[0]):
            j = int(dst[k])
            if j < 0 or j >= n:
                continue
            deg[j] += 1.0
            if int(et[k]) == R1_HBOND:
                n_r1[j] += 1.0
            elif int(et[k]) == R2_DEHYDRON:
                n_r2[j] += 1.0
    denom = np.maximum(deg, 1.0)
    rho = n_r1 / denom
    tau = n_r2 / denom
    ss = _ca_ss_onehot(np.asarray(coords, dtype=np.float64), n)
    sasa = _ca_sasa_proxy(np.asarray(coords, dtype=np.float64), n)
    return np.stack(
        [rho, tau, ss[:, 0], ss[:, 1], ss[:, 2], deg, sasa], axis=1
    ).astype(np.float32)


def _ca_ss_onehot(coords: np.ndarray, n: int) -> np.ndarray:
    """Very small CA-angle heuristic: coil default; helix-like if 70–120°; sheet if >140°."""
    ss = np.zeros((n, 3), dtype=np.float64)
    ss[:, 2] = 1.0
    if coords.shape[0] != n or n < 3:
        return ss
    for i in range(1, n - 1):
        a = coords[i - 1] - coords[i]
        b = coords[i + 1] - coords[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            continue
        ang = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))
        ss[i] = 0.0
        if 70.0 <= ang <= 120.0:
            ss[i, 0] = 1.0
        elif ang >= 140.0:
            ss[i, 1] = 1.0
        else:
            ss[i, 2] = 1.0
    return ss


def _ca_sasa_proxy(coords: np.ndarray, n: int, radius: float = 8.0) -> np.ndarray:
    if coords.shape[0] != n or n == 0:
        return np.ones(n, dtype=np.float64)
    sasa = np.ones(n, dtype=np.float64)
    r2 = float(radius) ** 2
    for i in range(n):
        d2 = np.sum((coords - coords[i]) ** 2, axis=1)
        neigh = int(np.sum((d2 > 1e-8) & (d2 <= r2)))
        sasa[i] = 1.0 / (1.0 + float(neigh))
    return sasa


def directed_pairs_for_type(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    type_code: int,
) -> list[tuple[int, int]]:
    """List directed ``(src, dst)`` rows matching ``type_code``."""
    ei = np.asarray(edge_index, dtype=np.int64)
    et = np.asarray(edge_type, dtype=np.int64).reshape(-1)
    out: list[tuple[int, int]] = []
    for k in range(et.shape[0]):
        if int(et[k]) == int(type_code):
            out.append((int(ei[0, k]), int(ei[1, k])))
    return out


def _append_bidir(
    src: list[int],
    dst: list[int],
    types: list[int],
    attrs: list[list[float]],
    i: int,
    j: int,
    type_code: int,
    *,
    wrap: float = 0.0,
    dist: float = 0.0,
    flags: int = 0,
    aux: float = 0.0,
) -> None:
    row = [float(wrap), float(dist), float(flags), float(aux), float(type_code)]
    src.extend([i, j])
    dst.extend([j, i])
    types.extend([type_code, type_code])
    attrs.append(row)
    attrs.append(list(row))


def _ca_coord(rec: ResidueRecord) -> np.ndarray | None:
    a = rec.get_atom("CA")
    if a is None:
        return None
    return np.asarray(a.coord, dtype=np.float64).reshape(3)


def _cb_coord(rec: ResidueRecord) -> np.ndarray | None:
    cb = rec.get_atom("CB")
    if cb is not None:
        return np.asarray(cb.coord, dtype=np.float64).reshape(3)
    return _ca_coord(rec)


def _sidechain_centroid(rec: ResidueRecord) -> np.ndarray | None:
    pts: list[np.ndarray] = []
    for atom in rec.atoms:
        if atom.atom_name.strip().upper() in _BACKBONE:
            continue
        if atom.element.strip().upper() == "H":
            continue
        pts.append(np.asarray(atom.coord, dtype=np.float64).reshape(3))
    if not pts:
        return _cb_coord(rec)
    return np.mean(np.stack(pts, axis=0), axis=0)


def _detect_backbone_hbonds(
    records: Sequence[ResidueRecord],
    *,
    index_by_auth: Mapping[int, int],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Undirected ``(i,j) → wrap_count`` + energies for DSSP-gated H-bonds.

    Sprint 8: admit only Kabsch–Sander ``E ≤ -0.5`` pairs; wrap via double cone.
    """
    del index_by_auth  # records are already in graph index order
    n = len(records)
    coords = np.zeros((n, 3), dtype=np.float64)
    seq = np.zeros(n, dtype=np.int64)
    for i, rec in enumerate(records):
        ca = _ca_coord(rec)
        if ca is None:
            raise ValueError(f"residue {rec.residue_index} missing CA")
        coords[i] = ca
        seq[i] = int(rec.residue_index)

    all_atoms = [a for r in records for a in r.atoms]
    carbon_coords = wrapping_carbon_coords(all_atoms)
    by_seq = {int(r.residue_index): r for r in records}

    wraps: dict[tuple[int, int], float] = {}
    energies: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            seq_sep = abs(int(seq[i]) - int(seq[j]))
            if seq_sep > HBOND_SEQ_CUTOFF or d >= HBOND_SPATIAL_CUTOFF:
                continue
            # Try both donor/acceptor orientations; keep most negative energy.
            best_energy: float | None = None
            best_wrap = -1.0
            for di, aj in ((i, j), (j, i)):
                donor = records[di]
                acceptor = records[aj]
                prev = None
                dseq = int(donor.residue_index)
                prev_rec = by_seq.get(dseq - 1)
                if prev_rec is not None and prev_rec.chain_label == donor.chain_label:
                    prev = prev_rec
                h = place_backbone_amide_h(donor, prev)
                energy = kabsch_sander_energy(
                    donor, acceptor, h_coord=h, prev_donor=prev
                )
                if energy is None or energy > float(DSSP_ENERGY_CUTOFF):
                    continue
                wrap = compute_bond_wrapping_count(
                    donor,
                    acceptor,
                    all_atoms,
                    wrapping_radius=WRAPPING_RADIUS_A,
                    carbon_coords=carbon_coords,
                    h_coord=h,
                    prev_donor=prev,
                    use_double_cone=True,
                )
                if wrap < 0:
                    continue
                if best_energy is None or energy < best_energy:
                    best_energy = float(energy)
                    best_wrap = float(wrap)
            if best_energy is not None and best_wrap >= 0:
                key = (i, j)
                wraps[key] = best_wrap
                energies[key] = best_energy
    return wraps, energies

def _detect_hydrophobic_pairs(
    records: Sequence[ResidueRecord],
    *,
    index_by_auth: Mapping[int, int],
    max_dist: float = HYDROPHOBIC_CENTROID_MAX_A,
) -> dict[tuple[int, int], float]:
    cents: list[tuple[int, np.ndarray]] = []
    for rec in records:
        name = (rec.residue_name or "").strip().upper()
        if name not in HYDROPHOBIC_RESIDUES:
            continue
        idx = index_by_auth.get(int(rec.residue_index))
        if idx is None:
            continue
        c = _sidechain_centroid(rec)
        if c is None:
            continue
        cents.append((idx, c))
    out: dict[tuple[int, int], float] = {}
    for a in range(len(cents)):
        i, ci = cents[a]
        for b in range(a + 1, len(cents)):
            j, cj = cents[b]
            if i == j:
                continue
            d = float(np.linalg.norm(ci - cj))
            if d <= float(max_dist):
                key = (i, j) if i < j else (j, i)
                out[key] = d
    return out


def _detect_r5_pairs(
    records: Sequence[ResidueRecord],
    *,
    max_dist: float = CB_NEIGHBORHOOD_A,
) -> dict[tuple[int, int], float]:
    cbs: list[tuple[int, np.ndarray]] = []
    for i, rec in enumerate(records):
        cb = _cb_coord(rec)
        if cb is None:
            continue
        cbs.append((i, cb))
    out: dict[tuple[int, int], float] = {}
    for a in range(len(cbs)):
        i, ci = cbs[a]
        for b in range(a + 1, len(cbs)):
            j, cj = cbs[b]
            d = float(np.linalg.norm(ci - cj))
            if d <= float(max_dist):
                key = (i, j) if i < j else (j, i)
                out[key] = d
    return out


def build_r0_r5_graph(
    residue_records: Sequence[ResidueRecord],
    *,
    pi_centroid_max: float = PI_CENTROID_MAX_A,
    decouple_r2_input: bool = False,
) -> R0R5GraphResult:
    """Build dual-layer R0–R5 sparse graph from residue atom records.

    ``decouple_r2_input=True`` removes the dehydron target rule from the graph the
    model sees: every atom-validated H-bond is typed R1 (never R2), the R2 bit is
    never set in ``secondary_flags``, and the raw wrap count is not written to
    ``edge_attr[0]``. The wrap-gated R2 typing is still computed and returned as
    ``label_edge_index`` / ``label_edge_type`` for label generation only. In this
    mode the model-facing typing also (a) lets a salt bridge outrank a coincident
    H-bond (R4 over R1) and (b) reaches R3 hydrophobic contacts out to
    ``HYDROPHOBIC_CENTROID_INPUT_MAX_A``; label typing keeps the frozen rules.
    """
    records = [r for r in residue_records if r.get_atom("CA") is not None]
    n = len(records)
    if n == 0:
        return R0R5GraphResult(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_type=np.zeros((0,), dtype=np.int64),
            edge_attr=np.zeros((0, EDGE_ATTR_DIM), dtype=np.float32),
            num_nodes=0,
            meta={"num_nodes": 0},
        )

    index_by_auth = {int(r.residue_index): i for i, r in enumerate(records)}
    if len(index_by_auth) != n:
        raise ValueError("duplicate residue_index in residue_records")

    src: list[int] = []
    dst: list[int] = []
    types: list[int] = []
    attrs: list[list[float]] = []
    # Label-side typing (legacy R2-by-wrap). Only populated when decoupling.
    l_src: list[int] = []
    l_dst: list[int] = []
    l_types: list[int] = []
    l_attrs: list[list[float]] = []

    # ----- Layer A: R0 peptide ±1 ------------------------------------------------
    r0_undirected = 0
    for i in range(n):
        for j in range(i + 1, n):
            if records[i].chain_label != records[j].chain_label:
                continue
            if abs(int(records[i].residue_index) - int(records[j].residue_index)) != 1:
                continue
            _append_bidir(src, dst, types, attrs, i, j, R0_COVALENT)
            if decouple_r2_input:
                _append_bidir(l_src, l_dst, l_types, l_attrs, i, j, R0_COVALENT)
            r0_undirected += 1

    # ----- Layer B candidates ----------------------------------------------------
    hbonds, hbond_energies = _detect_backbone_hbonds(
        records, index_by_auth=index_by_auth
    )
    salts = {
        ((i, j) if i < j else (j, i)): d
        for i, j, _cp, d in detect_salt_bridges(records, index_by_auth=index_by_auth)
    }
    pi_raw = detect_pi_stacks(
        records,
        index_by_auth=index_by_auth,
        max_centroid_dist=float(pi_centroid_max),
    )
    pi_pairs = {
        ((i, j) if i < j else (j, i)): (d, dih, align)
        for i, j, d, dih, align in pi_raw
    }
    hydro = _detect_hydrophobic_pairs(records, index_by_auth=index_by_auth)
    if decouple_r2_input:
        hydro_in = _detect_hydrophobic_pairs(
            records,
            index_by_auth=index_by_auth,
            max_dist=HYDROPHOBIC_CENTROID_INPUT_MAX_A,
        )
    else:
        hydro_in = hydro
    r5 = _detect_r5_pairs(records)

    candidate_keys: set[tuple[int, int]] = set()
    candidate_keys |= set(hbonds)
    candidate_keys |= set(salts)
    candidate_keys |= set(pi_pairs)
    candidate_keys |= set(hydro_in)  # superset of hydro (6.5 A reach >= 5.0 A)
    candidate_keys |= set(r5)

    counts = {
        "r0": r0_undirected,
        "r1": 0,
        "r2": 0,
        "r3": 0,
        "r4": 0,
        "r5": 0,
    }
    primary_name = {
        R1_HBOND: "r1",
        R2_DEHYDRON: "r2",
        R3_HYDROPHOBIC_PI: "r3",
        R4_SALT_BRIDGE: "r4",
        R5_LOCAL_NEIGHBORHOOD: "r5",
    }

    for key in sorted(candidate_keys):
        i, j = key
        wrap = hbonds.get(key)
        has_hbond = wrap is not None
        is_dehydron = False
        if has_hbond:
            assert wrap is not None
            is_dehydron = classify_hbond_vs_dehydron(wrap) == R2_DEHYDRON
        has_pi = key in pi_pairs or key in hydro
        has_pi_in = key in pi_pairs or key in hydro_in
        has_salt = key in salts
        has_r5n = key in r5
        primary, flags = resolve_layer_b_primary(
            has_hbond=has_hbond,
            is_dehydron=is_dehydron,
            has_pi_or_hydrophobic=has_pi,
            has_salt=has_salt,
            has_r5_neighborhood=has_r5n,
        )
        if primary is None and not decouple_r2_input:
            continue
        label_primary = primary
        if decouple_r2_input:
            if primary is not None:
                _append_bidir(l_src, l_dst, l_types, l_attrs, i, j, primary)
            # Input typing: same resolution with the dehydron rule switched off.
            primary, flags = resolve_layer_b_primary(
                has_hbond=has_hbond,
                is_dehydron=False,
                has_pi_or_hydrophobic=has_pi_in,
                has_salt=has_salt,
                has_r5_neighborhood=has_r5n,
                salt_first=True,
            )
            if primary is None:
                continue
        dist = 0.0
        aux = 0.0
        if primary in (R1_HBOND, R2_DEHYDRON):
            ca_i, ca_j = _ca_coord(records[i]), _ca_coord(records[j])
            if ca_i is not None and ca_j is not None:
                dist = float(np.linalg.norm(ca_i - ca_j))
        elif primary == R3_HYDROPHOBIC_PI:
            if key in pi_pairs:
                dist = float(pi_pairs[key][0])
                aux = float(pi_pairs[key][1])
            else:
                dist = float(hydro_in[key])
        elif primary == R4_SALT_BRIDGE:
            dist = float(salts[key])
        elif primary == R5_LOCAL_NEIGHBORHOOD:
            dist = float(r5[key])

        _append_bidir(
            src,
            dst,
            types,
            attrs,
            i,
            j,
            primary,
            wrap=(
                float(wrap) if (wrap is not None and not decouple_r2_input) else 0.0
            ),
            dist=dist,
            flags=flags,
            aux=aux,
        )
        if label_primary is not None:
            counts[primary_name[label_primary]] += 1

    if src:
        edge_index = np.stack(
            [np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)]
        )
        edge_type = np.asarray(types, dtype=np.int64)
        edge_attr = np.asarray(attrs, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_ATTR_DIM), dtype=np.float32)

    label_edge_index: np.ndarray | None = None
    label_edge_type: np.ndarray | None = None
    if decouple_r2_input:
        if l_src:
            label_edge_index = np.stack(
                [np.asarray(l_src, dtype=np.int64), np.asarray(l_dst, dtype=np.int64)]
            )
            label_edge_type = np.asarray(l_types, dtype=np.int64)
        else:
            label_edge_index = np.zeros((2, 0), dtype=np.int64)
            label_edge_type = np.zeros((0,), dtype=np.int64)
        assert_layer_b_exclusivity(label_edge_index, label_edge_type)
        # Input graph must never carry the dehydron class.
        assert not bool(np.any(edge_type == R2_DEHYDRON))

    wrap_counts = [float(v) for v in hbonds.values()]
    meta = {
        "decouple_r2_input": bool(decouple_r2_input),
        "num_nodes": n,
        "undirected_counts": counts,
        "n_edges_directed": int(edge_type.shape[0]),
        "dehydron_wrap_max": get_dehydron_wrap_max(),
        "wrapping_radius_a": WRAPPING_RADIUS_A,
        "dssp_energy_cutoff": float(DSSP_ENERGY_CUTOFF),
        "wrap_cone_half_angle_deg": float(WRAP_CONE_HALF_ANGLE_DEG),
        "wrap_cone_cos": float(WRAP_CONE_COS),
        "hbond_wrap_counts": wrap_counts,
        "hbond_energy_mean": (
            float(np.mean(list(hbond_energies.values()))) if hbond_energies else None
        ),
        "n_r0": int(counts["r0"]),
        "n_r1": int(counts["r1"]),
        "n_r2": int(counts["r2"]),
        "n_r3": int(counts["r3"]),
        "n_r4": int(counts["r4"]),
        "n_r5": int(counts["r5"]),
        **edge_type_directed_fractions(edge_type),
    }
    assert_layer_b_exclusivity(edge_index, edge_type)
    return R0R5GraphResult(
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        num_nodes=n,
        meta=meta,
        label_edge_index=label_edge_index,
        label_edge_type=label_edge_type,
    )


def load_r0_r5_from_pdb(
    pdb_path: str | Path,
    chain: str,
) -> R0R5GraphResult:
    """Parse a PDB chain into residue records and build the R0–R5 graph.

    Train/inference entry point. Does not call ingest or Normalizer.
    Drops incomplete residues that lack a Cα (common at chain termini).
    """
    records = parse_residue_records_from_pdb_chain(Path(pdb_path), chain)
    records = [r for r in records if r.get_atom("CA") is not None]
    if not records:
        raise ValueError(f"no CA-complete residues in {pdb_path} chain {chain}")
    return build_r0_r5_graph(records)


def attach_r0_r5_graph(
    data: Data,
    residue_records: Sequence[ResidueRecord],
    *,
    device: torch.device | str | None = None,
) -> Data:
    """Attach v8 R0–R5 tensors onto a PyG ``Data`` (model-side only)."""
    result = build_r0_r5_graph(residue_records)
    if device is None:
        if torch.is_tensor(getattr(data, "x", None)):
            device = data.x.device
        else:
            device = "cpu"
    data.edge_index = torch.tensor(result.edge_index, dtype=torch.long, device=device)
    data.edge_type = torch.tensor(result.edge_type, dtype=torch.long, device=device)
    data.edge_attr = torch.tensor(result.edge_attr, dtype=torch.float32, device=device)
    data.r0_r5_meta = result.meta
    data.r0_r5_graph = True
    return data


__all__ = [
    "CB_NEIGHBORHOOD_A",
    "DEHYDRON_WRAP_MAX",
    "EDGE_ATTR_DIM",
    "HYDROPHOBIC_CENTROID_MAX_A",
    "HYDROPHOBIC_CENTROID_INPUT_MAX_A",
    "PI_CENTROID_MAX_A",
    "R0_COVALENT",
    "R0R5GraphResult",
    "R1_HBOND",
    "R2_DEHYDRON",
    "R3_HYDROPHOBIC_PI",
    "R4_SALT_BRIDGE",
    "R5_LOCAL_NEIGHBORHOOD",
    "R_TYPE_TO_NAME",
    "WRAPPING_RADIUS_A",
    "attach_r0_r5_graph",
    "assert_layer_b_exclusivity",
    "build_r0_r5_graph",
    "chemistry_gate_features",
    "classify_hbond_vs_dehydron",
    "directed_pairs_for_type",
    "edge_type_directed_fractions",
    "get_dehydron_wrap_max",
    "load_r0_r5_from_pdb",
    "resolve_layer_b_primary",
    "set_dehydron_wrap_max",
]
