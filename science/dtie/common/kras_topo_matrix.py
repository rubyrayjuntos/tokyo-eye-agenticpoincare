"""KRAS topo-structural triangulation helpers (Phase A primary matrix).

Spec: ``docs/specs/kras-topo-structural-inference/design.md``
Historical triad: 4OBE (WT GDP) / 4DSO (G12D GDP) / 5VQ2 (G12V GppNHp).
Four-quadrant rematch (edge ΔE closeout): 4LPK / 6GOD / 5US4 / 6GOF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PRIMARY_WT_INACTIVE = "4OBE"
PRIMARY_MUT_INACTIVE = "4DSO"
PRIMARY_ACTIVE = "5VQ2"

# Inhibitor-free four-quadrant roster (rewiring / mimetic-inactive closeout).
FOUR_Q_WT_OFF = "4LPK"
FOUR_Q_WT_ON = "6GOD"
FOUR_Q_MUT_OFF = "5US4"
FOUR_Q_MUT_ON = "6GOF"

# Locked canonical residue sequence numbers (deposited numbering).
CRITICAL_RESSEQ = (81, 114, 156, 163, 12)
SWITCH_LOCK_PAIRS = ((12, 32), (12, 61))  # Asp12–Tyr32, Asp12–Gln61 on 4DSO
COUPLED_LOCK_PAIRS = ((12, 32), (12, 60), (12, 61))  # report-only companion
# Per-pair Cα cutoffs (Å): Switch-I 12–32 relaxed to 11.0 for snapshot breathing.
SWITCH_LOCK_CA_CUTOFFS_A: dict[tuple[int, int], float] = {
    (12, 32): 11.0,
    (12, 61): 10.0,
}
SWITCH_LOCK_CA_CUTOFF_A = 11.0  # legacy default / max of pair cutoffs
CA_CONTACT_CUTOFF_A = 8.0
# Historical triad: Pass used dehydron-wrapper; Cα was observational.
# Four-quadrant rematch: Pass uses Cα @ 8 Å; complementarity is telemetry.


def parse_resseq(residue_id: Any) -> int | None:
    """Extract integer sequence number from residue_id variants."""
    import re

    if residue_id is None:
        return None
    if isinstance(residue_id, (int, np.integer)):
        return int(residue_id)
    s = str(residue_id).strip()
    if not s:
        return None
    # Forms: "A:12", "A:12:", "12", "ASP114", "12A"
    if ":" in s:
        parts = [p for p in s.split(":") if p]
        # Prefer last numeric-looking segment (chain:resseq[:icode])
        for part in reversed(parts):
            m = re.search(r"-?\d+", part)
            if m:
                return int(m.group(0))
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def residue_index_map(residue_ids: Sequence[Any]) -> dict[int, int]:
    """Map deposited resseq → graph index (first wins)."""
    out: dict[int, int] = {}
    for i, rid in enumerate(residue_ids):
        rs = parse_resseq(rid)
        if rs is None or rs in out:
            continue
        out[rs] = i
    return out


def align_resseq_vectors(
    values_by_structure: Mapping[str, Sequence[float]],
    resseq_maps: Mapping[str, Mapping[int, int]],
    *,
    structures: Sequence[str] | None = None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Intersect residue numbers; return shared resseqs + aligned value vectors."""
    structs = list(structures) if structures is not None else list(values_by_structure.keys())
    if len(structs) < 2:
        raise ValueError("need ≥2 structures to align")
    shared = set(resseq_maps[structs[0]].keys())
    for s in structs[1:]:
        shared &= set(resseq_maps[s].keys())
    shared_sorted = sorted(shared)
    if not shared_sorted:
        raise ValueError("empty residue intersection across structures")
    aligned: dict[str, np.ndarray] = {}
    for s in structs:
        vals = np.asarray(values_by_structure[s], dtype=np.float64).reshape(-1)
        idx_map = resseq_maps[s]
        aligned[s] = np.asarray([vals[idx_map[r]] for r in shared_sorted], dtype=np.float64)
    return shared_sorted, aligned


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation; NaN if undefined."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa, bb = a[mask], b[mask]
    ra = aa.argsort().argsort().astype(np.float64)
    rb = bb.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def ca_contact_edges_by_resseq(
    ca_coords: np.ndarray,
    residue_ids: Sequence[Any],
    *,
    cutoff_angstrom: float = CA_CONTACT_CUTOFF_A,
) -> set[tuple[int, int]]:
    """Undirected residue-number edges from Cα contacts."""
    coords = np.asarray(ca_coords, dtype=np.float64)
    n = min(coords.shape[0], len(residue_ids))
    idx_to_rs: dict[int, int] = {}
    for i in range(n):
        rs = parse_resseq(residue_ids[i])
        if rs is not None:
            idx_to_rs[i] = rs
    edges: set[tuple[int, int]] = set()
    cutoff = float(cutoff_angstrom)
    keys = sorted(idx_to_rs.keys())
    for ii, i in enumerate(keys):
        for j in keys[ii + 1 :]:
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 1e-8 < d <= cutoff:
                a, b = idx_to_rs[i], idx_to_rs[j]
                edges.add((a, b) if a <= b else (b, a))
    return edges


def edge_symdiff_size(
    edges_a: set[tuple[int, int]],
    edges_b: set[tuple[int, int]],
    *,
    shared_resseqs: Sequence[int] | None = None,
) -> int:
    r"""Cardinality of edge symmetric difference, optional shared-residue filter."""
    if shared_resseqs is not None:
        shared = set(shared_resseqs)

        def _filt(edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
            return {(u, v) for u, v in edges if u in shared and v in shared}

        edges_a, edges_b = _filt(edges_a), _filt(edges_b)
    return len(edges_a.symmetric_difference(edges_b))


def ca_distance(
    ca_coords: np.ndarray,
    residue_ids: Sequence[Any],
    resseq_a: int,
    resseq_b: int,
) -> float:
    """Cα–Cα distance between two residue numbers; NaN if missing."""
    idx_map = residue_index_map(residue_ids)
    if resseq_a not in idx_map or resseq_b not in idx_map:
        return float("nan")
    coords = np.asarray(ca_coords, dtype=np.float64)
    i, j = idx_map[resseq_a], idx_map[resseq_b]
    if i >= coords.shape[0] or j >= coords.shape[0]:
        return float("nan")
    return float(np.linalg.norm(coords[i] - coords[j]))


def dehydron_wrapper_edges_by_resseq(
    pdb_path: str | Path,
    chain_id: str = "A",
    *,
    tau: float | None = None,
    max_no_dist: float = 3.5,
) -> dict[str, set[tuple[int, int]]]:
    """Build dehydron / wrapped-H-bond edge sets keyed by residue number.

    Uses backbone N–O pairs (same extractor as barcode midpoints). Underwrapped
    pairs (``wrapping_count < tau``) are dehydrons; the rest of qualifying N–O
    pairs within ``max_no_dist`` are wrapped H-bonds. Complementarity graph =
    dehydron ∪ wrapped.
    """
    from pathlib import Path as _Path

    from science.dtie.common.dehydron_barcode_features import (
        TAU,
        WRAPPING_RADIUS,
        _load_chain_structure_atoms,
        _wrapping_count_at_midpoint,
        _residue_key,
    )

    if tau is None:
        tau = float(TAU)
    atoms, rmap = _load_chain_structure_atoms(_Path(pdb_path), chain_id)
    idx_to_resseq = {idx: key[1] for key, idx in rmap.items()}

    donors: list[tuple[int, np.ndarray]] = []
    acceptors: list[tuple[int, np.ndarray]] = []
    for atom in atoms:
        key = _residue_key(atom)
        if key is None or key not in rmap:
            continue
        idx = rmap[key]
        if atom.atom_name == "N" and atom.element == "N":
            donors.append((idx, np.asarray(atom.coord, dtype=np.float64)))
        elif atom.atom_name == "O" and atom.element == "O":
            acceptors.append((idx, np.asarray(atom.coord, dtype=np.float64)))

    dehydron: set[tuple[int, int]] = set()
    wrapped: set[tuple[int, int]] = set()
    for donor_idx, n_coord in donors:
        for acceptor_idx, o_coord in acceptors:
            if donor_idx == acceptor_idx:
                continue
            rs_d = idx_to_resseq.get(donor_idx)
            rs_a = idx_to_resseq.get(acceptor_idx)
            if rs_d is None or rs_a is None:
                continue
            if abs(int(rs_d) - int(rs_a)) < 1:
                continue
            distance = float(np.linalg.norm(n_coord - o_coord))
            if distance >= max_no_dist:
                continue
            midpoint = (n_coord + o_coord) / 2.0
            wrap = _wrapping_count_at_midpoint(
                midpoint, atoms, wrapping_radius=WRAPPING_RADIUS
            )
            edge = (rs_d, rs_a) if rs_d <= rs_a else (rs_a, rs_d)
            if wrap < float(tau):
                dehydron.add(edge)
            else:
                wrapped.add(edge)

    complementarity = set(dehydron) | set(wrapped)
    return {
        "dehydron": dehydron,
        "wrapped_hbond": wrapped,
        "complementarity": complementarity,
    }


def switch_lock_verdict(
    distances_mut: Mapping[tuple[int, int], float],
    distances_wt: Mapping[tuple[int, int], float],
    *,
    cutoffs_a: Mapping[tuple[int, int], float] | None = None,
    cutoff_a: float | None = None,
    min_shortening_a: float = 1.0,
) -> dict[str, Any]:
    """Pass if both tethers form on mutant and are absent/longer on WT."""
    pair_cutoffs = dict(SWITCH_LOCK_CA_CUTOFFS_A)
    if cutoffs_a is not None:
        pair_cutoffs.update(dict(cutoffs_a))
    pair_reports: list[dict[str, Any]] = []
    all_pass = True
    for pair in SWITCH_LOCK_PAIRS:
        lim = float(
            pair_cutoffs.get(
                pair, cutoff_a if cutoff_a is not None else SWITCH_LOCK_CA_CUTOFF_A
            )
        )
        d_mut = float(distances_mut.get(pair, float("nan")))
        d_wt = float(distances_wt.get(pair, float("nan")))
        forms_mut = bool(np.isfinite(d_mut) and d_mut <= lim)
        longer_or_absent_wt = bool(
            (not np.isfinite(d_wt))
            or d_wt > lim
            or (np.isfinite(d_mut) and d_wt - d_mut >= min_shortening_a)
        )
        ok = forms_mut and longer_or_absent_wt
        if not ok:
            all_pass = False
        pair_reports.append(
            {
                "pair": list(pair),
                "distance_mut_a": d_mut,
                "distance_wt_a": d_wt,
                "cutoff_a": lim,
                "forms_on_mut": forms_mut,
                "absent_or_longer_on_wt": longer_or_absent_wt,
                "pass": ok,
            }
        )
    return {
        "pass": all_pass,
        "cutoffs_a": {f"{a}-{b}": pair_cutoffs[(a, b)] for a, b in SWITCH_LOCK_PAIRS},
        "pairs": pair_reports,
    }


def betweenness_proximity_verdict(
    rho_mut_active: float,
    rho_wt_active: float,
) -> dict[str, Any]:
    """Pass: ρ(mut, active) > ρ(wt, active)."""
    ok = bool(
        np.isfinite(rho_mut_active)
        and np.isfinite(rho_wt_active)
        and rho_mut_active > rho_wt_active
    )
    return {
        "pass": ok,
        "rho_mut_active": float(rho_mut_active),
        "rho_wt_active": float(rho_wt_active),
        "delta_rho": float(rho_mut_active - rho_wt_active)
        if np.isfinite(rho_mut_active) and np.isfinite(rho_wt_active)
        else float("nan"),
        "rule": "rho(4DSO,5VQ2) > rho(4OBE,5VQ2)",
    }


def edge_delta_verdict(delta_e_mut_active: int, delta_e_wt_active: int) -> dict[str, Any]:
    """Historical triad rule: |ΔE(mut, active)| < |ΔE(wt, active)|.

    Superseded as a blocking Pass by ``four_quadrant_edge_verdict``
    (4LPK/6GOD/5US4/6GOF). Retained for telemetry on the 4OBE/4DSO/5VQ2 roster.
    """
    ok = int(delta_e_mut_active) < int(delta_e_wt_active)
    return {
        "pass": ok,
        "report_only": True,
        "superseded_by": "four_quadrant_edge_delta",
        "delta_e_mut_active": int(delta_e_mut_active),
        "delta_e_wt_active": int(delta_e_wt_active),
        "rule": "|ΔE(4DSO,5VQ2)| < |ΔE(4OBE,5VQ2)| (historical; report-only)",
    }


def four_quadrant_edge_verdict(
    *,
    delta_e_mimetic: int,
    delta_e_wt_to_active: int,
    delta_e_mut_span: int,
    delta_e_wt_span: int,
) -> dict[str, Any]:
    """Four-quadrant rewiring Pass bars (Cα @ 8 Å primary).

    1. Mimetic inactive: |ΔE(5US4, 6GOD)| < |ΔE(4LPK, 6GOD)|
    2. Compressed mut span: |ΔE(5US4, 6GOF)| < |ΔE(4LPK, 6GOD)|
    """
    mimetic_ok = int(delta_e_mimetic) < int(delta_e_wt_to_active)
    span_ok = int(delta_e_mut_span) < int(delta_e_wt_span)
    ok = mimetic_ok and span_ok
    return {
        "pass": ok,
        "gates": {
            "mimetic_inactive": mimetic_ok,
            "compressed_mut_span": span_ok,
        },
        "delta_e_mimetic": int(delta_e_mimetic),
        "delta_e_wt_to_active": int(delta_e_wt_to_active),
        "delta_e_mut_span": int(delta_e_mut_span),
        "delta_e_wt_span": int(delta_e_wt_span),
        "roster": {
            "wt_off": FOUR_Q_WT_OFF,
            "wt_on": FOUR_Q_WT_ON,
            "mut_off": FOUR_Q_MUT_OFF,
            "mut_on": FOUR_Q_MUT_ON,
        },
        "rules": [
            f"|ΔE({FOUR_Q_MUT_OFF},{FOUR_Q_WT_ON})| < |ΔE({FOUR_Q_WT_OFF},{FOUR_Q_WT_ON})|",
            f"|ΔE({FOUR_Q_MUT_OFF},{FOUR_Q_MUT_ON})| < |ΔE({FOUR_Q_WT_OFF},{FOUR_Q_WT_ON})|",
        ],
        "outcome": "Pass" if ok else "Fail",
    }


def primary_matrix_verdict(
    *,
    betweenness: Mapping[str, Any],
    edge_delta: Mapping[str, Any],
    switch_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate Phase A gates (conductance hub migration tracked separately).

    Edge ΔE on the historical triad is **report-only** (superseded by the
    four-quadrant rematch). Blocking Pass = betweenness ∧ switch_lock.
    """
    blocking = {
        "betweenness_proximity": bool(betweenness.get("pass")),
        "switch_lock_tethers": bool(switch_lock.get("pass")),
    }
    report_only = {
        "edge_symmetric_difference": bool(edge_delta.get("pass")),
        "edge_status": "report_only_superseded_by_four_quadrant",
    }
    ok = all(blocking.values())
    return {
        "pass": ok,
        "gates": {**blocking, "edge_symmetric_difference": report_only["edge_symmetric_difference"]},
        "blocking_gates": blocking,
        "report_only": report_only,
        "outcome": "Pass" if ok else "Fail",
        "note": (
            "edge_symmetric_difference on 4OBE/4DSO/5VQ2 is report-only; "
            "rewiring closeout is four_quadrant_edge_delta (4LPK/6GOD/5US4/6GOF)"
        ),
    }


__all__ = [
    "COUPLED_LOCK_PAIRS",
    "FOUR_Q_MUT_OFF",
    "FOUR_Q_MUT_ON",
    "FOUR_Q_WT_OFF",
    "FOUR_Q_WT_ON",
    "PRIMARY_ACTIVE",
    "PRIMARY_MUT_INACTIVE",
    "PRIMARY_WT_INACTIVE",
    "SWITCH_LOCK_PAIRS",
    "SWITCH_LOCK_CA_CUTOFF_A",
    "SWITCH_LOCK_CA_CUTOFFS_A",
    "align_resseq_vectors",
    "betweenness_proximity_verdict",
    "ca_contact_edges_by_resseq",
    "ca_distance",
    "dehydron_wrapper_edges_by_resseq",
    "edge_delta_verdict",
    "edge_symdiff_size",
    "four_quadrant_edge_verdict",
    "parse_resseq",
    "primary_matrix_verdict",
    "residue_index_map",
    "spearman_rho",
    "switch_lock_verdict",
]
