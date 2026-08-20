"""KRAS basin observation classify helpers (sensor + referee smoke).

Pre-reg: docs/specs/kras-topo-structural-inference/kras-basin-observation-classify-prereg.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.kras_g12_graft import (
    SWITCH_I_RANGE,
    SWITCH_II_RANGE,
    neighborhood_n12,
)
from science.dtie.common.kras_topo_matrix import residue_index_map, spearman_rho

# Roster basin ids → expected guards
ROSTER: dict[str, dict[str, str]] = {
    "4LPK": {"basin": "OFF_WT", "nucleotide": "GDP", "allele": "G12", "chain": "A"},
    "5US4": {"basin": "OFF_MUT", "nucleotide": "GDP", "allele": "G12D", "chain": "A"},
    "6GOD": {"basin": "ON_WT", "nucleotide": "GppNHp", "allele": "G12", "chain": "A"},
    "6GOF": {"basin": "ON_MUT", "nucleotide": "GppNHp", "allele": "G12D", "chain": "A"},
}

# Deposit hetero names → nucleotide class
_OFF_LIGANDS = frozenset({"GDP"})
_ON_LIGANDS = frozenset({"GNP", "GSP", "GTP", "GTC", "GCP"})  # GNP = GppNHp


def parse_nucleotide_class(pdb_path: str | Path) -> str:
    """Return ``GDP`` or ``GppNHp`` from HETATM residue names; raise if ambiguous/missing."""
    path = Path(pdb_path)
    names: set[str] = set()
    with path.open() as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            # columns 18-20 resname (1-based PDB); Python slice [17:20]
            resname = line[17:20].strip().upper()
            if resname in _OFF_LIGANDS or resname in _ON_LIGANDS:
                names.add(resname)
    off = names & _OFF_LIGANDS
    on = names & _ON_LIGANDS
    if off and on:
        raise ValueError(f"{path.name}: both GDP and ON-ligands present: {sorted(names)}")
    if off and not on:
        return "GDP"
    if on and not off:
        return "GppNHp"
    raise ValueError(f"{path.name}: no GDP/GppNHp-class ligand found")


def parse_g12_allele(pdb_path: str | Path, *, chain: str = "A") -> str:
    """Return ``G12`` (Gly) or ``G12D`` (Asp) from ATOM lines at resseq 12."""
    path = Path(pdb_path)
    found: set[str] = set()
    with path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 26:
                continue
            ch = line[21]
            if ch != chain:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq != 12:
                continue
            resname = line[17:20].strip().upper()
            found.add(resname)
    if not found:
        raise ValueError(f"{path.name}: missing residue 12 on chain {chain}")
    if found == {"GLY"}:
        return "G12"
    if found == {"ASP"}:
        return "G12D"
    raise ValueError(f"{path.name}: residue 12 names {sorted(found)}; expected GLY or ASP")


def check_structure_guards(
    pdb_id: str,
    pdb_path: str | Path,
    *,
    chain: str = "A",
) -> dict[str, Any]:
    """Parse guards and compare to locked roster; return pass/fail detail."""
    expected = ROSTER[pdb_id]
    nuc = parse_nucleotide_class(pdb_path)
    allele = parse_g12_allele(pdb_path, chain=chain)
    ok = nuc == expected["nucleotide"] and allele == expected["allele"]
    return {
        "pdb_id": pdb_id,
        "pass": ok,
        "parsed": {"nucleotide": nuc, "allele": allele, "chain": chain},
        "expected": {
            "nucleotide": expected["nucleotide"],
            "allele": expected["allele"],
            "basin": expected["basin"],
        },
    }


def switch_resseqs() -> list[int]:
    return sorted(set(SWITCH_I_RANGE) | set(SWITCH_II_RANGE))


def r_star_for_structure(
    present: set[int],
    n12_arm: list[int],
) -> list[int]:
    """R★ = Switch I∪II ∪ N12_arm ∩ present."""
    base = set(switch_resseqs()) | set(n12_arm)
    return sorted(r for r in base if r in present)


def depth_on_resseqs(
    depth: np.ndarray,
    idx_map: dict[int, int],
    resseqs: list[int],
) -> tuple[list[int], np.ndarray]:
    """Return (shared resseqs present in map, depth values)."""
    shared = [r for r in resseqs if r in idx_map]
    vals = np.asarray([depth[idx_map[r]] for r in shared], dtype=np.float64)
    return shared, vals


def pairwise_spearman_on_rstar(
    depth_a: np.ndarray,
    map_a: dict[int, int],
    rstar_a: list[int],
    depth_b: np.ndarray,
    map_b: dict[int, int],
    rstar_b: list[int],
) -> dict[str, Any]:
    shared = sorted(set(rstar_a) & set(rstar_b) & set(map_a) & set(map_b))
    if len(shared) < 3:
        raise ValueError(f"too few shared R★ residues: {len(shared)}")
    va = np.asarray([depth_a[map_a[r]] for r in shared], dtype=np.float64)
    vb = np.asarray([depth_b[map_b[r]] for r in shared], dtype=np.float64)
    return {
        "rho": float(spearman_rho(va, vb)),
        "n_shared": len(shared),
        "shared_resseqs": shared,
    }


def allele_delta_site_specificity(
    depth_wt: np.ndarray,
    map_wt: dict[int, int],
    depth_mut: np.ndarray,
    map_mut: dict[int, int],
    *,
    rstar: list[int],
    n12: list[int],
    mut_prot: dict[str, Any],
) -> dict[str, Any]:
    """mean |Δ| on N12 ∩ shared R★ vs same-cardinality sites in R★\\N12.

    Child-2 distal donors lie outside Switch∪N12 by construction, so they cannot
    sit in R★. Control is therefore **within-R★ scramble**: sorted
    ``(R★ ∩ shared) \\ N12``, length-matched to N12 sites (site-non-specificity
    null inside the observation support).
    """
    del mut_prot  # reserved for future distal-full-chain report-only
    shared = sorted(set(rstar) & set(map_wt) & set(map_mut))
    if not shared:
        raise ValueError("empty shared R★ for allele delta")
    delta = {
        r: abs(float(depth_mut[map_mut[r]] - depth_wt[map_wt[r]])) for r in shared
    }
    n12_shared = [r for r in sorted(n12) if r in delta]
    if not n12_shared:
        raise ValueError("N12 ∩ shared empty")
    pool = [r for r in shared if r not in set(n12_shared)]
    if len(pool) < len(n12_shared):
        raise ValueError(
            f"need {len(n12_shared)} within-R★ scramble sites, only {len(pool)} in R★\\N12"
        )
    scr_shared = pool[: len(n12_shared)]
    mean_n12 = float(np.mean([delta[r] for r in n12_shared]))
    mean_scr = float(np.mean([delta[r] for r in scr_shared]))
    return {
        "mean_abs_delta_n12": mean_n12,
        "mean_abs_delta_scramble": mean_scr,
        "n12_sites": n12_shared,
        "scramble_sites": scr_shared,
        "scramble_policy": "within_R_star_minus_N12_sorted",
        "pass_arm": bool(mean_n12 > mean_scr),
    }


def basin_observation_verdict(
    *,
    guards_ok: bool,
    same_nuc: float,
    cross_nuc: float,
    allele_off_pass: bool,
    allele_on_pass: bool,
) -> dict[str, Any]:
    nuc_sep = bool(
        np.isfinite(same_nuc) and np.isfinite(cross_nuc) and same_nuc > cross_nuc
    )
    allele_ok = bool(allele_off_pass and allele_on_pass)
    ok = bool(guards_ok and nuc_sep and allele_ok)
    return {
        "pass": ok,
        "gates": {
            "guards_parse_all_four": bool(guards_ok),
            "same_nuc_exceeds_cross_nuc": nuc_sep,
            "allele_n12_exceeds_scramble_both_arms": allele_ok,
        },
        "same_nuc": float(same_nuc),
        "cross_nuc": float(cross_nuc),
        "rule": (
            "guards_ok AND same_nuc > cross_nuc AND "
            "mean|Δ|_N12 > mean|Δ|_scramble on OFF and ON"
        ),
        "outcome": "Pass" if ok else "Fail",
    }


def build_arm_n12(
    mut: dict[str, Any],
    wt: dict[str, Any],
) -> list[int]:
    return neighborhood_n12(mut, wt)


__all__ = [
    "ROSTER",
    "allele_delta_site_specificity",
    "basin_observation_verdict",
    "build_arm_n12",
    "check_structure_guards",
    "depth_on_resseqs",
    "pairwise_spearman_on_rstar",
    "parse_g12_allele",
    "parse_nucleotide_class",
    "r_star_for_structure",
    "residue_index_map",
    "spearman_rho",
    "switch_resseqs",
]
