"""Dehydron barcode input channel — midpoint extraction (Task 1), witness persistence (Task 2),
and per-residue scalar/binned aggregation (Task 3).

Witness points for Euclidean witness persistence are midpoints of **inter-residue**
backbone H-bonds (donor N … acceptor O), not per-residue local N–O midpoints used
for ρ in ``compute_dehydron_wrapping_count``.

Residue identity for ``residue_index_map`` keys uses ``(chain_label, residue_index)``
where ``residue_index`` is the PDB residue sequence number (``resseq``), matching
``ResidueRecord.residue_index`` / ``residue_features.residue_key`` conventions.
Insertion codes are not modeled in training graph assembly today; extend the key
tuple to ``(chain_label, residue_index, icode)`` if icode-aware graphs are added.

**First-arm scalar order (``SCALAR_NAMES``, length 3 — locked 2026-07-16):**

0. ``total_persistence_h1`` — log1p(structure-level sum of **H1** bar persistence)
1. ``fraction_long_lived_h1`` — fraction of **H1** bars above long-lived threshold
2. ``n_dehydrons_touching`` — log1p(count of midpoints touching residue)

Corpus z-score (μ/σ) is applied at cache-build over valid (non-missing) rows.
Binned output remains optional / deferred. Min-dehydron guard:
``MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS`` — below that, witness TDA is skipped and the
structure routes to the missing-mask path.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
from gudhi import WitnessComplex
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from science.dtie.common.residue_features import (
    POLAR_SIDECHAINS,
    TAU,
    WRAPPING_RADIUS,
    AtomRecord,
)

BARCODE_FEATURE_VERSION = "dehydron_barcode_v1_2"
BARCODE_SIDECAR_GLOB = f"*_{BARCODE_FEATURE_VERSION}.pt"
SCALAR_DIM = 3
BINNED_DIM = 40
# Local per-dehydron-pair edge channel (witness midpoints — not structure-global broadcast).
EDGE_BARCODE_DIM = 5

# Locked 2026-07-16 from Stage A-12 pure-TDA diagnostic
# (checkpoints/v65/diagnostics/dehydron_bar_length_v1/): pooled H1 p75 above
# noise floor 0.1 Å; dominance clear; no usable upper-half gap.
# See threshold_lock.json alongside that diagnostic for full justification.
LONG_LIVED_PERSISTENCE_ANGSTROM = 3.11
LONG_LIVED_THRESHOLD_METHOD = "pooled_h1_percentile_75"
LONG_LIVED_THRESHOLD_DIAGNOSTIC = (
    "checkpoints/v65/diagnostics/dehydron_bar_length_v1"
)

# Below this midpoint count, skip landmark/witness TDA → missing-mask (§4.3).
# Stage A-12 min midpoint count is 77; this only catches degenerate structures.
MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS = 5

# Orthogonality lock SSOT (within-SS inspection held for n_dehydrons_touching).
FIRST_ARM_SCALAR_LOCK = (
    "checkpoints/v65/diagnostics/dehydron_scalar_orthogonality_v1/scalar_subset_lock.json"
)
CORPUS_ZSCORE_EPS = 1e-6


def barcode_sidecar_filename(pdb_id: str, chain: str) -> str:
    """Versioned sidecar filename shared by writer and loader."""
    return f"{pdb_id.upper()}_{chain}_{BARCODE_FEATURE_VERSION}.pt"


EDGE_BARCODE_NAMES: list[str] = [
    "wrap_deficit",
    "log1p_wrapping",
    "witness_nn1_norm",
    "local_h1_persistence",
    "log1p_local_density",
]

SCALAR_NAMES: list[str] = [
    "total_persistence_h1",
    "fraction_long_lived_h1",
    "n_dehydrons_touching",
]

ResidueMapKey: TypeAlias = tuple[str, int]


def residue_map_key(chain_label: str, residue_index: int) -> ResidueMapKey:
    """Canonical key for ``residue_index_map`` — ``(chain, resseq)``."""
    return (chain_label, residue_index)


@dataclass(frozen=True)
class StructureAtom:
    """Atom with residue identity for backbone H-bond enumeration.

    Compatible with ``AtomRecord`` fields; adds ``chain_label`` and ``residue_index``
    required to group backbone N/O and map into the training-graph Cα order.
    """

    atom_name: str
    element: str
    coord: np.ndarray
    parent_residue_name: str = ""
    chain_label: str = "A"
    residue_index: int = 0


@dataclass(frozen=True)
class DehydronMidpoint:
    coord: np.ndarray
    donor_idx: int
    acceptor_idx: int
    wrapping_count: float


@dataclass(frozen=True)
class PersistenceBar:
    dim: int
    birth: float
    death: float
    persistence: float


def _compute_nearest_landmark_table(
    landmarks: np.ndarray,
    witnesses: np.ndarray,
) -> list[list[tuple[int, float]]]:
    """Full sorted nearest-landmark table for ``gudhi.WitnessComplex``."""
    dist_matrix = cdist(witnesses, landmarks)
    nearest_table: list[list[tuple[int, float]]] = []
    for i in range(len(witnesses)):
        sorted_indices = np.argsort(dist_matrix[i])
        row = [(int(j), float(dist_matrix[i, j])) for j in sorted_indices]
        nearest_table.append(row)
    return nearest_table


def _effective_death(death: float, *, max_alpha_angstrom: float) -> float:
    if math.isinf(death):
        return max_alpha_angstrom
    return float(death)


def compute_witness_persistence(
    midpoints: list[DehydronMidpoint],
    *,
    max_alpha_angstrom: float = 20.0,
    min_persistence_angstrom: float = 0.1,
    n_landmarks: int = 30,
    random_state: int = 42,
    min_dehydron_midpoints: int = MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS,
) -> list[PersistenceBar]:
    """Euclidean witness complex persistence on dehydron midpoints (H0 + H1).

    Structures with fewer than ``min_dehydron_midpoints`` midpoints return ``[]``
    so callers route to the missing-mask path rather than a degenerate barcode.
    """
    if len(midpoints) < int(min_dehydron_midpoints):
        return []

    witnesses = np.asarray([mp.coord for mp in midpoints], dtype=np.float64)
    n_witnesses = len(witnesses)
    n_clusters = min(n_landmarks, n_witnesses)

    landmarks = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    ).fit(witnesses).cluster_centers_

    w_complex = WitnessComplex(
        nearest_landmark_table=_compute_nearest_landmark_table(landmarks, witnesses)
    )
    max_alpha_square = max_alpha_angstrom**2
    simplex_tree = w_complex.create_simplex_tree(
        max_alpha_square=max_alpha_square,
        limit_dimension=2,
    )
    simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

    bars: list[PersistenceBar] = []
    for dim, (birth, death) in simplex_tree.persistence():
        if dim not in (0, 1):
            continue

        birth_f = float(birth)
        death_f = _effective_death(float(death), max_alpha_angstrom=max_alpha_angstrom)
        persistence = death_f - birth_f
        if persistence < min_persistence_angstrom:
            continue

        bars.append(
            PersistenceBar(
                dim=int(dim),
                birth=birth_f,
                death=death_f,
                persistence=persistence,
            )
        )

    return bars


def compute_midpoint_witness_nn1(
    midpoints: Sequence[DehydronMidpoint],
    *,
    n_landmarks: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """Nearest-landmark distance per dehydron midpoint (witness filtration entry proxy)."""
    if not midpoints:
        return np.zeros(0, dtype=np.float32)
    witnesses = np.asarray([mp.coord for mp in midpoints], dtype=np.float64)
    n_witnesses = len(witnesses)
    n_clusters = min(int(n_landmarks), n_witnesses)
    landmarks = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    ).fit(witnesses).cluster_centers_
    nn1 = cdist(witnesses, landmarks).min(axis=1)
    return nn1.astype(np.float32)


def local_h1_persistence_for_witness(
    witness_nn1: float,
    bars: Sequence[PersistenceBar],
) -> float:
    """Max H1 persistence among bars whose lifespan contains the witness scale."""
    scale = float(witness_nn1)
    best = 0.0
    for bar in bars:
        if bar.dim != 1:
            continue
        if bar.birth <= scale <= bar.death:
            best = max(best, float(bar.persistence))
    return best


def midpoint_local_density(
    midpoints: Sequence[DehydronMidpoint],
    *,
    radius: float = 4.0,
) -> np.ndarray:
    """Count of other dehydron midpoints within ``radius`` Å of each witness."""
    n = len(midpoints)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    coords = np.asarray([mp.coord for mp in midpoints], dtype=np.float64)
    dists = cdist(coords, coords)
    counts = (dists < float(radius)).sum(axis=1).astype(np.float32) - 1.0
    return np.clip(counts, 0.0, None)


def _dehydron_pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def aggregate_dehydron_edge_barcode_features(
    midpoints: Sequence[DehydronMidpoint],
    bars: Sequence[PersistenceBar],
    *,
    tau: float = TAU,
    max_alpha_angstrom: float = 20.0,
    n_landmarks: int = 30,
    random_state: int = 42,
    local_density_radius: float = 4.0,
) -> dict[str, np.ndarray]:
    """Per-dehydron-pair local barcode features keyed by midpoint H-bond pairs."""
    m = len(midpoints)
    edge_pairs = np.zeros((m, 2), dtype=np.int32)
    edge_scalars = np.zeros((m, EDGE_BARCODE_DIM), dtype=np.float32)
    if m == 0:
        return {"edge_pairs": edge_pairs, "edge_scalars": edge_scalars}

    nn1 = compute_midpoint_witness_nn1(
        midpoints,
        n_landmarks=n_landmarks,
        random_state=random_state,
    )
    density = midpoint_local_density(midpoints, radius=local_density_radius)
    alpha_scale = max(float(max_alpha_angstrom), 1e-6)
    tau_f = max(float(tau), 1e-6)

    for idx, mp in enumerate(midpoints):
        wrap = float(mp.wrapping_count)
        edge_pairs[idx, 0] = int(min(mp.donor_idx, mp.acceptor_idx))
        edge_pairs[idx, 1] = int(max(mp.donor_idx, mp.acceptor_idx))
        edge_scalars[idx, 0] = np.clip((tau_f - wrap) / tau_f, 0.0, 1.0)
        edge_scalars[idx, 1] = np.log1p(wrap)
        edge_scalars[idx, 2] = float(nn1[idx]) / alpha_scale
        edge_scalars[idx, 3] = local_h1_persistence_for_witness(float(nn1[idx]), bars)
        edge_scalars[idx, 4] = np.log1p(float(density[idx]))

    return {"edge_pairs": edge_pairs, "edge_scalars": edge_scalars}


def align_edge_barcode_to_graph(
    edge_pairs: np.ndarray,
    edge_scalars: np.ndarray,
    barcode_residue_indices: Sequence[int],
    graph_residue_indices: Sequence[int],
) -> dict[tuple[int, int], np.ndarray]:
    """Map PDB-order dehydron pairs onto training-graph residue indices."""
    pairs = np.asarray(edge_pairs, dtype=np.int32)
    scalars = np.asarray(edge_scalars, dtype=np.float32)
    if pairs.size == 0:
        return {}
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"edge_pairs must be [M, 2], got {pairs.shape}")
    if scalars.shape != (pairs.shape[0], EDGE_BARCODE_DIM):
        raise ValueError(
            f"edge_scalars must be [{pairs.shape[0]}, {EDGE_BARCODE_DIM}], "
            f"got {scalars.shape}"
        )

    src_resseqs = np.asarray(barcode_residue_indices, dtype=np.int32)
    graph_resseq_to_idx = {int(resseq): i for i, resseq in enumerate(graph_residue_indices)}
    lookup: dict[tuple[int, int], np.ndarray] = {}

    for row in range(pairs.shape[0]):
        i_pdb, j_pdb = int(pairs[row, 0]), int(pairs[row, 1])
        if i_pdb < 0 or j_pdb < 0 or i_pdb >= src_resseqs.shape[0] or j_pdb >= src_resseqs.shape[0]:
            continue
        gi = graph_resseq_to_idx.get(int(src_resseqs[i_pdb]))
        gj = graph_resseq_to_idx.get(int(src_resseqs[j_pdb]))
        if gi is None or gj is None:
            continue
        key = _dehydron_pair_key(gi, gj)
        vec = scalars[row]
        if key in lookup:
            lookup[key] = np.maximum(lookup[key], vec)
        else:
            lookup[key] = vec.copy()
    return lookup


def _h1_persistence_histogram(
    bars: Sequence[PersistenceBar],
    *,
    bin_width: float,
    bin_max: float,
) -> np.ndarray:
    n_bins = int(bin_max / bin_width)
    hist = np.zeros(n_bins, dtype=np.float64)
    for bar in bars:
        if bar.dim != 1:
            continue
        bin_idx = int(bar.persistence / bin_width)
        if 0 <= bin_idx < n_bins:
            hist[bin_idx] += 1.0
    return hist


def _first_arm_stats_from_h1_bars(
    bars: Sequence[PersistenceBar],
    *,
    long_lived_persistence_angstrom: float,
) -> tuple[float, float]:
    """Return (total_persistence_h1, fraction_long_lived_h1) from H1 bars only."""
    h1 = [bar for bar in bars if int(bar.dim) == 1]
    if not h1:
        return 0.0, 0.0
    persistences = [float(bar.persistence) for bar in h1]
    total = float(sum(persistences))
    frac = float(
        sum(p >= long_lived_persistence_angstrom for p in persistences) / len(persistences)
    )
    return total, frac


def aggregate_residue_barcode_features(
    n_residues: int,
    midpoints: Sequence[DehydronMidpoint],
    bars: Sequence[PersistenceBar],
    *,
    long_lived_persistence_angstrom: float = LONG_LIVED_PERSISTENCE_ANGSTROM,
    use_binned: bool = False,
    bin_width: float = 0.25,
    bin_max: float = 10.0,
) -> dict[str, np.ndarray | None]:
    """Aggregate locked first-arm H1 scalars (+ optional binned) to per-residue rows."""
    scalars = np.zeros((n_residues, SCALAR_DIM), dtype=np.float32)
    missing = np.ones((n_residues, 1), dtype=np.float32)
    binned: np.ndarray | None = (
        np.zeros((n_residues, BINNED_DIM), dtype=np.float32) if use_binned else None
    )

    if n_residues <= 0:
        return {"scalars": scalars, "binned": binned, "missing": missing}

    if not midpoints or not bars:
        return {"scalars": scalars, "binned": binned, "missing": missing}

    touching_by_residue: dict[int, list[DehydronMidpoint]] = defaultdict(list)
    for midpoint in midpoints:
        touching_by_residue[midpoint.donor_idx].append(midpoint)
        touching_by_residue[midpoint.acceptor_idx].append(midpoint)

    total_h1, frac_long_lived_h1 = _first_arm_stats_from_h1_bars(
        bars,
        long_lived_persistence_angstrom=long_lived_persistence_angstrom,
    )

    for residue_idx, touching_midpoints in touching_by_residue.items():
        if residue_idx < 0 or residue_idx >= n_residues:
            continue

        n_dehydrons_touching = float(len(touching_midpoints))
        scalars[residue_idx] = np.asarray(
            [
                np.log1p(total_h1),
                frac_long_lived_h1,
                np.log1p(n_dehydrons_touching),
            ],
            dtype=np.float32,
        )
        missing[residue_idx, 0] = 0.0

        if use_binned and binned is not None:
            hist = _h1_persistence_histogram(
                bars,
                bin_width=bin_width,
                bin_max=bin_max,
            )
            if hist.sum() > 0.0:
                binned[residue_idx] = (hist / hist.sum()).astype(np.float32)

    return {"scalars": scalars, "binned": binned, "missing": missing}


def compute_corpus_zscore_stats(
    scalar_arrays: Sequence[np.ndarray],
    missing_arrays: Sequence[np.ndarray],
    *,
    eps: float = CORPUS_ZSCORE_EPS,
) -> dict[str, Any]:
    """Corpus μ/σ over valid (missing==0) rows for each scalar channel."""
    valid_rows: list[np.ndarray] = []
    for scalars, missing in zip(scalar_arrays, missing_arrays, strict=True):
        sc = np.asarray(scalars, dtype=np.float64)
        miss = np.asarray(missing, dtype=np.float64).reshape(-1)
        if sc.ndim != 2 or sc.shape[1] != SCALAR_DIM:
            raise ValueError(f"scalars must be [N, {SCALAR_DIM}], got {sc.shape}")
        if miss.shape[0] != sc.shape[0]:
            raise ValueError("missing length must match scalar rows")
        keep = miss < 0.5
        if np.any(keep):
            valid_rows.append(sc[keep])
    if not valid_rows:
        mean = np.zeros(SCALAR_DIM, dtype=np.float64)
        std = np.ones(SCALAR_DIM, dtype=np.float64)
        n_valid = 0
    else:
        stacked = np.concatenate(valid_rows, axis=0)
        n_valid = int(stacked.shape[0])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=0)
        std = np.maximum(std, float(eps))
    return {
        "version": BARCODE_FEATURE_VERSION,
        "scalar_names": list(SCALAR_NAMES),
        "n_valid_rows": n_valid,
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
        "eps": float(eps),
    }


def apply_corpus_zscore(
    scalars: np.ndarray,
    missing: np.ndarray,
    stats: Mapping[str, Any],
) -> np.ndarray:
    """Z-score valid rows in-place-copy; leave missing rows as zeros."""
    sc = np.asarray(scalars, dtype=np.float32).copy()
    miss = np.asarray(missing, dtype=np.float32).reshape(-1)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if sc.shape[1] != SCALAR_DIM or mean.shape[0] != SCALAR_DIM:
        raise ValueError("z-score stats width must match SCALAR_DIM")
    valid = miss < 0.5
    if np.any(valid):
        sc[valid] = (sc[valid] - mean) / std
    sc[~valid] = 0.0
    return sc


def _residue_key(atom: StructureAtom | AtomRecord) -> ResidueMapKey | None:
    chain = getattr(atom, "chain_label", None)
    res_idx = getattr(atom, "residue_index", None)
    if chain is None or res_idx is None:
        return None
    return residue_map_key(str(chain), int(res_idx))


def _sequence_separation(
    donor_key: ResidueMapKey,
    acceptor_key: ResidueMapKey,
) -> int | None:
    donor_chain, donor_res = donor_key
    acceptor_chain, acceptor_res = acceptor_key
    if donor_chain != acceptor_chain:
        return None
    return abs(int(donor_res) - int(acceptor_res))


def _wrapping_count_at_midpoint(
    midpoint: np.ndarray,
    atoms: Sequence[StructureAtom | AtomRecord],
    *,
    wrapping_radius: float,
) -> float:
    """Carbon-shell wrapping count — same exclusions as ``compute_dehydron_wrapping_count``."""
    count = 0
    for atom in atoms:
        if atom.element != "C":
            continue
        parent = atom.parent_residue_name.strip().upper()
        if parent in POLAR_SIDECHAINS:
            continue
        if atom.atom_name == "C":
            continue
        dist = float(np.linalg.norm(atom.coord - midpoint))
        if dist >= wrapping_radius:
            continue
        count += 1
    return float(count)


def extract_dehydron_midpoints(
    structure_atoms: Sequence[StructureAtom | AtomRecord],
    residue_index_map: Mapping[ResidueMapKey, int],
    *,
    wrapping_radius: float = WRAPPING_RADIUS,
    tau: float = TAU,
    max_no_dist: float = 3.5,
) -> list[DehydronMidpoint]:
    """Extract underwrapped inter-residue backbone H-bond midpoints (dehydrons).

    Enumerates donor backbone N and acceptor backbone O from distinct residues with
    sequence separation ``|i-j| >= 1`` (within-chain) or any inter-chain pair.
    Keeps pairs with N–O distance ``< max_no_dist``, midpoint wrapping
    ``< tau``, and maps donor/acceptor to training-graph indices via
    ``residue_index_map``.
    """
    donors: list[tuple[ResidueMapKey, np.ndarray]] = []
    acceptors: list[tuple[ResidueMapKey, np.ndarray]] = []

    for atom in structure_atoms:
        key = _residue_key(atom)
        if key is None:
            continue
        if atom.atom_name == "N" and atom.element == "N":
            donors.append((key, np.asarray(atom.coord, dtype=np.float64)))
        elif atom.atom_name == "O" and atom.element == "O":
            acceptors.append((key, np.asarray(atom.coord, dtype=np.float64)))

    midpoints: list[DehydronMidpoint] = []
    for donor_key, n_coord in donors:
        donor_idx = residue_index_map.get(donor_key)
        if donor_idx is None:
            continue
        for acceptor_key, o_coord in acceptors:
            if donor_key == acceptor_key:
                continue
            sep = _sequence_separation(donor_key, acceptor_key)
            if sep is not None and sep < 1:
                continue

            acceptor_idx = residue_index_map.get(acceptor_key)
            if acceptor_idx is None:
                continue

            distance = float(np.linalg.norm(n_coord - o_coord))
            if distance >= max_no_dist:
                continue

            midpoint = (n_coord + o_coord) / 2.0
            wrapping_count = _wrapping_count_at_midpoint(
                midpoint,
                structure_atoms,
                wrapping_radius=wrapping_radius,
            )
            if wrapping_count >= tau:
                continue

            midpoints.append(
                DehydronMidpoint(
                    coord=midpoint,
                    donor_idx=int(donor_idx),
                    acceptor_idx=int(acceptor_idx),
                    wrapping_count=wrapping_count,
                )
            )

    return midpoints


def _load_chain_structure_atoms(
    pdb_path: Path | str,
    chain_id: str,
) -> tuple[list[StructureAtom], dict[ResidueMapKey, int]]:
    """Parse one PDB chain into ``StructureAtom`` list and training-graph index map.

    Residue order matches ``residue_features.build_from_pdb_chain`` (standard
    residues on ``chain_id``, PDB iteration order).
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("dehydron_barcode", str(pdb_path))
    residues_raw = [
        res
        for res in structure.get_residues()
        if res.get_id()[0] == " " and res.parent.id == chain_id
    ]

    structure_atoms: list[StructureAtom] = []
    residue_index_map: dict[ResidueMapKey, int] = {}
    for residue_idx, res in enumerate(residues_raw):
        res_name = res.get_resname().strip().upper()
        resseq = int(res.get_id()[1])
        residue_index_map[residue_map_key(chain_id, resseq)] = residue_idx
        for atom in res.get_atoms():
            structure_atoms.append(
                StructureAtom(
                    atom_name=atom.name,
                    element=(atom.element or "").upper(),
                    coord=np.asarray(atom.coord, dtype=np.float64),
                    parent_residue_name=res_name,
                    chain_label=chain_id,
                    residue_index=resseq,
                )
            )

    return structure_atoms, residue_index_map


def featurize_chain_dehydron_barcode(
    pdb_path: Path | str,
    chain: str,
    *,
    use_binned: bool = False,
    **kwargs: Any,
) -> dict[str, np.ndarray | None | dict[str, Any]]:
    """End-to-end dehydron barcode features for one PDB chain."""
    extract_kwargs = {
        key: kwargs[key]
        for key in ("wrapping_radius", "tau", "max_no_dist")
        if key in kwargs
    }
    persistence_kwargs = {
        key: kwargs[key]
        for key in (
            "max_alpha_angstrom",
            "min_persistence_angstrom",
            "n_landmarks",
            "random_state",
            "min_dehydron_midpoints",
        )
        if key in kwargs
    }
    aggregate_kwargs = {
        key: kwargs[key]
        for key in ("long_lived_persistence_angstrom", "bin_width", "bin_max")
        if key in kwargs
    }

    structure_atoms, residue_index_map = _load_chain_structure_atoms(pdb_path, chain)
    # Preserve PDB iteration order for row i ↔ residue_indices[i] (resseq).
    residue_indices = np.asarray(
        [
            resseq
            for (_chain, resseq), _idx in sorted(
                residue_index_map.items(), key=lambda item: item[1]
            )
        ],
        dtype=np.int32,
    )
    n_residues = int(residue_indices.shape[0])

    midpoints = extract_dehydron_midpoints(
        structure_atoms,
        residue_index_map,
        **extract_kwargs,
    )
    bars = compute_witness_persistence(midpoints, **persistence_kwargs)
    features = aggregate_residue_barcode_features(
        n_residues,
        midpoints,
        bars,
        use_binned=use_binned,
        **aggregate_kwargs,
    )
    edge_features = aggregate_dehydron_edge_barcode_features(
        midpoints,
        bars,
        tau=extract_kwargs.get("tau", TAU),
        max_alpha_angstrom=persistence_kwargs.get("max_alpha_angstrom", 20.0),
        n_landmarks=persistence_kwargs.get("n_landmarks", 30),
        random_state=persistence_kwargs.get("random_state", 42),
    )

    params = {
        "chain": chain,
        "use_binned": use_binned,
        "wrapping_radius": extract_kwargs.get("wrapping_radius", WRAPPING_RADIUS),
        "tau": extract_kwargs.get("tau", TAU),
        "max_no_dist": extract_kwargs.get("max_no_dist", 3.5),
        "max_alpha_angstrom": persistence_kwargs.get("max_alpha_angstrom", 20.0),
        "min_persistence_angstrom": persistence_kwargs.get(
            "min_persistence_angstrom", 0.1
        ),
        "n_landmarks": persistence_kwargs.get("n_landmarks", 30),
        "random_state": persistence_kwargs.get("random_state", 42),
        "min_dehydron_midpoints": persistence_kwargs.get(
            "min_dehydron_midpoints", MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS
        ),
        "long_lived_persistence_angstrom": aggregate_kwargs.get(
            "long_lived_persistence_angstrom", LONG_LIVED_PERSISTENCE_ANGSTROM
        ),
        "long_lived_threshold_method": LONG_LIVED_THRESHOLD_METHOD,
        "long_lived_threshold_diagnostic": LONG_LIVED_THRESHOLD_DIAGNOSTIC,
        "first_arm_scalar_lock": FIRST_ARM_SCALAR_LOCK,
        "scalar_names": list(SCALAR_NAMES),
        "bin_width": aggregate_kwargs.get("bin_width", 0.25),
        "bin_max": aggregate_kwargs.get("bin_max", 10.0),
    }

    return {
        "scalars": features["scalars"],
        "binned": features["binned"],
        "missing": features["missing"],
        "edge_pairs": edge_features["edge_pairs"],
        "edge_scalars": edge_features["edge_scalars"],
        "residue_indices": residue_indices,
        "metadata": {
            "version": BARCODE_FEATURE_VERSION,
            "params": params,
            "n_midpoints": len(midpoints),
            "n_bars": len(bars),
            "n_residues": n_residues,
        },
    }


def align_barcode_to_graph_residues(
    barcode: Mapping[str, np.ndarray | None],
    graph_residue_indices: Sequence[int],
    *,
    use_binned: bool = False,
) -> dict[str, np.ndarray | None]:
    """Reindex sidecar rows onto training-graph residue order by PDB resseq.

    Training graphs are often loaded from the governed DB and can differ in
    length from the PDB used at precompute. Rows without a sidecar match get
    zeros + ``missing=1``.
    """
    scalars_src = np.asarray(barcode["scalars"], dtype=np.float32)
    missing_src = np.asarray(barcode["missing"], dtype=np.float32)
    binned_src = barcode.get("binned")
    if binned_src is not None:
        binned_src = np.asarray(binned_src, dtype=np.float32)

    residue_indices = barcode.get("residue_indices")
    if residue_indices is None:
        if scalars_src.shape[0] != len(graph_residue_indices):
            raise ValueError(
                "barcode sidecar missing residue_indices and row count "
                f"{scalars_src.shape[0]} != graph N {len(graph_residue_indices)}; "
                "re-run make precompute-dehydron-barcodes"
            )
        return {
            "scalars": scalars_src,
            "missing": missing_src,
            "binned": binned_src,
        }

    src_resseqs = np.asarray(residue_indices, dtype=np.int32).reshape(-1)
    if src_resseqs.shape[0] != scalars_src.shape[0]:
        raise ValueError(
            "barcode residue_indices length "
            f"{src_resseqs.shape[0]} != scalars rows {scalars_src.shape[0]}"
        )
    src_by_resseq = {int(resseq): i for i, resseq in enumerate(src_resseqs)}

    n = len(graph_residue_indices)
    scalars = np.zeros((n, SCALAR_DIM), dtype=np.float32)
    missing = np.ones((n, 1), dtype=np.float32)
    binned = np.zeros((n, BINNED_DIM), dtype=np.float32) if use_binned else None

    for dst_i, resseq in enumerate(graph_residue_indices):
        src_i = src_by_resseq.get(int(resseq))
        if src_i is None:
            continue
        scalars[dst_i] = scalars_src[src_i]
        missing[dst_i] = missing_src[src_i]
        if use_binned and binned is not None and binned_src is not None:
            binned[dst_i] = binned_src[src_i]

    return {"scalars": scalars, "missing": missing, "binned": binned}


def stack_node_features_with_barcode(
    base_x: np.ndarray,
    barcode: Mapping[str, np.ndarray | None],
    *,
    use_binned: bool = False,
) -> np.ndarray:
    """Concatenate topology-three-vector with per-residue barcode features."""
    scalars = np.asarray(barcode["scalars"], dtype=np.float32)
    missing = np.asarray(barcode["missing"], dtype=np.float32)
    base = np.asarray(base_x, dtype=np.float32)

    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError(f"base_x must be [N, 3], got {base.shape}")
    if scalars.shape != (base.shape[0], SCALAR_DIM):
        raise ValueError(
            f"barcode scalars must be [{base.shape[0]}, {SCALAR_DIM}], got {scalars.shape}"
        )
    if missing.shape != (base.shape[0], 1):
        raise ValueError(
            f"barcode missing must be [{base.shape[0]}, 1], got {missing.shape}"
        )

    parts: list[np.ndarray] = [base, scalars]
    if use_binned:
        binned = barcode.get("binned")
        if binned is None:
            raise ValueError("use_binned=True requires barcode['binned']")
        binned_arr = np.asarray(binned, dtype=np.float32)
        if binned_arr.shape != (base.shape[0], BINNED_DIM):
            raise ValueError(
                f"barcode binned must be [{base.shape[0]}, {BINNED_DIM}], got {binned_arr.shape}"
            )
        parts.append(binned_arr)
    parts.append(missing)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)
