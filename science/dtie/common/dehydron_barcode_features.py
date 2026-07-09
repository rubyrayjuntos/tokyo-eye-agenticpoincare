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

**Scalar order (``SCALAR_NAMES``, length 11):**

0. ``n_bars`` — log1p(structure-level bar count)
1. ``n_h1_bars`` — log1p(structure-level H1 bar count)
2. ``total_persistence`` — log1p(structure-level sum of bar persistence)
3. ``max_persistence`` — structure-level max persistence
4. ``mean_persistence`` — structure-level mean persistence
5. ``std_persistence`` — structure-level std persistence
6. ``frac_long_lived`` — structure-level fraction above threshold
7. ``mean_birth_h1`` — structure-level H1 mean birth
8. ``mean_death_h1`` — structure-level H1 mean death
9. ``max_h1_persistence`` — structure-level max H1 persistence
10. ``n_dehydrons_touching`` — log1p(count of midpoints touching residue)

Scalars 0, 1, 2, and 10 receive ``log1p`` before return. Binned output (optional) is
an H1 persistence histogram on ``[0, bin_max)`` with ``bin_width`` (default 40 bins),
L1-normalized per residue when any mass is present.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

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

BARCODE_FEATURE_VERSION = "dehydron_barcode_v1"
SCALAR_DIM = 11
BINNED_DIM = 40

SCALAR_NAMES: list[str] = [
    "n_bars",
    "n_h1_bars",
    "total_persistence",
    "max_persistence",
    "mean_persistence",
    "std_persistence",
    "frac_long_lived",
    "mean_birth_h1",
    "mean_death_h1",
    "max_h1_persistence",
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
) -> list[PersistenceBar]:
    """Euclidean witness complex persistence on dehydron midpoints (H0 + H1)."""
    if len(midpoints) < 2:
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


def _scalar_stats_from_bars(
    bars: Sequence[PersistenceBar],
    *,
    long_lived_persistence_angstrom: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Raw (pre-log1p) per-dehydron scalar stats from one bar set."""
    if not bars:
        return (0.0,) * 10

    h1_bars = [bar for bar in bars if bar.dim == 1]
    persistences = [bar.persistence for bar in bars]
    h1_persistences = [bar.persistence for bar in h1_bars]

    n_bars = float(len(bars))
    n_h1_bars = float(len(h1_bars))
    total_persistence = float(sum(persistences))
    max_persistence = float(max(persistences))
    mean_persistence = float(np.mean(persistences))
    std_persistence = float(np.std(persistences)) if len(persistences) > 1 else 0.0
    frac_long_lived = float(
        sum(1 for persistence in persistences if persistence >= long_lived_persistence_angstrom)
        / len(persistences)
    )
    mean_birth_h1 = float(np.mean([bar.birth for bar in h1_bars])) if h1_bars else 0.0
    mean_death_h1 = float(np.mean([bar.death for bar in h1_bars])) if h1_bars else 0.0
    max_h1_persistence = float(max(h1_persistences)) if h1_persistences else 0.0

    return (
        n_bars,
        n_h1_bars,
        total_persistence,
        max_persistence,
        mean_persistence,
        std_persistence,
        frac_long_lived,
        mean_birth_h1,
        mean_death_h1,
        max_h1_persistence,
    )


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


def aggregate_residue_barcode_features(
    n_residues: int,
    midpoints: Sequence[DehydronMidpoint],
    bars: Sequence[PersistenceBar],
    *,
    long_lived_persistence_angstrom: float = 2.0,
    use_binned: bool = False,
    bin_width: float = 0.25,
    bin_max: float = 10.0,
) -> dict[str, np.ndarray | None]:
    """Aggregate global barcode features to per-residue scalars, optional binned vector, and mask."""
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

    for residue_idx, touching_midpoints in touching_by_residue.items():
        if residue_idx < 0 or residue_idx >= n_residues:
            continue

        structure_stats = _scalar_stats_from_bars(
            bars,
            long_lived_persistence_angstrom=long_lived_persistence_angstrom,
        )

        raw_n_bars = structure_stats[0]
        raw_n_h1_bars = structure_stats[1]
        raw_total_persistence = structure_stats[2]
        max_persistence = structure_stats[3]
        mean_persistence = structure_stats[4]
        std_persistence = structure_stats[5]
        frac_long_lived = structure_stats[6]
        mean_birth_h1 = structure_stats[7]
        mean_death_h1 = structure_stats[8]
        max_h1_persistence = structure_stats[9]
        n_dehydrons_touching = float(len(touching_midpoints))

        scalars[residue_idx] = np.asarray(
            [
                np.log1p(raw_n_bars),
                np.log1p(raw_n_h1_bars),
                np.log1p(raw_total_persistence),
                max_persistence,
                mean_persistence,
                std_persistence,
                frac_long_lived,
                mean_birth_h1,
                mean_death_h1,
                max_h1_persistence,
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
