"""SVD-based Kabsch superposition for structural alignment.

Computes the optimal rotation + translation that minimizes RMSD between
two aligned coordinate sets. Uses scipy for SVD computation and handles
the reflection case (det(R) = -1).

Requirements: 7.1, 7.2, 7.3, 7.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Minimum number of common residues required for superposition
MIN_COMMON_RESIDUES = 3


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ComparableCore:
    """The subset of residues used for superposition.

    Defines which residues were included and the criteria used,
    enabling reproducibility of the alignment.
    """

    residue_pairs: list[tuple[str, str]] = field(default_factory=list)
    uniprot_positions: list[int] = field(default_factory=list)
    criteria: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for storage."""
        return {
            "residue_pairs": self.residue_pairs,
            "uniprot_positions": self.uniprot_positions,
            "criteria": self.criteria,
        }


@dataclass
class SuperpositionResult:
    """Result of a Kabsch superposition between two structures."""

    rotation_matrix: NDArray[np.float64]  # 3x3
    translation: NDArray[np.float64]  # 3
    rmsd: float
    aligned_residue_count: int
    comparable_core: ComparableCore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_kabsch_superposition(
    query_coords: NDArray[np.float64],
    ref_coords: NDArray[np.float64],
    comparable_core: ComparableCore,
) -> SuperpositionResult | None:
    """Compute optimal rotation + translation minimizing RMSD.

    Uses SVD-based Kabsch algorithm. Handles the reflection case
    (det(R) = -1) by flipping the sign of the column corresponding
    to the smallest singular value.

    Args:
        query_coords: Nx3 Cα coordinates of the query structure.
        ref_coords: Nx3 Cα coordinates of the reference structure (same N).
        comparable_core: Definition of which residues are in the core.

    Returns:
        SuperpositionResult, or None if fewer than MIN_COMMON_RESIDUES
        coordinate pairs are provided.
    """
    if query_coords.shape[0] < MIN_COMMON_RESIDUES:
        logger.warning(
            "Skipping Kabsch: only %d common residues (need >= %d)",
            query_coords.shape[0],
            MIN_COMMON_RESIDUES,
        )
        return None

    if query_coords.shape != ref_coords.shape:
        raise ValueError(
            f"Coordinate shape mismatch: query {query_coords.shape} vs ref {ref_coords.shape}"
        )

    if query_coords.shape[1] != 3:
        raise ValueError(f"Expected Nx3 coordinates, got shape {query_coords.shape}")

    n_points = query_coords.shape[0]

    # Center both coordinate sets
    query_centroid = query_coords.mean(axis=0)
    ref_centroid = ref_coords.mean(axis=0)

    query_centered = query_coords - query_centroid
    ref_centered = ref_coords - ref_centroid

    # Compute cross-covariance matrix H = Q^T * R
    H = query_centered.T @ ref_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Determine sign correction for proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    # Optimal rotation matrix
    R = Vt.T @ sign_matrix @ U.T

    # Translation vector
    t = ref_centroid - R @ query_centroid

    # Compute RMSD after applying the rotation
    query_aligned = (R @ query_coords.T).T + t
    diff = query_aligned - ref_coords
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    return SuperpositionResult(
        rotation_matrix=R,
        translation=t,
        rmsd=rmsd,
        aligned_residue_count=n_points,
        comparable_core=comparable_core,
    )


# ---------------------------------------------------------------------------
# Comparable core construction
# ---------------------------------------------------------------------------


@dataclass
class ResidueCoordinate:
    """A residue with its Cα coordinate for alignment."""

    residue_id: str
    uniprot_position: int
    ca_coord: NDArray[np.float64]  # 3-element array
    occupancy: float = 1.0
    altloc: str | None = None
    is_tag: bool = False


def build_comparable_core(
    query_residues: list[ResidueCoordinate],
    ref_residues: list[ResidueCoordinate],
    min_occupancy: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], ComparableCore] | None:
    """Build the comparable core from two sets of residues.

    Selects residues that:
    - Share the same UniProt position
    - Have Cα coordinates in both structures
    - Meet minimum occupancy threshold
    - Are not tags/tails

    Args:
        query_residues: Query structure residues with coordinates.
        ref_residues: Reference structure residues with coordinates.
        min_occupancy: Minimum acceptable occupancy for inclusion.

    Returns:
        Tuple of (query_coords, ref_coords, comparable_core), or None
        if fewer than MIN_COMMON_RESIDUES overlap.
    """
    # Build lookup by UniProt position for each structure
    # For duplicates at same position, prefer highest occupancy (preferred altloc)
    query_by_pos = _best_by_position(query_residues, min_occupancy)
    ref_by_pos = _best_by_position(ref_residues, min_occupancy)

    # Find common UniProt positions
    common_positions = sorted(set(query_by_pos.keys()) & set(ref_by_pos.keys()))

    if len(common_positions) < MIN_COMMON_RESIDUES:
        logger.warning(
            "Insufficient common positions for alignment: %d (need >= %d)",
            len(common_positions),
            MIN_COMMON_RESIDUES,
        )
        return None

    # Extract coordinate arrays and build core definition
    query_coords_list: list[NDArray[np.float64]] = []
    ref_coords_list: list[NDArray[np.float64]] = []
    residue_pairs: list[tuple[str, str]] = []

    for pos in common_positions:
        q_res = query_by_pos[pos]
        r_res = ref_by_pos[pos]
        query_coords_list.append(q_res.ca_coord)
        ref_coords_list.append(r_res.ca_coord)
        residue_pairs.append((q_res.residue_id, r_res.residue_id))

    query_coords = np.array(query_coords_list, dtype=np.float64)
    ref_coords = np.array(ref_coords_list, dtype=np.float64)

    core = ComparableCore(
        residue_pairs=residue_pairs,
        uniprot_positions=common_positions,
        criteria={
            "min_occupancy": min_occupancy,
            "excluded_tags": True,
            "ca_required": True,
            "common_uniprot_positions": len(common_positions),
        },
    )

    return query_coords, ref_coords, core


def _best_by_position(
    residues: list[ResidueCoordinate],
    min_occupancy: float,
) -> dict[int, ResidueCoordinate]:
    """Select best residue per UniProt position (highest occupancy, no tags)."""
    best: dict[int, ResidueCoordinate] = {}
    for res in residues:
        # Exclude tags/tails
        if res.is_tag:
            continue
        # Exclude low occupancy
        if res.occupancy < min_occupancy:
            continue
        pos = res.uniprot_position
        if pos not in best or res.occupancy > best[pos].occupancy:
            best[pos] = res
    return best
