"""Site merger for binding site scan phase.

Merges GNN-derived cryptic site candidates with geometry-detected surface pockets
into a unified, ranked list of binding site candidates. Computes composite
druggability scores and assigns site ranks.

Requirements: 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field

from agent.tools.cryptic.pocket_detector import GeometryPocket
from agent.tools.cryptic.seed_generator import CandidateCluster

logger = logging.getLogger(__name__)

# Site types recognized by the classification heuristic
SITE_TYPES = frozenset({
    "cryptic_wedge",
    "structural_stent",
    "dynamic_lid",
    "allosteric_clamp",
    "strain_relief_insert",
    "surface_pocket",
})

# Site types that receive a druggability bonus
BONUS_SITE_TYPES = frozenset({"cryptic_wedge", "structural_stent"})

# Discovery method values
DISCOVERY_GNN = "gnn_strain"
DISCOVERY_GEOMETRY = "geometry"
DISCOVERY_HYBRID = "hybrid"


@dataclass
class UnifiedCandidate:
    """A merged binding site candidate from any detection source."""

    site_id: str
    residue_ids: list[str]
    centroid_xyz: tuple[float, float, float]
    site_type: str
    discovery_method: str  # "gnn_strain", "geometry", or "hybrid"
    druggability_score: float  # [0, 1]
    site_rank: int = 0
    composite_gnn_score: float | None = None
    fpocket_druggability: float | None = None
    volume_angstrom3: float | None = None
    provenance_gate: str = "scan_phase"
    heuristic_version: str = "v1.0"



def _euclidean_distance(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def compute_druggability_score(
    composite_gnn_score: float | None,
    fpocket_druggability: float | None,
    volume_angstrom3: float | None,
    site_type: str,
) -> float:
    """Compute unified druggability score from all available signals.

    Weighted combination:
    - GNN composite score (40% weight if available)
    - fpocket druggability (30% weight if available)
    - Volume score (20% weight, normalized by max 2000 Å³)
    - Site type bonus (10% — cryptic_wedge and structural_stent get +0.1)

    Returns score in [0, 1]. Missing signals reduce the denominator
    (weights re-normalize among available signals).

    Requirements: 3.2
    """
    components: list[tuple[float, float]] = []  # (value, weight)

    if composite_gnn_score is not None:
        # Clamp to [0, 1]
        clamped = max(0.0, min(1.0, composite_gnn_score))
        components.append((clamped, 0.4))

    if fpocket_druggability is not None:
        clamped = max(0.0, min(1.0, fpocket_druggability))
        components.append((clamped, 0.3))

    if volume_angstrom3 is not None:
        # Normalize volume: cap at 2000 Å³
        vol_score = max(0.0, min(volume_angstrom3 / 2000.0, 1.0))
        components.append((vol_score, 0.2))

    # Site type bonus component
    bonus = 0.1 if site_type in BONUS_SITE_TYPES else 0.0
    components.append((bonus / 0.1 if bonus > 0 else 0.0, 0.1))

    if not components:
        return 0.0

    # Weighted average with re-normalization
    total_weight = sum(w for _, w in components)
    if total_weight == 0.0:
        return 0.0

    score = sum(val * weight for val, weight in components) / total_weight

    # Final clamp to [0, 1]
    return max(0.0, min(1.0, score))


def merge_candidates(
    gnn_candidates: list[CandidateCluster],
    geometry_pockets: list[GeometryPocket],
    overlap_threshold_angstrom: float = 5.0,
) -> list[UnifiedCandidate]:
    """Merge GNN-derived and geometry-derived candidates into unified list.

    Rules:
    - If a geometry pocket centroid is within overlap_threshold of a GNN cluster
      centroid, merge into one UnifiedCandidate with discovery_method="hybrid"
    - If a geometry pocket has no GNN overlap, include as discovery_method="geometry"
    - GNN clusters without geometry overlap remain as discovery_method="gnn_strain"

    Returns merged list sorted by druggability_score descending with site_rank assigned.

    Requirements: 2.2, 2.3, 2.4, 2.5, 3.1, 3.3
    """
    merged: list[UnifiedCandidate] = []

    # Track which GNN candidates and geometry pockets have been merged
    gnn_merged: set[int] = set()  # indices into gnn_candidates
    geo_merged: set[int] = set()  # indices into geometry_pockets

    # Find spatial overlaps: geometry pocket within threshold of GNN cluster
    for gi, geo in enumerate(geometry_pockets):
        best_gnn_idx: int | None = None
        best_distance = float("inf")

        for ci, gnn in enumerate(gnn_candidates):
            if ci in gnn_merged:
                continue
            dist = _euclidean_distance(geo.centroid_xyz, gnn.centroid_xyz)
            if dist < overlap_threshold_angstrom and dist < best_distance:
                best_gnn_idx = ci
                best_distance = dist

        if best_gnn_idx is not None:
            # Merge: hybrid candidate
            gnn = gnn_candidates[best_gnn_idx]
            gnn_merged.add(best_gnn_idx)
            geo_merged.add(gi)

            # Combine residue IDs from both sources (deduplicated)
            combined_residues = sorted(set(gnn.residue_ids + geo.residue_ids))

            # Use GNN centroid as primary (it's derived from Cα positions)
            site_type = "surface_pocket"  # will be overridden by mapper in scan phase
            druggability = compute_druggability_score(
                composite_gnn_score=gnn.composite_score,
                fpocket_druggability=geo.druggability_score,
                volume_angstrom3=geo.volume_angstrom3,
                site_type=site_type,
            )

            merged.append(
                UnifiedCandidate(
                    site_id=str(uuid.uuid4()),
                    residue_ids=combined_residues,
                    centroid_xyz=gnn.centroid_xyz,
                    site_type=site_type,
                    discovery_method=DISCOVERY_HYBRID,
                    druggability_score=druggability,
                    composite_gnn_score=gnn.composite_score,
                    fpocket_druggability=geo.druggability_score,
                    volume_angstrom3=geo.volume_angstrom3,
                )
            )

    # GNN-only candidates (no geometry overlap)
    for ci, gnn in enumerate(gnn_candidates):
        if ci in gnn_merged:
            continue

        site_type = "surface_pocket"  # placeholder until mapper classifies
        druggability = compute_druggability_score(
            composite_gnn_score=gnn.composite_score,
            fpocket_druggability=None,
            volume_angstrom3=None,
            site_type=site_type,
        )

        merged.append(
            UnifiedCandidate(
                site_id=str(uuid.uuid4()),
                residue_ids=list(gnn.residue_ids),
                centroid_xyz=gnn.centroid_xyz,
                site_type=site_type,
                discovery_method=DISCOVERY_GNN,
                druggability_score=druggability,
                composite_gnn_score=gnn.composite_score,
                fpocket_druggability=None,
                volume_angstrom3=None,
            )
        )

    # Geometry-only pockets (no GNN overlap)
    for gi, geo in enumerate(geometry_pockets):
        if gi in geo_merged:
            continue

        druggability = compute_druggability_score(
            composite_gnn_score=None,
            fpocket_druggability=geo.druggability_score,
            volume_angstrom3=geo.volume_angstrom3,
            site_type="surface_pocket",
        )

        merged.append(
            UnifiedCandidate(
                site_id=str(uuid.uuid4()),
                residue_ids=list(geo.residue_ids),
                centroid_xyz=geo.centroid_xyz,
                site_type="surface_pocket",
                discovery_method=DISCOVERY_GEOMETRY,
                druggability_score=druggability,
                composite_gnn_score=None,
                fpocket_druggability=geo.druggability_score,
                volume_angstrom3=geo.volume_angstrom3,
            )
        )

    # Sort by druggability descending, assign ranks
    merged = assign_ranks(merged)

    return merged


def assign_ranks(candidates: list[UnifiedCandidate]) -> list[UnifiedCandidate]:
    """Assign site_rank 1..N to candidates ordered by druggability_score descending.

    Ties are broken by site_id for determinism.

    Requirements: 3.3
    """
    # Sort by druggability descending, then site_id ascending for tie-breaking
    candidates.sort(key=lambda c: (-c.druggability_score, c.site_id))

    for rank, candidate in enumerate(candidates, start=1):
        candidate.site_rank = rank

    return candidates
