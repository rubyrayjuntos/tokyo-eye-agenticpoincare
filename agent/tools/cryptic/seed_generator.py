"""Seed generator for exhaustive binding site scan.

Identifies all candidate binding site clusters from GNN node output using
DBSCAN spatial clustering on qualifying residues. This is the first step
in the full-structure scan phase (Phase 3.5).

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    from sklearn.cluster import DBSCAN
except ImportError as _sklearn_err:
    DBSCAN = None  # type: ignore[assignment,misc]
    _SKLEARN_MISSING = _sklearn_err
else:
    _SKLEARN_MISSING = None

if TYPE_CHECKING:
    from agent.tools.cryptic.gnn_channel_profile import GNNChannelProfile

logger = logging.getLogger(__name__)


@dataclass
class GNNNodeOutput:
    """Minimal GNN node output needed for seed generation."""

    residue_id: str
    epistemic_uncertainty: float
    cone_depth: float
    disc_r: float | None = None  # v6 shell boundary proxy (‖hyp_projections_2d‖)


@dataclass
class CandidateCluster:
    """A spatially-clustered group of high-signal residues."""

    cluster_id: int
    residue_ids: list[str]
    centroid_xyz: tuple[float, float, float]
    composite_score: float  # mean composite score of cluster members
    member_count: int


def _compute_residue_composite_score(
    epistemic_uncertainty: float, cone_depth: float
) -> float:
    """Compute a composite score for a single qualifying residue.

    Combines epistemic_uncertainty and cone_depth into a single [0, 1] signal.
    Both are normalized against practical maxima observed in real structures.
    """
    # Normalize uncertainty: typical range 0-30, cap at 30
    norm_uncertainty = min(epistemic_uncertainty / 30.0, 1.0)
    # Normalize cone_depth: typical range 0-20, cap at 20
    norm_depth = min(cone_depth / 20.0, 1.0)
    # Equal weighting
    return (norm_uncertainty + norm_depth) / 2.0


def filter_qualifying_residues(
    nodes: list[GNNNodeOutput],
    uncertainty_threshold: float = 9.5,
    cone_depth_threshold: float = 6.0,
) -> list[GNNNodeOutput]:
    """Filter residues that meet both uncertainty and cone_depth thresholds.

    A residue qualifies if and only if:
      epistemic_uncertainty >= uncertainty_threshold AND
      cone_depth >= cone_depth_threshold

    Requirements: 1.1
    """
    return [
        node
        for node in nodes
        if node.epistemic_uncertainty >= uncertainty_threshold
        and node.cone_depth >= cone_depth_threshold
    ]


def cluster_residues_dbscan(
    qualifying: list[GNNNodeOutput],
    ca_coords: dict[str, tuple[float, float, float]],
    eps_angstrom: float = 8.0,
    min_cluster_size: int = 3,
) -> list[CandidateCluster]:
    """Cluster qualifying residues using DBSCAN on Cα coordinates.

    Residues without Cα coordinates are silently skipped.

    Requirements: 1.2, 1.3
    """
    # Filter to residues with known coordinates
    residues_with_coords = [
        node for node in qualifying if node.residue_id in ca_coords
    ]

    if not residues_with_coords:
        return []

    # Build coordinate matrix
    coord_matrix = np.array(
        [ca_coords[node.residue_id] for node in residues_with_coords],
        dtype=np.float64,
    )

    # Run DBSCAN
    if DBSCAN is None:
        raise ImportError(
            "scikit-learn is required for cluster_residues_dbscan. "
            f"Install it with: pip install scikit-learn  (original error: {_SKLEARN_MISSING})"
        )
    clustering = DBSCAN(eps=eps_angstrom, min_samples=min_cluster_size, metric="euclidean")
    labels = clustering.fit_predict(coord_matrix)

    # Group residues by cluster label (label -1 = noise, discard)
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(idx)

    # Build CandidateCluster objects
    result: list[CandidateCluster] = []
    for cluster_id, member_indices in clusters.items():
        member_nodes = [residues_with_coords[i] for i in member_indices]
        member_coords = coord_matrix[member_indices]

        centroid = tuple(float(c) for c in member_coords.mean(axis=0))

        scores = [
            _compute_residue_composite_score(n.epistemic_uncertainty, n.cone_depth)
            for n in member_nodes
        ]
        composite_score = sum(scores) / len(scores)

        result.append(
            CandidateCluster(
                cluster_id=cluster_id,
                residue_ids=[n.residue_id for n in member_nodes],
                centroid_xyz=(centroid[0], centroid[1], centroid[2]),
                composite_score=composite_score,
                member_count=len(member_nodes),
            )
        )

    return result


def generate_seeds_from_gnn(
    gnn_nodes: list[GNNNodeOutput],
    ca_coords: dict[str, tuple[float, float, float]],
    uncertainty_threshold: float = 9.5,
    cone_depth_threshold: float = 6.0,
    eps_angstrom: float = 8.0,
    min_cluster_size: int = 3,
    max_clusters: int = 25,
    *,
    channel_profile: "GNNChannelProfile | None" = None,
) -> list[CandidateCluster]:
    """Identify all candidate binding site clusters from GNN output.

    Pipeline:
    1. Filter residues by epistemic + shell thresholds (profile or legacy kwargs)
    2. Extract Cα coordinates for qualifying residues
    3. Run DBSCAN with eps=eps_angstrom, min_samples=min_cluster_size
    4. Score each cluster by mean composite_score of its members
    5. Sort by score descending, cap at max_clusters

    Returns list of CandidateCluster sorted by composite_score descending.

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    from dataclasses import replace

    from agent.tools.cryptic.gnn_channel_profile import (
        V5_LEGACY,
        composite_score_profile,
        filter_qualifying_residues_profile,
        resolve_profile_thresholds,
    )

    if channel_profile is not None:
        profile = channel_profile
    elif uncertainty_threshold != 9.5 or cone_depth_threshold != 6.0:
        profile = replace(
            V5_LEGACY,
            epistemic_threshold=uncertainty_threshold,
            shell_threshold=cone_depth_threshold,
        )
    else:
        profile = V5_LEGACY

    # Step 1: Filter qualifying residues
    qualifying = filter_qualifying_residues_profile(gnn_nodes, profile)
    epi_thr, shell_thr = resolve_profile_thresholds(gnn_nodes, profile)

    if not qualifying:
        logger.info(
            "No qualifying residues (profile=%s, epi>=%.4f, shell>=%.4f)",
            profile.name,
            epi_thr,
            shell_thr,
        )
        return []

    # Steps 2-3: Cluster using DBSCAN
    clusters = cluster_residues_dbscan(
        qualifying, ca_coords, eps_angstrom, min_cluster_size
    )

    if not clusters:
        logger.info("DBSCAN produced no clusters from %d qualifying residues", len(qualifying))
        return []

    # Re-score clusters with profile-aware composite
    for cluster in clusters:
        member_by_id = {n.residue_id: n for n in qualifying}
        scores = [
            composite_score_profile(member_by_id[rid], profile)
            for rid in cluster.residue_ids
            if rid in member_by_id
        ]
        if scores:
            cluster.composite_score = sum(scores) / len(scores)

    # Step 4-5: Sort by composite_score descending
    clusters.sort(key=lambda c: c.composite_score, reverse=True)

    # Cap at max_clusters
    if len(clusters) > max_clusters:
        logger.warning(
            "Capping clusters from %d to %d (max_clusters limit). "
            "Discarding %d lower-scoring clusters.",
            len(clusters),
            max_clusters,
            len(clusters) - max_clusters,
        )
        clusters = clusters[:max_clusters]

    # Re-assign cluster_ids to be 0-based after sorting/capping
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i

    return clusters
