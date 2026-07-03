"""Offline binding-site scan (no DB) for diagnostics and pipeline demos."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.tools.cryptic.gnn_channel_profile import GNNChannelProfile
from agent.tools.cryptic.pocket_detector import GeometryPocket
from agent.tools.cryptic.scan_phase import classify_site_type, HEURISTIC_VERSION
from agent.tools.cryptic.seed_generator import GNNNodeOutput, generate_seeds_from_gnn
from agent.tools.cryptic.site_merger import (
    UnifiedCandidate,
    assign_ranks,
    compute_druggability_score,
    merge_candidates,
)

logger = logging.getLogger(__name__)


@dataclass
class OfflineScanResult:
    structure_id: str
    run_id: str
    channel_profile: str
    candidates: list[UnifiedCandidate]
    n_qualifying_residues: int
    thresholds: dict[str, float]
    scan_parameters: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def run_offline_binding_scan(
    *,
    structure_id: str,
    gnn_nodes: list[GNNNodeOutput],
    ca_coords: dict[str, tuple[float, float, float]],
    geometry_pockets: list[GeometryPocket],
    channel_profile: GNNChannelProfile,
    graph_metrics: dict[str, dict[str, Any]] | None = None,
    eps_angstrom: float = 8.0,
    min_cluster_size: int = 3,
    max_clusters: int = 25,
    overlap_threshold_angstrom: float = 5.0,
    run_id: str | None = None,
) -> OfflineScanResult:
    """Run seed → merge → classify → rank without persistence."""
    from agent.tools.cryptic.gnn_channel_profile import (
        filter_qualifying_residues_profile,
        resolve_profile_thresholds,
    )

    run_id = run_id or f"offline_{structure_id}_{uuid.uuid4().hex[:8]}"
    warnings: list[str] = []

    epi_thr, shell_thr = resolve_profile_thresholds(gnn_nodes, channel_profile)
    qualifying = filter_qualifying_residues_profile(gnn_nodes, channel_profile)

    clusters = generate_seeds_from_gnn(
        gnn_nodes=gnn_nodes,
        ca_coords=ca_coords,
        eps_angstrom=eps_angstrom,
        min_cluster_size=min_cluster_size,
        max_clusters=max_clusters,
        channel_profile=channel_profile,
    )

    cluster_type_map: dict[int, str] = {}
    for cluster in clusters:
        cluster_type_map[cluster.cluster_id] = classify_site_type(
            cluster, graph_metrics or None
        )

    merged = merge_candidates(
        gnn_candidates=clusters,
        geometry_pockets=geometry_pockets,
        overlap_threshold_angstrom=overlap_threshold_angstrom,
    )

    for candidate in merged:
        if candidate.discovery_method in ("gnn_strain", "hybrid"):
            matched_type = _match_cluster_type(candidate, clusters, cluster_type_map)
            if matched_type:
                candidate.site_type = matched_type
                candidate.druggability_score = compute_druggability_score(
                    composite_gnn_score=candidate.composite_gnn_score,
                    fpocket_druggability=candidate.fpocket_druggability,
                    volume_angstrom3=candidate.volume_angstrom3,
                    site_type=candidate.site_type,
                )
        candidate.heuristic_version = HEURISTIC_VERSION

    merged = assign_ranks(merged)

    if not geometry_pockets:
        warnings.append("No geometry pockets (fpocket unavailable or empty).")

    return OfflineScanResult(
        structure_id=structure_id,
        run_id=run_id,
        channel_profile=channel_profile.name,
        candidates=merged,
        n_qualifying_residues=len(qualifying),
        thresholds={
            "epistemic": epi_thr,
            "shell": shell_thr,
            "shell_field": channel_profile.shell_field,
        },
        scan_parameters={
            "eps_angstrom": eps_angstrom,
            "min_cluster_size": min_cluster_size,
            "max_clusters": max_clusters,
            "overlap_threshold_angstrom": overlap_threshold_angstrom,
        },
        warnings=warnings,
    )


def _match_cluster_type(
    candidate: UnifiedCandidate,
    clusters: list,
    cluster_type_map: dict[int, str],
) -> str | None:
    for cluster in clusters:
        if cluster.centroid_xyz == candidate.centroid_xyz:
            return cluster_type_map.get(cluster.cluster_id)
    return None
