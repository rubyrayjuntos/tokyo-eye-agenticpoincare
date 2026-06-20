"""Scan Phase Orchestrator for the binding site scan at ingestion.

Composes seed_generator + pocket_detector + site_merger into a full-structure
binding site scan. Fetches GNN node outputs and Cα coordinates from the database,
runs classification on each cluster, and persists all results.

This is the 'Phase 3.5' that executes automatically after GNN inference
and graph topology computation complete.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent.tools.cryptic.pocket_detector import detect_surface_pockets
from agent.tools.cryptic.seed_generator import (
    CandidateCluster,
    GNNNodeOutput,
    generate_seeds_from_gnn,
)
from agent.tools.cryptic.site_merger import (
    UnifiedCandidate,
    assign_ranks,
    compute_druggability_score,
    merge_candidates,
)

logger = logging.getLogger(__name__)

# Current heuristic version for site_type classification
HEURISTIC_VERSION = "v1.0"

# Model version (default, overridden by DB provenance if available)
DEFAULT_MODEL_VERSION = "GOSPConeMapper-v5"


@dataclass
class ScanResult:
    """Complete result of a full-structure binding site scan."""

    structure_id: str
    run_id: str
    candidates: list[UnifiedCandidate]
    scan_parameters: dict[str, Any]
    heuristic_version: str
    model_version: str
    duration_ms: int
    warnings: list[str] = field(default_factory=list)


def classify_site_type(
    cluster: CandidateCluster,
    graph_metrics: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Classify a candidate cluster into a site type using the v1 heuristic.

    Heuristic rules (v1.0):
    - High composite_score (>= 0.7) + bridge residues → "cryptic_wedge"
    - High composite_score (>= 0.7) + high betweenness (>= 0.1) → "structural_stent"
    - Moderate composite_score (>= 0.5) + high clustering coeff → "dynamic_lid"
    - Moderate composite_score (>= 0.5) + low clustering → "allosteric_clamp"
    - Lower composite_score (>= 0.3) → "strain_relief_insert"
    - Default fallback → "surface_pocket"

    This is a placeholder heuristic that will be calibrated against
    benchmark data (see Requirement 7).
    """
    score = cluster.composite_score

    # If we have graph metrics, use them for refined classification
    if graph_metrics:
        # Check if any cluster residue is a bridge node
        has_bridge = any(
            graph_metrics.get(rid, {}).get("is_bridge", False)
            for rid in cluster.residue_ids
        )
        # Average betweenness of cluster residues
        betweenness_vals = [
            graph_metrics.get(rid, {}).get("betweenness", 0.0)
            for rid in cluster.residue_ids
        ]
        avg_betweenness = (
            sum(betweenness_vals) / len(betweenness_vals) if betweenness_vals else 0.0
        )
        # Average clustering coefficient
        clustering_vals = [
            graph_metrics.get(rid, {}).get("clustering_coefficient", 0.0)
            for rid in cluster.residue_ids
        ]
        avg_clustering = (
            sum(clustering_vals) / len(clustering_vals) if clustering_vals else 0.0
        )

        if score >= 0.7 and has_bridge:
            return "cryptic_wedge"
        if score >= 0.7 and avg_betweenness >= 0.1:
            return "structural_stent"
        if score >= 0.5 and avg_clustering >= 0.3:
            return "dynamic_lid"
        if score >= 0.5:
            return "allosteric_clamp"
        if score >= 0.3:
            return "strain_relief_insert"
    else:
        # Without graph metrics, use score-only heuristic
        if score >= 0.7:
            return "cryptic_wedge"
        if score >= 0.5:
            return "structural_stent"
        if score >= 0.3:
            return "strain_relief_insert"

    return "surface_pocket"


async def _fetch_gnn_nodes(structure_id: str, db: Any) -> list[GNNNodeOutput]:
    """Fetch GNN node outputs from fact_gnn_node_embedding for a structure.

    Returns the most recent run's embeddings (by computed_at descending).
    """
    rows = await db.fetch_all(
        """
        SELECT e.residue_id, e.epistemic_uncertainty, e.cone_depth
        FROM fact_gnn_node_embedding e
        JOIN provenance_run p ON p.run_id = e.run_id
        WHERE e.structure_id = :structure_id
          AND p.run_type = 'inference'
        ORDER BY e.computed_at DESC
        """,
        {"structure_id": structure_id},
    )

    if not rows:
        return []

    # Deduplicate by residue_id (keep most recent via ORDER BY)
    seen: set[str] = set()
    nodes: list[GNNNodeOutput] = []
    for row in rows:
        rid = row["residue_id"]
        if rid in seen:
            continue
        seen.add(rid)
        nodes.append(
            GNNNodeOutput(
                residue_id=rid,
                epistemic_uncertainty=float(row.get("epistemic_uncertainty") or 0.0),
                cone_depth=float(row.get("cone_depth") or 0.0),
            )
        )

    return nodes


async def _fetch_ca_coordinates(
    structure_id: str, db: Any
) -> dict[str, tuple[float, float, float]]:
    """Fetch Cα coordinates from dim_atom for all residues of a structure.

    Returns dict mapping residue_id → (x, y, z).
    """
    rows = await db.fetch_all(
        """
        SELECT r.residue_id, a.x, a.y, a.z
        FROM dim_atom a
        JOIN dim_residue r ON r.residue_id = a.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
          AND a.atom_name = 'CA'
          AND a.x IS NOT NULL
        """,
        {"structure_id": structure_id},
    )

    return {
        row["residue_id"]: (float(row["x"]), float(row["y"]), float(row["z"]))
        for row in rows
    }


async def _fetch_graph_metrics(
    structure_id: str, db: Any
) -> dict[str, dict[str, Any]]:
    """Fetch graph node metrics for classification heuristic.

    Returns dict mapping residue_id → {betweenness, clustering_coefficient, is_bridge, ...}.
    """
    rows = await db.fetch_all(
        """
        SELECT residue_id, betweenness, clustering_coefficient, is_bridge,
               degree, closeness, eigenvector_centrality, conductance
        FROM fact_graph_node_metrics
        WHERE structure_id = :structure_id
        ORDER BY computed_at DESC
        """,
        {"structure_id": structure_id},
    )

    # Deduplicate by residue_id (keep most recent)
    metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = row["residue_id"]
        if rid in metrics:
            continue
        metrics[rid] = {
            "betweenness": float(row.get("betweenness") or 0.0),
            "clustering_coefficient": float(row.get("clustering_coefficient") or 0.0),
            "is_bridge": bool(row.get("is_bridge", False)),
            "degree": int(row.get("degree") or 0),
            "closeness": float(row.get("closeness") or 0.0),
            "eigenvector_centrality": float(row.get("eigenvector_centrality") or 0.0),
            "conductance": float(row.get("conductance") or 0.0),
        }

    return metrics


async def _fetch_model_version(structure_id: str, db: Any) -> str:
    """Fetch model_version from the most recent GNN inference run for this structure."""
    row = await db.fetch_one(
        """
        SELECT p.model_version
        FROM provenance_run p
        JOIN fact_gnn_node_embedding e ON e.run_id = p.run_id
        WHERE e.structure_id = :structure_id
          AND p.run_type = 'inference'
        ORDER BY p.started_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    if row and row.get("model_version"):
        return row["model_version"]
    return DEFAULT_MODEL_VERSION


async def _persist_scan_candidates(
    candidates: list[UnifiedCandidate],
    structure_id: str,
    run_id: str,
    db: Any,
) -> None:
    """Persist all candidates to fact_cryptic_site via direct SQL upsert.

    Follows the normalizer pattern: idempotent upsert on (site_id).
    """
    if not candidates:
        return

    params_list = [
        {
            "site_id": c.site_id,
            "structure_id": structure_id,
            "run_id": run_id,
            "residue_ids": c.residue_ids,
            "centroid_x": c.centroid_xyz[0],
            "centroid_y": c.centroid_xyz[1],
            "centroid_z": c.centroid_xyz[2],
            "site_type": c.site_type,
            "discovery_method": c.discovery_method,
            "druggability_score": c.druggability_score,
            "site_rank": c.site_rank,
            "composite_gnn_score": c.composite_gnn_score,
            "fpocket_druggability": c.fpocket_druggability,
            "volume_angstrom3": c.volume_angstrom3,
            "provenance_gate": c.provenance_gate,
            "heuristic_version": c.heuristic_version,
            "md_validation_status": "pending",
            "scan_run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for c in candidates
    ]

    await db.execute_many(
        """
        INSERT INTO fact_cryptic_site (
            site_id, structure_id, run_id, residue_ids,
            centroid_x, centroid_y, centroid_z,
            site_type, discovery_method, druggability_score, site_rank,
            composite_gnn_score, fpocket_druggability, volume_angstrom3,
            provenance_gate, heuristic_version, md_validation_status,
            scan_run_id, created_at
        ) VALUES (
            :site_id, :structure_id, :run_id, :residue_ids,
            :centroid_x, :centroid_y, :centroid_z,
            :site_type, :discovery_method, :druggability_score, :site_rank,
            :composite_gnn_score, :fpocket_druggability, :volume_angstrom3,
            :provenance_gate, :heuristic_version, :md_validation_status,
            :scan_run_id, :created_at
        )
        ON CONFLICT (site_id) DO UPDATE SET
            site_type = EXCLUDED.site_type,
            discovery_method = EXCLUDED.discovery_method,
            druggability_score = EXCLUDED.druggability_score,
            site_rank = EXCLUDED.site_rank,
            composite_gnn_score = EXCLUDED.composite_gnn_score,
            fpocket_druggability = EXCLUDED.fpocket_druggability,
            volume_angstrom3 = EXCLUDED.volume_angstrom3,
            provenance_gate = EXCLUDED.provenance_gate,
            heuristic_version = EXCLUDED.heuristic_version,
            scan_run_id = EXCLUDED.scan_run_id
        """,
        params_list,
    )


async def _persist_scan_metadata(
    structure_id: str,
    run_id: str,
    heuristic_version: str,
    model_version: str,
    scan_parameters: dict[str, Any],
    sites_found: int,
    duration_ms: int,
    status: str,
    db: Any,
) -> None:
    """Persist scan run metadata to fact_binding_site_scan."""
    await db.execute(
        """
        INSERT INTO fact_binding_site_scan (
            structure_id, run_id, heuristic_version, model_version,
            scan_parameters, sites_found, duration_ms, status, created_at
        ) VALUES (
            :structure_id, :run_id, :heuristic_version, :model_version,
            :scan_parameters, :sites_found, :duration_ms, :status, :created_at
        )
        ON CONFLICT (structure_id, run_id) DO UPDATE SET
            heuristic_version = EXCLUDED.heuristic_version,
            model_version = EXCLUDED.model_version,
            scan_parameters = EXCLUDED.scan_parameters,
            sites_found = EXCLUDED.sites_found,
            duration_ms = EXCLUDED.duration_ms,
            status = EXCLUDED.status
        """,
        {
            "structure_id": structure_id,
            "run_id": run_id,
            "heuristic_version": heuristic_version,
            "model_version": model_version,
            "scan_parameters": scan_parameters,
            "sites_found": sites_found,
            "duration_ms": duration_ms,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


async def _delete_previous_scan_results(structure_id: str, db: Any) -> None:
    """Delete previous scan results for a structure before re-scanning.

    Requirement 4.5: Re-scan replaces previous results.
    """
    await db.execute(
        "DELETE FROM fact_cryptic_site WHERE structure_id = :structure_id AND scan_run_id IS NOT NULL",
        {"structure_id": structure_id},
    )


async def run_full_structure_scan(
    structure_id: str,
    db: Any,
    uncertainty_threshold: float = 9.5,
    cone_depth_threshold: float = 6.0,
    eps_angstrom: float = 8.0,
    min_cluster_size: int = 3,
    max_clusters: int = 25,
    overlap_threshold_angstrom: float = 5.0,
) -> ScanResult:
    """Execute full-structure binding site scan.

    Pipeline:
    1. Fetch GNN node outputs from DB (fact_gnn_node_embedding)
    2. Fetch Cα coordinates from dim_atom
    3. Generate seed clusters via DBSCAN
    4. Fetch graph metrics for classification heuristic
    5. Run mapper classification on each cluster
    6. Run fpocket pocket detection
    7. Merge GNN + geometry candidates
    8. Classify and recalculate druggability with proper site_types
    9. Rank by druggability_score
    10. Delete old scan results for this structure
    11. Persist all candidates to fact_cryptic_site
    12. Persist scan metadata to fact_binding_site_scan
    13. Return ScanResult

    This function is called automatically by the pipeline orchestrator
    after Phase 3 + graph topology complete.

    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    start_time = time.monotonic()
    warnings: list[str] = []
    run_id = f"scan_{structure_id}_{uuid.uuid4().hex[:8]}"

    scan_parameters = {
        "uncertainty_threshold": uncertainty_threshold,
        "cone_depth_threshold": cone_depth_threshold,
        "eps_angstrom": eps_angstrom,
        "min_cluster_size": min_cluster_size,
        "max_clusters": max_clusters,
        "overlap_threshold_angstrom": overlap_threshold_angstrom,
    }

    # Step 1: Fetch GNN node outputs
    gnn_nodes = await _fetch_gnn_nodes(structure_id, db)
    if not gnn_nodes:
        logger.warning(
            "No GNN embeddings found for structure %s; scan phase skipped", structure_id
        )
        warnings.append("No GNN embeddings found. Run DTIE pipeline first.")
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Persist metadata indicating no data
        await _persist_scan_metadata(
            structure_id=structure_id,
            run_id=run_id,
            heuristic_version=HEURISTIC_VERSION,
            model_version=DEFAULT_MODEL_VERSION,
            scan_parameters=scan_parameters,
            sites_found=0,
            duration_ms=duration_ms,
            status="no_gnn_data",
            db=db,
        )

        return ScanResult(
            structure_id=structure_id,
            run_id=run_id,
            candidates=[],
            scan_parameters=scan_parameters,
            heuristic_version=HEURISTIC_VERSION,
            model_version=DEFAULT_MODEL_VERSION,
            duration_ms=duration_ms,
            warnings=warnings,
        )

    # Step 2: Fetch Cα coordinates
    ca_coords = await _fetch_ca_coordinates(structure_id, db)
    if not ca_coords:
        logger.warning(
            "No Cα coordinates found for structure %s; scan phase skipped", structure_id
        )
        warnings.append("No Cα coordinates found in dim_atom.")
        duration_ms = int((time.monotonic() - start_time) * 1000)

        await _persist_scan_metadata(
            structure_id=structure_id,
            run_id=run_id,
            heuristic_version=HEURISTIC_VERSION,
            model_version=DEFAULT_MODEL_VERSION,
            scan_parameters=scan_parameters,
            sites_found=0,
            duration_ms=duration_ms,
            status="no_coordinates",
            db=db,
        )

        return ScanResult(
            structure_id=structure_id,
            run_id=run_id,
            candidates=[],
            scan_parameters=scan_parameters,
            heuristic_version=HEURISTIC_VERSION,
            model_version=DEFAULT_MODEL_VERSION,
            duration_ms=duration_ms,
            warnings=warnings,
        )

    # Step 3: Generate seed clusters via DBSCAN
    clusters = generate_seeds_from_gnn(
        gnn_nodes=gnn_nodes,
        ca_coords=ca_coords,
        uncertainty_threshold=uncertainty_threshold,
        cone_depth_threshold=cone_depth_threshold,
        eps_angstrom=eps_angstrom,
        min_cluster_size=min_cluster_size,
        max_clusters=max_clusters,
    )

    # Step 4: Fetch graph metrics for classification
    graph_metrics = await _fetch_graph_metrics(structure_id, db)

    # Step 5: Classify each cluster using the heuristic
    # Build a mapping of cluster_id → site_type for post-merge application
    cluster_type_map: dict[int, str] = {}
    for cluster in clusters:
        cluster_type_map[cluster.cluster_id] = classify_site_type(
            cluster, graph_metrics or None
        )

    # Step 6: Run fpocket pocket detection
    geometry_pockets = await detect_surface_pockets(structure_id, db)
    if not geometry_pockets:
        warnings.append(
            "Geometry detection returned no pockets (fpocket unavailable or no pockets found)."
        )

    # Step 7: Merge GNN + geometry candidates
    merged = merge_candidates(
        gnn_candidates=clusters,
        geometry_pockets=geometry_pockets,
        overlap_threshold_angstrom=overlap_threshold_angstrom,
    )

    # Step 8: Apply classifier site_types to merged candidates and recalculate druggability
    # For GNN-originated and hybrid candidates, apply the classified type
    for candidate in merged:
        if candidate.discovery_method in ("gnn_strain", "hybrid"):
            # Find the matching cluster by centroid proximity
            matched_type = _match_cluster_type(candidate, clusters, cluster_type_map)
            if matched_type:
                candidate.site_type = matched_type
                # Recalculate druggability with the correct site_type
                candidate.druggability_score = compute_druggability_score(
                    composite_gnn_score=candidate.composite_gnn_score,
                    fpocket_druggability=candidate.fpocket_druggability,
                    volume_angstrom3=candidate.volume_angstrom3,
                    site_type=candidate.site_type,
                )
        # Geometry-only stays as "surface_pocket"
        candidate.heuristic_version = HEURISTIC_VERSION

    # Step 9: Re-rank after classification updates druggability scores
    merged = assign_ranks(merged)

    # Fetch model version from provenance
    model_version = await _fetch_model_version(structure_id, db)

    # Step 10: Delete old scan results
    await _delete_previous_scan_results(structure_id, db)

    # Step 11: Persist candidates
    await _persist_scan_candidates(merged, structure_id, run_id, db)

    # Step 12: Persist scan metadata
    duration_ms = int((time.monotonic() - start_time) * 1000)
    status = "no_sites_found" if not merged else "complete"

    await _persist_scan_metadata(
        structure_id=structure_id,
        run_id=run_id,
        heuristic_version=HEURISTIC_VERSION,
        model_version=model_version,
        scan_parameters=scan_parameters,
        sites_found=len(merged),
        duration_ms=duration_ms,
        status=status,
        db=db,
    )

    logger.info(
        "Full structure scan complete: structure=%s, sites=%d, duration=%dms",
        structure_id,
        len(merged),
        duration_ms,
    )

    return ScanResult(
        structure_id=structure_id,
        run_id=run_id,
        candidates=merged,
        scan_parameters=scan_parameters,
        heuristic_version=HEURISTIC_VERSION,
        model_version=model_version,
        duration_ms=duration_ms,
        warnings=warnings,
    )


def _match_cluster_type(
    candidate: UnifiedCandidate,
    clusters: list[CandidateCluster],
    cluster_type_map: dict[int, str],
) -> str | None:
    """Match a merged candidate back to its originating cluster for type assignment.

    Uses centroid matching (exact match since merge preserves GNN centroid).
    """
    for cluster in clusters:
        if cluster.centroid_xyz == candidate.centroid_xyz:
            return cluster_type_map.get(cluster.cluster_id)
    return None
