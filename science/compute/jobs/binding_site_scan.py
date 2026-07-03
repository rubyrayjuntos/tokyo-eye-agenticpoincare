"""Atomic job: binding_site_scan (Act 03 — Cryptic Pocket)."""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


async def run_binding_site_scan(
    db: Any,
    config: PipelineConfig,
    *,
    pipeline_run_id: str,
    job_params: dict[str, Any] | None = None,
) -> PhaseResult:
    """Run full-structure binding site scan after GNN + graph topology."""
    from agent.tools.cryptic.scan_phase import run_full_structure_scan

    params = job_params or {}
    scan_result = await run_full_structure_scan(
        structure_id=config.structure_id,
        db=db,
        run_id=pipeline_run_id,
        eps_angstrom=float(params.get("cluster_distance_angstrom", params.get("eps_angstrom", 8.0))),
        min_cluster_size=int(params.get("min_cluster_size", 3)),
        max_clusters=int(params.get("max_pockets", params.get("max_clusters", 25))),
    )

    scan_status = "complete" if scan_result.candidates else "no_sites_found"
    if scan_result.warnings and any("No GNN embeddings" in w for w in scan_result.warnings):
        scan_status = "no_gnn_data"

    sites: list[dict[str, Any]] = []
    for candidate in scan_result.candidates:
        sites.append({
            "site_id": candidate.site_id,
            "site_type": candidate.site_type,
            "residue_ids": candidate.residue_ids,
            "centroid_xyz": list(candidate.centroid_xyz),
            "druggability_score": candidate.druggability_score,
            "discovery_method": candidate.discovery_method,
            "site_rank": candidate.site_rank,
            "composite_gnn_score": candidate.composite_gnn_score,
            "volume_angstrom3": candidate.volume_angstrom3,
        })

    return PhaseResult(
        phase_name="binding_site_scan",
        structure_id=config.structure_id,
        model_version="binding-scan-v1",
        success=scan_status != "no_gnn_data",
        outputs={
            "sites_found": len(scan_result.candidates),
            "sites": sites,
            "scan_run_id": scan_result.run_id,
            "scan_status": scan_status,
            "duration_ms": scan_result.duration_ms,
            "heuristic_version": scan_result.heuristic_version,
            "warnings": scan_result.warnings,
            "pipeline_run_id": pipeline_run_id,
        },
        warnings=scan_result.warnings,
    )
