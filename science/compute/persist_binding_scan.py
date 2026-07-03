"""Governed persistence for binding-site scan results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from data.normalizer.core import Normalizer
from science.dtie.common.normalizer_payloads import (
    BindingSiteScanMetadata,
    BindingSiteScanPayload,
    CrypticSiteRecord,
    ProvenanceContext,
    RunType,
    SourceType,
)


async def persist_binding_site_scan(
    db: Any,
    *,
    run_id: str,
    structure_id: str,
    candidates: list[Any],
    heuristic_version: str,
    model_version: str,
    scan_parameters: dict[str, Any],
    sites_found: int,
    duration_ms: int,
    status: str,
    parent_run_id: str | None = None,
    code_version: str | None = None,
    caller_identity: str = "binding_site_scan",
) -> None:
    """Persist scan candidates and metadata through the Normalizer."""
    from science.dtie.common.keys import coerce_residue_ids

    sites = [
        CrypticSiteRecord(
            site_id=c.site_id,
            residue_ids=coerce_residue_ids(list(c.residue_ids), structure_id),
            centroid_x=float(c.centroid_xyz[0]),
            centroid_y=float(c.centroid_xyz[1]),
            centroid_z=float(c.centroid_xyz[2]),
            site_type=c.site_type,
            discovery_method=c.discovery_method,
            druggability_score=float(c.druggability_score),
            site_rank=c.site_rank,
            composite_gnn_score=c.composite_gnn_score,
            fpocket_druggability=c.fpocket_druggability,
            volume_angstrom3=c.volume_angstrom3,
            provenance_gate=c.provenance_gate,
            heuristic_version=c.heuristic_version,
            scan_run_id=run_id,
        )
        for c in candidates
    ]

    payload = BindingSiteScanPayload(
        provenance=ProvenanceContext(
            run_id=run_id,
            structure_id=structure_id,
            model_version=model_version,
            pipeline_name="discovery_story",
            parent_run_id=parent_run_id,
            code_version=code_version,
            run_type=RunType.ANALYSIS,
            source_type=SourceType.DETERMINISTIC,
        ),
        structure_id=structure_id,
        sites=sites,
        scan_metadata=BindingSiteScanMetadata(
            heuristic_version=heuristic_version,
            model_version=model_version,
            scan_parameters=scan_parameters,
            sites_found=sites_found,
            duration_ms=duration_ms,
            status=status,
        ),
        computed_at=datetime.now(timezone.utc),
    )

    normalizer = Normalizer(db=db, caller_identity=caller_identity)
    await normalizer.normalize_binding_site_scan(payload)
