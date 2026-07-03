"""Atomic job: hyperbolic_motifs (Act 02 — Persistent Leak, tier 2)."""

from __future__ import annotations

from typing import Any

from science.compute.motifs.discovery import discover_and_persist_motifs
from science.compute.provenance import pathway_pipeline_name
from science.dtie.common.interfaces import PhaseResult


async def run_hyperbolic_motif_discovery(
    db: Any,
    structure_id: str,
    *,
    run_id: str | None = None,
    pathway: str = "discovery_story",
    parent_run_id: str | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> PhaseResult:
    """Discover hyperbolic motifs from governed embeddings (HDBSCAN on disc)."""
    try:
        effective_run_id, motifs = await discover_and_persist_motifs(
            db,
            structure_id,
            run_id=run_id,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            caller_identity="compute_job_hyperbolic_motifs",
            pipeline_name=pathway_pipeline_name(pathway),
            parent_run_id=parent_run_id,
        )
    except ValueError as exc:
        return PhaseResult(
            phase_name="hyperbolic_motifs",
            structure_id=structure_id,
            model_version="motif-analyzer-v1",
            success=False,
            outputs={"error": str(exc)},
        )

    return PhaseResult(
        phase_name="hyperbolic_motifs",
        structure_id=structure_id,
        model_version="motif-analyzer-v1",
        success=True,
        outputs={"motif_count": len(motifs), "motifs": motifs, "run_id": effective_run_id},
    )
