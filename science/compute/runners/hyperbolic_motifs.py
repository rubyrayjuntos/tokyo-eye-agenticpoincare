"""Atomic runner: hyperbolic_motifs (Act 02 — Persistent Leak)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from science.compute.jobs.hyperbolic_motifs import run_hyperbolic_motif_discovery
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.common import (
    finalize_job_provenance,
    insert_job_provenance,
    resolve_gnn_parent_run_id,
)
from science.dtie.common.provenance_runtime import resolve_code_version

logger = logging.getLogger(__name__)

JOB_ID = "hyperbolic_motifs"


async def run_hyperbolic_motifs_job(db: Any, ctx: JobRunContext) -> JobRunResult:
    structure_id = ctx.structure_id.strip().lower()
    run_id = f"job_{JOB_ID}_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)
    gnn_run_id = ctx.parent_run_id or await resolve_gnn_parent_run_id(db, structure_id)

    await insert_job_provenance(
        db,
        run_id=run_id,
        structure_id=structure_id,
        job_id=JOB_ID,
        model_version="motif-analyzer-v1",
        ctx=ctx,
        code_version=code_version,
        parent_run_id=gnn_run_id,
    )

    phase_result = await run_hyperbolic_motif_discovery(
        db,
        structure_id,
        run_id=run_id,
        pathway=ctx.pathway,
        parent_run_id=gnn_run_id,
        min_cluster_size=int(ctx.job_params.get("min_cluster_size", 5)),
        min_samples=int(ctx.job_params.get("min_samples", 3)),
    )
    warnings = list(phase_result.warnings or [])

    await finalize_job_provenance(db, run_id=run_id, warnings=warnings or None)
    await db.commit()

    return JobRunResult(
        job_id=JOB_ID,
        run_id=run_id,
        structure_id=structure_id,
        success=phase_result.success,
        artifacts_produced=["motifs"] if phase_result.success else [],
        outputs=phase_result.outputs,
        warnings=warnings,
    )
