"""Atomic runner: witness_embedding (Act 01 — Signal)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from science.compute.jobs.witness_embedding import run_witness_embedding
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.common import (
    finalize_job_provenance,
    insert_job_provenance,
    resolve_gnn_parent_run_id,
)
from science.dtie.common.provenance_runtime import resolve_code_version
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

JOB_ID = "witness_embedding"


async def run_witness_embedding_job(db: Any, ctx: JobRunContext) -> JobRunResult:
    structure_id = ctx.structure_id.strip().lower()
    run_id = f"job_{JOB_ID}_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)
    gnn_run_id = ctx.parent_run_id or await resolve_gnn_parent_run_id(db, structure_id)

    if gnn_run_id is None:
        return JobRunResult(
            job_id=JOB_ID,
            run_id=run_id,
            structure_id=structure_id,
            success=False,
            outputs={"error": "No hyperbolic GNN embeddings found for structure"},
            warnings=["gnn_inference prerequisite not satisfied"],
        )

    await insert_job_provenance(
        db,
        run_id=run_id,
        structure_id=structure_id,
        job_id=JOB_ID,
        model_version="witness-embedding-v4",
        ctx=ctx,
        code_version=code_version,
        parent_run_id=gnn_run_id,
    )

    config = PipelineConfig(
        structure_id=structure_id,
        parent_run_id=ctx.computation_run_id or gnn_run_id,
        code_version=code_version,
        run_gnn=False,
    )

    phase_result = await run_witness_embedding(
        db, config, run_id=run_id, gnn_run_id=gnn_run_id
    )
    warnings = list(phase_result.warnings or [])

    if phase_result.success:
        try:
            await db.commit()
        except Exception as exc:
            logger.exception("Witness embedding commit failed for %s", structure_id)
            warnings.append(f"commit failed: {exc}")
            phase_result.success = False
            phase_result.outputs["error"] = str(exc)

    await finalize_job_provenance(db, run_id=run_id, warnings=warnings or None)
    await db.commit()

    return JobRunResult(
        job_id=JOB_ID,
        run_id=run_id,
        structure_id=structure_id,
        success=phase_result.success,
        artifacts_produced=["witness_embedding"] if phase_result.success else [],
        outputs=phase_result.outputs,
        warnings=warnings,
    )
