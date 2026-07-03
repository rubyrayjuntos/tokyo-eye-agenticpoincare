"""Generic phase-backed atomic job runner."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from science.compute.persist import persist_phase_result
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.common import (
    finalize_job_provenance,
    insert_job_provenance,
    resolve_gnn_parent_run_id,
)
from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.provenance_runtime import resolve_code_version
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

PhaseExecutor = Callable[[Any, PipelineConfig, str | None], Awaitable[PhaseResult]]


async def run_phase_job(
    db: Any,
    ctx: JobRunContext,
    *,
    job_id: str,
    model_version: str,
    artifacts: list[str],
    execute: PhaseExecutor,
    caller_identity: str | None = None,
    persist: bool = True,
) -> JobRunResult:
    """Run a phase function, optionally persist, and record provenance."""
    structure_id = ctx.structure_id.strip().lower()
    run_id = f"job_{job_id}_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)
    gnn_run_id = ctx.parent_run_id or await resolve_gnn_parent_run_id(db, structure_id)

    await insert_job_provenance(
        db,
        run_id=run_id,
        structure_id=structure_id,
        job_id=job_id,
        model_version=model_version,
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

    phase_result = await execute(db, config, gnn_run_id=gnn_run_id)
    warnings = list(phase_result.warnings or [])

    if persist and phase_result.success:
        try:
            await persist_phase_result(
                db,
                phase_result,
                run_id=run_id,
                gnn_run_id=gnn_run_id,
                config=config,
                caller_identity=caller_identity or f"compute_job_{job_id}",
            )
            await db.commit()
        except Exception as exc:
            logger.exception("%s persistence failed for %s", job_id, structure_id)
            warnings.append(f"persistence failed: {exc}")
            phase_result.success = False
            phase_result.outputs["error"] = str(exc)

    await finalize_job_provenance(db, run_id=run_id, warnings=warnings or None)
    await db.commit()

    return JobRunResult(
        job_id=job_id,
        run_id=run_id,
        structure_id=structure_id,
        success=phase_result.success,
        artifacts_produced=artifacts if phase_result.success else [],
        outputs=phase_result.outputs,
        warnings=warnings,
    )
