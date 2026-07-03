"""Atomic runner: fragment_screen (Act 04 — planned stub)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from psycopg.types.json import Json

from science.compute.provenance import job_provenance_parameters, pathway_pipeline_name
from science.compute.runners.base import JobRunContext, JobRunResult
from science.dtie.common.provenance_runtime import resolve_code_version

logger = logging.getLogger(__name__)

JOB_ID = "fragment_screen"


async def run_fragment_screen(db: Any, ctx: JobRunContext) -> JobRunResult:
    """Stub runner — fragment screening is planned; records provenance only."""
    structure_id = ctx.structure_id.strip().lower()
    run_id = f"job_{JOB_ID}_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)
    warnings = ["fragment_screen is planned — no hits produced in this release"]

    await db.execute(
        """
        INSERT INTO provenance_run (
            run_id, structure_id, model_version, pipeline_name,
            run_type, source_type, started_at, parameters, code_version, parent_run_id,
            completed_at, warnings
        )
        VALUES (
            :run_id, :structure_id, :model_version, :pipeline_name,
            :run_type, :source_type, NOW(), :parameters, :code_version, :parent_run_id,
            NOW(), :warnings
        )
        ON CONFLICT (run_id) DO NOTHING
        """,
        {
            "run_id": run_id,
            "structure_id": structure_id,
            "model_version": "fragment-screen-stub-v0",
            "pipeline_name": pathway_pipeline_name(ctx.pathway),
            "run_type": "analysis",
            "source_type": "deterministic",
            "parameters": Json(
                job_provenance_parameters(
                    JOB_ID,
                    pathway=ctx.pathway,
                    computation_run_id=ctx.computation_run_id,
                    extra={"status": "planned"},
                )
            ),
            "code_version": code_version,
            "parent_run_id": ctx.parent_run_id,
            "warnings": Json(warnings),
        },
    )
    await db.commit()

    logger.info("fragment_screen stub complete structure=%s run_id=%s", structure_id, run_id)

    return JobRunResult(
        job_id=JOB_ID,
        run_id=run_id,
        structure_id=structure_id,
        success=True,
        artifacts_produced=[],
        outputs={"status": "planned", "hits": 0},
        warnings=warnings,
    )
