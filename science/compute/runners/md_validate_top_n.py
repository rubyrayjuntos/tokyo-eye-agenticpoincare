"""Atomic runner: md_validate_top_n (Act 03 — Cryptic Pocket, tier 2 GPU)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from psycopg.types.json import Json

from science.compute.cryptic.md_validate import validate_top_n_sites
from science.compute.provenance import job_provenance_parameters, pathway_pipeline_name
from science.compute.runners.base import JobRunContext, JobRunResult
from science.dtie.common.provenance_runtime import resolve_code_version

logger = logging.getLogger(__name__)

JOB_ID = "md_validate_top_n"
DEFAULT_TOP_N = 5


async def run_md_validate_top_n(db: Any, ctx: JobRunContext) -> JobRunResult:
    """Run top-N MD validation as an atomic job (tier 2 — non-blocking on failure)."""
    structure_id = ctx.structure_id.strip().lower()
    run_id = f"job_{JOB_ID}_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)

    await db.execute(
        """
        INSERT INTO provenance_run (
            run_id, structure_id, model_version, pipeline_name,
            run_type, source_type, started_at, parameters, code_version, parent_run_id
        )
        VALUES (
            :run_id, :structure_id, :model_version, :pipeline_name,
            :run_type, :source_type, NOW(), :parameters, :code_version, :parent_run_id
        )
        ON CONFLICT (run_id) DO NOTHING
        """,
        {
            "run_id": run_id,
            "structure_id": structure_id,
            "model_version": "smd-runner-v1",
            "pipeline_name": pathway_pipeline_name(ctx.pathway),
            "run_type": "analysis",
            "source_type": "deterministic",
            "parameters": Json(
                job_provenance_parameters(
                    JOB_ID,
                    pathway=ctx.pathway,
                    computation_run_id=ctx.computation_run_id,
                    extra={"top_n": DEFAULT_TOP_N},
                )
            ),
            "code_version": code_version,
            "parent_run_id": ctx.parent_run_id,
        },
    )

    top_n = int(ctx.job_params.get("top_n", DEFAULT_TOP_N))
    site_id = ctx.job_params.get("site_id")

    if site_id:
        from science.compute.cryptic.md_validate import validate_site_md_in_process

        if ctx.job_params.get("dry_run"):
            batch = {
                "structure_id": structure_id,
                "validated": 1,
                "passed": 0,
                "failed": 0,
                "timed_out": 0,
                "results": [{"site_id": site_id, "dry_run": True, "success": True}],
                "success": True,
            }
        else:
            site_result = await validate_site_md_in_process(
                str(site_id),
                db,
                timeout_seconds=int(ctx.job_params.get("timeout_seconds", 1800)),
            )
            batch = {
                "structure_id": structure_id,
                "validated": 1,
                "passed": 1 if site_result.get("md_validation_status") == "passed" else 0,
                "failed": 1 if site_result.get("md_validation_status") == "failed" else 0,
                "timed_out": 1 if site_result.get("md_validation_status") == "timeout" else 0,
                "results": [site_result],
                "success": site_result.get("success", False),
            }
    else:
        batch = await validate_top_n_sites(structure_id, db, top_n=top_n)
    warnings: list[str] = []
    if batch.get("validated", 0) == 0:
        warnings.append("No sites eligible for MD validation")

    await db.execute(
        """
        UPDATE provenance_run
        SET completed_at = NOW(),
            warnings = :warnings,
            parameters = COALESCE(parameters, '{}'::jsonb) || :result_params::jsonb
        WHERE run_id = :run_id
        """,
        {
            "run_id": run_id,
            "warnings": Json(warnings) if warnings else None,
            "result_params": Json(
                {
                    "validated": batch.get("validated", 0),
                    "passed": batch.get("passed", 0),
                    "failed": batch.get("failed", 0),
                }
            ),
        },
    )
    await db.commit()

    success = batch.get("validated", 0) > 0 and batch.get("passed", 0) > 0
    if batch.get("validated", 0) == 0:
        success = True

    logger.info(
        "md_validate_top_n complete structure=%s run_id=%s validated=%s passed=%s",
        structure_id,
        run_id,
        batch.get("validated", 0),
        batch.get("passed", 0),
    )

    return JobRunResult(
        job_id=JOB_ID,
        run_id=run_id,
        structure_id=structure_id,
        success=success,
        artifacts_produced=["md_validation"] if batch.get("passed", 0) > 0 else [],
        outputs=batch,
        warnings=warnings,
    )
