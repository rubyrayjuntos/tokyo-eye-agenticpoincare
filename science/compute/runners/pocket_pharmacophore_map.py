"""Atomic runner: pocket_pharmacophore_map (Act 03 optional)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from psycopg.types.json import Json

from science.compute.jobs.pocket_pharmacophore_map import run_pocket_pharmacophore_map
from science.compute.persist import persist_phase_result
from science.compute.provenance import job_provenance_parameters, pathway_pipeline_name
from science.compute.runners.base import JobRunContext, JobRunResult
from science.dtie.common.provenance_runtime import resolve_code_version
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

JOB_ID = "pocket_pharmacophore_map"


async def run_pocket_pharmacophore_map_job(db: Any, ctx: JobRunContext) -> JobRunResult:
    """Run pocket-scoped pharmacophore mapping (tier 2)."""
    structure_id = ctx.structure_id.strip().lower()
    run_id = f"job_{JOB_ID}_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)
    gnn_run_id = ctx.parent_run_id

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
            "model_version": "pocket-pharmacophore-v1",
            "pipeline_name": pathway_pipeline_name(ctx.pathway),
            "run_type": "analysis",
            "source_type": "deterministic",
            "parameters": Json(
                job_provenance_parameters(
                    JOB_ID,
                    pathway=ctx.pathway,
                    computation_run_id=ctx.computation_run_id,
                )
            ),
            "code_version": code_version,
            "parent_run_id": gnn_run_id,
        },
    )

    config = PipelineConfig(
        structure_id=structure_id,
        parent_run_id=ctx.computation_run_id or gnn_run_id,
        code_version=code_version,
        run_gnn=False,
    )

    phase_result = await run_pocket_pharmacophore_map(
        db, config, gnn_run_id=gnn_run_id
    )
    warnings = list(phase_result.warnings or [])

    if phase_result.success:
        try:
            from science.dtie.common.interfaces import PhaseResult

            persist_result = PhaseResult(
                phase_name="phase5_pharmacophore",
                structure_id=phase_result.structure_id,
                model_version=phase_result.model_version,
                success=phase_result.success,
                outputs=phase_result.outputs,
                warnings=phase_result.warnings,
            )
            await persist_phase_result(
                db,
                persist_result,
                run_id=run_id,
                gnn_run_id=gnn_run_id,
                config=config,
                caller_identity="compute_job_pocket_pharmacophore_map",
            )
            await db.commit()
        except Exception as exc:
            logger.exception("Pocket pharmacophore persistence failed for %s", structure_id)
            warnings.append(f"persistence failed: {exc}")
            phase_result.success = False
            phase_result.outputs["error"] = str(exc)

    await db.execute(
        """
        UPDATE provenance_run
        SET completed_at = NOW(), warnings = :warnings
        WHERE run_id = :run_id
        """,
        {"run_id": run_id, "warnings": Json(warnings) if warnings else None},
    )
    await db.commit()

    return JobRunResult(
        job_id=JOB_ID,
        run_id=run_id,
        structure_id=structure_id,
        success=phase_result.success,
        artifacts_produced=["pocket_pharmacophore"] if phase_result.success else [],
        outputs=phase_result.outputs,
        warnings=warnings,
    )
