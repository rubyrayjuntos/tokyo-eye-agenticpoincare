"""Atomic runner: binding_site_scan (Act 03 — Cryptic Pocket)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from psycopg.types.json import Json

from science.compute.jobs.binding_site_scan import run_binding_site_scan
from science.compute.provenance import job_provenance_parameters, pathway_pipeline_name
from science.compute.runners.base import JobRunContext, JobRunResult
from science.dtie.common.provenance_runtime import resolve_code_version
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

JOB_ID = "binding_site_scan"


async def _resolve_gnn_parent_run_id(db: Any, structure_id: str) -> str | None:
    row = await db.fetch_one(
        """
        SELECT p.run_id
        FROM fact_gnn_node_embedding f
        JOIN provenance_run p ON p.run_id = f.run_id
        JOIN embedding_space es ON es.space_id = f.space_id
        WHERE f.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
          AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
        ORDER BY p.started_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    if row is None:
        return None
    return str(row["run_id"])


async def run_binding_site_scan_job(db: Any, ctx: JobRunContext) -> JobRunResult:
    """Run binding-site scan as an atomic governed job."""
    structure_id = ctx.structure_id.strip().lower()
    run_id = f"job_{JOB_ID}_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)

    gnn_run_id = ctx.parent_run_id or await _resolve_gnn_parent_run_id(db, structure_id)
    if gnn_run_id is None:
        return JobRunResult(
            job_id=JOB_ID,
            run_id=run_id,
            structure_id=structure_id,
            success=False,
            outputs={"error": "No hyperbolic GNN embeddings found for structure"},
            warnings=["gnn_inference prerequisite not satisfied"],
        )

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
            "model_version": "binding-scan-v1",
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

    phase_result = await run_binding_site_scan(
        db, config, pipeline_run_id=run_id, job_params=ctx.job_params
    )
    warnings = list(phase_result.warnings or [])

    await db.execute(
        """
        UPDATE provenance_run
        SET completed_at = NOW(),
            warnings = :warnings
        WHERE run_id = :run_id
        """,
        {
            "run_id": run_id,
            "warnings": Json(warnings) if warnings else None,
        },
    )
    await db.commit()

    logger.info(
        "binding_site_scan complete structure=%s run_id=%s success=%s sites=%s",
        structure_id,
        run_id,
        phase_result.success,
        phase_result.outputs.get("sites_found", 0),
    )

    return JobRunResult(
        job_id=JOB_ID,
        run_id=run_id,
        structure_id=structure_id,
        success=phase_result.success,
        artifacts_produced=["binding_scan"] if phase_result.success else [],
        outputs=phase_result.outputs,
        warnings=warnings,
    )
