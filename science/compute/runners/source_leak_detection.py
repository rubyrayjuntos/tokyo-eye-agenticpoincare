"""Atomic runner: source_leak_detection (Act 02 — Persistent Leak)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from psycopg.types.json import Json

from science.compute.gnn_loader import load_gnn_inference_result
from science.compute.jobs.source_leak_detection import detect_source_leaks
from science.compute.persist import persist_phase_result
from science.compute.provenance import job_provenance_parameters, pathway_pipeline_name
from science.compute.runners.base import JobRunContext, JobRunResult
from science.dtie.common.provenance_runtime import resolve_code_version
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

JOB_ID = "source_leak_detection"


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


async def run_source_leak_detection(db: Any, ctx: JobRunContext) -> JobRunResult:
    """Run source-leak detection as an atomic governed job."""
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
            "model_version": "discovery-source-leak-v1",
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
        detect_source_leaks=True,
    )

    gnn_result = await load_gnn_inference_result(
        db,
        structure_id,
        gnn_run_id=gnn_run_id,
    )
    if gnn_result is None or not gnn_result.nodes:
        return JobRunResult(
            job_id=JOB_ID,
            run_id=run_id,
            structure_id=structure_id,
            success=False,
            outputs={"error": "No hyperbolic GNN nodes available for source-leak detection"},
            warnings=["gnn_inference prerequisite not satisfied"],
        )

    phase_result = await detect_source_leaks(db, config, gnn_result=gnn_result)
    warnings: list[str] = []

    if phase_result.success:
        try:
            await persist_phase_result(
                db,
                phase_result,
                run_id=run_id,
                gnn_run_id=gnn_run_id,
                config=config,
                caller_identity="compute_job_source_leak_detection",
            )
            await db.commit()
        except Exception as exc:
            logger.exception("Source leak persistence failed for %s", structure_id)
            warnings.append(f"persistence failed: {exc}")
            phase_result.success = False
            phase_result.outputs["error"] = str(exc)
    else:
        warnings.append(phase_result.outputs.get("error", "detection failed"))

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
        "source_leak_detection complete structure=%s run_id=%s success=%s leaks=%s",
        structure_id,
        run_id,
        phase_result.success,
        phase_result.outputs.get("source_leak_count", 0),
    )

    return JobRunResult(
        job_id=JOB_ID,
        run_id=run_id,
        structure_id=structure_id,
        success=phase_result.success,
        artifacts_produced=["source_leaks"] if phase_result.success else [],
        outputs=phase_result.outputs,
        warnings=warnings,
    )
