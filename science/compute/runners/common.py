"""Shared helpers for atomic compute job runners."""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Json

from science.compute.provenance import job_provenance_parameters, pathway_pipeline_name
from science.compute.runners.base import JobRunContext


async def resolve_gnn_parent_run_id(db: Any, structure_id: str) -> str | None:
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
    return str(row["run_id"]) if row else None


async def insert_job_provenance(
    db: Any,
    *,
    run_id: str,
    structure_id: str,
    job_id: str,
    model_version: str,
    ctx: JobRunContext,
    code_version: str | None,
    parent_run_id: str | None,
) -> None:
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
            "model_version": model_version,
            "pipeline_name": pathway_pipeline_name(ctx.pathway),
            "run_type": "analysis",
            "source_type": "deterministic",
            "parameters": Json(
                job_provenance_parameters(
                    job_id,
                    pathway=ctx.pathway,
                    computation_run_id=ctx.computation_run_id,
                )
            ),
            "code_version": code_version,
            "parent_run_id": parent_run_id,
        },
    )


async def finalize_job_provenance(
    db: Any,
    *,
    run_id: str,
    warnings: list[str] | None,
) -> None:
    await db.execute(
        """
        UPDATE provenance_run
        SET completed_at = NOW(), warnings = :warnings
        WHERE run_id = :run_id
        """,
        {"run_id": run_id, "warnings": Json(warnings) if warnings else None},
    )
