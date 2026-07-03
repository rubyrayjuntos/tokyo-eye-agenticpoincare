"""Atomic runner: graph_topology (Act 01 — Signal)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from science.compute.jobs.graph_topology import run_graph_topology
from science.compute.provenance import pathway_pipeline_name
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.common import (
    finalize_job_provenance,
    insert_job_provenance,
    resolve_gnn_parent_run_id,
)
from science.dtie.common.provenance_runtime import resolve_code_version
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

JOB_ID = "graph_topology"


async def run_graph_topology_job(db: Any, ctx: JobRunContext) -> JobRunResult:
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
        model_version="graph-topology-v1",
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

    phase_result = await run_graph_topology(
        db,
        config,
        run_id=run_id,
        gnn_run_id=gnn_run_id,
        pipeline_name=pathway_pipeline_name(ctx.pathway),
        contact_cutoff_angstrom=float(
            ctx.job_params.get("contact_cutoff_angstrom", 8.0)
        ),
        chain_filter=ctx.job_params.get("chain_filter"),
    )
    warnings = list(phase_result.warnings or [])

    if phase_result.success:
        try:
            await db.commit()
        except Exception as exc:
            logger.exception("Graph topology commit failed for %s", structure_id)
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
        artifacts_produced=["graph"] if phase_result.success else [],
        outputs=phase_result.outputs,
        warnings=warnings,
    )
