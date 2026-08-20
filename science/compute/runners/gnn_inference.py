"""Atomic runner: gnn_inference (foundation)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from science.compute.jobs.gnn_inference import run_gnn_inference
from science.compute.provenance import pathway_pipeline_name
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.common import finalize_job_provenance, insert_job_provenance
from science.contracts.model_registry import (
    get_production_model_version,
    resolve_production_checkpoint,
    resolve_model_version_for_checkpoint,
)
from science.tokyo_eye.governance.resolve import ResolveError
from science.dtie.common.provenance_runtime import resolve_code_version
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

JOB_ID = "gnn_inference"


async def run_gnn_inference_job(db: Any, ctx: JobRunContext) -> JobRunResult:
    structure_id = ctx.structure_id.strip().lower()
    job_run_id = f"job_{JOB_ID}_{uuid.uuid4().hex[:12]}"
    gnn_run_id = f"run_{uuid.uuid4().hex[:12]}"
    code_version = resolve_code_version(ctx.code_version)
    try:
        checkpoint, model_version, restore_metadata = _resolve_checkpoint_for_job(ctx)
    except (ResolveError, ValueError) as exc:
        return JobRunResult(
            job_id=JOB_ID,
            run_id=job_run_id,
            structure_id=structure_id,
            success=False,
            artifacts_produced=[],
            outputs={
                "error": str(exc),
                "restore_source": "mlflow_alias",
            },
            warnings=[],
        )

    await insert_job_provenance(
        db,
        run_id=job_run_id,
        structure_id=structure_id,
        job_id=JOB_ID,
        model_version=model_version,
        ctx=ctx,
        code_version=code_version,
        parent_run_id=ctx.computation_run_id,
    )

    config = PipelineConfig(
        structure_id=structure_id,
        checkpoint_path=checkpoint,
        parent_run_id=ctx.computation_run_id or job_run_id,
        code_version=code_version,
        run_gnn=False,
        run_graph_topology=False,
    )

    phase_result, _gnn_result = await run_gnn_inference(
        db,
        config,
        run_id=gnn_run_id,
        pipeline_name=pathway_pipeline_name(ctx.pathway),
        device=ctx.device,
        job_params=ctx.job_params,
    )
    warnings = list(phase_result.warnings or [])

    if phase_result.success:
        try:
            await db.commit()
        except Exception as exc:
            logger.exception("GNN inference commit failed for %s", structure_id)
            warnings.append(f"commit failed: {exc}")
            phase_result.success = False
            phase_result.outputs["error"] = str(exc)

    await finalize_job_provenance(db, run_id=job_run_id, warnings=warnings or None)
    await db.commit()

    artifacts: list[str] = []
    if phase_result.success:
        artifacts = ["gnn_hyp", "gnn_euc"]
        if phase_result.outputs.get("interactive_viewer"):
            artifacts.append("gnn_interactive_view")

    return JobRunResult(
        job_id=JOB_ID,
        run_id=job_run_id,
        structure_id=structure_id,
        success=phase_result.success,
        artifacts_produced=artifacts,
        outputs={
            **phase_result.outputs,
            "job_run_id": job_run_id,
            "embedding_run_id": phase_result.outputs.get("gnn_run_id"),
            "restore_metadata": restore_metadata,
            "checkpoint_sha256": restore_metadata.get("sha256"),
            "checkpoint_sha256_16": restore_metadata.get("sha256_16"),
        },
        warnings=warnings,
    )


def _resolve_checkpoint_for_job(ctx: JobRunContext) -> tuple[str, str, dict[str, Any]]:
    requested = (ctx.checkpoint_path or "").strip()
    params = ctx.job_params or {}
    allow_legacy = bool(params.get("allow_legacy_checkpoint")) or (
        str(params.get("restore_source") or "").strip().lower() == "legacy_checkpoint"
    )

    if requested:
        if requested.startswith("models:/TokyoEye@"):
            alias = requested.rsplit("@", 1)[-1]
            resolved = resolve_production_checkpoint(alias=alias)
            return resolved["cache_path"], resolved["model_version"], resolved
        if not allow_legacy:
            raise ValueError(
                "Direct checkpoint_path is legacy/compare-only for TokyoEye. "
                "Omit checkpoint_path to resolve models:/TokyoEye@champion, or pass "
                "job_params.allow_legacy_checkpoint=true for archaeology."
            )
        return (
            requested,
            resolve_model_version_for_checkpoint(requested) or "legacy-checkpoint",
            {
                "restore_source": "legacy_checkpoint",
                "path": requested,
                "model_version": resolve_model_version_for_checkpoint(requested)
                or "legacy-checkpoint",
            },
        )

    resolved = resolve_production_checkpoint()
    return (
        resolved["cache_path"],
        resolved["model_version"] or get_production_model_version(),
        resolved,
    )
