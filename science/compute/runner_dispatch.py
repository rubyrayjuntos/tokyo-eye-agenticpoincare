"""Dispatch atomic compute jobs by registry job_id."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from science.compute.registry import JOB_REGISTRY
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.gnn_inference import run_gnn_inference_job
from science.compute.runners.allosteric_site_detection import run_allosteric_site_detection_job
from science.compute.runners.binding_site_scan import run_binding_site_scan_job
from science.compute.runners.drug_candidate_ranking import run_drug_candidate_ranking_job
from science.compute.runners.fragment_screen import run_fragment_screen
from science.compute.runners.graph_topology import run_graph_topology_job
from science.compute.runners.hyperbolic_motifs import run_hyperbolic_motifs_job
from science.compute.runners.md_validate_top_n import run_md_validate_top_n
from science.compute.runners.pharmacophore_identification import run_pharmacophore_identification_job
from science.compute.runners.pocket_pharmacophore_map import run_pocket_pharmacophore_map_job
from science.compute.runners.resistance_pathway_map import run_resistance_pathway_map_job
from science.compute.runners.source_leak_detection import run_source_leak_detection
from science.compute.runners.strain_vulnerability_scan import run_strain_vulnerability_scan_job
from science.compute.runners.topological_lift import run_topological_lift_job
from science.compute.runners.witness_embedding import run_witness_embedding_job

JobRunner = Callable[[Any, JobRunContext], Awaitable[JobRunResult]]

JOB_RUNNERS: dict[str, JobRunner] = {
    "gnn_inference": run_gnn_inference_job,
    "source_leak_detection": run_source_leak_detection,
    "graph_topology": run_graph_topology_job,
    "witness_embedding": run_witness_embedding_job,
    "strain_vulnerability_scan": run_strain_vulnerability_scan_job,
    "hyperbolic_motifs": run_hyperbolic_motifs_job,
    "binding_site_scan": run_binding_site_scan_job,
    "md_validate_top_n": run_md_validate_top_n,
    "pocket_pharmacophore_map": run_pocket_pharmacophore_map_job,
    "pharmacophore_identification": run_pharmacophore_identification_job,
    "fragment_screen": run_fragment_screen,
    "drug_candidate_ranking": run_drug_candidate_ranking_job,
    "topological_lift": run_topological_lift_job,
    "resistance_pathway_map": run_resistance_pathway_map_job,
    "allosteric_site_detection": run_allosteric_site_detection_job,
}

# Jobs executed outside the monolith by the pathway scheduler.
PEELED_JOBS: frozenset[str] = frozenset(JOB_RUNNERS.keys())

EXTERNAL_JOBS: frozenset[str] = frozenset(
    {
        "ingest_dims",
        "assign_computation_scope",
        "alignment_sidecar",
    }
)

# Planned jobs may register stub runners until the real implementation ships.
PLANNED_RUNNER_ALLOWLIST: frozenset[str] = frozenset({"fragment_screen"})


def validate_runners_against_registry() -> list[str]:
    """Return validation errors when peeled runners drift from the job registry."""
    errors: list[str] = []
    for job_id in JOB_RUNNERS:
        if job_id not in JOB_REGISTRY:
            errors.append(f"JOB_RUNNERS[{job_id!r}] is not in JOB_REGISTRY")
    for job_id in PEELED_JOBS:
        job = JOB_REGISTRY[job_id]
        if job.status == "planned" and job_id not in PLANNED_RUNNER_ALLOWLIST:
            errors.append(f"planned job {job_id!r} must not be in JOB_RUNNERS")
    for job_id, job in JOB_REGISTRY.items():
        if job_id in EXTERNAL_JOBS or job.status == "planned":
            continue
        if job_id not in JOB_RUNNERS and job.status == "implemented":
            errors.append(f"implemented job {job_id!r} is missing from JOB_RUNNERS")
    return errors


async def dispatch_compute_job(
    db: Any,
    job_id: str,
    ctx: JobRunContext,
    *,
    skip_preconditions: bool = False,
) -> JobRunResult:
    """Run a single atomic compute job."""
    from science.compute.preconditions import check_job_preconditions

    if job_id not in JOB_REGISTRY:
        raise ValueError(f"Unknown job_id: {job_id}")
    runner = JOB_RUNNERS.get(job_id)
    if runner is None:
        raise ValueError(f"Job {job_id} is registered but has no runner implementation yet")

    if not skip_preconditions:
        pre = await check_job_preconditions(
            db,
            job_id,
            ctx.structure_id,
            pipeline_job_id=ctx.pipeline_job_id,
        )
        if not pre.satisfied:
            return JobRunResult(
                job_id=job_id,
                run_id="",
                structure_id=ctx.structure_id,
                success=False,
                outputs={
                    "error": pre.error,
                    "missing_artifacts": list(pre.missing_artifacts),
                },
                warnings=[f"precondition failed: {pre.error}"],
            )

        from science.compute.job_skip import (
            job_artifacts_already_present,
            skip_result_for_existing_artifacts,
        )

        if await job_artifacts_already_present(db, job_id, ctx.structure_id):
            return await skip_result_for_existing_artifacts(db, job_id, ctx.structure_id)

    dispatch_warnings = _geometric_dispatch_warnings(job_id)
    ctx = await _enrich_hyperbolic_context(db, job_id, ctx)
    if dispatch_warnings:
        from shared.audit.instrumentation import audit_geometric_messages

        await audit_geometric_messages(
            db,
            structure_id=ctx.structure_id,
            job_name=job_id,
            messages=dispatch_warnings,
            pipeline_job_id=ctx.pipeline_job_id,
            run_id=ctx.computation_run_id,
        )
    result = await runner(db, ctx)
    for message in dispatch_warnings:
        result.warnings.append(message)
    return await _apply_result_validation(
        db,
        result,
        learned_curvature=ctx.learned_curvature,
        pipeline_job_id=ctx.pipeline_job_id,
    )


async def _enrich_hyperbolic_context(db: Any, job_id: str, ctx: JobRunContext) -> JobRunContext:
    from science.contracts.geometric_runtime import load_structure_learned_curvature
    from science.contracts.onboard_contract import job_requires_hyperbolic
    from shared.audit.instrumentation import audit_curvature_passthrough

    if not job_requires_hyperbolic(job_id):
        return ctx
    if job_id == "gnn_inference":
        return ctx

    learned = await load_structure_learned_curvature(db, ctx.structure_id)
    await audit_curvature_passthrough(
        db,
        structure_id=ctx.structure_id,
        job_name=job_id,
        learned_curvature=learned,
        pipeline_job_id=ctx.pipeline_job_id,
        run_id=ctx.computation_run_id,
    )
    if learned is None:
        return ctx
    job_params = dict(ctx.job_params)
    job_params["learned_curvature"] = learned
    return replace(ctx, learned_curvature=learned, job_params=job_params)


def _geometric_dispatch_warnings(job_id: str) -> list[str]:
    from science.contracts.validation import validate_job_geometric_constraints

    return validate_job_geometric_constraints(job_id)


async def _apply_result_validation(
    db: Any,
    result: JobRunResult,
    *,
    learned_curvature: float | None = None,
    pipeline_job_id: str | None = None,
) -> JobRunResult:
    from science.contracts.geometric_runtime import apply_enforcement
    from science.contracts.validation import validate_job_run_result
    from shared.audit.instrumentation import (
        audit_curvature_learned,
        audit_enforcement_decision,
        audit_geometric_messages,
        audit_job_run_complete,
    )

    messages = validate_job_run_result(result, learned_curvature=learned_curvature)
    if messages:
        await audit_geometric_messages(
            db,
            structure_id=result.structure_id,
            job_name=result.job_id,
            messages=messages,
            pipeline_job_id=pipeline_job_id,
            run_id=result.run_id or None,
        )
    warnings, fail = apply_enforcement(result.job_id, messages)
    await audit_enforcement_decision(
        db,
        structure_id=result.structure_id,
        job_name=result.job_id,
        messages=warnings,
        failed=fail,
        pipeline_job_id=pipeline_job_id,
        run_id=result.run_id or None,
    )
    for message in warnings:
        result.warnings.append(message)
    if fail:
        result.success = False
        result.outputs = dict(result.outputs)
        result.outputs.setdefault("error", warnings[0])
    elif result.success and learned_curvature is not None and job_id_requires_curvature_output(result.job_id):
        result.outputs = dict(result.outputs)
        result.outputs.setdefault("curvature", learned_curvature)
    if result.success and result.job_id == "gnn_inference":
        curvature = result.outputs.get("curvature")
        if curvature is not None:
            await audit_curvature_learned(
                db,
                structure_id=result.structure_id,
                curvature=float(curvature),
                run_id=result.run_id or None,
                pipeline_job_id=pipeline_job_id,
            )
    await audit_job_run_complete(
        db,
        structure_id=result.structure_id,
        job_name=result.job_id,
        success=result.success,
        run_id=result.run_id or "",
        artifacts_produced=list(result.artifacts_produced),
        warning_count=len(result.warnings),
        pipeline_job_id=pipeline_job_id,
        learned_curvature=learned_curvature,
    )
    return result


def job_id_requires_curvature_output(job_id: str) -> bool:
    from science.contracts.onboard_contract import job_requires_hyperbolic

    return job_requires_hyperbolic(job_id) and job_id != "gnn_inference"
