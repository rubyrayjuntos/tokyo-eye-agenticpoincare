"""Precondition checks for atomic compute jobs.

Every job entry point (scheduler, POST /compute/jobs/{id}, legacy aliases)
runs these checks before dispatch so standalone and pathway runs fail the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data.readiness import ProbeInfrastructureError, probe_artifact
from science.compute.registry import JOB_REGISTRY
from science.contracts.geometric_runtime import load_structure_learned_curvature
from science.contracts.onboard_contract import job_requires_hyperbolic

STRUCTURE_EXISTS_SQL = (
    "SELECT 1 FROM dim_structure WHERE structure_id = :structure_id LIMIT 1"
)


@dataclass(frozen=True)
class PreconditionResult:
    satisfied: bool
    error: str | None = None
    missing_artifacts: tuple[str, ...] = ()


async def _structure_exists(db: Any, structure_id: str) -> bool:
    row = await db.fetch_one(STRUCTURE_EXISTS_SQL, {"structure_id": structure_id})
    return row is not None


async def _artifact_ready(db: Any, structure_id: str, artifact: str) -> bool:
    return await probe_artifact(db, structure_id, artifact)


async def _check_artifact_ready(
    db: Any,
    structure_id: str,
    artifact: str,
    *,
    missing: list[str],
    probe_errors: list[str],
) -> None:
    try:
        if not await _artifact_ready(db, structure_id, artifact):
            missing.append(artifact)
    except ProbeInfrastructureError as exc:
        probe_errors.append(str(exc))


async def check_job_preconditions(
    db: Any,
    job_id: str,
    structure_id: str,
    *,
    pipeline_job_id: str | None = None,
) -> PreconditionResult:
    """Return whether prerequisites for *job_id* are satisfied."""
    structure_id = structure_id.strip().lower()
    job = JOB_REGISTRY.get(job_id)
    if job is None:
        return PreconditionResult(False, f"Unknown job_id: {job_id}")

    if not await _structure_exists(db, structure_id):
        return PreconditionResult(
            False,
            f"Structure '{structure_id}' not found — run ingest first",
        )

    missing: list[str] = []
    probe_errors: list[str] = []

    for required_job in sorted(job.requires):
        required = JOB_REGISTRY.get(required_job)
        if required is None:
            continue
        for artifact in sorted(required.produces):
            await _check_artifact_ready(
                db, structure_id, artifact, missing=missing, probe_errors=probe_errors
            )

    if job_id == "gnn_inference":
        await _check_artifact_ready(
            db, structure_id, "scope", missing=missing, probe_errors=probe_errors
        )
        await _check_artifact_ready(
            db, structure_id, "dims", missing=missing, probe_errors=probe_errors
        )

    if job_id == "md_validate_top_n":
        await _check_artifact_ready(
            db, structure_id, "binding_scan", missing=missing, probe_errors=probe_errors
        )

    if job_requires_hyperbolic(job_id) and job_id != "gnn_inference":
        try:
            gnn_ready = await _artifact_ready(db, structure_id, "gnn_hyp")
        except ProbeInfrastructureError as exc:
            probe_errors.append(str(exc))
            gnn_ready = False
        if gnn_ready:
            learned = await load_structure_learned_curvature(db, structure_id)
            if learned is None:
                missing.append("learned_curvature")

    if probe_errors:
        unique_errors = tuple(dict.fromkeys(probe_errors))
        from shared.audit.instrumentation import audit_precondition_failed
        from shared.context import get_context

        ctx = get_context()
        effective_pipeline_job_id = (
            pipeline_job_id or ctx.run_id or ctx.extra.get("pipeline_job_id")
        )
        error = "; ".join(unique_errors)
        await audit_precondition_failed(
            db,
            structure_id=structure_id,
            job_name=job_id,
            missing_artifacts=[],
            error=error,
            pipeline_job_id=effective_pipeline_job_id,
        )
        return PreconditionResult(False, error, missing_artifacts=())

    if missing:
        unique = tuple(dict.fromkeys(missing))
        from shared.audit.instrumentation import audit_precondition_failed
        from shared.context import get_context

        ctx = get_context()
        effective_pipeline_job_id = (
            pipeline_job_id or ctx.run_id or ctx.extra.get("pipeline_job_id")
        )
        await audit_precondition_failed(
            db,
            structure_id=structure_id,
            job_name=job_id,
            missing_artifacts=list(unique),
            error=f"Preconditions not met for {job_id}: missing {', '.join(unique)}",
            pipeline_job_id=effective_pipeline_job_id,
        )
        return PreconditionResult(
            False,
            f"Preconditions not met for {job_id}: missing {', '.join(unique)}",
            missing_artifacts=unique,
        )

    return PreconditionResult(True)


# Single resolver for scheduler, pathway executor, POST /compute/jobs/{id},
# and dispatch_helpers — import this symbol in tests to prove entrypoint unity.
PRECONDITION_RESOLVER = check_job_preconditions


def precondition_dispatch_entrypoints() -> tuple[str, ...]:
    """Documented call chain roots that must share PRECONDITION_RESOLVER."""
    return (
        "science.compute.scheduler",
        "science.compute.pathway_executor.InProcessJobBackend.run_compute_job",
        "science.compute.dispatch_helpers.dispatch_job_in_process",
        "science.api.routers.compute.run_compute_job",
    )
