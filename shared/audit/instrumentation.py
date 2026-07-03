"""Pipeline audit instrumentation helpers."""

from __future__ import annotations

import json
from typing import Any

from shared.audit.bus import emit_audit_event
from shared.audit.events import (
    EVENT_CONTRACT_VALIDATION,
    EVENT_CURVATURE_LEARNED,
    EVENT_CURVATURE_PASSTHROUGH,
    EVENT_ENFORCEMENT_DECISION,
    EVENT_GEOMETRIC_READINESS,
    EVENT_GEOMETRIC_VALIDATION,
    EVENT_JOB_RUN_COMPLETE,
    EVENT_ONBOARD_GEOMETRIC_NOTE,
    EVENT_PATHWAY_COMPLETE,
    EVENT_PATHWAY_FAILED,
    EVENT_PATHWAY_STARTED,
    EVENT_PRECONDITION_FAILED,
)

_last_geometric_readiness_fingerprint: dict[str, str] = {}


async def audit_precondition_failed(
    db: Any,
    *,
    structure_id: str,
    job_name: str,
    missing_artifacts: list[str],
    error: str | None = None,
    pipeline_job_id: str | None = None,
) -> None:
    await emit_audit_event(
        EVENT_PRECONDITION_FAILED,
        "error",
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        job_name=job_name,
        details={
            "missing_artifacts": missing_artifacts,
            "error": error,
        },
        db=db,
    )


async def audit_geometric_messages(
    db: Any,
    *,
    structure_id: str,
    job_name: str,
    messages: list[str],
    pipeline_job_id: str | None = None,
    run_id: str | None = None,
) -> None:
    for message in messages:
        await emit_audit_event(
            EVENT_GEOMETRIC_VALIDATION,
            "warning",
            structure_id=structure_id,
            pipeline_job_id=pipeline_job_id,
            job_name=job_name,
            details={"message": message},
            correlation_id=run_id,
            db=db,
        )


async def audit_enforcement_decision(
    db: Any,
    *,
    structure_id: str,
    job_name: str,
    messages: list[str],
    failed: bool,
    pipeline_job_id: str | None = None,
    run_id: str | None = None,
) -> None:
    if not messages:
        return
    await emit_audit_event(
        EVENT_ENFORCEMENT_DECISION,
        "error" if failed else "warning",
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        job_name=job_name,
        details={
            "messages": messages,
            "failed_job": failed,
        },
        correlation_id=run_id,
        db=db,
    )


async def audit_curvature_learned(
    db: Any,
    *,
    structure_id: str,
    curvature: float,
    run_id: str | None = None,
    pipeline_job_id: str | None = None,
) -> None:
    await emit_audit_event(
        EVENT_CURVATURE_LEARNED,
        "info",
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        job_name="gnn_inference",
        details={"curvature": curvature},
        correlation_id=run_id,
        db=db,
    )


async def audit_curvature_passthrough(
    db: Any,
    *,
    structure_id: str,
    job_name: str,
    learned_curvature: float | None,
    pipeline_job_id: str | None = None,
    run_id: str | None = None,
) -> None:
    severity = "info" if learned_curvature is not None else "warning"
    await emit_audit_event(
        EVENT_CURVATURE_PASSTHROUGH,
        severity,
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        job_name=job_name,
        details={"learned_curvature": learned_curvature},
        correlation_id=run_id,
        db=db,
    )


async def audit_onboard_geometric_notes(
    db: Any,
    *,
    structure_id: str,
    notes: list[str],
) -> None:
    for note in notes:
        severity = "warning" if any(
            kw in note.lower() for kw in ("fail", "no residues", "not yet", "block")
        ) else "info"
        await emit_audit_event(
            EVENT_ONBOARD_GEOMETRIC_NOTE,
            severity,
            structure_id=structure_id,
            details={"note": note},
            db=db,
        )


async def audit_geometric_readiness(
    db: Any,
    *,
    structure_id: str,
    geometric_readiness: dict[str, Any],
) -> None:
    fingerprint = json.dumps(geometric_readiness, sort_keys=True, default=str)
    if _last_geometric_readiness_fingerprint.get(structure_id) == fingerprint:
        return
    _last_geometric_readiness_fingerprint[structure_id] = fingerprint

    severity = "info"
    if geometric_readiness.get("requires_hyperbolic") and not geometric_readiness.get(
        "hyperbolic_ready"
    ):
        severity = "warning"
    if geometric_readiness.get("curvature_ready") is False:
        severity = "warning"
    await emit_audit_event(
        EVENT_GEOMETRIC_READINESS,
        severity,
        structure_id=structure_id,
        details=geometric_readiness,
        db=db,
    )


async def audit_job_run_complete(
    db: Any,
    *,
    structure_id: str,
    job_name: str,
    success: bool,
    run_id: str,
    artifacts_produced: list[str],
    warning_count: int,
    pipeline_job_id: str | None = None,
    learned_curvature: float | None = None,
) -> None:
    details: dict[str, Any] = {
        "success": success,
        "artifacts_produced": artifacts_produced,
        "warning_count": warning_count,
    }
    if learned_curvature is not None:
        details["learned_curvature"] = learned_curvature
    await emit_audit_event(
        EVENT_JOB_RUN_COMPLETE,
        "info" if success else "error",
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        job_name=job_name,
        details=details,
        correlation_id=run_id,
        db=db,
    )


async def audit_contract_validation(
    db: Any | None,
    *,
    messages: list[str],
    context: str,
) -> None:
    for message in messages:
        await emit_audit_event(
            EVENT_CONTRACT_VALIDATION,
            "warning",
            details={"message": message, "context": context},
            db=db,
        )


async def audit_pathway_started(
    db: Any | None,
    *,
    structure_id: str,
    pipeline_job_id: str,
    pathway_id: str,
) -> None:
    from shared.context import get_context, set_context, RequestContext

    ctx = get_context()
    if ctx.run_id != pipeline_job_id:
        set_context(
            RequestContext(
                request_id=ctx.request_id,
                session_id=ctx.session_id,
                run_id=pipeline_job_id,
                user_id=ctx.user_id,
                extra={**ctx.extra, "pipeline_job_id": pipeline_job_id},
            )
        )
    await emit_audit_event(
        EVENT_PATHWAY_STARTED,
        "info",
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        details={"pathway_id": pathway_id},
        correlation_id=pipeline_job_id,
        db=db,
    )


async def audit_pathway_complete(
    db: Any | None,
    *,
    structure_id: str,
    pipeline_job_id: str | None,
    jobs_complete: list[str],
    pathway_id: str,
    run_id: str | None = None,
) -> None:
    await emit_audit_event(
        EVENT_PATHWAY_COMPLETE,
        "info",
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        details={
            "pathway_id": pathway_id,
            "jobs_complete": jobs_complete,
            "run_id": run_id,
        },
        correlation_id=pipeline_job_id or run_id,
        db=db,
    )


async def audit_pathway_failed(
    db: Any | None,
    *,
    structure_id: str,
    pipeline_job_id: str | None,
    error: str,
    pathway_id: str = "discovery_story",
) -> None:
    await emit_audit_event(
        EVENT_PATHWAY_FAILED,
        "error",
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        details={"pathway_id": pathway_id, "error": error},
        correlation_id=pipeline_job_id,
        db=db,
    )
