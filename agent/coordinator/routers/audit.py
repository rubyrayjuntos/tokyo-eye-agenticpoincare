"""Pipeline audit query API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from agent.coordinator.auth import get_current_user
from agent.coordinator.deps import get_db
from data.audit.pipeline_events import get_audit_events_for_structure, query_audit_events
from data.audit.retention import query_daily_summary, run_audit_retention
from shared.audit.bus import recent_events

router = APIRouter(tags=["audit"])


class AuditRetentionRequest(BaseModel):
    retention_days: int = Field(default=90, ge=1, le=3650)
    aggregate: bool = True
    dry_run: bool = False


@router.get("/api/structures/{structure_id}/audit")
async def get_structure_audit(
    structure_id: str,
    severity: Literal["info", "warning", "error"] | None = None,
    event_type: str | None = None,
    since: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Any = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return persisted audit events for a structure (newest first)."""
    events = await query_audit_events(
        db,
        structure_id=structure_id,
        severity=severity,
        event_type=event_type,
        since=since,
        limit=limit,
    )
    return {
        "structure_id": structure_id.strip().lower(),
        "count": len(events),
        "events": events,
    }


@router.get("/api/pipeline/audit")
async def get_pipeline_audit(
    structure_id: str | None = None,
    pipeline_job_id: str | None = None,
    job_name: str | None = None,
    event_type: str | None = None,
    severity: Literal["info", "warning", "error"] | None = None,
    since: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include_memory: bool = False,
    db: Any = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Query pipeline audit events across structures with optional filters."""
    events = await query_audit_events(
        db,
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        job_name=job_name,
        event_type=event_type,
        severity=severity,
        since=since,
        limit=limit,
        offset=offset,
    )
    payload: dict[str, Any] = {
        "count": len(events),
        "events": events,
        "offset": offset,
        "limit": limit,
    }
    if include_memory:
        payload["recent_memory_events"] = [e.to_dict() for e in recent_events(limit=50)]
    return payload


@router.get("/api/pipeline/audit/summary")
async def get_pipeline_audit_summary(
    structure_id: str | None = None,
    since_days: int = Query(default=30, ge=1, le=365),
    db: Any = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return rolled-up daily audit counts (post-retention aggregates)."""
    rows = await query_daily_summary(
        db,
        since_days=since_days,
        structure_id=structure_id,
    )
    return {"count": len(rows), "summaries": rows}


@router.post("/api/pipeline/audit/retention")
async def run_pipeline_audit_retention(
    request: AuditRetentionRequest,
    db: Any = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Aggregate and prune detailed audit events older than retention window."""
    result = await run_audit_retention(
        db,
        retention_days=request.retention_days,
        aggregate=request.aggregate,
        dry_run=request.dry_run,
    )
    if not request.dry_run:
        await db.commit()
    return result
