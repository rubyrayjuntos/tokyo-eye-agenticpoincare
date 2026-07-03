"""Audit event bus: structured logging + optional DB persistence."""

from __future__ import annotations

import os
from collections import deque
from typing import Any

from shared.audit.events import AuditEvent, AuditSeverity, severity_at_least
from shared.logging import get_logger

logger = get_logger(__name__)

# Recent in-memory ring buffer for dev / fallback when DB unavailable.
_RECENT_EVENTS: deque[AuditEvent] = deque(maxlen=500)


def _contract_version() -> str:
    try:
        from science.contracts.onboard_contract import load_contract

        return str(load_contract().get("version", "unknown"))
    except Exception:
        return "unknown"


def _enforcement_snapshot(job_name: str | None) -> str:
    try:
        from science.contracts.geometric_runtime import effective_enforcement_level

        if job_name:
            return effective_enforcement_level(job_name)
        return os.getenv("GEOMETRIC_ENFORCEMENT_LEVEL", "warning").strip().lower() or "warning"
    except Exception:
        return os.getenv("GEOMETRIC_ENFORCEMENT_LEVEL", "warning")


def _correlation_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        from shared.context import get_context

        ctx = get_context()
        return ctx.run_id or ctx.request_id
    except Exception:
        return ""


def _persist_enabled() -> bool:
    return os.getenv("AUDIT_PIPELINE_PERSIST", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _min_persist_severity() -> AuditSeverity:
    raw = os.getenv("AUDIT_PIPELINE_MIN_SEVERITY", "warning").strip().lower()
    if raw in ("info", "warning", "error"):
        return raw  # type: ignore[return-value]
    return "warning"


def recent_events(limit: int = 100) -> list[AuditEvent]:
    """Return recent in-memory audit events (newest last)."""
    items = list(_RECENT_EVENTS)
    if limit < len(items):
        return items[-limit:]
    return items


async def emit_audit_event(
    event_type: str,
    severity: AuditSeverity,
    *,
    structure_id: str | None = None,
    pipeline_job_id: str | None = None,
    job_name: str | None = None,
    details: dict[str, Any] | None = None,
    correlation_id: str | None = None,
    db: Any | None = None,
) -> AuditEvent:
    """Emit a structured audit event to logs and optionally persist to Aurora."""
    event = AuditEvent(
        event_type=event_type,
        severity=severity,
        structure_id=structure_id.strip().lower() if structure_id else None,
        pipeline_job_id=pipeline_job_id,
        job_name=job_name,
        contract_version=_contract_version(),
        details=dict(details or {}),
        correlation_id=_correlation_id(correlation_id),
        enforcement_level=_enforcement_snapshot(job_name),
    )
    _RECENT_EVENTS.append(event)

    log_fn = logger.info
    if severity == "warning":
        log_fn = logger.warning
    elif severity == "error":
        log_fn = logger.error
    log_fn(f"audit:{event_type}", **event.to_log_dict())

    if (
        db is not None
        and _persist_enabled()
        and severity_at_least(severity, _min_persist_severity())
    ):
        try:
            from data.audit.pipeline_events import persist_audit_event

            await persist_audit_event(db, event)
        except Exception as exc:
            logger.warning(
                "audit persist failed",
                audit_event_id=event.event_id,
                error=str(exc),
            )

    return event
