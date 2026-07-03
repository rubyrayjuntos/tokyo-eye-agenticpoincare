"""Persist and query pipeline audit events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from shared.audit.events import AuditEvent, AuditSeverity

INSERT_SQL = """
INSERT INTO audit_pipeline_events (
    event_id,
    timestamp,
    event_type,
    severity,
    structure_id,
    pipeline_job_id,
    job_name,
    contract_version,
    details,
    correlation_id,
    enforcement_level
) VALUES (
    :event_id,
    :timestamp,
    :event_type,
    :severity,
    :structure_id,
    :pipeline_job_id,
    :job_name,
    :contract_version,
    :details,
    :correlation_id,
    :enforcement_level
)
"""


async def persist_audit_event(db: Any, event: AuditEvent) -> None:
    """Append one audit event (idempotent on event_id)."""
    from psycopg.types.json import Json

    await db.execute(
        INSERT_SQL,
        {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "severity": event.severity,
            "structure_id": event.structure_id,
            "pipeline_job_id": event.pipeline_job_id,
            "job_name": event.job_name,
            "contract_version": event.contract_version,
            "details": Json(event.details),
            "correlation_id": event.correlation_id or None,
            "enforcement_level": event.enforcement_level,
        },
    )


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    details = row.get("details")
    if isinstance(details, str):
        import json

        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {"raw": details}
    ts = row.get("timestamp")
    if isinstance(ts, datetime):
        ts = ts.isoformat()
    return {
        "event_id": row["event_id"],
        "timestamp": ts,
        "event_type": row["event_type"],
        "severity": row["severity"],
        "structure_id": row.get("structure_id"),
        "pipeline_job_id": row.get("pipeline_job_id"),
        "job_name": row.get("job_name"),
        "contract_version": row.get("contract_version"),
        "details": details or {},
        "correlation_id": row.get("correlation_id"),
        "enforcement_level": row.get("enforcement_level"),
    }


async def query_audit_events(
    db: Any,
    *,
    structure_id: str | None = None,
    pipeline_job_id: str | None = None,
    job_name: str | None = None,
    event_type: str | None = None,
    severity: AuditSeverity | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Query persisted audit events with optional filters."""
    clauses: list[str] = []
    params: dict[str, Any] = {"limit": max(1, min(limit, 500)), "offset": max(0, offset)}

    if structure_id:
        clauses.append("structure_id = :structure_id")
        params["structure_id"] = structure_id.strip().lower()
    if pipeline_job_id:
        clauses.append("pipeline_job_id = :pipeline_job_id")
        params["pipeline_job_id"] = pipeline_job_id
    if job_name:
        clauses.append("job_name = :job_name")
        params["job_name"] = job_name
    if event_type:
        clauses.append("event_type = :event_type")
        params["event_type"] = event_type
    if severity:
        clauses.append("severity = :severity")
        params["severity"] = severity
    if since:
        clauses.append("timestamp >= :since")
        params["since"] = since
    if until:
        clauses.append("timestamp <= :until")
        params["until"] = until

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT event_id, timestamp, event_type, severity, structure_id,
               pipeline_job_id, job_name, contract_version, details,
               correlation_id, enforcement_level
        FROM audit_pipeline_events
        {where}
        ORDER BY timestamp DESC
        LIMIT :limit OFFSET :offset
    """
    rows = await db.fetch_all(sql, params)
    return [_row_to_dict(row) for row in rows]


async def get_audit_events_for_structure(
    db: Any,
    structure_id: str,
    *,
    since: datetime | None = None,
    severity: AuditSeverity | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Convenience wrapper for structure-scoped audit history."""
    return await query_audit_events(
        db,
        structure_id=structure_id,
        since=since,
        severity=severity,
        limit=limit,
    )


def parse_since_duration(raw: str) -> datetime | None:
    """Parse CLI duration like 7d, 24h, 30m into a UTC since timestamp."""
    raw = raw.strip().lower()
    if not raw:
        return None
    now = datetime.now(timezone.utc)
    if raw.endswith("d"):
        return now - timedelta(days=int(raw[:-1]))
    if raw.endswith("h"):
        return now - timedelta(hours=int(raw[:-1]))
    if raw.endswith("m"):
        return now - timedelta(minutes=int(raw[:-1]))
    return None
