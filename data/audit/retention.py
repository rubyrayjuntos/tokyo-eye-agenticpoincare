"""Retention and aggregation for pipeline audit events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

AGGREGATE_SQL = """
INSERT INTO audit_pipeline_daily_summary (
    summary_date,
    event_type,
    severity,
    job_name,
    structure_id,
    event_count,
    sample_details,
    updated_at
)
SELECT
    (timestamp AT TIME ZONE 'UTC')::date AS summary_date,
    event_type,
    severity,
    COALESCE(job_name, '') AS job_name,
    COALESCE(structure_id, '') AS structure_id,
    COUNT(*)::int AS event_count,
    '{}'::jsonb AS sample_details,
    NOW() AS updated_at
FROM audit_pipeline_events
WHERE timestamp < :cutoff
GROUP BY
    (timestamp AT TIME ZONE 'UTC')::date,
    event_type,
    severity,
    COALESCE(job_name, ''),
    COALESCE(structure_id, '')
ON CONFLICT (summary_date, event_type, severity, job_name, structure_id)
DO UPDATE SET
    event_count = audit_pipeline_daily_summary.event_count + EXCLUDED.event_count,
    updated_at = NOW()
"""

DELETE_SQL = """
DELETE FROM audit_pipeline_events
WHERE timestamp < :cutoff
"""

COUNT_SQL = """
SELECT COUNT(*)::int AS cnt FROM audit_pipeline_events WHERE timestamp < :cutoff
"""


async def run_audit_retention(
    db: Any,
    *,
    retention_days: int = 90,
    aggregate: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Aggregate then prune detailed audit events older than retention_days."""
    retention_days = max(1, retention_days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    count_row = await db.fetch_one(COUNT_SQL, {"cutoff": cutoff})
    stale_count = int(count_row["cnt"]) if count_row else 0

    if dry_run:
        return {
            "dry_run": True,
            "retention_days": retention_days,
            "cutoff": cutoff.isoformat(),
            "events_to_prune": stale_count,
            "aggregated": False,
            "deleted": 0,
        }

    aggregated = False
    if aggregate and stale_count > 0:
        await db.execute(AGGREGATE_SQL, {"cutoff": cutoff})
        aggregated = True

    deleted = 0
    if stale_count > 0:
        await db.execute(DELETE_SQL, {"cutoff": cutoff})
        deleted = stale_count

    return {
        "dry_run": False,
        "retention_days": retention_days,
        "cutoff": cutoff.isoformat(),
        "events_to_prune": stale_count,
        "aggregated": aggregated,
        "deleted": deleted,
    }


async def query_daily_summary(
    db: Any,
    *,
    since_days: int = 30,
    structure_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return rolled-up audit counts for dashboards."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=max(1, since_days))).date()
    clauses = ["summary_date >= :since"]
    params: dict[str, Any] = {"since": since_date.isoformat()}
    if structure_id:
        clauses.append("structure_id = :structure_id")
        params["structure_id"] = structure_id.strip().lower()

    sql = f"""
        SELECT summary_date, event_type, severity, job_name, structure_id,
               event_count, sample_details, updated_at
        FROM audit_pipeline_daily_summary
        WHERE {' AND '.join(clauses)}
        ORDER BY summary_date DESC, event_count DESC
        LIMIT 500
    """
    rows = await db.fetch_all(sql, params)
    return [dict(row) for row in rows]
