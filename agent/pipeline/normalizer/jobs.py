"""Write and update fact_job_status rows inside normalizer transactions."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import sqlalchemy as sa


@dataclass
class JobStatus:
    structure_id: str
    job_type: str
    status: str              # queued | running | complete | failed
    job_status_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_message: Optional[str] = None
    cloud_task_name: Optional[str] = None


_UPSERT = sa.text("""
    INSERT INTO fact_job_status
        (job_status_id, structure_id, job_type, status,
         error_message, cloud_task_name,
         queued_at, started_at, finished_at)
    VALUES
        (:job_status_id, :structure_id, :job_type, :status,
         :error_message, :cloud_task_name,
         :queued_at, :started_at, :finished_at)
    ON CONFLICT (job_status_id) DO UPDATE SET
        status          = EXCLUDED.status,
        error_message   = EXCLUDED.error_message,
        started_at      = COALESCE(fact_job_status.started_at, EXCLUDED.started_at),
        finished_at     = EXCLUDED.finished_at
""")


async def write_job_status(
    conn,
    structure_id: str,
    job_type: str,
    status: str,
    job_status_id: Optional[str] = None,
    error_message: Optional[str] = None,
    cloud_task_name: Optional[str] = None,
) -> str:
    now = datetime.now(timezone.utc)
    jid = job_status_id or str(uuid.uuid4())
    await conn.execute(_UPSERT, {
        "job_status_id":   jid,
        "structure_id":    structure_id,
        "job_type":        job_type,
        "status":          status,
        "error_message":   error_message,
        "cloud_task_name": cloud_task_name,
        "queued_at":       now if status == "queued" else None,
        "started_at":      now if status == "running" else None,
        "finished_at":     now if status in ("complete", "failed") else None,
    })
    return jid
