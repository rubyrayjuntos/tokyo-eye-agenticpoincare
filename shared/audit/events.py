"""Versioned audit event model for pipeline instrumentation."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

AuditSeverity = Literal["info", "warning", "error"]

# Stable event_type identifiers (extend as instrumentation grows).
EVENT_GEOMETRIC_VALIDATION = "geometric_validation"
EVENT_CURVATURE_LEARNED = "curvature_learned"
EVENT_CURVATURE_PASSTHROUGH = "curvature_passthrough"
EVENT_ENFORCEMENT_DECISION = "enforcement_decision"
EVENT_PRECONDITION_FAILED = "precondition_failed"
EVENT_ONBOARD_GEOMETRIC_NOTE = "onboard_geometric_note"
EVENT_GEOMETRIC_READINESS = "geometric_readiness"
EVENT_CONTRACT_VALIDATION = "contract_validation"
EVENT_JOB_RUN_COMPLETE = "job_run_complete"
EVENT_PATHWAY_STARTED = "pathway_started"
EVENT_PATHWAY_COMPLETE = "pathway_complete"
EVENT_PATHWAY_FAILED = "pathway_failed"

ALL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_GEOMETRIC_VALIDATION,
        EVENT_CURVATURE_LEARNED,
        EVENT_CURVATURE_PASSTHROUGH,
        EVENT_ENFORCEMENT_DECISION,
        EVENT_PRECONDITION_FAILED,
        EVENT_ONBOARD_GEOMETRIC_NOTE,
        EVENT_GEOMETRIC_READINESS,
        EVENT_CONTRACT_VALIDATION,
        EVENT_JOB_RUN_COMPLETE,
        EVENT_PATHWAY_STARTED,
        EVENT_PATHWAY_COMPLETE,
        EVENT_PATHWAY_FAILED,
    }
)

_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}


@dataclass(frozen=True)
class AuditEvent:
    """Structured audit record emitted by pipeline instrumentation."""

    event_type: str
    severity: AuditSeverity
    structure_id: str | None = None
    pipeline_job_id: str | None = None
    job_name: str | None = None
    contract_version: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    enforcement_level: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    def to_log_dict(self) -> dict[str, Any]:
        """Flatten for structured logging (audit_event namespace)."""
        return {
            "audit_event_id": self.event_id,
            "audit_event_type": self.event_type,
            "audit_severity": self.severity,
            "structure_id": self.structure_id,
            "pipeline_job_id": self.pipeline_job_id,
            "job_name": self.job_name,
            "contract_version": self.contract_version,
            "correlation_id": self.correlation_id,
            "enforcement_level": self.enforcement_level,
            **{f"audit_{k}": v for k, v in self.details.items()},
        }


def severity_at_least(severity: AuditSeverity, minimum: AuditSeverity) -> bool:
    return _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[minimum]
