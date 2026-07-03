"""Tests for structured pipeline audit events."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from shared.audit.bus import emit_audit_event, recent_events
from shared.audit.events import (
    EVENT_ENFORCEMENT_DECISION,
    EVENT_PRECONDITION_FAILED,
    severity_at_least,
)
from shared.audit.instrumentation import audit_enforcement_decision, audit_precondition_failed


class TestAuditEventModel:
    def test_severity_ordering(self) -> None:
        assert severity_at_least("error", "warning")
        assert severity_at_least("warning", "warning")
        assert not severity_at_least("info", "warning")

    def test_event_to_dict(self) -> None:
        event = __import__("shared.audit.events", fromlist=["AuditEvent"]).AuditEvent(
            event_type="test",
            severity="info",
            structure_id="4obe",
        )
        payload = event.to_dict()
        assert payload["event_type"] == "test"
        assert payload["structure_id"] == "4obe"
        assert "timestamp" in payload


class TestAuditBus:
    @pytest.mark.asyncio
    async def test_emit_logs_without_db(self) -> None:
        with patch.dict(os.environ, {"AUDIT_PIPELINE_PERSIST": "false"}, clear=False):
            event = await emit_audit_event(
                "geometric_validation",
                "warning",
                structure_id="4obe",
                job_name="gnn_inference",
                details={"message": "test"},
            )
        assert event.structure_id == "4obe"
        assert recent_events()[-1].event_id == event.event_id

    @pytest.mark.asyncio
    async def test_persist_respects_min_severity(self) -> None:
        db = AsyncMock()
        with patch.dict(
            os.environ,
            {"AUDIT_PIPELINE_PERSIST": "true", "AUDIT_PIPELINE_MIN_SEVERITY": "warning"},
            clear=False,
        ):
            await emit_audit_event("job_run_complete", "info", structure_id="4obe", db=db)
            await emit_audit_event(
                EVENT_ENFORCEMENT_DECISION,
                "warning",
                structure_id="4obe",
                db=db,
            )
        assert db.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_instrumentation_precondition_failed(self) -> None:
        db = AsyncMock()
        with patch.dict(os.environ, {"AUDIT_PIPELINE_PERSIST": "false"}, clear=False):
            await audit_precondition_failed(
                db,
                structure_id="4obe",
                job_name="source_leak_detection",
                missing_artifacts=["learned_curvature"],
                error="missing learned_curvature",
            )
        events = recent_events()
        assert events[-1].event_type == EVENT_PRECONDITION_FAILED
        assert events[-1].severity == "error"

    @pytest.mark.asyncio
    async def test_enforcement_decision_error_on_fail(self) -> None:
        db = AsyncMock()
        with patch.dict(os.environ, {"AUDIT_PIPELINE_PERSIST": "false"}, clear=False):
            await audit_enforcement_decision(
                db,
                structure_id="4obe",
                job_name="gnn_inference",
                messages=["curvature missing"],
                failed=True,
            )
        assert recent_events()[-1].severity == "error"


class TestAuditQueryHelpers:
    def test_parse_since_duration(self) -> None:
        from data.audit.pipeline_events import parse_since_duration
        from datetime import datetime, timezone

        since = parse_since_duration("7d")
        assert since is not None
        assert since.tzinfo == timezone.utc
        assert (datetime.now(timezone.utc) - since).days >= 6


class TestPathwayAuditWiring:
    @pytest.mark.asyncio
    async def test_execute_pathway_passes_pipeline_job_id(self, monkeypatch) -> None:
        from science.compute.pathway_executor import execute_pathway

        captured: list[dict] = []

        class _Backend:
            async def run_compute_job(self, job_id: str, structure_id: str, **kwargs):
                captured.append({"job_id": job_id, "structure_id": structure_id, **kwargs})
                return {"success": False, "outputs": {"error": "skip"}}

            async def run_post_source_leak_phases(self, *args, **kwargs):
                return {"phases_run": []}

        monkeypatch.setattr(
            "science.compute.pathway_executor.PEELED_JOBS",
            frozenset({"gnn_inference"}),
        )
        monkeypatch.setattr(
            "science.compute.pathway_executor.jobs_for_pathway",
            lambda *_a, **_k: ["gnn_inference"],
        )
        monkeypatch.setattr(
            "science.compute.pathway_executor.JOB_REGISTRY",
            {
                "gnn_inference": type(
                    "J",
                    (),
                    {"tier": 1, "produces": []},
                )()
            },
        )

        with pytest.raises(RuntimeError):
            await execute_pathway(
                "4obe",
                _Backend(),
                pipeline_job_id="pipe-123",
            )

        assert captured
        assert captured[0]["pipeline_job_id"] == "pipe-123"


class TestAuditRetention:
    @pytest.mark.asyncio
    async def test_retention_dry_run(self) -> None:
        from data.audit.retention import run_audit_retention

        class _DB:
            async def fetch_one(self, query: str, params: dict):
                return {"cnt": 0}

        result = await run_audit_retention(_DB(), retention_days=90, dry_run=True)
        assert result["dry_run"] is True
        assert result["events_to_prune"] == 0
