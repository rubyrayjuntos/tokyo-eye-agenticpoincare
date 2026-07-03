"""Tests for audit-only ingest discovery re-queue policy."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from data.readiness import should_requeue_discovery_after_audit_ingest


class _RequeueDB:
    def __init__(
        self,
        *,
        dims: bool = True,
        scope: bool = True,
        gnn_hyp: bool = False,
        job: dict[str, Any] | None = None,
    ):
        self._dims = dims
        self._scope = scope
        self._gnn_hyp = gnn_hyp
        self._job = job

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        q = query.lower()
        if "from dim_residue" in q:
            return {"count": 10} if self._dims else {"count": 0}
        if "structure_computation_scope" in q:
            return {"ok": 1} if self._scope else None
        if "fact_gnn_node_embedding" in q:
            return {"ok": 1} if self._gnn_hyp else None
        if "from pipeline_job" in q:
            return self._job
        if "from dim_structure" in q:
            return {"ok": 1}
        return None


@pytest.mark.asyncio
async def test_requeue_when_gnn_hyp_missing_and_foundation_present():
    db = _RequeueDB(gnn_hyp=False, job={"status": "failed", "error": "old"})
    assert await should_requeue_discovery_after_audit_ingest("11qe", db) is True


@pytest.mark.asyncio
async def test_requeue_when_latest_job_failed_even_if_gnn_present():
    db = _RequeueDB(
        gnn_hyp=True,
        job={"status": "failed", "error": "transient"},
    )
    assert await should_requeue_discovery_after_audit_ingest("4uj1", db) is True


@pytest.mark.asyncio
async def test_no_requeue_when_job_running():
    db = _RequeueDB(
        gnn_hyp=False,
        job={"status": "running", "current_step": "gnn_inference"},
    )
    assert await should_requeue_discovery_after_audit_ingest("11qe", db) is False


@pytest.mark.asyncio
async def test_no_requeue_when_foundation_incomplete():
    db = _RequeueDB(scope=False, gnn_hyp=False)
    assert await should_requeue_discovery_after_audit_ingest("11qe", db) is False


@pytest.mark.asyncio
async def test_no_requeue_when_complete_with_gnn():
    db = _RequeueDB(
        gnn_hyp=True,
        job={"status": "complete", "current_step": "complete"},
    )
    assert await should_requeue_discovery_after_audit_ingest("4uj1", db) is False


@pytest.mark.asyncio
async def test_coordinator_audit_ingest_queues_pipeline_when_policy_true():
    mock_conn = AsyncMock()
    mock_conn.__aenter__.return_value = object()
    mock_conn.__aexit__.return_value = False

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics", new_callable=AsyncMock) as mock_diag,
        patch("agent.tools.science_client.ScienceClient") as mock_client_cls,
        patch("agent.coordinator.routers.dashboard.get_connection", return_value=mock_conn),
        patch("agent.coordinator.routers.dashboard.DBAdapter"),
        patch(
            "data.readiness.should_requeue_discovery_after_audit_ingest",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "agent.coordinator.routers.dashboard._create_job_db",
            new_callable=AsyncMock,
            return_value="new-job-id",
        ) as mock_create_job,
        patch("agent.coordinator.routers.dashboard._run_pipeline_background", new_callable=AsyncMock),
    ):
        mock_diag.return_value.all_passed = True
        mock_client = mock_client_cls.return_value
        mock_client.ingest_structure = AsyncMock(
            return_value={
                "structure_id": "11qe",
                "pdb_id": "11QE",
                "audit_only": True,
                "already_existed": True,
                "chain_count": 1,
                "residue_count": 322,
                "atom_count": 1563,
            }
        )

        from httpx import ASGITransport, AsyncClient
        from agent.coordinator.app import app
        from agent.coordinator.auth import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/ingest", json={"pdb_id": "11QE"})
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["audit_only"] is True
    assert body["already_existed"] is True
    assert body["pipeline_status"] == "queued"
    assert body["pipeline_job_id"] == "new-job-id"
    mock_create_job.assert_awaited_once_with("11qe")
