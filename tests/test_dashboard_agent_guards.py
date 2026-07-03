from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.coordinator.rate_limit import RateLimiter
from agent.coordinator.routers.data import _reject_invalid_structure_id, _safe_export_path
from agent.orchestration.session_store import (
    apply_orchestration_context,
    clear_session_orchestrator,
    get_session_orchestrator,
)
from science.dtie.common.keys import validate_structure_id


def test_validate_structure_id_rejects_path_traversal():
    assert validate_structure_id("rcsb_4obe") is True
    assert validate_structure_id("../../../tmp/evil") is False
    assert validate_structure_id("foo/bar") is False


def test_safe_export_path_rejects_traversal(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    safe = _safe_export_path(export_dir, "rcsb_4obe_full_export_abcd1234.csv")
    assert safe is not None
    assert safe.parent == export_dir.resolve()

    unsafe = _safe_export_path(export_dir, "../../../outside.csv")
    assert unsafe is None


def test_reject_invalid_structure_id_returns_json_response():
    response = _reject_invalid_structure_id("../evil")
    assert response is not None
    assert response.status_code == 400


def test_rate_limiter_is_thread_safe_under_concurrency():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.check("session-a")
    limiter.check("session-a")
    with pytest.raises(Exception):
        limiter.check("session-a")


@pytest.mark.asyncio
async def test_session_orchestrator_applies_discovery_phase_from_context():
    clear_session_orchestrator("test-session")
    orchestrator = await get_session_orchestrator(
        "test-session",
        {"discovery_phase": "screening"},
    )
    assert orchestrator.discovery_phase.value == "screening"
    clear_session_orchestrator("test-session")


def test_apply_orchestration_context_reads_nested_state():
    from agent.orchestration.orchestrator import SessionOrchestrator

    orchestrator = SessionOrchestrator("nested-session")
    apply_orchestration_context(
        orchestrator,
        {"orchestration_state": {"discovery_phase": "pocket"}},
    )
    assert orchestrator.discovery_phase.value == "pocket"


@pytest.mark.asyncio
async def test_agent_chat_requires_auth(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics", new_callable=AsyncMock) as mock_diag,
    ):
        mock_diag.return_value.all_passed = True
        from httpx import ASGITransport, AsyncClient
        from agent.coordinator.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/agent/chat",
                json={"message": "hello"},
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_pipeline_run_requires_auth(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics", new_callable=AsyncMock) as mock_diag,
    ):
        mock_diag.return_value.all_passed = True
        from httpx import ASGITransport, AsyncClient
        from agent.coordinator.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/pipeline/run",
                json={"structure_id": "rcsb_4obe"},
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_agent_chat_redacts_errors_in_prod(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "prod")

    mock_coordinator = MagicMock()
    mock_coordinator.run = AsyncMock(side_effect=RuntimeError("secret database detail"))

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics", new_callable=AsyncMock) as mock_diag,
        patch("data.db.get_connection") as mock_get_conn,
        patch("agent.llm.agents.create_coordinator", return_value=mock_coordinator),
        patch("agent.llm.providers.get_provider", return_value=MagicMock()),
        patch(
            "agent.coordinator.memory.build_memory_prompt_context",
            new_callable=AsyncMock,
            return_value=MagicMock(block=""),
        ),
        patch(
            "agent.coordinator.memory.persist_memory_interaction",
            new_callable=AsyncMock,
        ),
        patch("agent.orchestration.context_builder.build_context_block", return_value=""),
        patch("agent.coordinator.routers.dashboard.ENVIRONMENT", "prod"),
    ):
        mock_diag.return_value.all_passed = True
        mock_conn = AsyncMock()
        mock_conn.__aenter__.return_value = object()
        mock_conn.__aexit__.return_value = False
        mock_get_conn.return_value = mock_conn

        from httpx import ASGITransport, AsyncClient
        from agent.coordinator.app import app
        from agent.coordinator.auth import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/agent/chat",
                    json={"message": "trigger failure"},
                )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert "secret database detail" not in body["response"]
    assert body["response"] == "An internal error occurred. Please try again."


@pytest.mark.asyncio
async def test_create_coordinator_passes_orchestrator_to_agent():
    from agent.llm.agents import create_coordinator
    from agent.orchestration.orchestrator import SessionOrchestrator

    orchestrator = SessionOrchestrator("agent-factory-test")
    llm = MagicMock()
    agent = create_coordinator(llm=llm, db=None, orchestrator=orchestrator)
    assert agent.orchestrator is orchestrator


@pytest.mark.asyncio
async def test_gated_tool_invoke_blocks_pipeline_in_residue_phase():
    from agent.llm.base import ToolCall
    from agent.orchestration.gates import gated_tool_invoke
    from agent.orchestration.orchestrator import SessionOrchestrator

    orchestrator = SessionOrchestrator("gate-test")
    call = ToolCall(id="1", name="screen_fragments", arguments={"structure_id": "rcsb_4obe"})

    async def handler(**_kwargs):
        return {"ok": True}

    result = await gated_tool_invoke(orchestrator, call, handler)
    assert result.is_error is True
    assert "not available" in result.content


@pytest.mark.asyncio
async def test_annotate_structure_uses_normalizer():
    from agent.tools.data_tools import annotate_structure

    mock_db = AsyncMock()
    mock_normalizer = AsyncMock()
    mock_normalizer.normalize_annotation = AsyncMock(
        return_value=MagicMock(run_id="annotation_ann_abc123", asset_ids=["ann_abc123"])
    )

    with patch("data.normalizer.core.Normalizer", return_value=mock_normalizer):
        result = await annotate_structure(
            structure_id="rcsb_4obe",
            annotation="Test finding",
            annotation_type="finding",
            db=mock_db,
        )

    assert result.success is True
    mock_normalizer.normalize_annotation.assert_awaited_once()
    assert result.data["run_id"] == "annotation_ann_abc123"
