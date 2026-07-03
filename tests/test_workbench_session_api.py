"""Tests for workbench session API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_orchestration_snapshot_requires_auth(monkeypatch):
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
            response = await client.get("/api/session/test-session/orchestration")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_orchestration_snapshot_returns_phase():
    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics", new_callable=AsyncMock) as mock_diag,
    ):
        mock_diag.return_value.all_passed = True
        from httpx import ASGITransport, AsyncClient
        from agent.coordinator.app import app
        from agent.coordinator.auth import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {
            "sub": "test-user",
            "session_id": "test-session",
        }
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/session/test-session/orchestration")
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["discovery_phase"] == "residue"
    assert "policy" in body


@pytest.mark.asyncio
async def test_workspace_layout_put_and_get():
    mock_db = AsyncMock()
    mock_db.fetch_one = AsyncMock(return_value=None)
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()

    async def _override_get_db():
        return mock_db

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics", new_callable=AsyncMock) as mock_diag,
    ):
        mock_diag.return_value.all_passed = True
        from httpx import ASGITransport, AsyncClient
        from agent.coordinator.app import app
        from agent.coordinator.auth import get_current_user
        from agent.coordinator.deps import get_db

        app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}
        app.dependency_overrides[get_db] = _override_get_db
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                put_response = await client.put(
                    "/api/session/ws-session-1/workspace-layout",
                    json={
                        "layout_json": {"panels": []},
                        "active_phase_group": "exploration",
                    },
                )
                mock_db.fetch_one = AsyncMock(
                    return_value={
                        "workspace_id": "default",
                        "layout_json": {"panels": []},
                        "active_phase_group": "exploration",
                        "updated_at": "2026-06-25T00:00:00Z",
                    }
                )
                get_response = await client.get("/api/session/ws-session-1/workspace-layout")
        finally:
            app.dependency_overrides.clear()

    assert put_response.status_code == 200
    assert put_response.json()["saved"] is True
    assert get_response.status_code == 200
    assert get_response.json()["active_phase_group"] == "exploration"
    mock_db.execute.assert_awaited()
