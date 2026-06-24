from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def _mock_conn_cm() -> AsyncMock:
    conn = AsyncMock()
    conn.__aenter__.return_value = object()
    conn.__aexit__.return_value = False
    return conn


@pytest.mark.asyncio
async def test_duplicate_ingest_returns_existing_summary_and_records_audit():
    mock_db = AsyncMock()
    mock_db.fetch_one = AsyncMock(
        side_effect=[
            {"structure_id": "rcsb_4obe"},
            {
                "primary_chain_ids": ["A"],
                "reference_chain": "A",
                "exclude_chain_ids": [],
                "scope_source": "auto",
                "selection_reason": "test",
                "normalization_protocol": "graph_default",
            },
            {"chains": 2, "residues": 10, "atoms": 80},
        ]
    )
    mock_db.execute = AsyncMock()

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("science.api.routers.ingest.get_connection", return_value=_mock_conn_cm()),
        patch("science.api.routers.ingest.DBAdapter", return_value=mock_db),
    ):
        from httpx import ASGITransport, AsyncClient
        from science.api.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/compute/ingest-full",
                json={
                    "pdb_id": "4obe",
                    "force_reingest": True,
                    "requested_by": "allowed-user",
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["structure_id"] == "4obe"
    assert body["already_existed"] is True
    assert body["audit_only"] is True
    assert body["audit_run_id"].startswith("ingest_audit_")
    assert body["chain_count"] == 2
    assert body["residue_count"] == 10
    assert body["atom_count"] == 80

    assert mock_db.execute.await_count == 2
    provenance_run_query = mock_db.execute.await_args_list[0].args[0]
    provenance_run_params = mock_db.execute.await_args_list[0].args[1]
    provenance_event_query = mock_db.execute.await_args_list[1].args[0]
    provenance_event_params = mock_db.execute.await_args_list[1].args[1]

    assert "INSERT INTO provenance_run" in provenance_run_query
    assert provenance_run_params["run_id"].startswith("ingest_audit_")
    assert provenance_run_params["parameters"].obj["audit_only"] is True
    assert provenance_run_params["parameters"].obj["force_reingest_requested"] is True
    assert provenance_run_params["parameters"].obj["requested_by"] == "allowed-user"
    assert "INSERT INTO provenance_event" in provenance_event_query
    assert provenance_event_params["event_type"] == "duplicate_ingest_forced_blocked"


@pytest.mark.asyncio
async def test_ingest_allowlist_blocks_unauthorized_subject(monkeypatch):
    monkeypatch.setenv("SCIENCE_INGEST_ALLOWED_SUBJECTS", "allowed-user")

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics", new_callable=AsyncMock) as mock_diag,
    ):
        mock_diag.return_value.all_passed = True
        from httpx import ASGITransport, AsyncClient
        from agent.coordinator.app import app
        from agent.coordinator.routers.ingest import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {"sub": "blocked-user"}
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/ingest",
                    json={"pdb_id": "4obe"},
                )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 403
    assert response.json()["detail"] == "Ingest not permitted for subject 'blocked-user'"
