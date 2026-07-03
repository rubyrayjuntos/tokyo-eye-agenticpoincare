"""Tests that governed ingest is the sole onboard/compute trigger."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.rcsb import INGEST_ONLY_MESSAGE, ingest_structure


class TestIngestSingleTrigger:
    @pytest.mark.asyncio
    async def test_lightweight_rcsb_ingest_disabled(self):
        result = await ingest_structure(pdb_id="4obe", db=AsyncMock())
        assert "error" in result
        assert INGEST_ONLY_MESSAGE in result["error"]

    @pytest.mark.asyncio
    async def test_pipeline_run_returns_410(self):
        from httpx import ASGITransport, AsyncClient

        from tests.test_e2e_ingest_pipeline_hydrate import _auth_override, _clear_auth_override

        with (
            patch("data.db.open_pool", new_callable=AsyncMock),
            patch("data.db.close_pool", new_callable=AsyncMock),
            patch("agent.tools.diagnostics.run_startup_diagnostics") as mock_diag,
        ):
            mock_report = MagicMock()
            mock_report.all_passed = True
            mock_diag.return_value = mock_report

            app = _auth_override()
            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/api/pipeline/run",
                        json={"structure_id": "rcsb_4obe"},
                    )
            finally:
                _clear_auth_override(app)

        assert resp.status_code == 410
        assert "POST /api/ingest" in resp.json()["detail"]
