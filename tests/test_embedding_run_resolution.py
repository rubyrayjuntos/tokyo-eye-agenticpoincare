"""Tests for hyperbolic embedding run resolution in dashboard read paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

HYPERBOLIC_RUN = {
    "run_id": "dtie_d8c34c16f0de_hyp",
    "pipeline_name": "dtie_v5",
    "model_version": "GOSPConeMapper-v6",
    "run_ts": "2026-06-24T20:05:00+00:00",
}

EMBEDDING_ROWS = [
    {
        "residue_id": "4uj1:A:1",
        "residue_index": 1,
        "residue_name": "MET",
        "chain_label": "A",
        "hyp_projection_2d": "[0.12,-0.08]",
        "hyp_projections": None,
        "cone_depth": 3.0,
        "epistemic_uncertainty": 0.4,
        "aleatoric_uncertainty": 0.1,
        "curvature": 1.0,
    },
    {
        "residue_id": "4uj1:A:2",
        "residue_index": 2,
        "residue_name": "GLU",
        "chain_label": "A",
        "hyp_projection_2d": "[0.20,-0.10]",
        "hyp_projections": None,
        "cone_depth": 2.5,
        "epistemic_uncertainty": 0.3,
        "aleatoric_uncertainty": 0.1,
        "curvature": 1.0,
    },
]


@pytest.mark.asyncio
async def test_resolve_latest_productive_run_metadata_uses_hyperbolic_embeddings():
    from agent.coordinator.routers import dashboard as dash

    mock_db = MagicMock()
    with patch.object(
        dash,
        "_resolve_latest_hyperbolic_embedding_run",
        new_callable=AsyncMock,
        return_value=HYPERBOLIC_RUN,
    ) as mock_hyp:
        with patch(
            "agent.tools.dtie.tools.ToolDB.fetch_one",
            new_callable=AsyncMock,
            return_value={"run_id": "dtie_d8c34c16f0de", "pipeline_name": "dtie_v5"},
        ):
            metadata = await dash._resolve_latest_productive_run_metadata("4uj1", mock_db)

    mock_hyp.assert_awaited_once_with("4uj1", mock_db)
    assert metadata["embeddings"]["run_id"] == "dtie_d8c34c16f0de_hyp"
    assert metadata["graph_metrics"]["run_id"] == "dtie_d8c34c16f0de"


@pytest.mark.asyncio
async def test_get_embeddings_pins_to_latest_hyperbolic_run():
    mock_db = AsyncMock()

    async def _override_get_db():
        return mock_db

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch("agent.tools.diagnostics.run_startup_diagnostics") as mock_diag,
        patch(
            "agent.coordinator.routers.dashboard._resolve_latest_hyperbolic_embedding_run",
            new_callable=AsyncMock,
            return_value=HYPERBOLIC_RUN,
        ),
        patch(
            "agent.coordinator.routers.dashboard._fetch_embeddings_for_hydration",
            new_callable=AsyncMock,
            return_value={
                "structure_id": "4uj1",
                "curvature": 1.0,
                "residues": [
                    {"residue_id": "4uj1:A:1", "residue_index": 1, "chain_label": "A", "x": 0.1, "y": -0.05},
                    {"residue_id": "4uj1:A:2", "residue_index": 2, "chain_label": "A", "x": 0.2, "y": -0.1},
                ],
            },
        ) as mock_fetch,
    ):
        mock_report = MagicMock()
        mock_report.all_passed = True
        mock_diag.return_value = mock_report

        from agent.coordinator.app import app
        from agent.coordinator.deps import get_db

        app.dependency_overrides[get_db] = _override_get_db
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/structures/4uj1/embeddings")
        finally:
            app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "dtie_d8c34c16f0de_hyp"
    assert body["residue_count"] == 2
    mock_fetch.assert_awaited_once_with("4uj1", mock_db, run_id="dtie_d8c34c16f0de_hyp")


@pytest.mark.asyncio
async def test_fetch_embeddings_for_hydration_filters_by_run_id():
    from agent.coordinator.routers import dashboard as dash

    mock_db = MagicMock()
    with patch(
        "agent.tools.dtie.tools.ToolDB.fetch_all",
        new_callable=AsyncMock,
        return_value=EMBEDDING_ROWS,
    ) as mock_fetch_all:
        result = await dash._fetch_embeddings_for_hydration(
            "4uj1",
            mock_db,
            run_id="dtie_d8c34c16f0de_hyp",
        )

    assert result is not None
    assert len(result["residues"]) == 2
    sql = mock_fetch_all.await_args.args[0]
    assert "e.run_id = :run_id" in sql
    assert mock_fetch_all.await_args.args[1]["run_id"] == "dtie_d8c34c16f0de_hyp"
