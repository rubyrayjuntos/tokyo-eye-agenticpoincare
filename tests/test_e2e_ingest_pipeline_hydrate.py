"""End-to-end checkpoint test: ingest → pipeline → hydrate.

Validates the full flow:
1. POST /api/ingest — structure is ingested and dimensional tables populated
2. POST /api/pipeline/run — pipeline dispatches to ScienceClient and completes
3. GET /api/structures/{id}/hydrate — returns populated embedding data

This is task 10 (final checkpoint) in the science-container-api spec.

Mocks: RCSB fetch (no network), ScienceClient (no science container),
        DB pool (uses in-memory state). The test validates the full
        router-level flow through the agent coordinator app.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: mock data that simulates a successful pipeline run
# ---------------------------------------------------------------------------

STRUCTURE_ID = "rcsb:4obe"
PDB_ID = "4OBE"

MOCK_INGEST_RESULT = {
    "structure_id": STRUCTURE_ID,
    "pdb_id": PDB_ID,
    "chains": 1,
    "residues": 5,
    "atoms": 40,
    "source": "rcsb",
}

MOCK_PIPELINE_RESULT = {
    "run_id": "run_4obe_test123",
    "structure_id": STRUCTURE_ID,
    "phases_run": ["gnn_inference", "source_leak_detection"],
    "assets_created": 10,
    "duration_ms": 1234.5,
    "warnings": [],
}

MOCK_EMBEDDING_ROWS = [
    {
        "residue_id": f"rcsb:4obe:A:{i}",
        "residue_index": i,
        "residue_name": "ALA",
        "chain_label": "A",
        "hyp_projection_2d": None,
        "hyp_projections": [0.1 * i, -0.05 * i],
        "cone_depth": 3.0 + i * 0.5,
        "epistemic_uncertainty": 0.1 + i * 0.02,
        "aleatoric_uncertainty": 0.05,
        "curvature": 1.0,
    }
    for i in range(1, 6)
]

MOCK_STRUCTURE_SNAPSHOT = {
    "structure": {
        "structure_id": STRUCTURE_ID,
        "pdb_id": PDB_ID,
        "title": "Mock 4OBE",
        "method": "X-RAY",
        "resolution": 2.1,
        "source": "rcsb",
        "organism": "Homo sapiens",
        "release_date": "2024-01-01",
        "polymer_composition": "protein",
    },
    "scope": {
        "primary_chain_ids": ["A"],
        "reference_chain": "A",
        "exclude_chain_ids": [],
        "normalization_protocol": "graph_default",
        "scope_source": "auto",
        "selection_reason": "mock",
    },
    "provenance": {
        "latest_run_ids_by_pipeline": {"embeddings": "run_mock"},
        "latest_model_versions": {"embeddings": "GOSPConeMapper-v6"},
    },
    "curvature": 1.0,
    "residues": [
        {
            "residue_id": r["residue_id"],
            "residue_index": r["residue_index"],
            "residue_name": "ALA",
            "chain_label": r["chain_label"],
            "x": r["hyp_projections"][0],
            "y": r["hyp_projections"][1],
            "cone_depth": r["cone_depth"],
            "epistemic_uncertainty": r.get("epistemic_uncertainty", 0.1),
            "aleatoric_uncertainty": r.get("aleatoric_uncertainty", 0.05),
        }
        for r in MOCK_EMBEDDING_ROWS
    ],
    "graph_metrics": {"structure_id": STRUCTURE_ID, "metrics": {"nodes": 5}},
    "findings": {
        "source_leaks": {"structure_id": STRUCTURE_ID, "source_leaks": [], "count": 0},
        "allosteric_sites": {"structure_id": STRUCTURE_ID, "sites": [], "count": 0},
        "vulnerability_doorways": None,
        "resistance": None,
        "pharmacophores": None,
        "drug_candidates": None,
    },
    "status": {
        "embeddings_persisted": True,
        "graph_persisted": True,
        "sites_persisted": False,
        "phase2_persisted": False,
        "phase4_persisted": False,
        "phase5_persisted": False,
        "phase6_persisted": False,
    },
}


@asynccontextmanager
async def _mock_get_connection():
    """Mock get_connection that yields an AsyncMock."""
    mock_conn = AsyncMock()
    yield mock_conn


# ---------------------------------------------------------------------------
# Test class: End-to-end ingest → pipeline → hydrate
# ---------------------------------------------------------------------------


class TestE2EIngestPipelineHydrate:
    """End-to-end test verifying the full ingest → pipeline → hydrate flow."""

    @pytest.mark.asyncio
    async def test_ingest_endpoint_returns_structure(self):
        """POST /api/ingest should return structure metadata on success."""
        with (
            patch("data.db.open_pool", new_callable=AsyncMock),
            patch("data.db.close_pool", new_callable=AsyncMock),
            patch("agent.tools.diagnostics.run_startup_diagnostics") as mock_diag,
            patch(
                "data.db.get_connection",
                side_effect=_mock_get_connection,
            ),
            patch(
                "agent.tools.science_client.ScienceClient.ingest_structure",
                new_callable=AsyncMock,
                return_value=MOCK_INGEST_RESULT,
            ),
            patch(
                "agent.coordinator.routers.dashboard._create_job_db",
                new_callable=AsyncMock,
                return_value="ingest-job-001",
            ),
            patch(
                "agent.coordinator.routers.dashboard._run_pipeline_background",
                new_callable=AsyncMock,
            ),
        ):
            mock_report = MagicMock()
            mock_report.all_passed = True
            mock_diag.return_value = mock_report

            from httpx import ASGITransport, AsyncClient
            from agent.coordinator.app import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/api/ingest", json={"pdb_id": "4obe"})

            assert resp.status_code == 200
            body = resp.json()
            assert body["structure_id"] == STRUCTURE_ID
            assert body["pdb_id"] == PDB_ID
            assert body["residues"] == 5
            assert body["pipeline_status"] == "queued"
            assert body["pipeline_job_id"] == "ingest-job-001"

    @pytest.mark.asyncio
    async def test_pipeline_run_dispatches_job(self):
        """POST /api/pipeline/run should create a job and return job_id."""
        with (
            patch("data.db.open_pool", new_callable=AsyncMock),
            patch("data.db.close_pool", new_callable=AsyncMock),
            patch("agent.tools.diagnostics.run_startup_diagnostics") as mock_diag,
            patch(
                "agent.coordinator.routers.dashboard._create_job_db",
                new_callable=AsyncMock,
                return_value="test-job-id-001",
            ),
            patch(
                "agent.coordinator.routers.dashboard._run_pipeline_background",
                new_callable=AsyncMock,
            ),
        ):
            mock_report = MagicMock()
            mock_report.all_passed = True
            mock_diag.return_value = mock_report

            from httpx import ASGITransport, AsyncClient
            from agent.coordinator.app import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/pipeline/run",
                    json={"structure_id": STRUCTURE_ID},
                )

            assert resp.status_code == 200
            body = resp.json()
            assert body["job_id"] == "test-job-id-001"
            assert body["status"] == "queued"
            assert "status_url" in body

    @pytest.mark.asyncio
    async def test_pipeline_background_calls_science_client(self):
        """The background task should call ScienceClient.run_pipeline()."""
        with (
            patch(
                "agent.coordinator.routers.dashboard._update_job_db",
                new_callable=AsyncMock,
            ),
            patch(
                "agent.tools.science_client.ScienceClient.run_pipeline",
                new_callable=AsyncMock,
                return_value=MOCK_PIPELINE_RESULT,
            ) as mock_pipeline,
        ):
            from agent.coordinator.routers.dashboard import (
                _pipeline_jobs,
                _run_pipeline_background,
            )

            # Set up in-memory job
            job_id = "test-job-bg-001"
            _pipeline_jobs[job_id] = {
                "job_id": job_id,
                "structure_id": STRUCTURE_ID,
                "status": "queued",
                "current_step": "ingestion",
                "progress": 0,
                "modules": [],
                "started_at": "2026-06-22T00:00:00Z",
                "completed_at": None,
                "error": None,
            }

            await _run_pipeline_background(job_id, STRUCTURE_ID)

            mock_pipeline.assert_called_once_with(structure_id=STRUCTURE_ID)
            assert _pipeline_jobs[job_id]["status"] == "complete"
            assert _pipeline_jobs[job_id]["progress"] == 100

            # Cleanup
            del _pipeline_jobs[job_id]

    @pytest.mark.asyncio
    async def test_hydrate_returns_embeddings_after_pipeline(self):
        """GET /api/structures/{id}/hydrate should return embedding data."""
        # Override the get_db dependency to provide a mock
        mock_db = AsyncMock()

        async def _override_get_db():
            return mock_db

        with (
            patch("data.db.open_pool", new_callable=AsyncMock),
            patch("data.db.close_pool", new_callable=AsyncMock),
            patch("agent.tools.diagnostics.run_startup_diagnostics") as mock_diag,
            patch(
                "agent.coordinator.routers.dashboard._fetch_structure_analysis_snapshot",
                new_callable=AsyncMock,
                return_value=MOCK_STRUCTURE_SNAPSHOT,
            ),
            patch(
                "agent.tools.hypothesis.tools.get_hypotheses",
                new_callable=AsyncMock,
                return_value=MagicMock(success=True, data={"hypotheses": []}),
            ),
            patch(
                "agent.tools.data_tools.get_provenance_lineage",
                new_callable=AsyncMock,
                return_value=MagicMock(success=True, data={"runs": []}),
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_annotations_for_hydration",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase2_vulnerability",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase4_resistance",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase5_pharmacophore",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase6_drug_candidates",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_report = MagicMock()
            mock_report.all_passed = True
            mock_diag.return_value = mock_report

            from agent.coordinator.app import app
            from agent.coordinator.deps import get_db
            from httpx import ASGITransport, AsyncClient

            app.dependency_overrides[get_db] = _override_get_db

            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.get(f"/api/structures/{STRUCTURE_ID}/hydrate")

                assert resp.status_code == 200
                body = resp.json()
                assert body["structure_snapshot"] is not None
                assert body["structure_snapshot"]["structure"]["structure_id"] == STRUCTURE_ID
                # Embeddings should be populated (not null)
                assert body["embeddings"] is not None
                assert body["embeddings"]["structure_id"] == STRUCTURE_ID
                assert body["embeddings"]["curvature"] == 1.0
                assert len(body["embeddings"]["residues"]) == 5
                # First residue check
                r0 = body["embeddings"]["residues"][0]
                assert r0["residue_id"] == "rcsb:4obe:A:1"
                assert r0["chain_label"] == "A"
                assert r0["x"] == pytest.approx(0.1)
                assert r0["y"] == pytest.approx(-0.05)
            finally:
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_full_flow_ingest_to_hydrate(self):
        """Integration: ingest returns structure, pipeline dispatches, hydrate returns data.

        This test exercises the full sequence in one flow, verifying that
        the endpoints cooperate correctly.
        """
        mock_db = AsyncMock()

        async def _override_get_db():
            return mock_db

        with (
            patch("data.db.open_pool", new_callable=AsyncMock),
            patch("data.db.close_pool", new_callable=AsyncMock),
            patch("agent.tools.diagnostics.run_startup_diagnostics") as mock_diag,
            patch(
                "data.db.get_connection",
                side_effect=_mock_get_connection,
            ),
            patch(
                "agent.tools.science_client.ScienceClient.ingest_structure",
                new_callable=AsyncMock,
                return_value=MOCK_INGEST_RESULT,
            ),
            patch(
                "agent.coordinator.routers.dashboard._create_job_db",
                new_callable=AsyncMock,
                return_value="full-flow-job-001",
            ),
            patch(
                "agent.coordinator.routers.dashboard._run_pipeline_background",
                new_callable=AsyncMock,
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_structure_analysis_snapshot",
                new_callable=AsyncMock,
                return_value=MOCK_STRUCTURE_SNAPSHOT,
            ),
            patch(
                "agent.tools.hypothesis.tools.get_hypotheses",
                new_callable=AsyncMock,
                return_value=MagicMock(success=True, data={"hypotheses": []}),
            ),
            patch(
                "agent.tools.data_tools.get_provenance_lineage",
                new_callable=AsyncMock,
                return_value=MagicMock(success=True, data={"runs": []}),
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_annotations_for_hydration",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase2_vulnerability",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase4_resistance",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase5_pharmacophore",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "agent.coordinator.routers.dashboard._fetch_phase6_drug_candidates",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_report = MagicMock()
            mock_report.all_passed = True
            mock_diag.return_value = mock_report

            from agent.coordinator.app import app
            from agent.coordinator.deps import get_db
            from httpx import ASGITransport, AsyncClient

            app.dependency_overrides[get_db] = _override_get_db

            try:
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    # Step 1: Ingest
                    ingest_resp = await client.post(
                        "/api/ingest", json={"pdb_id": "4obe"}
                    )
                    assert ingest_resp.status_code == 200
                    ingest_body = ingest_resp.json()
                    structure_id = ingest_body["structure_id"]
                    assert structure_id == STRUCTURE_ID
                    assert ingest_body["pipeline_status"] == "queued"
                    assert ingest_body["pipeline_job_id"] == "full-flow-job-001"

                    # Step 2: Trigger pipeline
                    pipeline_resp = await client.post(
                        "/api/pipeline/run",
                        json={"structure_id": structure_id},
                    )
                    assert pipeline_resp.status_code == 200
                    pipeline_body = pipeline_resp.json()
                    assert pipeline_body["status"] == "queued"
                    assert pipeline_body["job_id"] == "full-flow-job-001"

                    # Step 3: Hydrate (simulates after pipeline completes)
                    hydrate_resp = await client.get(
                        f"/api/structures/{structure_id}/hydrate"
                    )
                    assert hydrate_resp.status_code == 200
                    hydrate_body = hydrate_resp.json()
                    assert hydrate_body["structure_snapshot"] is not None

                    # Verify embeddings are populated
                    assert hydrate_body["embeddings"] is not None
                    assert hydrate_body["embeddings"]["structure_id"] == STRUCTURE_ID
                    assert len(hydrate_body["embeddings"]["residues"]) == 5

                    # Verify graph metrics are populated
                    assert hydrate_body["graph_metrics"] is not None
            finally:
                app.dependency_overrides.clear()
