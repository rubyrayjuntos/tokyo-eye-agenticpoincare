"""Smoke tests for the Science API — checkpoint verification.

Verifies:
1. The FastAPI app starts and the /health endpoint responds
2. The /compute/gnn endpoint handles requests (schema validation, 404 on missing structure)
3. All compute router models are importable and valid

These tests mock the DB layer to run without a real database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Health endpoint responds with correct structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint_responds():
    """Verify /health endpoint returns valid JSON with expected fields."""
    with (
        patch("data.db.open_pool", new_callable=AsyncMock) as mock_open,
        patch("data.db.close_pool", new_callable=AsyncMock) as mock_close,
        patch("science.api.routers.health._check_db", new_callable=AsyncMock) as mock_db,
        patch("science.api.routers.health._detect_gpu") as mock_gpu,
        patch("science.api.routers.health._list_checkpoints") as mock_ckpts,
    ):
        mock_open.return_value = None
        mock_close.return_value = None
        mock_db.return_value = {"connected": True}
        mock_gpu.return_value = {"available": False, "device_count": 0, "device_name": None}
        mock_ckpts.return_value = ["v5_stage4_11prot.pt"]

        from httpx import ASGITransport, AsyncClient
        from science.api.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "gpu" in body
        assert "checkpoints" in body
        assert "db" in body
        assert "contract_version" in body
        assert "gnn_production" in body
        assert "job_registry" in body
        assert body["status"] == "healthy"
        assert body["contract_version"] == "1.6"
        assert body["gnn_production"]["model_id"] == "tokyo_eye_v8"
        assert body["gpu"]["available"] is False
        assert body["checkpoints"] == ["v5_stage4_11prot.pt"]


# ---------------------------------------------------------------------------
# 2. GNN endpoint returns 404 when structure has no residues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gnn_endpoint_fails_when_structure_missing():
    """Verify POST /compute/gnn returns error when GNN dispatch fails."""
    from science.compute.runners.base import JobRunResult

    failed = JobRunResult(
        job_id="gnn_inference",
        run_id="run_fail",
        structure_id="fake",
        success=False,
        artifacts_produced=[],
        outputs={"error": "No residues found for structure 'FAKE'"},
    )

    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch(
            "science.api.routers.compute._compute_checkpoint_hash",
            return_value="abc123hash",
        ),
        patch("science.api.routers.compute.get_connection") as mock_conn_ctx,
        patch(
            "science.compute.dispatch_helpers.dispatch_job_in_process",
            new_callable=AsyncMock,
            return_value=failed,
        ),
    ):
        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_conn_ctx.return_value = mock_db

        from httpx import ASGITransport, AsyncClient
        from science.api.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/compute/gnn",
                json={
                    "structure_id": "FAKE",
                    "model_version": "v6",
                    "device": "cpu",
                    "checkpoint_path": "checkpoints/v6/tokyo_eyes_v6.pt",
                },
            )

        assert resp.status_code == 500
        body = resp.json()
        assert "detail" in body
        assert "No residues found" in body["detail"]


# ---------------------------------------------------------------------------
# 3. GNN endpoint request validation (Pydantic model)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gnn_endpoint_validates_request_body():
    """Verify POST /compute/gnn rejects invalid request body."""
    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
    ):
        from httpx import ASGITransport, AsyncClient
        from science.api.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Missing required field structure_id
            resp = await client.post("/compute/gnn", json={})

        assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# 4. All compute models are importable and valid
# ---------------------------------------------------------------------------


def test_compute_models_importable():
    """Verify all request/response Pydantic models can be instantiated."""
    from science.api.routers.compute import (
        CrypticScanRequest,
        CrypticScanResponse,
        GNNRequest,
        GNNResponse,
        MDValidateRequest,
        MDValidateResponse,
        MotifAnalysisRequest,
        MotifAnalysisResponse,
        PipelineRequest,
        PipelineResponse,
    )

    # Test with default values where available
    gnn_req = GNNRequest(structure_id="test_4obe")
    assert gnn_req.structure_id == "test_4obe"
    assert gnn_req.model_version == "v8"
    assert gnn_req.checkpoint_path == "checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt"
    assert gnn_req.device == "cpu"

    gnn_resp = GNNResponse(
        run_id="run_abc",
        structure_id="test_4obe",
        node_count=100,
        checkpoint_version_hash="deadbeef",
        duration_ms=42.5,
    )
    assert gnn_resp.node_count == 100

    pipeline_req = PipelineRequest(structure_id="test_4obe")
    assert pipeline_req.source_leak_only is False

    pipeline_resp = PipelineResponse(
        run_id="run_abc",
        structure_id="test_4obe",
        phases_run=["gnn_inference", "phase2"],
        assets_created=10,
        duration_ms=100.0,
    )
    assert pipeline_resp.assets_created == 10

    cryptic_req = CrypticScanRequest(structure_id="test_4obe")
    assert cryptic_req.candidate_percentile == 75.0

    motif_req = MotifAnalysisRequest(structure_id="test_4obe")
    assert motif_req.min_cluster_size == 5

    md_req = MDValidateRequest(structure_id="test_4obe", site_id="site_1")
    assert md_req.dry_run is False
    assert md_req.force_field == "amber14-all"


# ---------------------------------------------------------------------------
# 5. Health endpoint reports degraded when DB is down
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_degraded_when_db_unavailable():
    """Verify /health returns 'degraded' status when DB connection fails."""
    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
        patch(
            "science.api.routers.health._check_db",
            new_callable=AsyncMock,
            return_value={"connected": False, "error": "Connection refused"},
        ),
        patch(
            "science.api.routers.health._detect_gpu",
            return_value={"available": False, "device_count": 0, "device_name": None},
        ),
        patch("science.api.routers.health._list_checkpoints", return_value=[]),
    ):
        from httpx import ASGITransport, AsyncClient
        from science.api.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
