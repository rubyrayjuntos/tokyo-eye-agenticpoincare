"""Integration gates: real DBAdapter on ingest foundation → gnn_inference preconditions.

Requires TEST_DATABASE_URL or DATABASE_URL with migrations applied.
Skips automatically when no DB is configured.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from data.readiness import assess_structure_readiness, probe_dims
from science.compute.preconditions import check_job_preconditions
from tests.helpers.science_container_fixtures import seed_ingest_foundation


@pytest.mark.integration
class TestScienceContainerReadinessIntegration:
    @pytest.mark.asyncio
    async def test_probe_dims_true_with_real_dbadapter(self, integration_db_with_schema) -> None:
        structure_id = await seed_ingest_foundation(integration_db_with_schema, pdb_id="11qe")
        assert await probe_dims(structure_id, integration_db_with_schema) is True

    @pytest.mark.asyncio
    async def test_gnn_inference_preconditions_pass_after_ingest_seed(
        self, integration_db_with_schema
    ) -> None:
        structure_id = await seed_ingest_foundation(integration_db_with_schema, pdb_id="4obe")
        result = await check_job_preconditions(
            integration_db_with_schema,
            "gnn_inference",
            structure_id,
        )
        assert result.satisfied, result.error

    @pytest.mark.asyncio
    async def test_readiness_dims_true_after_ingest_seed(
        self, integration_db_with_schema
    ) -> None:
        structure_id = await seed_ingest_foundation(integration_db_with_schema, pdb_id="4uj1")
        readiness = await assess_structure_readiness(structure_id, integration_db_with_schema)
        assert readiness.tier1.get("dims") is True
        assert readiness.tier1.get("scope") is True
        assert not readiness.probe_errors.get("dims")


@pytest.mark.integration
class TestScienceApiPreconditionsIntegration:
    @pytest.mark.asyncio
    async def test_science_api_compute_job_uses_dbadapter_preconditions(
        self,
        integration_db_with_schema,
        monkeypatch,
    ) -> None:
        """Science API must not bypass DBAdapter when evaluating gnn_inference preconditions."""
        from httpx import ASGITransport, AsyncClient

        structure_id = await seed_ingest_foundation(integration_db_with_schema, pdb_id="gate2")
        captured: dict[str, object] = {}

        async def _fake_dispatch(db, job_id, **kwargs):
            captured["db_type"] = type(db).__name__
            captured["job_id"] = job_id
            captured["kwargs"] = kwargs
            from science.compute.runners.base import JobRunResult

            return JobRunResult(
                job_id=job_id,
                run_id="",
                structure_id=kwargs["structure_id"],
                success=False,
                outputs={"error": "stub stop before runner"},
            )

        conn = integration_db_with_schema._conn  # noqa: SLF001 — test fixture access

        class _ConnCM:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *_args):
                return False

        monkeypatch.setattr(
            "science.api.routers.compute.get_connection",
            lambda: _ConnCM(),
        )
        monkeypatch.setattr(
            "science.compute.dispatch_helpers.dispatch_job_in_process",
            _fake_dispatch,
        )

        from science.api.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/compute/jobs/gnn_inference",
                json={
                    "structure_id": structure_id,
                    "pipeline_job_id": "pipe-integration-001",
                    "device": "cpu",
                },
            )

        assert response.status_code in (409, 422, 500)
        assert captured["db_type"] == "DBAdapter"
        assert captured["job_id"] == "gnn_inference"
        assert captured["kwargs"]["pipeline_job_id"] == "pipe-integration-001"

    @pytest.mark.asyncio
    async def test_precondition_audit_receives_pipeline_job_id_without_request_context(
        self,
        integration_db_with_schema,
        monkeypatch,
    ) -> None:
        structure_id = await seed_ingest_foundation(integration_db_with_schema, pdb_id="gate3")
        audit_calls: list[dict] = []

        async def _capture_audit(*_args, **kwargs):
            audit_calls.append(kwargs)

        monkeypatch.setattr(
            "shared.audit.instrumentation.audit_precondition_failed",
            _capture_audit,
        )
        monkeypatch.setattr(
            "science.compute.preconditions.probe_artifact",
            AsyncMock(return_value=False),
        )

        result = await check_job_preconditions(
            integration_db_with_schema,
            "gnn_inference",
            structure_id,
            pipeline_job_id="pipe-explicit-777",
        )
        assert not result.satisfied
        assert audit_calls
        assert audit_calls[0]["pipeline_job_id"] == "pipe-explicit-777"
