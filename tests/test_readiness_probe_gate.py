"""P5 gate: readiness probes must distinguish infra errors from absent artifacts."""

from __future__ import annotations

from typing import Any

import pytest

from data.readiness import (
    PROBE_ERROR_PREFIX,
    ProbeInfrastructureError,
    assess_structure_readiness,
    probe_artifact,
    probe_dims,
)
from science.compute.preconditions import check_job_preconditions


class _StructureExistsDB:
    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "dim_structure" in query.lower():
            return {"ok": 1}
        if "pipeline_job" in query.lower():
            return None
        if "embedding_space" in query.lower() or "fact_gnn_node_embedding" in query.lower():
            return None
        return None


class _BrokenDimsDB(_StructureExistsDB):
    """Simulates production DBAdapter missing a method used by a broken probe."""

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "dim_residue" in query.lower() and "count" in query.lower():
            raise AttributeError("'DBAdapter' object has no attribute 'fetch_val'")
        return await super().fetch_one(query, params)


class _EmptyDimsDB(_StructureExistsDB):
    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "dim_residue" in query.lower() and "count" in query.lower():
            return {"count": 0}
        if "structure_computation_scope" in query.lower():
            return None
        return await super().fetch_one(query, params)


class _HealthyDimsDB(_StructureExistsDB):
    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "dim_residue" in query.lower() and "count" in query.lower():
            return {"count": 42}
        if "structure_computation_scope" in query.lower():
            return {"ok": 1}
        return await super().fetch_one(query, params)


@pytest.mark.asyncio
async def test_probe_dims_infra_error_surfaces_in_readiness_probe_errors() -> None:
    result = await assess_structure_readiness("11qe", _BrokenDimsDB())
    assert result.tier1.get("dims") is False
    assert "dims" in result.probe_errors
    assert "AttributeError" in result.probe_errors["dims"]
    assert any(PROBE_ERROR_PREFIX in reason for reason in result.degraded_reasons)
    assert "dims" not in result.missing_artifacts
    assert not any(
        reason.startswith("Missing tier-1 artifact: dims") for reason in result.degraded_reasons
    )


@pytest.mark.asyncio
async def test_probe_dims_absent_differs_from_infra_error() -> None:
    absent = await assess_structure_readiness("11qe", _EmptyDimsDB())
    broken = await assess_structure_readiness("11qe", _BrokenDimsDB())

    assert absent.tier1.get("dims") is False
    assert broken.tier1.get("dims") is False
    assert absent.probe_errors == {}
    assert broken.probe_errors
    assert "dims" in absent.missing_artifacts
    assert "dims" not in broken.missing_artifacts
    assert any("Missing tier-1 artifact: dims" in r for r in absent.degraded_reasons)
    assert not any("Missing tier-1 artifact: dims" in r for r in broken.degraded_reasons)


@pytest.mark.asyncio
async def test_probe_artifact_raises_probe_infrastructure_error() -> None:
    with pytest.raises(ProbeInfrastructureError) as exc_info:
        await probe_artifact(_BrokenDimsDB(), "11qe", "dims")
    assert exc_info.value.artifact == "dims"
    assert "AttributeError" in str(exc_info.value)


@pytest.mark.asyncio
async def test_preconditions_report_probe_error_not_missing_dims() -> None:
    result = await check_job_preconditions(_BrokenDimsDB(), "gnn_inference", "11qe")
    assert not result.satisfied
    assert result.error is not None
    assert PROBE_ERROR_PREFIX in result.error
    assert "missing dims" not in result.error.lower()
    assert result.missing_artifacts == ()


@pytest.mark.asyncio
async def test_preconditions_still_report_missing_dims_when_probe_succeeds() -> None:
    result = await check_job_preconditions(_EmptyDimsDB(), "gnn_inference", "11qe")
    assert not result.satisfied
    assert result.missing_artifacts
    assert "dims" in result.missing_artifacts
    assert PROBE_ERROR_PREFIX not in (result.error or "")


@pytest.mark.asyncio
async def test_probe_witness_embedding_uses_safe_literals() -> None:
    from data.readiness import probe_witness_embedding

    class _WitnessDB:
        def __init__(self) -> None:
            self.query = ""

        async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
            self.query = query
            return None

    db = _WitnessDB()
    assert await probe_witness_embedding("11qe", db) is False
    assert "phase1_witness_embedding" in db.query
    assert "%witness%" not in db.query


@pytest.mark.asyncio
async def test_probe_dims_healthy_path() -> None:
    assert await probe_dims("11qe", _HealthyDimsDB()) is True
