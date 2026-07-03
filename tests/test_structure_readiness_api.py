"""Tests for structure readiness HTTP API semantics."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from data.readiness import StructureReadiness


@pytest.mark.asyncio
async def test_readiness_returns_200_for_failed_compute(monkeypatch):
    from agent.coordinator.routers.structures import get_structure_readiness

    readiness = StructureReadiness(
        structure_id="11qe",
        readiness_status="failed",
        pathway="discovery_story",
        current_act=1,
        tier1={
            "dims": True,
            "scope": True,
            "gnn_hyp": False,
            "graph": False,
            "source_leaks": False,
            "binding_scan": False,
        },
        tier2={},
        missing_artifacts=["gnn_hyp"],
        degraded_reasons=["Pipeline job error: Preconditions not met for gnn_inference: missing dims"],
        pipeline_job={
            "job_id": "job-1",
            "structure_id": "11qe",
            "status": "failed",
            "error": "Preconditions not met for gnn_inference: missing dims",
        },
        foundation={"dims": True, "scope": True, "gnn_hyp": False},
        acts={},
        artifacts={},
    )

    async def _assess(_structure_id: str, _db: Any) -> StructureReadiness:
        return readiness

    monkeypatch.setattr(
        "agent.coordinator.routers.structures.assess_structure_readiness",
        _assess,
    )

    result = await get_structure_readiness("11qe", db=AsyncMock(), _user={})
    assert result["readiness_status"] == "failed"
    assert result["pipeline_job"]["status"] == "failed"


@pytest.mark.asyncio
async def test_readiness_404_only_when_structure_missing_from_dim(monkeypatch):
    from agent.coordinator.routers.structures import get_structure_readiness

    readiness = StructureReadiness(
        structure_id="missing",
        readiness_status="failed",
        pathway="discovery_story",
        current_act=0,
        tier1=dict.fromkeys(
            ["dims", "scope", "gnn_hyp", "graph", "source_leaks", "binding_scan"],
            False,
        ),
        tier2={},
        missing_artifacts=["dims"],
        degraded_reasons=["Structure not found in dim_structure"],
        pipeline_job=None,
        foundation={},
        acts={},
        artifacts={},
    )

    async def _assess(_structure_id: str, _db: Any) -> StructureReadiness:
        return readiness

    monkeypatch.setattr(
        "agent.coordinator.routers.structures.assess_structure_readiness",
        _assess,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_structure_readiness("missing", db=AsyncMock(), _user={})

    assert exc_info.value.status_code == 404
    assert "missing" in str(exc_info.value.detail)
