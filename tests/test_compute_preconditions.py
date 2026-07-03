"""Tests for compute job preconditions."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from science.compute.preconditions import check_job_preconditions


class _FakeDB:
    def __init__(self, *, structure: bool = True, artifacts: set[str] | None = None):
        self.structure = structure
        self.artifacts = artifacts or set()

    async def fetch_one(self, query: str, params: dict):
        if "dim_structure" in query:
            return {"ok": 1} if self.structure else None
        return None


@pytest.mark.asyncio
async def test_unknown_structure_fails(monkeypatch):
    db = _FakeDB(structure=False)

    async def _probe(_db, _sid, artifact):
        return artifact in db.artifacts

    monkeypatch.setattr(
        "science.compute.preconditions.probe_artifact",
        _probe,
    )
    result = await check_job_preconditions(db, "gnn_inference", "4obe")
    assert not result.satisfied
    assert "not found" in (result.error or "")


@pytest.mark.asyncio
async def test_gnn_requires_dims_and_scope(monkeypatch):
    db = _FakeDB(artifacts={"dims"})

    async def _probe(_db, _sid, artifact):
        return artifact in db.artifacts

    monkeypatch.setattr("science.compute.preconditions.probe_artifact", _probe)
    result = await check_job_preconditions(db, "gnn_inference", "4obe")
    assert not result.satisfied
    assert "scope" in result.missing_artifacts


@pytest.mark.asyncio
async def test_source_leak_requires_learned_curvature(monkeypatch):
    db = _FakeDB(artifacts={"dims", "scope", "gnn_hyp"})

    async def _probe(_db, _sid, artifact):
        return artifact in db.artifacts

    async def _no_curvature(_db, _sid):
        return None

    monkeypatch.setattr("science.compute.preconditions.probe_artifact", _probe)
    monkeypatch.setattr(
        "science.compute.preconditions.load_structure_learned_curvature",
        _no_curvature,
    )
    result = await check_job_preconditions(db, "source_leak_detection", "4obe")
    assert not result.satisfied
    assert "learned_curvature" in result.missing_artifacts


@pytest.mark.asyncio
async def test_graph_topology_requires_gnn(monkeypatch):
    db = _FakeDB(artifacts={"dims", "scope"})

    async def _probe(_db, _sid, artifact):
        return artifact in {"dims", "scope"}

    monkeypatch.setattr("science.compute.preconditions.probe_artifact", _probe)
    result = await check_job_preconditions(db, "graph_topology", "4obe")
    assert not result.satisfied
    assert "gnn_hyp" in result.missing_artifacts


@pytest.mark.asyncio
async def test_source_leak_passes_with_learned_curvature(monkeypatch):
    db = _FakeDB(artifacts={"dims", "scope", "gnn_hyp", "gnn_euc"})

    async def _probe(_db, _sid, artifact):
        return artifact in db.artifacts

    async def _curvature(_db, _sid):
        return 0.92

    monkeypatch.setattr("science.compute.preconditions.probe_artifact", _probe)
    monkeypatch.setattr(
        "science.compute.preconditions.load_structure_learned_curvature",
        _curvature,
    )
    result = await check_job_preconditions(db, "source_leak_detection", "4obe")
    assert result.satisfied
