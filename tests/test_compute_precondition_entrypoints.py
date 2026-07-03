"""Prove scheduler and API endpoints share one precondition resolver."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from science.compute.preconditions import (
    PRECONDITION_RESOLVER,
    PreconditionResult,
    check_job_preconditions,
    precondition_dispatch_entrypoints,
)
from science.compute.runners.base import JobRunContext


class _FakeDB:
    def __init__(self, *, structure: bool = True, artifacts: set[str] | None = None):
        self.structure = structure
        self.artifacts = artifacts or set()

    async def fetch_one(self, query: str, params: dict):
        if "dim_structure" in query:
            return {"ok": 1} if self.structure else None
        return None


def _result_from_job_run(result) -> PreconditionResult:
    if result.success:
        return PreconditionResult(True)
    return PreconditionResult(
        False,
        error=result.outputs.get("error"),
        missing_artifacts=tuple(result.outputs.get("missing_artifacts", [])),
    )


@pytest.fixture(autouse=True)
def _noop_audit_precondition_failed(monkeypatch):
    monkeypatch.setattr(
        "shared.audit.instrumentation.audit_precondition_failed",
        AsyncMock(),
    )


def test_precondition_resolver_is_canonical_function():
    assert PRECONDITION_RESOLVER is check_job_preconditions


def test_dispatch_helpers_imports_canonical_resolver():
    from science.compute import dispatch_helpers, preconditions

    assert dispatch_helpers.check_job_preconditions is preconditions.check_job_preconditions


def test_pathway_and_api_entrypoints_use_dispatch_helper():
    from science.compute.pathway_executor import InProcessJobBackend

    pathway_src = inspect.getsource(InProcessJobBackend.run_compute_job)
    assert "dispatch_job_in_process" in pathway_src

    from science.api.routers import compute as compute_router

    compute_src = inspect.getsource(compute_router)
    assert "dispatch_job_in_process" in compute_src


def test_documented_entrypoints_cover_scheduler_path():
    entrypoints = precondition_dispatch_entrypoints()
    assert "science.compute.dispatch_helpers.dispatch_job_in_process" in entrypoints
    assert "science.compute.pathway_executor.InProcessJobBackend.run_compute_job" in entrypoints


@pytest.mark.parametrize(
    ("job_id", "artifacts", "expected_missing"),
    [
        ("gnn_inference", {"dims"}, ("scope",)),
        (
            "source_leak_detection",
            {"dims", "scope", "gnn_hyp", "gnn_euc"},
            ("learned_curvature",),
        ),
    ],
)
@pytest.mark.parametrize("via", ["direct", "dispatch_helper", "runner_dispatch"])
@pytest.mark.asyncio
async def test_precondition_parity_across_entrypoints(
    via: str,
    job_id: str,
    artifacts: set[str],
    expected_missing: tuple[str, ...],
    monkeypatch,
):
    db = _FakeDB(artifacts=artifacts)

    async def _probe(_db, _sid, artifact: str) -> bool:
        return artifact in db.artifacts

    async def _no_curvature(_db, _sid):
        return None

    monkeypatch.setattr("science.compute.preconditions.probe_artifact", _probe)
    monkeypatch.setattr(
        "science.compute.preconditions.load_structure_learned_curvature",
        _no_curvature,
    )

    direct = await check_job_preconditions(db, job_id, "4obe")
    assert not direct.satisfied
    assert expected_missing == direct.missing_artifacts

    if via == "direct":
        return

    if via == "dispatch_helper":
        from science.compute.dispatch_helpers import dispatch_job_in_process

        wrapped = await dispatch_job_in_process(db, job_id, structure_id="4obe")
        via_result = _result_from_job_run(wrapped)
    else:
        from science.compute.runner_dispatch import dispatch_compute_job

        ctx = JobRunContext(
            structure_id="4obe",
            job_id=job_id,
            pathway="discovery_story",
        )
        wrapped = await dispatch_compute_job(
            db,
            job_id,
            ctx,
            skip_preconditions=False,
        )
        via_result = _result_from_job_run(wrapped)

    assert via_result.satisfied == direct.satisfied
    assert via_result.missing_artifacts == direct.missing_artifacts
    assert (via_result.error or "") == (direct.error or "")
