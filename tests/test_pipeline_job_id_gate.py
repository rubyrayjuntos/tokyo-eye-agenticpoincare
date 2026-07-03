"""Gate: pipeline_job_id propagates through pathway dispatch and audit surfaces."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from science.compute.dispatch_helpers import dispatch_job_in_process
from science.compute.preconditions import check_job_preconditions
from science.compute.runners.base import JobRunContext, JobRunResult


@pytest.fixture(autouse=True)
def _clear_request_context():
    from shared.context import RequestContext, set_context

    set_context(RequestContext())
    yield
    set_context(RequestContext())


def test_dispatch_helpers_passes_pipeline_job_id_to_preconditions():
    source = inspect.getsource(dispatch_job_in_process)
    assert "pipeline_job_id=pipeline_job_id" in source
    assert "check_job_preconditions" in source


def test_preconditions_accepts_explicit_pipeline_job_id_parameter():
    signature = inspect.signature(check_job_preconditions)
    assert "pipeline_job_id" in signature.parameters


@pytest.mark.asyncio
async def test_dispatch_job_in_process_forwards_pipeline_job_id_to_context(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_dispatch(db, job_id, ctx, **kwargs):
        captured["pipeline_job_id"] = ctx.pipeline_job_id
        return JobRunResult(
            job_id=job_id,
            run_id="run-1",
            structure_id=ctx.structure_id,
            success=True,
            outputs={},
        )

    monkeypatch.setattr(
        "science.compute.dispatch_helpers.check_job_preconditions",
        AsyncMock(return_value=type("R", (), {"satisfied": True})()),
    )
    monkeypatch.setattr(
        "science.compute.dispatch_helpers.dispatch_compute_job",
        _fake_dispatch,
    )

    await dispatch_job_in_process(
        object(),
        "graph_topology",
        structure_id="4obe",
        pipeline_job_id="pipe-dispatch-42",
    )
    assert captured["pipeline_job_id"] == "pipe-dispatch-42"


@pytest.mark.asyncio
async def test_precondition_failure_audit_uses_explicit_pipeline_job_id(monkeypatch):
    audit_calls: list[dict] = []

    async def _capture(*_args, **kwargs):
        audit_calls.append(kwargs)

    monkeypatch.setattr(
        "shared.audit.instrumentation.audit_precondition_failed",
        _capture,
    )
    monkeypatch.setattr(
        "science.compute.preconditions._structure_exists",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "science.compute.preconditions._artifact_ready",
        AsyncMock(return_value=False),
    )

    result = await check_job_preconditions(
        object(),
        "gnn_inference",
        "4obe",
        pipeline_job_id="pipe-audit-99",
    )
    assert not result.satisfied
    assert audit_calls
    assert audit_calls[0]["pipeline_job_id"] == "pipe-audit-99"


@pytest.mark.asyncio
async def test_runner_dispatch_forwards_pipeline_job_id_to_preconditions(monkeypatch):
    captured: dict[str, object] = {}

    async def _capture_pre(db, job_id, structure_id, *, pipeline_job_id=None):
        captured["pipeline_job_id"] = pipeline_job_id
        return type("R", (), {"satisfied": True})()

    async def _fake_runner(db, ctx):
        return JobRunResult(
            job_id=ctx.job_id,
            run_id="run-1",
            structure_id=ctx.structure_id,
            success=True,
            outputs={},
        )

    monkeypatch.setitem(
        __import__("science.compute.runner_dispatch", fromlist=["JOB_RUNNERS"]).JOB_RUNNERS,
        "graph_topology",
        _fake_runner,
    )
    monkeypatch.setattr(
        "science.compute.preconditions.check_job_preconditions",
        _capture_pre,
    )
    monkeypatch.setattr(
        "science.compute.runner_dispatch._enrich_hyperbolic_context",
        AsyncMock(side_effect=lambda _db, _job, ctx: ctx),
    )
    monkeypatch.setattr(
        "science.compute.runner_dispatch._apply_result_validation",
        AsyncMock(side_effect=lambda _db, result, **kwargs: result),
    )

    from science.compute.runner_dispatch import dispatch_compute_job

    ctx = JobRunContext(
        structure_id="4obe",
        job_id="graph_topology",
        pathway="discovery_story",
        pipeline_job_id="pipe-runner-11",
    )
    await dispatch_compute_job(object(), "graph_topology", ctx)
    assert captured["pipeline_job_id"] == "pipe-runner-11"


@pytest.mark.asyncio
async def test_science_api_request_model_exposes_pipeline_job_id():
    from science.api.routers.compute import ComputeJobRequest

    fields = ComputeJobRequest.model_fields
    assert "pipeline_job_id" in fields
