"""Tests for skipping jobs when governed artifacts already exist."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from science.compute.job_skip import job_artifacts_already_present, skip_result_for_existing_artifacts
from science.compute.runner_dispatch import dispatch_compute_job
from science.compute.runners.base import JobRunContext
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput


class _SkipDB:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        self.queries.append(query)
        return {"ok": 1}

    async def fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.queries.append(query)
        return []


@pytest.mark.asyncio
async def test_job_artifacts_already_present_requires_all_produces(monkeypatch):
    async def _probe_all(_db, _sid, artifact: str) -> bool:
        return artifact in {"gnn_hyp", "gnn_euc"}

    monkeypatch.setattr("science.compute.job_skip.probe_artifact", _probe_all)
    db = _SkipDB()
    assert await job_artifacts_already_present(db, "gnn_inference", "11qe") is True

    async def _probe_partial(_db, _sid, artifact: str) -> bool:
        return artifact == "gnn_hyp"

    monkeypatch.setattr("science.compute.job_skip.probe_artifact", _probe_partial)
    assert await job_artifacts_already_present(db, "gnn_inference", "11qe") is False


@pytest.mark.asyncio
async def test_skip_result_includes_embedding_run_id():
    db = _SkipDB()
    gnn_result = GNNInferenceResult(
        structure_id="11qe",
        model_version="v6",
        checkpoint_path=None,
        nodes=[
            GNNNodeOutput(
                residue_index=1,
                chain_label="A",
                input_features=np.zeros(4),
                projections=np.zeros(64),
                cone_depth=2.0,
                cone_width=0.1,
                epistemic_uncertainty=0.4,
            )
        ],
        space_type="hyperbolic",
        metadata={"run_id": "gnn-run-existing"},
    )
    with patch(
        "science.compute.gnn_loader.load_gnn_inference_result",
        new_callable=AsyncMock,
        return_value=gnn_result,
    ):
        result = await skip_result_for_existing_artifacts(db, "gnn_inference", "11qe")

    assert result.success is True
    assert result.outputs.get("skipped") is True
    assert result.outputs.get("embedding_run_id") == "gnn-run-existing"
    assert "gnn_hyp" in result.artifacts_produced


@pytest.mark.asyncio
async def test_dispatch_skips_gnn_when_embeddings_present(monkeypatch):
    db = _SkipDB()
    runner_called = False

    async def _probe(_db, _sid, artifact: str) -> bool:
        return artifact in {"dims", "scope", "gnn_hyp", "gnn_euc"}

    monkeypatch.setattr("science.compute.preconditions.probe_artifact", _probe)
    monkeypatch.setattr("science.compute.job_skip.probe_artifact", _probe)

    async def _should_not_run(_db, _ctx):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner should not be invoked")

    monkeypatch.setitem(
        __import__("science.compute.runner_dispatch", fromlist=["JOB_RUNNERS"]).JOB_RUNNERS,
        "gnn_inference",
        _should_not_run,
    )
    monkeypatch.setattr(
        "science.compute.runner_dispatch._enrich_hyperbolic_context",
        AsyncMock(side_effect=lambda _db, _job, ctx: ctx),
    )
    monkeypatch.setattr(
        "science.compute.runner_dispatch._apply_result_validation",
        AsyncMock(side_effect=lambda _db, result, **kwargs: result),
    )

    gnn_result = GNNInferenceResult(
        structure_id="11qe",
        model_version="v6",
        checkpoint_path=None,
        nodes=[],
        space_type="hyperbolic",
        metadata={"run_id": "gnn-run-existing"},
    )
    with patch(
        "science.compute.gnn_loader.load_gnn_inference_result",
        new_callable=AsyncMock,
        return_value=gnn_result,
    ):
        ctx = JobRunContext(structure_id="11qe", job_id="gnn_inference")
        result = await dispatch_compute_job(db, "gnn_inference", ctx)

    assert result.success is True
    assert result.outputs.get("skipped") is True
    assert runner_called is False
