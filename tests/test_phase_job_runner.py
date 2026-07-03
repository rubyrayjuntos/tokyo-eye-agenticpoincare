"""Tests for phase-backed compute job runner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from science.compute.runners.base import JobRunContext
from science.compute.runners.phase_job_runner import run_phase_job
from science.dtie.common.interfaces import PhaseResult


@pytest.mark.asyncio
async def test_run_phase_job_passes_gnn_run_id_as_keyword():
    """Phase executors use keyword-only gnn_run_id; runner must not pass it positionally."""
    execute = AsyncMock(
        return_value=PhaseResult(
            phase_name="phase2",
            structure_id="11qe",
            model_version="v5",
            success=True,
            outputs={"residues_scanned": 10},
        )
    )
    db = MagicMock()
    db.commit = AsyncMock()
    ctx = JobRunContext(
        structure_id="11qe",
        job_id="strain_vulnerability_scan",
        parent_run_id="gnn_run_abc",
        computation_run_id="comp_run_1",
    )

    with (
        patch(
            "science.compute.runners.phase_job_runner.insert_job_provenance",
            new_callable=AsyncMock,
        ),
        patch(
            "science.compute.runners.phase_job_runner.finalize_job_provenance",
            new_callable=AsyncMock,
        ),
        patch(
            "science.compute.runners.phase_job_runner.persist_phase_result",
            new_callable=AsyncMock,
        ),
    ):
        result = await run_phase_job(
            db,
            ctx,
            job_id="strain_vulnerability_scan",
            model_version="v5",
            artifacts=["strain_vulnerability"],
            execute=execute,
            persist=False,
        )

    assert result.success is True
    execute.assert_awaited_once()
    _, kwargs = execute.call_args
    assert kwargs == {"gnn_run_id": "gnn_run_abc"}
    assert execute.call_args[0][0] is db
    assert execute.call_args[0][1].structure_id == "11qe"
