"""Shared in-process compute job dispatch for the Science API."""

from __future__ import annotations

from typing import Any

from science.compute.preconditions import check_job_preconditions
from science.compute.runner_dispatch import dispatch_compute_job
from science.compute.runners.base import JobRunContext, JobRunResult


async def dispatch_job_in_process(
    db: Any,
    job_id: str,
    *,
    structure_id: str,
    pathway: str = "discovery_story",
    computation_run_id: str | None = None,
    parent_run_id: str | None = None,
    pipeline_job_id: str | None = None,
    device: str = "cpu",
    checkpoint_path: str | None = None,
    job_params: dict[str, Any] | None = None,
    skip_preconditions: bool = False,
) -> JobRunResult:
    """Run a single atomic job in-process (canonical write path)."""
    structure_id = structure_id.strip().lower()
    job_id = job_id.strip()

    if not skip_preconditions:
        pre = await check_job_preconditions(
            db,
            job_id,
            structure_id,
            pipeline_job_id=pipeline_job_id,
        )
        if not pre.satisfied:
            return JobRunResult(
                job_id=job_id,
                run_id="",
                structure_id=structure_id,
                success=False,
                outputs={
                    "error": pre.error,
                    "missing_artifacts": list(pre.missing_artifacts),
                },
            )

    ctx = JobRunContext(
        structure_id=structure_id,
        job_id=job_id,
        pathway=pathway,
        computation_run_id=computation_run_id,
        parent_run_id=parent_run_id,
        pipeline_job_id=pipeline_job_id,
        device=device,
        checkpoint_path=checkpoint_path,
        job_params=job_params or {},
    )
    return await dispatch_compute_job(
        db,
        job_id,
        ctx,
        skip_preconditions=True,
    )
