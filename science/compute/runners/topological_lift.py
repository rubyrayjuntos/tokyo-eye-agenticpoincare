"""Atomic runner: topological_lift (Act 05 — Verdict)."""

from __future__ import annotations

from typing import Any

from science.compute.jobs.topological_lift import run_topological_lift
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.phase_job_runner import run_phase_job

JOB_ID = "topological_lift"


async def run_topological_lift_job(db: Any, ctx: JobRunContext) -> JobRunResult:
    return await run_phase_job(
        db,
        ctx,
        job_id=JOB_ID,
        model_version="topological-lift-v1",
        artifacts=["buffering_atlas"],
        execute=run_topological_lift,
        caller_identity="compute_job_topological_lift",
    )
