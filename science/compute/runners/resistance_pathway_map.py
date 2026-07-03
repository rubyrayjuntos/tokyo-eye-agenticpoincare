"""Atomic runner: resistance_pathway_map (Act 05 — Verdict)."""

from __future__ import annotations

from typing import Any

from science.compute.jobs.resistance_pathway_map import run_resistance_pathway_map
from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.phase_job_runner import run_phase_job

JOB_ID = "resistance_pathway_map"


async def run_resistance_pathway_map_job(db: Any, ctx: JobRunContext) -> JobRunResult:
    return await run_phase_job(
        db,
        ctx,
        job_id=JOB_ID,
        model_version="resistance-pathway-v1",
        artifacts=["resistance_pathway"],
        execute=run_resistance_pathway_map,
        caller_identity="compute_job_resistance_pathway_map",
    )
