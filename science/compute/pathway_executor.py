"""Pathway execution — single orchestration path for onboard and API.

All jobs dispatch through ``dispatch_job_in_process`` (Science API) or
``HttpJobBackend`` (coordinator agent calling Science over HTTP).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from science.compute.registry import DEFAULT_PATHWAY, JOB_REGISTRY, PATHWAY_JOBS, topological_order
from science.compute.runner_dispatch import PEELED_JOBS

logger = logging.getLogger(__name__)


def _job_exception_message(exc: Exception) -> str:
    """Prefer structured API error detail over generic wrapper messages."""
    detail = getattr(exc, "detail", None)
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return str(exc).strip() or type(exc).__name__


EXTERNAL_JOBS = frozenset(
    {
        "ingest_dims",
        "assign_computation_scope",
        "alignment_sidecar",
    }
)

PHASE_RESULT_TO_JOB: dict[str, str] = {
    "gnn_inference": "gnn_inference",
    "binding_site_scan": "binding_site_scan",
    "phase2": "strain_vulnerability_scan",
    "phase35": "topological_lift",
    "phase4": "resistance_pathway_map",
    "phase5": "pharmacophore_identification",
    "phase6_drug_discovery": "drug_candidate_ranking",
    "source_leak_detection": "source_leak_detection",
    "allosteric_sites": "allosteric_site_detection",
    "buffering_atlas": "topological_lift",
}

IMPLICIT_JOBS_ON_GNN_SUCCESS = frozenset({"graph_topology"})

JobCallback = Callable[[str, str, dict[str, Any] | None], Awaitable[None]]


def jobs_for_pathway(pathway_id: str, *, source_leak_only: bool = False) -> list[str]:
    if source_leak_only:
        pathway_id = "viewport_explore"
    job_ids = PATHWAY_JOBS.get(pathway_id)
    if job_ids is None:
        raise ValueError(f"Unknown pathway: {pathway_id}")
    return topological_order(job_ids)


def map_phases_to_jobs(phases_run: list[str]) -> set[str]:
    completed: set[str] = set()
    phase_set = set(phases_run)
    for phase_name, job_id in PHASE_RESULT_TO_JOB.items():
        if phase_name in phase_set:
            completed.add(job_id)
    if "gnn_inference" in phase_set:
        completed |= IMPLICIT_JOBS_ON_GNN_SUCCESS - PEELED_JOBS
    return completed


class JobBackend(Protocol):
    async def run_compute_job(
        self,
        job_id: str,
        structure_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    async def run_post_source_leak_phases(
        self,
        structure_id: str,
        parent_run_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class HttpJobBackend:
    """Dispatch jobs via Science HTTP API (coordinator → science container)."""

    def __init__(self) -> None:
        from agent.tools.science_client import ScienceClient

        self._client = ScienceClient()

    async def run_compute_job(
        self,
        job_id: str,
        structure_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._client.run_compute_job(job_id, structure_id, **kwargs)

    async def run_post_source_leak_phases(
        self,
        structure_id: str,
        parent_run_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._client.run_post_source_leak_phases(
            structure_id,
            parent_run_id,
            **kwargs,
        )


class InProcessJobBackend:
    """Dispatch jobs in-process inside the Science API container."""

    def __init__(self, db: Any) -> None:
        self._db = db

    async def run_compute_job(
        self,
        job_id: str,
        structure_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from science.compute.dispatch_helpers import dispatch_job_in_process

        job_params = kwargs.pop("job_params", None)
        result = await dispatch_job_in_process(
            self._db,
            job_id,
            structure_id=structure_id,
            pathway=kwargs.get("pathway", DEFAULT_PATHWAY),
            computation_run_id=kwargs.get("computation_run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            pipeline_job_id=kwargs.get("pipeline_job_id"),
            device=kwargs.get("device", "cpu"),
            checkpoint_path=kwargs.get("checkpoint_path"),
            job_params=job_params,
        )
        return result.to_dict()

    async def run_post_source_leak_phases(
        self,
        structure_id: str,
        parent_run_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from science.compute.jobs.post_source_leak import run_post_source_leak_tail

        phases = await run_post_source_leak_tail(
            self._db,
            structure_id=structure_id,
            parent_run_id=parent_run_id,
            gnn_run_id=kwargs.get("gnn_run_id"),
            identify_allosteric_sites=kwargs.get("identify_allosteric_sites", False),
            run_buffering_atlas=kwargs.get("run_buffering_atlas", True),
        )
        await self._db.commit()
        return {
            "run_id": parent_run_id,
            "structure_id": structure_id,
            "phases_run": [name for name, pr in phases.items() if pr.success],
        }


async def execute_pathway(
    structure_id: str,
    backend: JobBackend,
    *,
    pathway_id: str = DEFAULT_PATHWAY,
    source_leak_only: bool = False,
    computation_run_id: str | None = None,
    pipeline_job_id: str | None = None,
    on_job: JobCallback | None = None,
) -> dict[str, Any]:
    """Execute a discovery pathway using the provided job backend."""
    structure_id = structure_id.strip().lower()
    effective_pathway = "viewport_explore" if source_leak_only else pathway_id
    ordered_jobs = jobs_for_pathway(pathway_id, source_leak_only=source_leak_only)
    pathway_set = set(ordered_jobs)

    from shared.audit.instrumentation import audit_pathway_started

    await audit_pathway_started(
        None,
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id or computation_run_id or structure_id,
        pathway_id=effective_pathway,
    )

    peeled_in_pathway = [j for j in ordered_jobs if j in PEELED_JOBS]
    monolith_jobs = [
        j for j in ordered_jobs if j not in EXTERNAL_JOBS and j not in PEELED_JOBS
    ]

    completed_jobs: set[str] = set()
    pathway_result: dict[str, Any] = {}
    phases_run: list[str] = []
    gnn_embedding_run_id: str | None = None

    async def _notify(job_id: str, status: str, metadata: dict[str, Any] | None = None) -> None:
        if on_job:
            await on_job(job_id, status, metadata)

    for job_id in monolith_jobs:
        if job_id not in pathway_set:
            continue
        await _notify(
            job_id,
            "skipped",
            {"computation_run_id": computation_run_id, "pathway": effective_pathway},
        )

    for job_id in peeled_in_pathway:
        if job_id not in pathway_set:
            continue
        await _notify(
            job_id,
            "running",
            {"computation_run_id": computation_run_id, "pathway": effective_pathway},
        )
        job_kwargs: dict[str, Any] = {
            "pathway": effective_pathway,
            "computation_run_id": computation_run_id,
            "pipeline_job_id": pipeline_job_id,
        }
        if gnn_embedding_run_id and job_id != "gnn_inference":
            job_kwargs["parent_run_id"] = gnn_embedding_run_id

        try:
            job_result = await backend.run_compute_job(
                job_id,
                structure_id=structure_id,
                **job_kwargs,
            )
        except Exception as exc:
            await _notify(
                job_id,
                "failed",
                {
                    "computation_run_id": computation_run_id,
                    "error": _job_exception_message(exc),
                },
            )
            if JOB_REGISTRY[job_id].tier == 1:
                raise
            logger.warning("Job %s failed (tier %s): %s", job_id, JOB_REGISTRY[job_id].tier, exc)
            continue

        if job_result.get("success"):
            completed_jobs.add(job_id)
            if job_id == "gnn_inference":
                gnn_embedding_run_id = job_result.get("outputs", {}).get(
                    "embedding_run_id"
                ) or job_result.get("outputs", {}).get("gnn_run_id")
            await _notify(
                job_id,
                "complete",
                {
                    "computation_run_id": computation_run_id,
                    "pathway": effective_pathway,
                    "run_id": job_result.get("run_id"),
                },
            )
        else:
            await _notify(
                job_id,
                "failed",
                {
                    "computation_run_id": computation_run_id,
                    "error": job_result.get("outputs", {}).get("error"),
                },
            )
            if JOB_REGISTRY[job_id].tier == 1:
                raise RuntimeError(
                    job_result.get("outputs", {}).get("error", f"Job {job_id} failed")
                )

    parent_run_id = (
        pathway_result.get("run_id")
        or computation_run_id
        or gnn_embedding_run_id
    )
    if (
        "source_leak_detection" in peeled_in_pathway
        and "source_leak_detection" in completed_jobs
        and not source_leak_only
        and parent_run_id
        and "allosteric_site_detection" not in completed_jobs
    ):
        try:
            tail = await backend.run_post_source_leak_phases(
                structure_id=structure_id,
                parent_run_id=parent_run_id,
                gnn_run_id=gnn_embedding_run_id,
                identify_allosteric_sites=False,
                run_buffering_atlas=True,
            )
            tail_phases = tail.get("phases_run", [])
            phases_run = list(dict.fromkeys(phases_run + tail_phases))
            completed_jobs |= map_phases_to_jobs(tail_phases)
        except Exception as exc:
            logger.warning("Post-source-leak tail failed for %s: %s", structure_id, exc)

    logger.info(
        "Pathway complete structure=%s pathway=%s jobs_complete=%s",
        structure_id,
        effective_pathway,
        sorted(completed_jobs),
    )
    from shared.audit.instrumentation import audit_pathway_complete

    await audit_pathway_complete(
        None,
        structure_id=structure_id,
        pipeline_job_id=pipeline_job_id,
        jobs_complete=sorted(completed_jobs),
        pathway_id=effective_pathway,
        run_id=parent_run_id or computation_run_id,
    )
    return {
        **pathway_result,
        "run_id": parent_run_id or computation_run_id,
        "pathway": effective_pathway,
        "jobs_complete": sorted(completed_jobs),
        "jobs_scheduled": ordered_jobs,
        "phases_run": phases_run,
    }
