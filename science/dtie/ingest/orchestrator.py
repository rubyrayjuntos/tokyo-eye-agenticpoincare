"""Onboard compute orchestration after structure ingest.

Runs the structure-scoped compute pathway via the job scheduler.
See: docs/specs/ingest-compute-contract/requirements.md
      docs/specs/discovery-story-pathway/design.md
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from science.compute.registry import DEFAULT_PATHWAY, JOB_REGISTRY, topological_order

logger = logging.getLogger(__name__)

# Registry job IDs executed during full onboard (discovery_story pathway).
ONBOARD_STAGE_IDS: tuple[str, ...] = tuple(topological_order())

StageCallback = Callable[[str, str, dict[str, Any] | None], Awaitable[None]]


async def run_onboard_compute(
    structure_id: str,
    *,
    job_id: str | None = None,
    pipeline_job_id: str | None = None,
    source_leak_only: bool = False,
    pathway_id: str = DEFAULT_PATHWAY,
    on_stage: StageCallback | None = None,
) -> dict[str, Any]:
    """Execute the onboard compute pathway for a structure."""
    from science.compute.scheduler import run_pathway

    effective_pipeline_job_id = pipeline_job_id or job_id

    async def _on_job(
        registry_job_id: str,
        status: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        if on_stage:
            await on_stage(registry_job_id, status, metadata)

    return await run_pathway(
        structure_id,
        pathway_id=pathway_id,
        source_leak_only=source_leak_only,
        computation_run_id=job_id,
        pipeline_job_id=effective_pipeline_job_id,
        on_job=_on_job if on_stage else None,
    )


def job_module_entry(
    job_id: str,
    status: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pipeline_job.modules JSON entry for a registry job."""
    job = JOB_REGISTRY.get(job_id)
    entry: dict[str, Any] = {
        "job_id": job_id,
        "stage_id": job_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if job is not None:
        entry["discovery_act"] = job.discovery_act
        entry["priority_group"] = job.priority_group
        entry["resource_class"] = job.resource_class
        entry["artifacts_produced"] = sorted(job.produces)
    if metadata:
        entry.update(metadata)
    return entry


def stage_module_entry(
    stage_id: str,
    status: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper — stage_id is treated as job_id when known."""
    if stage_id in JOB_REGISTRY:
        return job_module_entry(stage_id, status, metadata=metadata)
    entry: dict[str, Any] = {
        "stage_id": stage_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry.update(metadata)
    return entry
