"""Pathway scheduler — dispatches atomic compute jobs for onboard compute.

See: docs/specs/discovery-story-pathway/design.md
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from science.compute.pathway_executor import (
    HttpJobBackend,
    JobCallback,
    execute_pathway,
    jobs_for_pathway,
    map_phases_to_jobs,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "execute_pathway",
    "jobs_for_pathway",
    "map_phases_to_jobs",
    "run_pathway",
]


async def run_pathway(
    structure_id: str,
    *,
    pathway_id: str = "discovery_story",
    source_leak_only: bool = False,
    computation_run_id: str | None = None,
    pipeline_job_id: str | None = None,
    on_job: JobCallback | None = None,
) -> dict[str, Any]:
    """Execute a discovery pathway for a structure (HTTP backend → Science API)."""
    from agent.tools.science_client import ScienceComputeError, ScienceTimeoutError

    try:
        return await execute_pathway(
            structure_id,
            HttpJobBackend(),
            pathway_id=pathway_id,
            source_leak_only=source_leak_only,
            computation_run_id=computation_run_id,
            pipeline_job_id=pipeline_job_id,
            on_job=on_job,
        )
    except RuntimeError as exc:
        raise ScienceComputeError("/compute/pathway", 500, str(exc)) from exc
