"""Atomic compute job registry and scheduling (Discovery Story pathway).

Runner dispatch stays lazy: agent readiness and structure APIs only need the
job catalog, not torch / TokyoEye.
"""

from __future__ import annotations

from typing import Any

from science.compute.registry import (
    ACT_JOB_MAP,
    DEFAULT_PATHWAY,
    JOB_REGISTRY,
    PATHWAY_JOBS,
    ComputeJob,
    topological_order,
)

__all__ = [
    "ACT_JOB_MAP",
    "DEFAULT_PATHWAY",
    "JOB_REGISTRY",
    "PATHWAY_JOBS",
    "PEELED_JOBS",
    "ComputeJob",
    "dispatch_compute_job",
    "map_phases_to_jobs",
    "run_pathway",
    "topological_order",
]


def __getattr__(name: str) -> Any:
    if name in {"PEELED_JOBS", "dispatch_compute_job"}:
        from science.compute.runner_dispatch import PEELED_JOBS, dispatch_compute_job

        return PEELED_JOBS if name == "PEELED_JOBS" else dispatch_compute_job
    if name in {"map_phases_to_jobs", "run_pathway"}:
        from science.compute.scheduler import map_phases_to_jobs, run_pathway

        return map_phases_to_jobs if name == "map_phases_to_jobs" else run_pathway
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
