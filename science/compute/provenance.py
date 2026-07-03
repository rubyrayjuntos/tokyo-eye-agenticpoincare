"""Provenance helpers for Discovery Story compute jobs."""

from __future__ import annotations

from typing import Any

from science.compute.registry import DEFAULT_PATHWAY, JOB_REGISTRY

# Historical pipeline name retained only for reading legacy provenance rows.
LEGACY_PIPELINE_NAME = "dtie_v5"


def pathway_pipeline_name(pathway: str = DEFAULT_PATHWAY) -> str:
    """Canonical pipeline_name for new onboard compute runs."""
    return pathway


def job_provenance_parameters(
    job_id: str | None,
    *,
    pathway: str = DEFAULT_PATHWAY,
    computation_run_id: str | None = None,
    phases_run: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provenance_run.parameters / provenance_event.metadata payload."""
    job = JOB_REGISTRY.get(job_id) if job_id else None
    params: dict[str, Any] = {"pathway": pathway}
    if job_id:
        params["job_id"] = job_id
    if job is not None:
        params["discovery_act"] = job.discovery_act
        params["priority_group"] = job.priority_group
        params["resource_class"] = job.resource_class
    if computation_run_id:
        params["computation_run_id"] = computation_run_id
    if phases_run:
        params["phases_run"] = phases_run
    if extra:
        params.update(extra)
    return params


def monolith_orchestrator_parameters(
    *,
    pathway: str = DEFAULT_PATHWAY,
    computation_run_id: str | None = None,
) -> dict[str, Any]:
    """Parameters for the compatibility monolith parent provenance run."""
    params: dict[str, Any] = {
        "pathway": pathway,
        "orchestrator_mode": "monolith_facade",
        "legacy_pipeline_name": LEGACY_PIPELINE_NAME,
    }
    if computation_run_id:
        params["computation_run_id"] = computation_run_id
    return params


async def record_job_provenance_events(
    db: Any,
    *,
    parent_run_id: str,
    structure_id: str,
    jobs_complete: list[str],
    pathway: str = DEFAULT_PATHWAY,
    computation_run_id: str | None = None,
) -> None:
    """Insert provenance_event rows for each completed registry job."""
    from psycopg.types.json import Json

    for job_id in jobs_complete:
        job = JOB_REGISTRY.get(job_id)
        if job is None:
            continue
        metadata = job_provenance_parameters(
            job_id,
            pathway=pathway,
            computation_run_id=computation_run_id,
        )
        metadata["structure_id"] = structure_id
        await db.execute(
            """
            INSERT INTO provenance_event (
                run_id, event_type, description, metadata
            )
            VALUES (
                :run_id, :event_type, :description, :metadata
            )
            """,
            {
                "run_id": parent_run_id,
                "event_type": f"compute_job.{job_id}",
                "description": f"Completed compute job {job_id} ({job.discovery_act})",
                "metadata": Json(metadata),
            },
        )
