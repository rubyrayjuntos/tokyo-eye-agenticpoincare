"""Skip re-running compute jobs when governed artifacts already exist."""

from __future__ import annotations

from typing import Any

from data.readiness import probe_artifact
from science.compute.registry import JOB_REGISTRY
from science.compute.runners.base import JobRunResult


async def job_artifacts_already_present(
    db: Any,
    job_id: str,
    structure_id: str,
) -> bool:
    """Return True when every artifact in the job's produces set is already present."""
    job = JOB_REGISTRY.get(job_id)
    if job is None or not job.produces:
        return False
    structure_id = structure_id.strip().lower()
    for artifact in sorted(job.produces):
        if not await probe_artifact(db, structure_id, artifact):
            return False
    return True


async def skip_result_for_existing_artifacts(
    db: Any,
    job_id: str,
    structure_id: str,
) -> JobRunResult:
    """Build a successful skip result without re-executing the job runner."""
    structure_id = structure_id.strip().lower()
    job = JOB_REGISTRY[job_id]
    artifacts = sorted(job.produces)
    outputs: dict[str, Any] = {
        "skipped": True,
        "reason": "artifacts already present",
    }

    if job_id == "gnn_inference":
        from science.compute.gnn_loader import load_gnn_inference_result

        gnn_result = await load_gnn_inference_result(db, structure_id)
        if gnn_result is not None:
            run_id = gnn_result.metadata.get("run_id")
            if run_id:
                outputs["embedding_run_id"] = run_id
                outputs["gnn_run_id"] = run_id

    return JobRunResult(
        job_id=job_id,
        run_id="",
        structure_id=structure_id,
        success=True,
        artifacts_produced=artifacts,
        outputs=outputs,
        warnings=["skipped: job outputs already present"],
    )
