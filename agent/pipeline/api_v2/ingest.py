"""
POST /ingest — fetch structure from RCSB, normalize to dim tables, enqueue jobs.

Contract (per Task 18):
    - Body: {"pdb_id": str}
    - PDB ID validated: non-empty, 4-character alphanumeric
    - Calls fetch_and_normalize_structure() → {"structure_id": str, "structure": ...}
    - Calls enqueue_all_jobs(structure_id, structure) → {"fast_path": str, ...}
    - Returns {"structure_id": str, "pdb_id": str, "job_ids": dict}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, field_validator

router = APIRouter(tags=["ingest"])

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

_PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")


class IngestRequest(BaseModel):
    pdb_id: str

    @field_validator("pdb_id")
    @classmethod
    def _validate_pdb_id(cls, v: str) -> str:
        if not v or not _PDB_ID_RE.match(v):
            raise ValueError(
                "pdb_id must be a non-empty 4-character alphanumeric string"
            )
        return v.upper()


class IngestResponse(BaseModel):
    structure_id: str
    pdb_id: str
    job_ids: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers (extracted so tests can patch them at the module level)
# ---------------------------------------------------------------------------

async def fetch_and_normalize_structure(pdb_id: str) -> Dict[str, Any]:
    """Fetch BinaryCIF from RCSB and write dim tables.  Returns structure_id."""
    from gosp.services.ingestion_broker import ingest_structure_broker
    from gosp.normalizer.structure import normalize_structure
    from gosp.db.connection import get_engine
    from sqlalchemy.ext.asyncio import AsyncSession

    loop = asyncio.get_running_loop()
    def _fetch_with_format(fmt: str):
        return ingest_structure_broker(
            input_type="pdb",
            input_value=pdb_id,
            fmt=fmt,
        )

    # Prefer BinaryCIF for speed/size, but fall back to text CIF if decoding fails.
    try:
        broker_result = await loop.run_in_executor(
            None,
            lambda: _fetch_with_format("bcif"),
        )
    except Exception as exc:
        logging.getLogger("gosp.ingest").warning(
            "BinaryCIF ingest failed for %s (%s). Retrying with CIF.",
            pdb_id,
            exc,
        )
        broker_result = await loop.run_in_executor(
            None,
            lambda: _fetch_with_format("cif"),
        )
    structure = broker_result.structure

    structure_id = str(uuid.uuid4())
    engine = get_engine()
    async with AsyncSession(engine) as session:
        async with session.begin():
            structure_id = await normalize_structure(
                conn=session,
                structure_id=structure_id,
                structure=structure,
            )

    return {"structure_id": structure_id, "structure": structure}


async def enqueue_all_jobs(structure_id: str, structure) -> Dict[str, str]:
    """Enqueue Tier 1 (fast_path) + Tier 2 jobs.  Returns task-name map."""
    project = os.environ.get("GCP_PROJECT_ID", "")
    location = os.environ.get("CLOUD_TASKS_LOCATION", "us-central1")
    queue = os.environ.get("CLOUD_TASKS_QUEUE", "tier2-jobs")
    tasks_url = os.environ.get("CLOUD_TASKS_HANDLER_URL", "")
    if tasks_url.endswith("/tasks/run"):
        tasks_url = tasks_url[: -len("/tasks/run")]

    # validation_mining is NOT enqueued here — it fires from Tier 1 after
    # gnn-ready is published, ensuring dehydrons are already in the DB.
    job_types = ["void", "cdd", "immunogenicity", "metabolism", "lerp", "energy"]

    log = logging.getLogger("gosp.ingest")
    from gosp.db.connection import get_engine
    from gosp.jobs.tier1 import run_tier1
    from sqlalchemy.ext.asyncio import AsyncSession

    results: Dict[str, str] = {}
    engine = get_engine()

    # Tier 1 must complete before any Tier 2 work can safely begin.
    async with AsyncSession(engine) as session:
        async with session.begin():
            try:
                tier1_result = await run_tier1(session, structure_id, structure)
                results["fast_path"] = "complete"
                log.info("Tier 1 complete: %s", tier1_result)
            except Exception as exc:
                results["fast_path"] = f"failed: {exc}"
                log.error("Tier 1 failed: %s", exc)
                return results

    if not project or not tasks_url:
        # Local dev: run Tier 1 inline (SASA, dehydrons, GNN payload),
        # then run Tier 2 jobs in background tasks.
        log.info("Local dev mode — running jobs inline for %s", structure_id)

        from gosp.jobs.tier2 import (
            run_void_job, run_cdd_job, run_immunogenicity_job,
            run_metabolism_job, run_lerp_job, run_energy_job,
        )

        # GNN inference — call the host inference server (port 8090)
        # The inference server runs on the host with torch/e3nn/geoopt.
        import requests as http_requests
        gnn_server = os.environ.get("GNN_INFERENCE_URL", "http://host.docker.internal:8090")
        try:
            resp = http_requests.post(
                f"{gnn_server}/infer",
                json={"structure_id": structure_id},
                timeout=120,
            )
            if resp.status_code == 200:
                results["gnn_inference"] = "complete"
                log.info("GNN inference complete: %s", resp.json())
            else:
                results["gnn_inference"] = f"failed: {resp.status_code} {resp.text[:200]}"
                log.warning("GNN inference failed: %s", resp.text[:200])
        except Exception as exc:
            results["gnn_inference"] = f"failed: {exc}"
            log.warning("GNN inference server unreachable: %s", exc)

        # Tier 2 jobs — run sequentially in local dev
        async with AsyncSession(engine) as session:
            async with session.begin():
                for job_name, job_fn in [
                    ("void", run_void_job),
                    ("cdd", run_cdd_job),
                    ("immunogenicity", run_immunogenicity_job),
                    ("metabolism", run_metabolism_job),
                    ("lerp", run_lerp_job),
                    ("energy", run_energy_job),
                ]:
                    try:
                        await job_fn(session, structure_id, structure)
                        results[job_name] = "complete"
                    except Exception as exc:
                        results[job_name] = f"failed: {exc}"
                        log.warning("Tier 2 job %s failed: %s", job_name, exc)

        return results

    try:
        from google.cloud import tasks_v2  # type: ignore[import]

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(project, location, queue)
        sa_email = os.environ.get(
            "CLOUD_TASKS_SA_EMAIL",
            "578264867059-compute@developer.gserviceaccount.com",
        )
        task_names: Dict[str, str] = {}
        for job_type in job_types:
            http_request = tasks_v2.HttpRequest(
                http_method=tasks_v2.HttpMethod.POST,
                url=f"{tasks_url}/tasks/run",
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {"job_type": job_type, "structure_id": structure_id}
                ).encode("utf-8"),
                oidc_token=tasks_v2.OidcToken(
                    service_account_email=sa_email,
                    audience=tasks_url,
                ),
            )
            task = tasks_v2.Task(http_request=http_request)
            response = client.create_task(request={"parent": parent, "task": task})
            task_names[job_type] = response.name
        return {**results, **task_names}
    except Exception as exc:
        logging.getLogger("gosp.ingest").error(
            "Cloud Tasks enqueue failed for %s: %s", structure_id, exc
        )
        return {**results, **{jt: f"failed-enqueue-{jt}" for jt in job_types}}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def ingest_structure(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Ingest a structure by PDB ID.

    1. Validate PDB ID (4-char alphanumeric).
    2. Fetch BinaryCIF from RCSB and normalize to dim tables.
    3. Enqueue Tier 1 + Tier 2 jobs via Cloud Tasks (local stub if unconfigured).
    4. Return structure_id immediately.
    """
    try:
        result = await fetch_and_normalize_structure(request.pdb_id)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logging.getLogger("gosp.ingest").error("Ingest failed for %s: %s\n%s", request.pdb_id, exc, tb)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to fetch or normalize {request.pdb_id}: {type(exc).__name__}: {exc}",
        )

    structure_id = result["structure_id"]
    structure = result["structure"]

    job_ids = await enqueue_all_jobs(structure_id, structure)

    return IngestResponse(
        structure_id=structure_id,
        pdb_id=request.pdb_id,
        job_ids=job_ids,
    )
