"""Ingest API router — agent-side endpoint for structure ingestion.

Delegates to ScienceClient.ingest_structure() and triggers the computation
pipeline in parallel with the alignment sidecar (which runs inside the
science container as a background task).

Requirements: 9.1, 9.2
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from agent.tools.science_client import (
    ScienceClient,
    ScienceComputeError,
    ScienceTimeoutError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class AgentIngestRequest(BaseModel):
    pdb_id: str
    force_reingest: bool = False
    run_pipeline: bool = True


class AgentIngestResponse(BaseModel):
    structure_id: str
    chain_count: int
    residue_count: int
    atom_count: int
    computation_scope: dict[str, Any]
    alignment_status: str
    pipeline_status: str  # "started", "skipped", "failed"
    already_existed: bool


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=AgentIngestResponse)
async def ingest_structure(
    request: AgentIngestRequest,
    background_tasks: BackgroundTasks,
) -> AgentIngestResponse:
    """Ingest a PDB structure and trigger computation pipeline in parallel.

    Delegates to the science container's /compute/ingest-full endpoint,
    then fires the computation pipeline as a background task (non-blocking).
    The alignment sidecar is already triggered within the science container.

    Args:
        request: PDB ID and options.
        background_tasks: FastAPI background task runner.

    Returns:
        AgentIngestResponse with structure info and pipeline status.
    """
    client = ScienceClient()

    # Call science container for full ingestion
    try:
        ingest_result = await client.ingest_structure(
            pdb_id=request.pdb_id,
            force_reingest=request.force_reingest,
        )
    except ScienceTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Science container timed out during ingestion: {e}",
        )
    except ScienceComputeError as e:
        raise HTTPException(
            status_code=e.status,
            detail=f"Ingestion failed: {e.detail}",
        )

    structure_id = ingest_result["structure_id"]
    already_existed = ingest_result.get("already_existed", False)

    # Trigger computation pipeline in parallel (non-blocking)
    pipeline_status = "skipped"
    if request.run_pipeline and not already_existed:
        background_tasks.add_task(
            _run_computation_pipeline,
            client=client,
            structure_id=structure_id,
        )
        pipeline_status = "started"
    elif already_existed and request.run_pipeline:
        # For existing structures, still allow pipeline re-run
        background_tasks.add_task(
            _run_computation_pipeline,
            client=client,
            structure_id=structure_id,
        )
        pipeline_status = "started"

    return AgentIngestResponse(
        structure_id=structure_id,
        chain_count=ingest_result.get("chain_count", 0),
        residue_count=ingest_result.get("residue_count", 0),
        atom_count=ingest_result.get("atom_count", 0),
        computation_scope=ingest_result.get("computation_scope", {}),
        alignment_status=ingest_result.get("alignment_status", "unknown"),
        pipeline_status=pipeline_status,
        already_existed=already_existed,
    )


# ---------------------------------------------------------------------------
# Background task: computation pipeline
# ---------------------------------------------------------------------------


async def _run_computation_pipeline(client: ScienceClient, structure_id: str) -> None:
    """Run the DTIE computation pipeline as a background task.

    This runs in parallel with the alignment sidecar (which is handled
    inside the science container). Failures are logged but don't propagate.
    """
    try:
        result = await client.run_pipeline(structure_id=structure_id)
        logger.info(
            "Computation pipeline completed for %s: phases=%s, duration=%.0fms",
            structure_id,
            result.get("phases_run", []),
            result.get("duration_ms", 0),
        )
    except (ScienceTimeoutError, ScienceComputeError) as e:
        logger.error(
            "Computation pipeline failed for %s: %s",
            structure_id,
            e,
        )
    except Exception as e:
        logger.error(
            "Unexpected error in computation pipeline for %s: %s",
            structure_id,
            e,
            exc_info=True,
        )
