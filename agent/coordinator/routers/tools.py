"""Tool endpoints — direct API access to DTIE tools."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from agent.coordinator.auth import get_current_user
from agent.tools.dtie.tools import (
    ToolResult,
    compare_wt_mutant,
    get_high_uncertainty_residues,
    get_residue_state,
    get_source_leaks,
    run_gnn_inference,
    run_phase,
)

router = APIRouter(prefix="/api/tools", tags=["tools"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class GNNInferenceRequest(BaseModel):
    structure_id: str
    model_version: str = "v5"
    checkpoint_path: str | None = None


class PhaseRequest(BaseModel):
    structure_id: str
    phase: str
    model_version: str = "v5"
    parameters: dict[str, Any] | None = None


class PipelineRequest(BaseModel):
    structure_id: str
    checkpoint_path: str | None = None


class SourceLeakRequest(BaseModel):
    structure_id: str
    uncertainty_threshold: float = 0.3
    min_depth: float = 1.5


class UncertaintyRequest(BaseModel):
    structure_id: str
    top_n: int = 20
    uncertainty_type: str = "epistemic"


class ResidueStateRequest(BaseModel):
    structure_id: str
    residue_ids: list[str] | None = None


class CompareRequest(BaseModel):
    wt_structure_id: str
    mutant_structure_id: str
    focus_residues: list[str] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run-pipeline")
async def api_run_pipeline(req: PipelineRequest, user: dict = Depends(get_current_user)):
    """Run the full DTIE v5 pipeline on a structure via Science Container API."""
    from agent.tools.science_client import ScienceClient, ScienceComputeError, ScienceTimeoutError

    client = ScienceClient()
    try:
        result = await client.run_pipeline(
            structure_id=req.structure_id,
            checkpoint_path=req.checkpoint_path,
        )
        return {
            "success": True,
            "run_id": result.get("run_id"),
            "phases_run": result.get("phases_run", []),
            "assets_created": result.get("assets_created", 0),
            "duration_ms": result.get("duration_ms"),
            "warnings": result.get("warnings", []),
        }
    except ScienceTimeoutError as e:
        return {"success": False, "error": str(e)}
    except ScienceComputeError as e:
        return {"success": False, "error": e.detail, "status": e.status}


@router.post("/gnn-inference")
async def api_run_gnn(req: GNNInferenceRequest, user: dict = Depends(get_current_user)):
    """Run GNN inference only."""
    from data.db import DBAdapter, get_connection

    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await run_gnn_inference(
            structure_id=req.structure_id,
            model_version=req.model_version,
            checkpoint_path=req.checkpoint_path,
            db=db,
        )
    return _result_to_dict(result)


@router.post("/source-leaks")
async def api_source_leaks(req: SourceLeakRequest, user: dict = Depends(get_current_user)):
    """Identify source-leak candidates."""
    from data.db import DBAdapter, get_connection

    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await get_source_leaks(
            structure_id=req.structure_id,
            uncertainty_threshold=req.uncertainty_threshold,
            min_depth=req.min_depth,
            db=db,
        )
    return _result_to_dict(result)


@router.post("/high-uncertainty")
async def api_high_uncertainty(req: UncertaintyRequest, user: dict = Depends(get_current_user)):
    """Get highest-uncertainty residues."""
    from data.db import DBAdapter, get_connection

    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await get_high_uncertainty_residues(
            structure_id=req.structure_id,
            top_n=req.top_n,
            uncertainty_type=req.uncertainty_type,
            db=db,
        )
    return _result_to_dict(result)


@router.post("/residue-state")
async def api_residue_state(req: ResidueStateRequest, user: dict = Depends(get_current_user)):
    """Get governed state of residues."""
    from data.db import DBAdapter, get_connection

    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await get_residue_state(
            structure_id=req.structure_id,
            residue_ids=req.residue_ids,
            db=db,
        )
    return _result_to_dict(result)


@router.post("/compare")
async def api_compare(req: CompareRequest, user: dict = Depends(get_current_user)):
    """Compare WT and mutant embeddings."""
    from data.db import DBAdapter, get_connection

    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await compare_wt_mutant(
            wt_structure_id=req.wt_structure_id,
            mutant_structure_id=req.mutant_structure_id,
            focus_residues=req.focus_residues,
            db=db,
        )
    return _result_to_dict(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_dict(result: ToolResult) -> dict:
    return {
        "success": result.success,
        "data": result.data,
        "message": result.message,
        "viewport_directives": [d.model_dump(mode="json") for d in result.viewport_directives],
        "warnings": result.warnings,
    }
