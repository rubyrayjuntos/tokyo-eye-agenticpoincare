"""Tool endpoints — direct API access to discovery signal tools."""

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
    run_phase,
)

router = APIRouter(prefix="/api/tools", tags=["tools"])


class PhaseRequest(BaseModel):
    structure_id: str
    phase: str
    model_version: str = "v5"
    parameters: dict[str, Any] | None = None


class SourceLeakRequest(BaseModel):
    structure_id: str
    uncertainty_threshold: float = 0.3
    min_depth: float = 1.5
    ranking: str = "physics_rim"
    top_n: int = 50


class UncertaintyRequest(BaseModel):
    structure_id: str
    top_n: int = 20
    uncertainty_type: str = "cone_depth"


class ResidueStateRequest(BaseModel):
    structure_id: str
    residue_ids: list[str] | None = None


class CompareRequest(BaseModel):
    wt_structure_id: str
    mutant_structure_id: str
    focus_residues: list[str] | None = None


@router.post("/source-leaks")
async def api_source_leaks(req: SourceLeakRequest, user: dict = Depends(get_current_user)):
    """Identify source-leak candidates (physics-rim default)."""
    from data.db import DBAdapter, get_connection

    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await get_source_leaks(
            structure_id=req.structure_id,
            uncertainty_threshold=req.uncertainty_threshold,
            min_depth=req.min_depth,
            ranking=req.ranking,
            top_n=req.top_n,
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


def _result_to_dict(result: ToolResult) -> dict:
    return {
        "success": result.success,
        "data": result.data,
        "message": result.message,
        "viewport_directives": [d.model_dump(mode="json") for d in result.viewport_directives],
        "warnings": result.warnings,
    }
