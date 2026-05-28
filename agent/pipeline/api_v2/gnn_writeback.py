from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gosp.services.gnn_writeback import write_results_to_db

router = APIRouter(tags=["gnn-writeback"])


class GNNWritebackRequest(BaseModel):
    structure_id: str
    payload: Dict[str, Any]
    model_output: Dict[str, Any]
    source_type: str = "probabilistic"


@router.post("/gnn/writeback", response_model=Dict[str, Any])
async def post_gnn_writeback(request: GNNWritebackRequest) -> Dict[str, Any]:
    try:
        gnn_run_id = await write_results_to_db(
            request.structure_id,
            request.payload,
            request.model_output,
            source_type=request.source_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"gnn_run_id": gnn_run_id, "structure_id": request.structure_id}