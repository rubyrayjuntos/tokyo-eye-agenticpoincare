"""Structure readiness and metadata API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from agent.coordinator.auth import get_current_user
from agent.coordinator.deps import get_db
from data.readiness import assess_structure_readiness
from shared.gnn_viewer_paths import resolve_disc_html_path, resolve_viewer_html_path, viewer_output_dir

router = APIRouter(prefix="/api/structures", tags=["structures"])


@router.get("/{structure_id}/readiness")
async def get_structure_readiness(
    structure_id: str,
    db: Any = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Return ingest–compute readiness checklist for a structure.

    Probes tier-1 and tier-2 artifacts, derives global status, and returns
    Discovery Story act-scoped progress (pathway, current_act, acts).
    """
    readiness = await assess_structure_readiness(structure_id, db)
    if readiness.readiness_status == "failed" and any(
        "Structure not found in dim_structure" in reason
        for reason in readiness.degraded_reasons
    ):
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")
    payload = readiness.to_dict()
    viewer_path = resolve_viewer_html_path(structure_id)
    if viewer_path is not None:
        payload["gnn_viewer_url"] = f"/api/structures/{structure_id.strip().lower()}/gnn-viewer"
    disc_path = resolve_disc_html_path(structure_id)
    if disc_path is not None:
        payload["gnn_disc_viewer_url"] = (
            f"/api/structures/{structure_id.strip().lower()}/gnn-viewer/disc"
        )
    return payload


@router.get("/{structure_id}/gnn-viewer")
async def serve_gnn_interactive_viewer(
    structure_id: str,
    _user: dict = Depends(get_current_user),
) -> FileResponse:
    """Serve the latest GNN interactive NGL HTML for a structure."""
    file_path = resolve_viewer_html_path(structure_id)
    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No GNN interactive viewer for structure '{structure_id}'. "
            "Run ingest / gnn_inference first.",
        )

    try:
        file_path.resolve().relative_to(viewer_output_dir().resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc

    return FileResponse(
        path=str(file_path),
        media_type="text/html",
        filename=file_path.name,
    )


@router.get("/{structure_id}/gnn-viewer/disc")
async def serve_gnn_disc_viewer(
    structure_id: str,
    _user: dict = Depends(get_current_user),
) -> FileResponse:
    """Serve the Poincaré disc interactive HTML for a structure."""
    file_path = resolve_disc_html_path(structure_id)
    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Poincaré disc viewer for structure '{structure_id}'. "
            "Run ingest / gnn_inference or export_corpus_viewers first.",
        )

    try:
        file_path.resolve().relative_to(viewer_output_dir().resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc

    return FileResponse(
        path=str(file_path),
        media_type="text/html",
        filename=file_path.name,
    )
