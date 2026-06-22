"""Plots API router — REST endpoints for matplotlib figure generation.

Exposes 2 endpoints:
- POST /api/plots/generate — wraps generate_plot, returns file path + download URL
- GET /api/plots/{filename} — serves generated PNG from PLOT_OUTPUT_DIR

Wraps the existing tool functions in agent/tools/plotting/tools.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.coordinator.deps import get_db
from agent.tools.plotting.tools import PLOT_OUTPUT_DIR, PLOT_TYPES
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/plots", tags=["plots"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class PlotGenerateRequest(BaseModel):
    structure_id: str = Field(..., description="Structure to generate plot for")
    plot_type: str = Field(..., description="Plot type (e.g., poincare_disc, uncertainty_profile)")
    parameters: dict[str, Any] | None = Field(None, description="Plot-specific parameters")


# ---------------------------------------------------------------------------
# POST /api/plots/generate
# ---------------------------------------------------------------------------


@router.post("/generate")
async def generate_plot_endpoint(request: PlotGenerateRequest, db=Depends(get_db)):
    """Generate a matplotlib figure from pipeline data.

    Renders the requested plot type server-side and returns the PNG
    file path and a download URL.
    """
    from agent.tools.plotting.tools import generate_plot

    if request.plot_type not in PLOT_TYPES:
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_plot_type",
                "message": f"Unknown plot type '{request.plot_type}'. Valid types: {', '.join(sorted(PLOT_TYPES))}",
            },
        )

    result = await generate_plot(
        structure_id=request.structure_id,
        plot_type=request.plot_type,
        parameters=request.parameters,
        db=db,
    )

    if not result.success:
        return JSONResponse(
            status_code=404,
            content={"error": "missing_data", "message": result.message},
        )

    filename = result.file_path.split("/")[-1] if result.file_path else None
    download_url = f"/api/plots/{filename}" if filename else None

    return {
        "success": True,
        "file_path": result.file_path,
        "plot_type": result.plot_type,
        "message": result.message,
        "download_url": download_url,
    }


# ---------------------------------------------------------------------------
# GET /api/plots/{filename}
# ---------------------------------------------------------------------------


@router.get("/{filename}")
async def serve_plot(filename: str):
    """Serve a generated plot PNG file.

    Returns the image file from PLOT_OUTPUT_DIR for display or download.
    """
    file_path = PLOT_OUTPUT_DIR / filename

    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "file_not_found", "message": f"Plot file '{filename}' not found"},
        )

    # Security: ensure the resolved path is within PLOT_OUTPUT_DIR
    try:
        file_path.resolve().relative_to(PLOT_OUTPUT_DIR.resolve())
    except ValueError:
        return JSONResponse(
            status_code=403,
            content={"error": "forbidden", "message": "Access denied"},
        )

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename,
    )
