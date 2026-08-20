"""Poincaré data endpoint — serves disc visualization data."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from science.dtie.common.curvature_values import (
    MissingLearnedCurvatureError,
    learned_curvature_from_row,
)

router = APIRouter(prefix="/api", tags=["poincare"])

logger = logging.getLogger(__name__)


@router.get("/poincare-data")
async def get_poincare_data(
    structure_id: str = Query(..., description="Structure ID"),
    condition: str = Query("gdp", description="Condition: gdp or gtp"),
):
    """Return per-residue Poincaré disc data from the governed layer.

    Reads from fact_gnn_node_embedding + dimensions (governed view).
    """
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            rows = await db.fetch_all(
                """
                SELECT r.residue_id, r.residue_index, r.residue_name,
                       c.chain_label,
                       e.hyp_projections, e.hyp_projection_2d,
                       e.cone_depth, e.cone_width, e.epistemic_uncertainty,
                       es.curvature, es.name AS space_name, e.model_version
                FROM fact_gnn_node_embedding e
                JOIN dim_residue r ON r.residue_id = e.residue_id
                JOIN dim_chain c ON c.chain_id = r.chain_id
                JOIN embedding_space es ON es.space_id = e.space_id
                WHERE e.structure_id = :structure_id
                ORDER BY r.residue_index
                """,
                {"structure_id": structure_id},
            )

        if not rows:
            return JSONResponse(
                status_code=404,
                content={"error": f"No GNN results for '{structure_id}'. Run the DTIE pipeline first."},
            )

        residues = []
        for r in rows:
            x, y = 0.0, 0.0
            if r.get("hyp_projection_2d"):
                coords = r["hyp_projection_2d"]
                # pgvector returns a string like "[0.1,-0.2]" — parse it
                if isinstance(coords, str):
                    coords = [float(v) for v in coords.strip("[]").split(",")]
                x, y = float(coords[0]), float(coords[1])
            elif r.get("hyp_projections"):
                proj = r["hyp_projections"]
                if isinstance(proj, list) and len(proj) >= 2:
                    x, y = float(proj[0]), float(proj[1])

            residues.append({
                "residue_id": r["residue_id"],
                "residue_index": r["residue_index"],
                "residue_name": r["residue_name"],
                "chain_label": r["chain_label"],
                "x": x,
                "y": y,
                "cone_depth": r["cone_depth"],
                "cone_width": r.get("cone_width"),
                "epistemic_uncertainty": r.get("epistemic_uncertainty"),
            })

        try:
            curvature_c = learned_curvature_from_row(rows[0], context="poincare-data")
        except MissingLearnedCurvatureError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})

        return {
            "structure_id": structure_id,
            "condition": condition,
            "curvature_c": curvature_c,
            "model_version": rows[0].get("model_version") or "TokyoEye-v7",
            "residues": residues,
        }

    except Exception as e:
        logger.exception("poincare-data failed for %s", structure_id)
        return JSONResponse(status_code=503, content={"error": "Internal server error"})
