"""Compare API router — REST endpoints for structure comparison.

Exposes direct GET endpoints for comparing two protein structures:
- Embedding displacement comparison
- Graph topology comparison

Wraps existing tool functions for direct frontend consumption
without going through the agent chat interface.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from agent.coordinator.deps import get_db
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/compare", tags=["compare"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _compute_displacements(id_a: str, id_b: str, db: Any) -> list[dict] | None:
    """Compute per-residue embedding displacements between two structures.

    Returns ALL matched residues (no truncation), sorted by magnitude descending.
    Returns None if either structure has no embeddings.
    """
    from agent.tools.dtie.tools import ToolDB

    tool_db = ToolDB(db)

    query = """
        SELECT e.residue_id, e.cone_depth,
               r.residue_index, c.chain_label
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
        ORDER BY e.computed_at DESC
    """

    rows_a = await tool_db.fetch_all(query, {"structure_id": id_a})
    rows_b = await tool_db.fetch_all(query, {"structure_id": id_b})

    if not rows_a or not rows_b:
        return None

    # Index by canonical position (chain_label, residue_index)
    by_pos_a = {(r["chain_label"], r["residue_index"]): r for r in rows_a}
    by_pos_b = {(r["chain_label"], r["residue_index"]): r for r in rows_b}

    displacements = []
    for pos, ra in by_pos_a.items():
        if pos in by_pos_b:
            rb = by_pos_b[pos]
            depth_a = ra["cone_depth"]
            depth_b = rb["cone_depth"]
            depth_change = abs(depth_b - depth_a) if depth_a is not None and depth_b is not None else None
            displacements.append({
                "residue_index": pos[1],
                "chain_label": pos[0],
                "wt_residue_id": ra["residue_id"],
                "mut_residue_id": rb["residue_id"],
                "wt_depth": depth_a,
                "mut_depth": depth_b,
                "depth_change": depth_change,
            })

    # Sort by displacement magnitude descending
    displacements.sort(key=lambda d: d.get("depth_change") or 0, reverse=True)
    return displacements


# ---------------------------------------------------------------------------
# GET /api/compare/embeddings/{id_a}/{id_b}
# ---------------------------------------------------------------------------


@router.get("/embeddings/{id_a}/{id_b}")
async def compare_embeddings(
    id_a: str,
    id_b: str,
    threshold: float = Query(default=0.5, description="Displacement threshold for summary count"),
    db=Depends(get_db),
):
    """Compare hyperbolic embeddings between two structures.

    Returns per-residue displacements sorted by magnitude (descending),
    plus summary statistics (mean, max, count above threshold).

    Unlike the agent tool (which truncates to top 20), this endpoint
    returns ALL matched residues for full frontend table display.
    """
    displacements = await _compute_displacements(id_a, id_b, db)

    if displacements is None:
        return JSONResponse(
            status_code=404,
            content={"error": "no_embedding_data", "message": "No embeddings found for one or both structures"},
        )

    # Build rows with required fields
    rows = []
    for d in displacements:
        rows.append({
            "residue_id": d["wt_residue_id"],
            "chain": d["chain_label"],
            "index": d["residue_index"],
            "primary_depth": d["wt_depth"],
            "secondary_depth": d["mut_depth"],
            "depth_delta": d["depth_change"],
            "displacement": d["depth_change"] if d["depth_change"] is not None else 0.0,
        })

    # Sort by displacement magnitude descending
    rows.sort(key=lambda r: r["displacement"], reverse=True)

    # Compute summary statistics
    magnitudes = [r["displacement"] for r in rows]
    mean_displacement = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
    max_displacement = max(magnitudes) if magnitudes else 0.0
    movers_above_threshold = sum(1 for m in magnitudes if m > threshold)

    return {
        "structure_a": id_a,
        "structure_b": id_b,
        "displacements": rows,
        "total_matched": len(rows),
        "summary": {
            "mean_displacement": mean_displacement,
            "max_displacement": max_displacement,
            "movers_above_threshold": movers_above_threshold,
            "threshold": threshold,
        },
    }


# ---------------------------------------------------------------------------
# GET /api/compare/graphs/{id_a}/{id_b}
# ---------------------------------------------------------------------------


@router.get("/graphs/{id_a}/{id_b}")
async def compare_graphs_endpoint(
    id_a: str,
    id_b: str,
    run_id_a: str | None = Query(default=None, description="Specific run for structure A"),
    run_id_b: str | None = Query(default=None, description="Specific run for structure B"),
    db=Depends(get_db),
):
    """Compare graph topology between two structures.

    Returns edge diff counts (gained/lost/changed), H-bond gain/loss
    breakdown, and per-residue metric deltas.
    """
    from agent.tools.graph_tools import compare_graphs

    result = await compare_graphs(
        structure_id_a=id_a,
        structure_id_b=id_b,
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        db=db,
    )

    if not result.success:
        return JSONResponse(
            status_code=404,
            content={"error": "no_graph_data", "message": result.message},
        )

    edge_diff = result.data.get("edge_diff", {})
    metric_diff = result.data.get("metric_diff", [])

    # Build per-residue metric delta rows with required fields
    metric_deltas = []
    for m in metric_diff:
        metric_deltas.append({
            "residue_id": m.get("position", ""),
            "chain": m.get("position", "").split(":")[0] if ":" in m.get("position", "") else "",
            "betweenness_delta": m.get("betweenness_delta", 0.0),
            "degree_delta": m.get("degree_delta", 0.0),
            "clustering_delta": m.get("clustering_coefficient_delta", 0.0),
        })

    return {
        "structure_a": id_a,
        "structure_b": id_b,
        "edge_diff": {
            "gained_count": edge_diff.get("gained_count", 0),
            "lost_count": edge_diff.get("lost_count", 0),
            "changed_count": edge_diff.get("changed_count", 0),
            "gained": edge_diff.get("gained", []),
            "lost": edge_diff.get("lost", []),
            "changed": edge_diff.get("changed", []),
        },
        "hbond": {
            "gained": result.data.get("hbond_gained", 0),
            "lost": result.data.get("hbond_lost", 0),
        },
        "metric_deltas": metric_deltas,
        "total_matched_residues": result.data.get("total_matched_residues", 0),
    }
