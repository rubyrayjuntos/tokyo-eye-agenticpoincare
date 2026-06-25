"""Graph Topology API router — REST endpoints for graph analysis tools.

Exposes 5 endpoints for querying molecular contact graph topology:
metrics, bridges, H-bonds, shortest path, and graph comparison.
Wraps the existing tool functions in agent/tools/graph_tools.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.coordinator.deps import get_db
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class ShortestPathRequest(BaseModel):
    source_residue_id: str = Field(..., description="Starting residue ID")
    target_residue_id: str = Field(..., description="Ending residue ID")
    run_id: str | None = Field(None, description="Specific run_id (latest if omitted)")


class GraphCompareRequest(BaseModel):
    structure_id_a: str = Field(..., description="First structure (e.g., wild-type)")
    structure_id_b: str = Field(..., description="Second structure (e.g., mutant)")
    run_id_a: str | None = Field(None, description="Specific run for structure A")
    run_id_b: str | None = Field(None, description="Specific run for structure B")


# ---------------------------------------------------------------------------
# GET /api/graph/{structure_id}/metrics
# ---------------------------------------------------------------------------


@router.get("/{structure_id}/metrics")
async def graph_metrics(structure_id: str, run_id: str | None = None, db=Depends(get_db)):
    """Per-residue graph metrics for a structure.

    Returns degree, betweenness, clustering coefficient, closeness,
    eigenvector centrality, bridge status, and conductance.
    """
    from agent.tools.graph_tools import get_graph_metrics

    result = await get_graph_metrics(structure_id=structure_id, run_id=run_id, db=db)

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "no_graph_data", "message": result.message})

    return {
        "structure_id": result.data.get("structure_id"),
        "metrics": result.data.get("metrics", []),
        "count": result.data.get("count", 0),
    }


# ---------------------------------------------------------------------------
# GET /api/graph/{structure_id}/bridges
# ---------------------------------------------------------------------------


@router.get("/{structure_id}/bridges")
async def graph_bridges(structure_id: str, run_id: str | None = None, db=Depends(get_db)):
    """Bridge/articulation-point residues in the contact graph.

    Bridge residues are communication bottlenecks whose removal
    disconnects the graph.
    """
    from agent.tools.graph_tools import find_graph_bridges

    result = await find_graph_bridges(structure_id=structure_id, run_id=run_id, db=db)

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "no_graph_data", "message": result.message})

    return {
        "structure_id": result.data.get("structure_id"),
        "bridges": result.data.get("bridges", []),
        "count": result.data.get("count", 0),
    }


# ---------------------------------------------------------------------------
# GET /api/graph/{structure_id}/hbonds
# ---------------------------------------------------------------------------


@router.get("/{structure_id}/hbonds")
async def graph_hbonds(structure_id: str, run_id: str | None = None, db=Depends(get_db)):
    """H-bond subgraph for a structure.

    Returns all edges of type 'h_bond'.
    """
    from agent.tools.graph_tools import get_hbond_network

    result = await get_hbond_network(structure_id=structure_id, run_id=run_id, db=db)

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "no_graph_data", "message": result.message})

    return {
        "structure_id": result.data.get("structure_id"),
        "edges": result.data.get("edges", []),
        "count": result.data.get("count", 0),
        "residue_count": result.data.get("residue_count", 0),
    }


# ---------------------------------------------------------------------------
# POST /api/graph/{structure_id}/shortest-path
# ---------------------------------------------------------------------------


@router.post("/{structure_id}/shortest-path")
async def graph_shortest_path(structure_id: str, request: ShortestPathRequest, db=Depends(get_db)):
    """Shortest path between two residues in the contact graph.

    Uses edge weights (distance_angstrom) for path computation.
    Returns disconnected=true if no path exists.
    """
    from agent.tools.graph_tools import get_shortest_paths

    result = await get_shortest_paths(
        structure_id=structure_id,
        source_residue_id=request.source_residue_id,
        target_residue_id=request.target_residue_id,
        run_id=request.run_id,
        db=db,
    )

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "no_graph_data", "message": result.message})

    return {
        "structure_id": result.data.get("structure_id"),
        "source": result.data.get("source"),
        "target": result.data.get("target"),
        "path": result.data.get("path", []),
        "path_length": result.data.get("path_length", 0),
        "total_distance": result.data.get("total_distance", 0.0),
        "disconnected": result.data.get("disconnected", False),
    }


# ---------------------------------------------------------------------------
# POST /api/graph/compare
# ---------------------------------------------------------------------------


@router.post("/compare")
async def graph_compare(request: GraphCompareRequest, db=Depends(get_db)):
    """Compare graph topology between two structures.

    Returns edge diff (gained/lost/changed) and per-residue metric deltas.
    """
    from agent.tools.graph_tools import compare_graphs

    result = await compare_graphs(
        structure_id_a=request.structure_id_a,
        structure_id_b=request.structure_id_b,
        run_id_a=request.run_id_a,
        run_id_b=request.run_id_b,
        db=db,
    )

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "no_graph_data", "message": result.message})

    return {
        "structure_a": result.data.get("structure_a"),
        "structure_b": result.data.get("structure_b"),
        "edge_diff": result.data.get("edge_diff", {}),
        "metric_diff": result.data.get("metric_diff", []),
        "total_matched_residues": result.data.get("total_matched_residues", 0),
    }
