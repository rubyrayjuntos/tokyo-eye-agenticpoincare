# Migrated from: new (Phase 4 implementation) on 2026-05-27
"""Visualizer server — serves Poincaré disc data to the React frontend.

Endpoints:
- GET /api/structures — list available structures
- GET /api/structure/{id}/disc-data — get disc coordinates for rendering
- WS /ws/viewport — real-time viewport directive channel

The server reads from the governed data layer (production views)
and pushes agent-generated ViewportDirectives to connected frontends.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Tokyo Eye Visualizer Server",
    description="Serves Poincaré disc data and viewport directives",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DiscNode(BaseModel):
    """A single node for Poincaré disc rendering."""

    residue_id: str
    residue_index: int
    residue_name: str
    chain_label: str
    x: float  # Disc x coordinate (-1 to 1)
    y: float  # Disc y coordinate (-1 to 1)
    cone_depth: float
    cone_width: float
    epistemic_uncertainty: float | None = None
    aleatoric_uncertainty: float | None = None
    site_types: list[str] | None = None
    is_dehydron: bool = False


class DiscData(BaseModel):
    """Complete disc data for one structure."""

    structure_id: str
    pdb_id: str | None
    curvature: float
    space_name: str
    model_version: str
    nodes: list[DiscNode]
    edges: list[tuple[int, int]] | None = None


class StructureSummary(BaseModel):
    """Summary of an available structure."""

    structure_id: str
    pdb_id: str | None
    chain_count: int
    residue_count: int
    has_embeddings: bool
    latest_run: str | None


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/structures", response_model=list[StructureSummary])
async def list_structures() -> list[StructureSummary]:
    """List all structures with available disc data.

    Queries dim_structure joined with governed_asset to find
    structures that have GNN embeddings.
    """
    from data.db import get_connection, DBAdapter

    async with get_connection() as conn:
        db = DBAdapter(conn)
        rows = await db.fetch_all(
            """
            SELECT s.structure_id, s.pdb_id,
                   COUNT(DISTINCT c.chain_id) AS chain_count,
                   COUNT(DISTINCT r.residue_id) AS residue_count,
                   EXISTS(
                       SELECT 1 FROM fact_gnn_node_embedding e
                       WHERE e.structure_id = s.structure_id
                   ) AS has_embeddings,
                   (SELECT p.run_id FROM provenance_run p
                    WHERE p.structure_id = s.structure_id
                    ORDER BY p.started_at DESC LIMIT 1) AS latest_run
            FROM dim_structure s
            LEFT JOIN dim_chain c ON c.structure_id = s.structure_id
            LEFT JOIN dim_residue r ON r.chain_id = c.chain_id
            GROUP BY s.structure_id, s.pdb_id
            ORDER BY s.structure_id
            """,
            {},
        )

    return [
        StructureSummary(
            structure_id=r["structure_id"],
            pdb_id=r["pdb_id"],
            chain_count=r["chain_count"] or 0,
            residue_count=r["residue_count"] or 0,
            has_embeddings=r["has_embeddings"] or False,
            latest_run=r["latest_run"],
        )
        for r in rows
    ]


@app.get("/api/structure/{structure_id}/disc-data", response_model=DiscData)
async def get_disc_data(structure_id: str) -> DiscData:
    """Get Poincaré disc coordinates for a structure.

    Queries v_viz_poincare_disc_data for the latest hyperbolic
    projections and formats them for the React frontend.
    """
    from data.db import get_connection, DBAdapter

    async with get_connection() as conn:
        db = DBAdapter(conn)
        rows = await db.fetch_all(
            """
            SELECT residue_id, residue_index, residue_name, chain_label,
                   hyp_projections, hyp_projection_2d,
                   cone_depth, cone_width, epistemic_uncertainty,
                   curvature, space_name, model_version
            FROM v_viz_poincare_disc_data
            WHERE structure_id = :structure_id
            ORDER BY residue_index
            """,
            {"structure_id": structure_id},
        )

    if not rows:
        return DiscData(
            structure_id=structure_id,
            pdb_id=None,
            curvature=1.0,
            space_name="none",
            model_version="none",
            nodes=[],
        )

    nodes = []
    for r in rows:
        # Extract 2D disc coordinates
        x, y = 0.0, 0.0
        if r.get("hyp_projection_2d"):
            coords = r["hyp_projection_2d"]
            x, y = float(coords[0]), float(coords[1])
        elif r.get("hyp_projections"):
            proj = r["hyp_projections"]
            if isinstance(proj, list) and len(proj) >= 2:
                x, y = float(proj[0]), float(proj[1])

        nodes.append(DiscNode(
            residue_id=r["residue_id"],
            residue_index=r["residue_index"],
            residue_name=r["residue_name"] or "?",
            chain_label=r["chain_label"],
            x=x,
            y=y,
            cone_depth=r["cone_depth"] or 0.0,
            cone_width=r.get("cone_width") or 0.0,
            epistemic_uncertainty=r.get("epistemic_uncertainty"),
        ))

    return DiscData(
        structure_id=structure_id,
        pdb_id=None,
        curvature=rows[0].get("curvature") or 1.0,
        space_name=rows[0].get("space_name") or "unknown",
        model_version=rows[0].get("model_version") or "unknown",
        nodes=nodes,
    )


# ---------------------------------------------------------------------------
# WebSocket — Viewport Directive Channel
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages WebSocket connections from visualizer frontends."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def broadcast_directive(self, directive: dict[str, Any]) -> None:
        """Push a viewport directive to all connected frontends."""
        for connection in self.active_connections:
            try:
                await connection.send_json(directive)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/viewport")
async def viewport_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time viewport communication.

    Frontend connects here to:
    1. Register its viewport capabilities
    2. Receive ViewportDirectives from the agent
    3. Send viewport events (hover, select, click) back
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle incoming viewport events from frontend
            # (hover, selection, click events)
            event_type = data.get("type")
            if event_type == "register":
                # Frontend registering its capabilities
                pass
            elif event_type == "selection":
                # User selected residues in the viewer
                pass
            elif event_type == "hover":
                # User hovering over a residue
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/viewport/directive")
async def push_directive(directive: dict[str, Any]) -> dict[str, str]:
    """Push a viewport directive to all connected frontends.

    Called by the agent coordinator when it wants to update the view.
    """
    await manager.broadcast_directive(directive)
    return {"status": "broadcast_sent", "connections": len(manager.active_connections)}
