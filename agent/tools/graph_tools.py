"""Graph topology agent tools.

Provides tools for querying, comparing, and analyzing persisted
molecular contact graph topology. All reads go through the governed
fact tables (fact_graph_edge, fact_graph_node_metrics).

Tools:
- get_graph_metrics: Query per-residue graph metrics
- compare_graphs: Compare topology of two structures (edge + metric diff)
- get_hbond_network: Extract H-bond subgraph
- find_graph_bridges: Identify bridge/articulation-point residues
- get_shortest_paths: Find shortest path between two residues
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent.models.viewport import (
    DirectiveAction,
    HighlightGroup,
    HighlightStyle,
    ViewportDirective,
)
from agent.tools.dtie.tools import ToolDB, ToolResult


# ---------------------------------------------------------------------------
# Tool 1: get_graph_metrics
# ---------------------------------------------------------------------------


async def get_graph_metrics(
    structure_id: str,
    run_id: str | None = None,
    residue_ids: list[str] | None = None,
    metric_type: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Retrieve per-residue graph metrics for a structure.

    Returns degree, betweenness, clustering coefficient, closeness,
    eigenvector centrality, bridge status, and conductance.

    Args:
        structure_id: Structure to query metrics for.
        run_id: Specific run_id (latest if omitted).
        residue_ids: Optional filter to specific residues.
        metric_type: Optional filter to a single metric column.
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Determine which columns to select
    all_metrics = [
        "degree", "betweenness", "clustering_coefficient",
        "closeness", "eigenvector_centrality", "is_bridge", "conductance",
    ]

    valid_metric_types = set(all_metrics)
    if metric_type and metric_type not in valid_metric_types:
        return ToolResult(
            success=False,
            message=f"Invalid metric_type '{metric_type}'. Valid: {sorted(valid_metric_types)}",
        )

    if metric_type:
        select_cols = f"m.residue_id, m.run_id, m.structure_id, m.{metric_type}"
    else:
        select_cols = (
            "m.residue_id, m.run_id, m.structure_id, "
            "m.degree, m.betweenness, m.clustering_coefficient, "
            "m.closeness, m.eigenvector_centrality, m.is_bridge, m.conductance"
        )

    # Build query
    conditions = ["m.structure_id = :structure_id"]
    params: dict[str, Any] = {"structure_id": structure_id}

    if run_id:
        conditions.append("m.run_id = :run_id")
        params["run_id"] = run_id

    if residue_ids:
        placeholders = ", ".join(f":r{i}" for i in range(len(residue_ids)))
        conditions.append(f"m.residue_id IN ({placeholders})")
        params.update({f"r{i}": rid for i, rid in enumerate(residue_ids)})

    where_clause = " AND ".join(conditions)

    # If no run_id specified, get the latest run
    if not run_id:
        order_clause = "ORDER BY m.computed_at DESC"
    else:
        order_clause = "ORDER BY m.residue_id"

    rows = await tool_db.fetch_all(
        f"""
        SELECT {select_cols}
        FROM fact_graph_node_metrics m
        WHERE {where_clause}
        {order_clause}
        """,
        params,
    )

    if not rows:
        return ToolResult(
            success=True,
            data={"structure_id": structure_id, "metrics": [], "count": 0},
            message=f"No graph metrics found for {structure_id}",
        )

    # Highlight high-betweenness residues in the viewer
    high_betweenness_ids = [
        r["residue_id"] for r in rows
        if r.get("betweenness") is not None and r.get("betweenness", 0) > 0.1
    ]

    directives = []
    if high_betweenness_ids:
        directives.append(ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id=structure_id,
            highlight_groups=[
                HighlightGroup(
                    residue_ids=high_betweenness_ids,
                    color="#e67e22",
                    style=HighlightStyle.GLOW,
                    label="High Betweenness",
                )
            ],
            message=f"{len(high_betweenness_ids)} high-betweenness residues",
        ))

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "metrics": rows,
            "count": len(rows),
            "metric_type": metric_type,
        },
        message=f"Retrieved graph metrics for {len(rows)} residues in {structure_id}",
        viewport_directives=directives,
    )


# ---------------------------------------------------------------------------
# Tool 2: compare_graphs
# ---------------------------------------------------------------------------


async def compare_graphs(
    structure_id_a: str,
    structure_id_b: str,
    run_id_a: str | None = None,
    run_id_b: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Compare graph topology between two structures (e.g., WT vs mutant).

    Computes:
    - Edge diff: gained, lost, and weight-changed edges
    - Metric diff: per-residue delta for all graph metrics

    Residues are matched across structures using canonical residue_id
    alignment (same chain + residue_index).

    Args:
        structure_id_a: First structure (e.g., wild-type).
        structure_id_b: Second structure (e.g., mutant).
        run_id_a: Specific run for structure A (latest if omitted).
        run_id_b: Specific run for structure B (latest if omitted).
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Fetch edges for both structures
    edges_a = await _fetch_edges_for_structure(tool_db, structure_id_a, run_id_a)
    edges_b = await _fetch_edges_for_structure(tool_db, structure_id_b, run_id_b)

    if not edges_a:
        return ToolResult(success=False, message=f"No graph edges found for {structure_id_a}")
    if not edges_b:
        return ToolResult(success=False, message=f"No graph edges found for {structure_id_b}")

    # Compute edge diff using canonical key (source, target, edge_type) minus run_id
    def _edge_key(e: dict) -> tuple:
        # Use chain:index portion of residue_id for cross-structure alignment
        src = _canonical_position(e["source_residue_id"])
        tgt = _canonical_position(e["target_residue_id"])
        return (src, tgt, e["edge_type"])

    edges_a_by_key = {_edge_key(e): e for e in edges_a}
    edges_b_by_key = {_edge_key(e): e for e in edges_b}

    keys_a = set(edges_a_by_key.keys())
    keys_b = set(edges_b_by_key.keys())

    gained_keys = keys_b - keys_a
    lost_keys = keys_a - keys_b
    common_keys = keys_a & keys_b

    # Detect weight changes in common edges
    changed = []
    for key in common_keys:
        ea = edges_a_by_key[key]
        eb = edges_b_by_key[key]
        w_a = ea.get("weight") or 1.0
        w_b = eb.get("weight") or 1.0
        if abs(w_a - w_b) > 1e-6:
            changed.append({
                "source": key[0],
                "target": key[1],
                "edge_type": key[2],
                "weight_a": w_a,
                "weight_b": w_b,
                "weight_delta": w_b - w_a,
            })

    gained = [
        {"source": k[0], "target": k[1], "edge_type": k[2]}
        for k in gained_keys
    ]
    lost = [
        {"source": k[0], "target": k[1], "edge_type": k[2]}
        for k in lost_keys
    ]

    # Fetch metrics for both structures and compute deltas
    metrics_a = await _fetch_metrics_for_structure(tool_db, structure_id_a, run_id_a)
    metrics_b = await _fetch_metrics_for_structure(tool_db, structure_id_b, run_id_b)

    metrics_a_by_pos = {_canonical_position(m["residue_id"]): m for m in metrics_a}
    metrics_b_by_pos = {_canonical_position(m["residue_id"]): m for m in metrics_b}

    common_positions = set(metrics_a_by_pos.keys()) & set(metrics_b_by_pos.keys())

    metric_fields = ["degree", "betweenness", "clustering_coefficient", "closeness", "eigenvector_centrality"]
    metric_deltas = []
    for pos in sorted(common_positions):
        ma = metrics_a_by_pos[pos]
        mb = metrics_b_by_pos[pos]
        delta = {"position": pos}
        for field in metric_fields:
            val_a = ma.get(field) or 0
            val_b = mb.get(field) or 0
            delta[f"{field}_a"] = val_a
            delta[f"{field}_b"] = val_b
            delta[f"{field}_delta"] = val_b - val_a
        metric_deltas.append(delta)

    # Sort by largest betweenness change
    metric_deltas.sort(key=lambda d: abs(d.get("betweenness_delta", 0)), reverse=True)

    # Highlight gained/lost H-bonds in viewer
    hbond_gained = [g for g in gained if g["edge_type"] == "h_bond"]
    hbond_lost = [l for l in lost if l["edge_type"] == "h_bond"]

    highlight_groups = []
    if hbond_gained:
        gained_residues = list({r for g in hbond_gained for r in _residue_ids_from_position(g["source"], edges_b) + _residue_ids_from_position(g["target"], edges_b)})
        if gained_residues:
            highlight_groups.append(HighlightGroup(
                residue_ids=gained_residues,
                color="#2ecc71",
                style=HighlightStyle.GLOW,
                label="Gained H-bonds",
            ))
    if hbond_lost:
        lost_residues = list({r for l in hbond_lost for r in _residue_ids_from_position(l["source"], edges_a) + _residue_ids_from_position(l["target"], edges_a)})
        if lost_residues:
            highlight_groups.append(HighlightGroup(
                residue_ids=lost_residues,
                color="#e74c3c",
                style=HighlightStyle.PULSE,
                label="Lost H-bonds",
            ))

    directives = []
    if highlight_groups:
        directives.append(ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id=structure_id_a,
            highlight_groups=highlight_groups,
            message=f"Gained {len(hbond_gained)} / Lost {len(hbond_lost)} H-bonds",
        ))

    return ToolResult(
        success=True,
        data={
            "structure_a": structure_id_a,
            "structure_b": structure_id_b,
            "edge_diff": {
                "gained": gained,
                "lost": lost,
                "changed": changed,
                "gained_count": len(gained),
                "lost_count": len(lost),
                "changed_count": len(changed),
            },
            "metric_diff": metric_deltas[:20],
            "total_matched_residues": len(common_positions),
            "hbond_gained": len(hbond_gained),
            "hbond_lost": len(hbond_lost),
        },
        message=(
            f"Compared {structure_id_a} vs {structure_id_b}: "
            f"+{len(gained)} edges, -{len(lost)} edges, "
            f"~{len(changed)} weight changes"
        ),
        viewport_directives=directives,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_position(residue_id: str) -> str:
    """Extract chain:index from a canonical residue_id for cross-structure matching.

    Canonical format: structure_id:chain:index[:insertion]
    Returns: chain:index (the position-identifying portion).
    """
    parts = residue_id.split(":")
    if len(parts) >= 3:
        return f"{parts[1]}:{parts[2]}"
    return residue_id


def _residue_ids_from_position(position: str, edges: list[dict]) -> list[str]:
    """Find full residue_ids matching a canonical position in an edge list."""
    results = []
    for e in edges:
        if _canonical_position(e["source_residue_id"]) == position:
            results.append(e["source_residue_id"])
        if _canonical_position(e["target_residue_id"]) == position:
            results.append(e["target_residue_id"])
    return list(set(results))


async def _fetch_edges_for_structure(
    tool_db: ToolDB, structure_id: str, run_id: str | None = None
) -> list[dict[str, Any]]:
    """Fetch all edges for a structure, optionally filtered by run_id."""
    conditions = ["structure_id = :structure_id"]
    params: dict[str, Any] = {"structure_id": structure_id}

    if run_id:
        conditions.append("run_id = :run_id")
        params["run_id"] = run_id

    where = " AND ".join(conditions)
    return await tool_db.fetch_all(
        f"""
        SELECT source_residue_id, target_residue_id, edge_type,
               distance_angstrom, weight, run_id
        FROM fact_graph_edge
        WHERE {where}
        ORDER BY source_residue_id, target_residue_id
        """,
        params,
    )


async def _fetch_metrics_for_structure(
    tool_db: ToolDB, structure_id: str, run_id: str | None = None
) -> list[dict[str, Any]]:
    """Fetch all node metrics for a structure, optionally filtered by run_id."""
    conditions = ["structure_id = :structure_id"]
    params: dict[str, Any] = {"structure_id": structure_id}

    if run_id:
        conditions.append("run_id = :run_id")
        params["run_id"] = run_id

    where = " AND ".join(conditions)
    return await tool_db.fetch_all(
        f"""
        SELECT residue_id, degree, betweenness, clustering_coefficient,
               closeness, eigenvector_centrality, is_bridge, conductance, run_id
        FROM fact_graph_node_metrics
        WHERE {where}
        ORDER BY residue_id
        """,
        params,
    )


# ---------------------------------------------------------------------------
# Tool 3: get_hbond_network
# ---------------------------------------------------------------------------


async def get_hbond_network(
    structure_id: str,
    run_id: str | None = None,
    residue_ids: list[str] | None = None,
    db: Any = None,
) -> ToolResult:
    """Extract the H-bond subgraph for a structure or region.

    Returns all edges of type 'h_bond'. When a residue_ids filter is
    provided, returns only H-bond edges involving at least one of the
    specified residues.

    Args:
        structure_id: Structure to query.
        run_id: Specific run_id (latest if omitted).
        residue_ids: Optional filter — edges must involve at least one.
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    conditions = ["e.structure_id = :structure_id", "e.edge_type = 'h_bond'"]
    params: dict[str, Any] = {"structure_id": structure_id}

    if run_id:
        conditions.append("e.run_id = :run_id")
        params["run_id"] = run_id

    if residue_ids:
        placeholders = ", ".join(f":r{i}" for i in range(len(residue_ids)))
        conditions.append(
            f"(e.source_residue_id IN ({placeholders}) OR e.target_residue_id IN ({placeholders}))"
        )
        params.update({f"r{i}": rid for i, rid in enumerate(residue_ids)})

    where_clause = " AND ".join(conditions)

    rows = await tool_db.fetch_all(
        f"""
        SELECT e.source_residue_id, e.target_residue_id, e.edge_type,
               e.distance_angstrom, e.weight, e.run_id
        FROM fact_graph_edge e
        WHERE {where_clause}
        ORDER BY e.source_residue_id, e.target_residue_id
        """,
        params,
    )

    # Collect unique residue_ids for highlighting
    hbond_residues = list({
        rid
        for row in rows
        for rid in (row["source_residue_id"], row["target_residue_id"])
    })

    directives = []
    if hbond_residues:
        directives.append(ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id=structure_id,
            highlight_groups=[
                HighlightGroup(
                    residue_ids=hbond_residues,
                    color="#3498db",
                    style=HighlightStyle.GLOW,
                    label="H-bond Network",
                )
            ],
            message=f"{len(rows)} H-bonds involving {len(hbond_residues)} residues",
        ))

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "edges": rows,
            "count": len(rows),
            "residue_count": len(hbond_residues),
        },
        message=f"Found {len(rows)} H-bond edges in {structure_id}",
        viewport_directives=directives,
    )


# ---------------------------------------------------------------------------
# Tool 4: find_graph_bridges
# ---------------------------------------------------------------------------


async def find_graph_bridges(
    structure_id: str,
    run_id: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Identify bridge residues (articulation points) in the contact graph.

    Bridge residues are those whose removal disconnects the graph,
    making them potential allosteric communication bottlenecks.

    Args:
        structure_id: Structure to query.
        run_id: Specific run_id (latest if omitted).
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    conditions = ["m.structure_id = :structure_id", "m.is_bridge = TRUE"]
    params: dict[str, Any] = {"structure_id": structure_id}

    if run_id:
        conditions.append("m.run_id = :run_id")
        params["run_id"] = run_id

    where_clause = " AND ".join(conditions)

    rows = await tool_db.fetch_all(
        f"""
        SELECT m.residue_id, m.degree, m.betweenness, m.is_bridge, m.run_id
        FROM fact_graph_node_metrics m
        WHERE {where_clause}
        ORDER BY m.betweenness DESC
        """,
        params,
    )

    bridge_ids = [r["residue_id"] for r in rows]

    directives = []
    if bridge_ids:
        directives.append(ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id=structure_id,
            highlight_groups=[
                HighlightGroup(
                    residue_ids=bridge_ids,
                    color="#9b59b6",
                    style=HighlightStyle.PULSE,
                    label="Bridge Residues",
                )
            ],
            message=f"{len(bridge_ids)} bridge residues (communication bottlenecks)",
        ))

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "bridges": rows,
            "count": len(rows),
        },
        message=f"Found {len(rows)} bridge residues in {structure_id}",
        viewport_directives=directives,
    )


# ---------------------------------------------------------------------------
# Tool 5: get_shortest_paths
# ---------------------------------------------------------------------------


async def get_shortest_paths(
    structure_id: str,
    source_residue_id: str,
    target_residue_id: str,
    run_id: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Find the shortest path between two residues in the contact graph.

    Loads edges into a networkx graph (in a background thread) and
    computes the shortest path weighted by distance_angstrom.

    Args:
        structure_id: Structure to query.
        source_residue_id: Starting residue.
        target_residue_id: Ending residue.
        run_id: Specific run_id (latest if omitted).
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Fetch all edges for the structure
    edges = await _fetch_edges_for_structure(tool_db, structure_id, run_id)

    if not edges:
        return ToolResult(
            success=False,
            message=f"No graph edges found for {structure_id}",
        )

    # Compute shortest path in a background thread
    path_result = await asyncio.to_thread(
        _compute_shortest_path, edges, source_residue_id, target_residue_id
    )

    if path_result["disconnected"]:
        return ToolResult(
            success=True,
            data={
                "structure_id": structure_id,
                "source": source_residue_id,
                "target": target_residue_id,
                "path": [],
                "total_distance": 0.0,
                "disconnected": True,
            },
            message=(
                f"No path exists between {source_residue_id} and "
                f"{target_residue_id} — residues are in disconnected components"
            ),
        )

    path_ids = path_result["path"]

    directives = []
    if path_ids:
        directives.append(ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id=structure_id,
            highlight_groups=[
                HighlightGroup(
                    residue_ids=path_ids,
                    color="#1abc9c",
                    style=HighlightStyle.GLOW,
                    label="Shortest Path",
                )
            ],
            message=f"Path length: {len(path_ids)} residues, {path_result['total_distance']:.2f} Å",
        ))

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "source": source_residue_id,
            "target": target_residue_id,
            "path": path_ids,
            "path_length": len(path_ids),
            "total_distance": path_result["total_distance"],
            "disconnected": False,
        },
        message=(
            f"Shortest path from {source_residue_id} to {target_residue_id}: "
            f"{len(path_ids)} residues, {path_result['total_distance']:.2f} Å total"
        ),
        viewport_directives=directives,
    )


def _compute_shortest_path(
    edges: list[dict[str, Any]],
    source: str,
    target: str,
) -> dict[str, Any]:
    """Build networkx graph and compute shortest path (CPU-bound, runs in thread)."""
    import networkx as nx

    G = nx.Graph()
    for e in edges:
        weight = e.get("distance_angstrom") or e.get("weight") or 1.0
        G.add_edge(
            e["source_residue_id"],
            e["target_residue_id"],
            distance_angstrom=weight,
        )

    # Check if both nodes exist
    if source not in G:
        return {"path": [], "total_distance": 0.0, "disconnected": True}
    if target not in G:
        return {"path": [], "total_distance": 0.0, "disconnected": True}

    # Check connectivity
    if not nx.has_path(G, source, target):
        return {"path": [], "total_distance": 0.0, "disconnected": True}

    # Compute shortest path
    path = nx.shortest_path(G, source, target, weight="distance_angstrom")

    # Compute total distance along the path
    total_distance = 0.0
    for i in range(len(path) - 1):
        edge_data = G[path[i]][path[i + 1]]
        total_distance += edge_data.get("distance_angstrom", 1.0)

    return {"path": path, "total_distance": total_distance, "disconnected": False}
