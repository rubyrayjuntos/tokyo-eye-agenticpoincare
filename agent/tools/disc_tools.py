"""Agent tool wrappers for Poincaré disc topology.

Provides high-level tools that the agent can invoke:
- get_disc_topology: Fetches coordinates from DB, computes or retrieves cached topology
- get_disc_neighborhood: Returns neighborhood using cached topology

Feature: poincare-visual-context
Requirements: 1.1, 2.1
"""

from __future__ import annotations

import logging
from typing import Any

from agent.models.viewport import (
    DirectiveAction,
    HighlightGroup,
    HighlightStyle,
    ViewportDirective,
)
from agent.tools.disc_topology import (
    DiscTopologyResult,
    compute_disc_neighborhood,
    compute_disc_topology,
)
from science.dtie.common.curvature_values import (
    MissingLearnedCurvatureError,
    learned_curvature_from_rows,
)
from agent.tools.disc_topology_cache import TopologyCache
from agent.tools.dtie.tools import ToolDB, ToolResult

logger = logging.getLogger(__name__)

# Module-level singleton cache shared across tool invocations
_topology_cache = TopologyCache(max_size=50)


def get_topology_cache() -> TopologyCache:
    """Return the module-level topology cache singleton."""
    return _topology_cache


# ---------------------------------------------------------------------------
# Tool: get_disc_topology
# ---------------------------------------------------------------------------


async def get_disc_topology(
    structure_id: str,
    run_id: str | None = None,
    min_cluster_size: int = 5,
    db: Any = None,
) -> ToolResult:
    """Compute or retrieve cached disc topology for a structure.

    Fetches hyp_projection_2d coordinates from the governed data layer,
    computes HDBSCAN clustering on the hyperbolic distance matrix, and
    returns cluster assignments, hubs, bridges, and density profile.

    Args:
        structure_id: Structure to compute topology for.
        run_id: Specific run_id (latest if omitted).
        min_cluster_size: Minimum cluster size for HDBSCAN.
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    # Resolve run_id if not provided
    tool_db = ToolDB(db)
    if not run_id:
        latest = await tool_db.fetch_one(
            """
            SELECT e.run_id
            FROM fact_gnn_node_embedding e
            WHERE e.structure_id = :structure_id
            ORDER BY e.computed_at DESC
            LIMIT 1
            """,
            {"structure_id": structure_id},
        )
        if not latest:
            return ToolResult(
                success=False,
                message=f"No GNN embeddings found for {structure_id}. Run GNN inference first.",
            )
        run_id = latest["run_id"]

    # Check cache
    cached = _topology_cache.get(structure_id, run_id)
    if cached is not None:
        return _topology_to_tool_result(cached, from_cache=True)

    # Fetch coordinates from DB
    rows = await tool_db.fetch_all(
        """
        SELECT r.residue_id, e.hyp_projection_2d, es.curvature
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND e.run_id = :run_id
          AND e.hyp_projection_2d IS NOT NULL
        ORDER BY r.residue_index
        """,
        {"structure_id": structure_id, "run_id": run_id},
    )

    if not rows:
        return ToolResult(
            success=False,
            message=f"No hyp_projection_2d data for {structure_id} run {run_id}.",
        )

    try:
        curvature_c = learned_curvature_from_rows(rows, context="get_disc_topology")
    except MissingLearnedCurvatureError as exc:
        return ToolResult(success=False, message=str(exc))

    coordinates: list[tuple[str, float, float]] = []
    for r in rows:
        coords = r["hyp_projection_2d"]
        if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            coordinates.append((r["residue_id"], float(coords[0]), float(coords[1])))

    if len(coordinates) < 3:
        return ToolResult(
            success=False,
            message=f"Insufficient coordinates ({len(coordinates)}) for topology computation.",
        )

    # Compute topology
    result = compute_disc_topology(
        coordinates,
        curvature_c=curvature_c,
        min_cluster_size=min_cluster_size,
        structure_id=structure_id,
        run_id=run_id,
    )

    # Cache the result
    _topology_cache.put(result)

    return _topology_to_tool_result(result, from_cache=False)


# ---------------------------------------------------------------------------
# Tool: get_disc_neighborhood
# ---------------------------------------------------------------------------


async def get_disc_neighborhood(
    structure_id: str,
    residue_id: str,
    k: int = 8,
    run_id: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Get k-nearest neighbors on the Poincaré disc for a residue.

    Uses cached topology if available; otherwise computes it first.

    Args:
        structure_id: Structure to query.
        residue_id: Target residue to find neighbors for.
        k: Number of nearest neighbors (default 8).
        run_id: Specific run_id (latest if omitted).
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    # Resolve run_id if not provided
    tool_db = ToolDB(db)
    if not run_id:
        latest = await tool_db.fetch_one(
            """
            SELECT e.run_id
            FROM fact_gnn_node_embedding e
            WHERE e.structure_id = :structure_id
            ORDER BY e.computed_at DESC
            LIMIT 1
            """,
            {"structure_id": structure_id},
        )
        if not latest:
            return ToolResult(
                success=False,
                message=f"No GNN embeddings found for {structure_id}.",
            )
        run_id = latest["run_id"]

    # Get or compute topology
    topology = _topology_cache.get(structure_id, run_id)
    if topology is None:
        # Need to compute — fetch coordinates
        topo_result = await get_disc_topology(
            structure_id=structure_id, run_id=run_id, db=db,
        )
        if not topo_result.success:
            return topo_result
        topology = _topology_cache.get(structure_id, run_id)
        if topology is None:
            return ToolResult(
                success=False,
                message="Failed to compute disc topology.",
            )

    # Fetch coordinates for neighborhood computation
    rows = await tool_db.fetch_all(
        """
        SELECT r.residue_id, e.hyp_projection_2d, es.curvature
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND e.run_id = :run_id
          AND e.hyp_projection_2d IS NOT NULL
        ORDER BY r.residue_index
        """,
        {"structure_id": structure_id, "run_id": run_id},
    )

    try:
        curvature_c = learned_curvature_from_rows(rows, context="get_disc_neighborhood")
    except MissingLearnedCurvatureError as exc:
        return ToolResult(success=False, message=str(exc))

    coordinates: list[tuple[str, float, float]] = []
    for r in rows:
        coords = r["hyp_projection_2d"]
        if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            coordinates.append((r["residue_id"], float(coords[0]), float(coords[1])))

    # Check target exists
    residue_ids_in_coords = [c[0] for c in coordinates]
    if residue_id not in residue_ids_in_coords:
        return ToolResult(
            success=False,
            message=f"Residue {residue_id} not found in embeddings for {structure_id}.",
        )

    # Compute neighborhood
    try:
        neighborhood = compute_disc_neighborhood(
            target_residue_id=residue_id,
            coordinates=coordinates,
            topology=topology,
            curvature_c=curvature_c,
            k=k,
        )
    except ValueError as e:
        return ToolResult(success=False, message=str(e))

    # Build viewport directive to highlight neighbors
    neighbor_ids = [n.residue_id for n in neighborhood.neighbors]
    directives = []
    if neighbor_ids:
        highlight_groups = [
            HighlightGroup(
                residue_ids=[residue_id],
                color="#ffffff",
                style=HighlightStyle.GLOW,
                label="Target",
            ),
            HighlightGroup(
                residue_ids=neighbor_ids,
                color="#4ecdc4",
                style=HighlightStyle.GLOW,
                label=f"k={k} Neighbors",
            ),
        ]
        directives.append(ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id=structure_id,
            highlight_groups=highlight_groups,
            message=f"Neighborhood of {residue_id}: {k} nearest on disc",
        ))

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "target_residue_id": neighborhood.target_residue_id,
            "target_cluster_id": neighborhood.target_cluster_id,
            "is_hub": neighborhood.is_hub,
            "is_peripheral": neighborhood.is_peripheral,
            "neighbors": [
                {
                    "residue_id": n.residue_id,
                    "hyperbolic_distance": n.hyperbolic_distance,
                    "cluster_id": n.cluster_id,
                    "cone_depth": n.cone_depth,
                    "epistemic_uncertainty": n.epistemic_uncertainty,
                }
                for n in neighborhood.neighbors
            ],
            "k": k,
        },
        message=(
            f"Neighborhood of {residue_id}: "
            f"{'hub' if neighborhood.is_hub else 'non-hub'}, "
            f"{'peripheral' if neighborhood.is_peripheral else 'interior'}, "
            f"cluster {neighborhood.target_cluster_id}"
        ),
        viewport_directives=directives,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topology_to_tool_result(topology: DiscTopologyResult, from_cache: bool) -> ToolResult:
    """Convert a DiscTopologyResult into a ToolResult for the agent."""
    clusters_data = [
        {
            "cluster_id": c.cluster_id,
            "size": c.size,
            "angular_sector": c.angular_sector,
            "centroid_angle_deg": c.centroid_angle_deg,
            "centroid_radius": c.centroid_radius,
            "hub_residue_id": c.hub_residue_id,
        }
        for c in topology.clusters
    ]

    return ToolResult(
        success=True,
        data={
            "structure_id": topology.structure_id,
            "run_id": topology.run_id,
            "curvature_c": topology.curvature_c,
            "total_residues": topology.total_residues,
            "cluster_count": topology.cluster_count,
            "clusters": clusters_data,
            "bridge_residues": topology.bridge_residues,
            "peripheral_residues": topology.peripheral_residues,
            "radial_density": topology.radial_density,
            "from_cache": from_cache,
        },
        message=(
            f"Disc topology for {topology.structure_id}: "
            f"{topology.cluster_count} clusters, "
            f"{topology.total_residues} residues"
            f"{' (cached)' if from_cache else ''}"
        ),
    )
