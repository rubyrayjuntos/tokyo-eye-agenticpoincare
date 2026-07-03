"""Discovery signal tools for the agent coordinator.

Concrete tools the coordinator calls for read-only analysis of governed
pathway artifacts. Each tool queries production views and returns
structured results plus optional viewport directives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.models.viewport import (
    DirectiveAction,
    HighlightGroup,
    HighlightStyle,
    ViewportDirective,
)


class ToolDB:
    """Database adapter for tool queries. Wraps the DBAdapter protocol."""

    def __init__(self, db: Any):
        self._db = db

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return await self._db.fetch_all(query, params or {})

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return await self._db.fetch_one(query, params or {})


@dataclass
class ToolResult:
    """Standard result from any agent tool."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    viewport_directives: list[ViewportDirective] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


async def run_phase(
    structure_id: str,
    phase: str,
    model_version: str = "v4",
    parameters: dict[str, Any] | None = None,
    db: Any = None,
) -> ToolResult:
    """Run a specific discovery pathway phase on a structure.

    Requires GNN embeddings to already exist in the governed layer.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    # Verify embeddings exist
    tool_db = ToolDB(db)
    existing = await tool_db.fetch_one(
        """
        SELECT COUNT(*) as cnt FROM fact_gnn_node_embedding e
        JOIN provenance_run p ON p.run_id = e.run_id
        WHERE e.structure_id = :structure_id AND p.run_type = 'inference'
        """,
        {"structure_id": structure_id},
    )

    if not existing or existing["cnt"] == 0:
        return ToolResult(
            success=False,
            message=f"No GNN embeddings found for {structure_id}. Run GNN inference first.",
        )

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "phase": phase,
            "model_version": model_version,
            "status": "phase_execution_ready",
            "parameters": parameters,
        },
        message=f"Phase {phase} ({model_version}) ready for {structure_id}",
    )


async def get_source_leaks(
    structure_id: str,
    uncertainty_threshold: float = 0.3,
    min_depth: float = 1.5,
    db: Any = None,
) -> ToolResult:
    """Identify source-leak candidates from governed data.

    Queries v_agent_high_uncertainty_residues filtered by depth.
    Source leaks = high epistemic uncertainty + significant depth.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        """
        SELECT residue_id, residue_index, residue_name, chain_label,
               epistemic_uncertainty, cone_depth, model_version
        FROM v_agent_high_uncertainty_residues
        WHERE structure_id = :structure_id
          AND epistemic_uncertainty >= :threshold
          AND cone_depth >= :min_depth
        ORDER BY epistemic_uncertainty DESC
        LIMIT 50
        """,
        {
            "structure_id": structure_id,
            "threshold": uncertainty_threshold,
            "min_depth": min_depth,
        },
    )

    residue_ids = [r["residue_id"] for r in rows]

    directive = ViewportDirective(
        action=DirectiveAction.HIGHLIGHT,
        structure_id=structure_id,
        highlight_groups=[
            HighlightGroup(
                residue_ids=residue_ids,
                color="#ff4444",
                style=HighlightStyle.PULSE,
                label="Source Leaks",
            )
        ] if residue_ids else [],
        message=f"Found {len(residue_ids)} source-leak candidates",
    )

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "source_leaks": rows,
            "count": len(rows),
            "threshold": uncertainty_threshold,
            "min_depth": min_depth,
        },
        message=f"Found {len(rows)} source-leak candidates in {structure_id}",
        viewport_directives=[directive],
    )


async def get_high_uncertainty_residues(
    structure_id: str,
    top_n: int = 20,
    uncertainty_type: str = "epistemic",
    db: Any = None,
) -> ToolResult:
    """Get the highest-uncertainty residues for a structure.

    Queries v_agent_high_uncertainty_residues.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    # Validate uncertainty_type against allowlist to prevent SQL injection
    VALID_UNCERTAINTY_TYPES = {"epistemic", "aleatoric", "total"}
    if uncertainty_type not in VALID_UNCERTAINTY_TYPES:
        return ToolResult(
            success=False,
            message=f"Invalid uncertainty_type '{uncertainty_type}'. Must be one of: {', '.join(sorted(VALID_UNCERTAINTY_TYPES))}",
        )

    col = f"{uncertainty_type}_uncertainty"
    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        f"""
        SELECT residue_id, residue_index, residue_name, chain_label,
               epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty,
               cone_depth, model_version
        FROM v_agent_high_uncertainty_residues
        WHERE structure_id = :structure_id
          AND {col} IS NOT NULL
        ORDER BY {col} DESC
        LIMIT :top_n
        """,
        {"structure_id": structure_id, "top_n": top_n},
    )

    residue_ids = [r["residue_id"] for r in rows]

    directive = ViewportDirective(
        action=DirectiveAction.SHOW_UNCERTAINTY,
        structure_id=structure_id,
        metric=uncertainty_type,
        highlight_groups=[
            HighlightGroup(
                residue_ids=residue_ids,
                color="#ffaa00",
                style=HighlightStyle.GLOW,
                label=f"Top {uncertainty_type}",
            )
        ] if residue_ids else [],
        message=f"Top {len(rows)} residues by {uncertainty_type} uncertainty",
    )

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "residues": rows,
            "count": len(rows),
            "uncertainty_type": uncertainty_type,
        },
        message=f"Top {len(rows)} {uncertainty_type} uncertainty residues in {structure_id}",
        viewport_directives=[directive],
    )


async def get_residue_state(
    structure_id: str,
    residue_ids: list[str] | None = None,
    db: Any = None,
) -> ToolResult:
    """Get the current governed state of residues.

    Queries v_agent_residue_state for latest embeddings, uncertainty,
    dehydron status, and site membership.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    if residue_ids:
        # Query specific residues
        placeholders = ", ".join(f":r{i}" for i in range(len(residue_ids)))
        params: dict[str, Any] = {"structure_id": structure_id}
        params.update({f"r{i}": rid for i, rid in enumerate(residue_ids)})

        rows = await tool_db.fetch_all(
            f"""
            SELECT * FROM v_agent_residue_state
            WHERE structure_id = :structure_id
              AND residue_id IN ({placeholders})
            """,
            params,
        )
    else:
        rows = await tool_db.fetch_all(
            """
            SELECT * FROM v_agent_residue_state
            WHERE structure_id = :structure_id
            ORDER BY residue_index
            """,
            {"structure_id": structure_id},
        )

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "residues": rows,
            "count": len(rows),
        },
        message=f"Retrieved state for {len(rows)} residues in {structure_id}",
    )


async def compare_wt_mutant(
    wt_structure_id: str,
    mutant_structure_id: str,
    focus_residues: list[str] | None = None,
    db: Any = None,
) -> ToolResult:
    """Compare wild-type and mutant embeddings in hyperbolic space.

    Computes hyperbolic displacement between corresponding residues
    in WT and mutant structures.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Get latest hyperbolic embeddings for both structures
    wt_rows = await tool_db.fetch_all(
        """
        SELECT e.residue_id, e.embedding, e.cone_depth, e.epistemic_uncertainty,
               r.residue_index, c.chain_label
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
        ORDER BY e.computed_at DESC
        """,
        {"structure_id": wt_structure_id},
    )

    mut_rows = await tool_db.fetch_all(
        """
        SELECT e.residue_id, e.embedding, e.cone_depth, e.epistemic_uncertainty,
               r.residue_index, c.chain_label
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
        ORDER BY e.computed_at DESC
        """,
        {"structure_id": mutant_structure_id},
    )

    # Match by residue_index + chain_label
    wt_by_pos = {(r["chain_label"], r["residue_index"]): r for r in wt_rows}
    mut_by_pos = {(r["chain_label"], r["residue_index"]): r for r in mut_rows}

    displacements = []
    for pos, wt_r in wt_by_pos.items():
        if pos in mut_by_pos:
            mut_r = mut_by_pos[pos]
            displacements.append({
                "residue_index": pos[1],
                "chain_label": pos[0],
                "wt_residue_id": wt_r["residue_id"],
                "mut_residue_id": mut_r["residue_id"],
                "wt_depth": wt_r["cone_depth"],
                "mut_depth": mut_r["cone_depth"],
                "depth_change": abs(mut_r["cone_depth"] - wt_r["cone_depth"]) if wt_r["cone_depth"] and mut_r["cone_depth"] else None,
            })

    # Sort by displacement magnitude
    displacements.sort(key=lambda d: d.get("depth_change") or 0, reverse=True)

    # Highlight top movers
    top_movers = [d["wt_residue_id"] for d in displacements[:10] if d.get("depth_change")]

    directive = ViewportDirective(
        action=DirectiveAction.COMPARE_RUNS,
        structure_id=wt_structure_id,
        highlight_groups=[
            HighlightGroup(
                residue_ids=top_movers,
                color="#ff00ff",
                style=HighlightStyle.PULSE,
                label="High displacement",
            )
        ] if top_movers else [],
        parameters={
            "wt": wt_structure_id,
            "mutant": mutant_structure_id,
        },
        message=f"Comparing {wt_structure_id} vs {mutant_structure_id}",
    )

    return ToolResult(
        success=True,
        data={
            "wt_structure_id": wt_structure_id,
            "mutant_structure_id": mutant_structure_id,
            "displacements": displacements[:20],
            "total_matched": len(displacements),
            "top_movers": top_movers,
        },
        message=f"Compared {len(displacements)} residues between WT and mutant",
        viewport_directives=[directive],
    )
