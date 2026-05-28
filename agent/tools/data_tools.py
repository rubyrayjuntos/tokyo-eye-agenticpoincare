"""Data access and export tools for the agent.

These tools provide the agent with capabilities to:
- Export pipeline results in various formats (CSV, JSON)
- Search and filter residues with flexible criteria
- List analyzed structures
- Retrieve allosteric site information
- Query provenance lineage
- Summarize pipeline runs
- Annotate structures with findings
- Compare pipeline runs

All reads go through the governed views/fact tables.
All writes go through the Normalizer or governed_asset catalog.
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.models.viewport import (
    DirectiveAction,
    HighlightGroup,
    HighlightStyle,
    ViewportDirective,
)
from agent.tools.dtie.tools import ToolDB, ToolResult

EXPORT_OUTPUT_DIR = Path(os.getenv("EXPORT_OUTPUT_DIR", "./data/local_objects/exports"))


# ---------------------------------------------------------------------------
# Tool 1: export_structure_data
# ---------------------------------------------------------------------------


async def export_structure_data(
    structure_id: str,
    format: str = "csv",
    include_fields: list[str] | None = None,
    db: Any = None,
) -> ToolResult:
    """Export pipeline results for a structure as CSV or JSON.

    Exports per-residue data including embeddings, uncertainty, cone depth,
    and site membership. Useful for external tools (PyMOL, ChimeraX) or
    for inclusion in publications.

    Args:
        structure_id: Structure to export.
        format: 'csv' or 'json'.
        include_fields: Specific fields to include (None = all available).
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    valid_formats = {"csv", "json"}
    if format not in valid_formats:
        return ToolResult(success=False, message=f"Invalid format '{format}'. Use: {valid_formats}")

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, r.residue_name,
               c.chain_label,
               e.cone_depth, e.cone_width,
               e.epistemic_uncertainty, e.aleatoric_uncertainty, e.total_uncertainty,
               e.model_version, e.computed_at
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
        ORDER BY r.residue_index
        """,
        {"structure_id": structure_id},
    )

    if not rows:
        return ToolResult(success=False, message=f"No data found for '{structure_id}'")

    # Filter fields if requested
    if include_fields:
        rows = [{k: v for k, v in row.items() if k in include_fields} for row in rows]

    # Generate output
    EXPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{structure_id}_export_{uuid.uuid4().hex[:8]}.{format}"
    file_path = EXPORT_OUTPUT_DIR / filename

    if format == "csv":
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(file_path, "w") as f:
            json.dump({"structure_id": structure_id, "residues": rows}, f, indent=2, default=str)

    return ToolResult(
        success=True,
        data={
            "file_path": str(file_path),
            "format": format,
            "residue_count": len(rows),
            "fields": list(rows[0].keys()) if rows else [],
        },
        message=f"Exported {len(rows)} residues to {filename}",
    )


# ---------------------------------------------------------------------------
# Tool 2: get_allosteric_sites
# ---------------------------------------------------------------------------


async def get_allosteric_sites(
    structure_id: str,
    db: Any = None,
) -> ToolResult:
    """Retrieve identified allosteric site clusters for a structure.

    Returns site membership, contributing residues, and aggregate metrics
    for each identified allosteric site.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Get sites
    sites = await tool_db.fetch_all(
        """
        SELECT s.site_id, s.site_type, s.site_label, s.confidence_score,
               s.metadata
        FROM dim_site s
        WHERE s.structure_id = :structure_id
        ORDER BY s.confidence_score DESC NULLS LAST
        """,
        {"structure_id": structure_id},
    )

    if not sites:
        return ToolResult(
            success=True,
            data={"structure_id": structure_id, "sites": [], "count": 0},
            message=f"No allosteric sites identified for {structure_id}. Run the full pipeline first.",
        )

    # Get residue membership for each site
    site_details = []
    all_residue_ids = []
    for site in sites:
        members = await tool_db.fetch_all(
            """
            SELECT br.residue_id, r.residue_index, r.residue_name, c.chain_label
            FROM bridge_site_residue br
            JOIN dim_residue r ON r.residue_id = br.residue_id
            JOIN dim_chain c ON c.chain_id = r.chain_id
            WHERE br.site_id = :site_id
            ORDER BY r.residue_index
            """,
            {"site_id": site["site_id"]},
        )
        member_ids = [m["residue_id"] for m in members]
        all_residue_ids.extend(member_ids)
        site_details.append({
            **site,
            "residues": members,
            "residue_count": len(members),
        })

    directive = ViewportDirective(
        action=DirectiveAction.HIGHLIGHT,
        structure_id=structure_id,
        highlight_groups=[
            HighlightGroup(
                residue_ids=all_residue_ids,
                color="#9b59b6",
                style=HighlightStyle.GLOW,
                label="Allosteric Sites",
            )
        ] if all_residue_ids else [],
        message=f"Showing {len(sites)} allosteric sites",
    )

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "sites": site_details,
            "count": len(sites),
            "total_residues": len(all_residue_ids),
        },
        message=f"Found {len(sites)} allosteric sites ({len(all_residue_ids)} total residues)",
        viewport_directives=[directive],
    )


# ---------------------------------------------------------------------------
# Tool 3: get_provenance_lineage
# ---------------------------------------------------------------------------


async def get_provenance_lineage(
    run_id: str | None = None,
    structure_id: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Query provenance lineage for a run or structure.

    Shows what pipeline produced a result, which checkpoint was used,
    parent/child run relationships, and timing.

    Args:
        run_id: Specific run to query lineage for.
        structure_id: Get all runs for a structure.
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    if not run_id and not structure_id:
        return ToolResult(success=False, message="Provide either run_id or structure_id")

    tool_db = ToolDB(db)

    if run_id:
        # Get specific run + its parent chain
        runs = await tool_db.fetch_all(
            """
            WITH RECURSIVE lineage AS (
                SELECT run_id, structure_id, model_version, checkpoint_uri,
                       code_version, pipeline_name, run_type, source_type,
                       parent_run_id, started_at, 0 AS depth
                FROM provenance_run
                WHERE run_id = :run_id
                UNION ALL
                SELECT p.run_id, p.structure_id, p.model_version, p.checkpoint_uri,
                       p.code_version, p.pipeline_name, p.run_type, p.source_type,
                       p.parent_run_id, p.started_at, l.depth + 1
                FROM provenance_run p
                JOIN lineage l ON p.run_id = l.parent_run_id
                WHERE l.depth < 10
            )
            SELECT * FROM lineage ORDER BY depth
            """,
            {"run_id": run_id},
        )
    else:
        # Get all runs for a structure
        runs = await tool_db.fetch_all(
            """
            SELECT run_id, structure_id, model_version, checkpoint_uri,
                   code_version, pipeline_name, run_type, source_type,
                   parent_run_id, started_at
            FROM provenance_run
            WHERE structure_id = :structure_id
            ORDER BY started_at DESC
            LIMIT 50
            """,
            {"structure_id": structure_id},
        )

    if not runs:
        return ToolResult(
            success=True,
            data={"runs": [], "count": 0},
            message="No provenance records found",
        )

    return ToolResult(
        success=True,
        data={
            "runs": runs,
            "count": len(runs),
            "query": {"run_id": run_id, "structure_id": structure_id},
        },
        message=f"Found {len(runs)} provenance records",
    )


# ---------------------------------------------------------------------------
# Tool 4: search_residues
# ---------------------------------------------------------------------------


async def search_residues(
    structure_id: str,
    chain: str | None = None,
    residue_name: str | None = None,
    min_uncertainty: float | None = None,
    max_uncertainty: float | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    uncertainty_type: str = "epistemic",
    limit: int = 100,
    db: Any = None,
) -> ToolResult:
    """Search and filter residues with flexible criteria.

    Allows natural-language-style queries like "all glycines with
    uncertainty > 0.4 in chain A" to be translated into structured filters.

    Args:
        structure_id: Structure to search within.
        chain: Filter by chain label (e.g., 'A').
        residue_name: Filter by residue name (e.g., 'G', 'ALA').
        min_uncertainty: Minimum uncertainty threshold.
        max_uncertainty: Maximum uncertainty threshold.
        min_depth: Minimum cone depth.
        max_depth: Maximum cone depth.
        uncertainty_type: Which uncertainty to filter on (epistemic/aleatoric/total).
        limit: Max results to return.
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    # Validate uncertainty_type
    valid_types = {"epistemic", "aleatoric", "total"}
    if uncertainty_type not in valid_types:
        return ToolResult(success=False, message=f"Invalid uncertainty_type. Use: {valid_types}")

    uncertainty_col = f"{uncertainty_type}_uncertainty"

    # Build dynamic WHERE clauses
    conditions = ["e.structure_id = :structure_id", "es.space_type = 'hyperbolic'"]
    params: dict[str, Any] = {"structure_id": structure_id, "limit": limit}

    if chain:
        conditions.append("c.chain_label = :chain")
        params["chain"] = chain

    if residue_name:
        conditions.append("r.residue_name = :residue_name")
        params["residue_name"] = residue_name.upper()

    if min_uncertainty is not None:
        conditions.append(f"e.{uncertainty_col} >= :min_uncertainty")
        params["min_uncertainty"] = min_uncertainty

    if max_uncertainty is not None:
        conditions.append(f"e.{uncertainty_col} <= :max_uncertainty")
        params["max_uncertainty"] = max_uncertainty

    if min_depth is not None:
        conditions.append("e.cone_depth >= :min_depth")
        params["min_depth"] = min_depth

    if max_depth is not None:
        conditions.append("e.cone_depth <= :max_depth")
        params["max_depth"] = max_depth

    where_clause = " AND ".join(conditions)

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        f"""
        SELECT r.residue_id, r.residue_index, r.residue_name,
               c.chain_label,
               e.cone_depth, e.cone_width,
               e.epistemic_uncertainty, e.aleatoric_uncertainty, e.total_uncertainty
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE {where_clause}
        ORDER BY e.{uncertainty_col} DESC NULLS LAST
        LIMIT :limit
        """,
        params,
    )

    residue_ids = [r["residue_id"] for r in rows]

    # Build filter description for the message
    filters = []
    if chain:
        filters.append(f"chain={chain}")
    if residue_name:
        filters.append(f"name={residue_name}")
    if min_uncertainty is not None:
        filters.append(f"{uncertainty_type}≥{min_uncertainty}")
    if max_uncertainty is not None:
        filters.append(f"{uncertainty_type}≤{max_uncertainty}")
    if min_depth is not None:
        filters.append(f"depth≥{min_depth}")
    if max_depth is not None:
        filters.append(f"depth≤{max_depth}")
    filter_desc = ", ".join(filters) if filters else "no filters"

    directive = ViewportDirective(
        action=DirectiveAction.HIGHLIGHT,
        structure_id=structure_id,
        highlight_groups=[
            HighlightGroup(
                residue_ids=residue_ids,
                color="#f39c12",
                style=HighlightStyle.GLOW,
                label=f"Search: {filter_desc}",
            )
        ] if residue_ids else [],
        message=f"Found {len(rows)} residues matching: {filter_desc}",
    )

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "residues": rows,
            "count": len(rows),
            "filters_applied": filter_desc,
        },
        message=f"Found {len(rows)} residues matching: {filter_desc}",
        viewport_directives=[directive],
    )


# ---------------------------------------------------------------------------
# Tool 5: annotate_structure
# ---------------------------------------------------------------------------


async def annotate_structure(
    structure_id: str,
    residue_ids: list[str] | None = None,
    annotation: str = "",
    annotation_type: str = "finding",
    db: Any = None,
) -> ToolResult:
    """Add a text annotation to a structure or specific residues.

    Annotations are stored in the governed layer and can be retrieved
    later for reports or collaboration.

    Args:
        structure_id: Structure to annotate.
        residue_ids: Specific residues (None = structure-level annotation).
        annotation: The annotation text.
        annotation_type: Type of annotation (finding, hypothesis, note, warning).
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    if not annotation.strip():
        return ToolResult(success=False, message="Annotation text cannot be empty")

    valid_types = {"finding", "hypothesis", "note", "warning"}
    if annotation_type not in valid_types:
        return ToolResult(success=False, message=f"Invalid type. Use: {valid_types}")

    annotation_id = f"ann_{uuid.uuid4().hex[:12]}"

    await db.execute(
        """
        INSERT INTO governed_asset (
            asset_id, asset_type, structure_id, run_id, access_level, created_at
        ) VALUES (
            :asset_id, :asset_type, :structure_id, :run_id, :access_level, :created_at
        )
        ON CONFLICT (asset_id) DO NOTHING
        """,
        {
            "asset_id": annotation_id,
            "asset_type": f"annotation_{annotation_type}",
            "structure_id": structure_id,
            "run_id": annotation_id,  # Self-referencing for annotations
            "access_level": "internal",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    await db.commit()

    directive = ViewportDirective(
        action=DirectiveAction.ANNOTATE,
        structure_id=structure_id,
        highlight_groups=[
            HighlightGroup(
                residue_ids=residue_ids or [],
                color="#2ecc71",
                style=HighlightStyle.OUTLINE,
                label=annotation_type.title(),
            )
        ] if residue_ids else [],
        message=annotation,
    )

    return ToolResult(
        success=True,
        data={
            "annotation_id": annotation_id,
            "structure_id": structure_id,
            "residue_ids": residue_ids,
            "annotation": annotation,
            "type": annotation_type,
        },
        message=f"Annotation saved: '{annotation[:60]}...' " if len(annotation) > 60 else f"Annotation saved: '{annotation}'",
        viewport_directives=[directive],
    )


# ---------------------------------------------------------------------------
# Tool 6: list_structures
# ---------------------------------------------------------------------------


async def list_structures(
    db: Any = None,
) -> ToolResult:
    """List all structures that have been analyzed (have GNN embeddings).

    Returns structure IDs, model versions used, residue counts, and
    when they were last computed.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        """
        SELECT e.structure_id,
               e.model_version,
               COUNT(DISTINCT e.residue_id) AS residue_count,
               MAX(e.computed_at) AS last_computed,
               COUNT(DISTINCT e.run_id) AS run_count
        FROM fact_gnn_node_embedding e
        GROUP BY e.structure_id, e.model_version
        ORDER BY MAX(e.computed_at) DESC
        LIMIT 100
        """,
        {},
    )

    return ToolResult(
        success=True,
        data={
            "structures": rows,
            "count": len(rows),
        },
        message=f"Found {len(rows)} analyzed structures",
    )


# ---------------------------------------------------------------------------
# Tool 7: get_run_summary
# ---------------------------------------------------------------------------


async def get_run_summary(
    run_id: str,
    db: Any = None,
) -> ToolResult:
    """Summarize a pipeline run: what was computed, timing, warnings.

    Args:
        run_id: The pipeline run ID to summarize.
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Get provenance record
    run = await tool_db.fetch_one(
        """
        SELECT run_id, structure_id, model_version, checkpoint_uri,
               code_version, pipeline_name, run_type, source_type,
               parameters, parent_run_id, started_at
        FROM provenance_run
        WHERE run_id = :run_id
        """,
        {"run_id": run_id},
    )

    if not run:
        return ToolResult(success=False, message=f"No run found with ID '{run_id}'")

    # Get asset count
    assets = await tool_db.fetch_one(
        """
        SELECT COUNT(*) AS asset_count
        FROM governed_asset
        WHERE run_id = :run_id
        """,
        {"run_id": run_id},
    )

    # Get audit records
    audits = await tool_db.fetch_all(
        """
        SELECT payload_type, status, assets_created, duration_ms, error_message
        FROM normalization_audit
        WHERE run_id = :run_id
        ORDER BY duration_ms DESC
        """,
        {"run_id": run_id},
    )

    total_duration = sum(a.get("duration_ms") or 0 for a in audits)
    errors = [a for a in audits if a.get("status") == "write_error"]

    return ToolResult(
        success=True,
        data={
            "run": run,
            "assets_created": assets["asset_count"] if assets else 0,
            "audit_records": audits,
            "total_duration_ms": total_duration,
            "error_count": len(errors),
            "errors": errors,
        },
        message=(
            f"Run {run_id}: {run['pipeline_name']} on {run['structure_id']} "
            f"({run['model_version']}), {assets['asset_count'] if assets else 0} assets, "
            f"{total_duration}ms total"
        ),
    )


# ---------------------------------------------------------------------------
# Tool 8: compare_runs
# ---------------------------------------------------------------------------


async def compare_runs(
    run_id_a: str,
    run_id_b: str,
    db: Any = None,
) -> ToolResult:
    """Compare two pipeline runs on the same structure.

    Useful for comparing different model versions, checkpoints, or
    parameter settings. Shows per-residue differences in cone depth
    and uncertainty.

    Args:
        run_id_a: First run ID.
        run_id_b: Second run ID.
        db: Database adapter.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Get embeddings from both runs
    rows_a = await tool_db.fetch_all(
        """
        SELECT e.residue_id, r.residue_index, c.chain_label,
               e.cone_depth, e.epistemic_uncertainty, e.model_version
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE e.run_id = :run_id
        ORDER BY r.residue_index
        """,
        {"run_id": run_id_a},
    )

    rows_b = await tool_db.fetch_all(
        """
        SELECT e.residue_id, r.residue_index, c.chain_label,
               e.cone_depth, e.epistemic_uncertainty, e.model_version
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE e.run_id = :run_id
        ORDER BY r.residue_index
        """,
        {"run_id": run_id_b},
    )

    if not rows_a:
        return ToolResult(success=False, message=f"No data for run_id '{run_id_a}'")
    if not rows_b:
        return ToolResult(success=False, message=f"No data for run_id '{run_id_b}'")

    # Match by residue_id
    map_a = {r["residue_id"]: r for r in rows_a}
    map_b = {r["residue_id"]: r for r in rows_b}

    common_ids = set(map_a.keys()) & set(map_b.keys())
    diffs = []
    for rid in sorted(common_ids, key=lambda x: map_a[x]["residue_index"]):
        a = map_a[rid]
        b = map_b[rid]
        depth_a = a.get("cone_depth") or 0
        depth_b = b.get("cone_depth") or 0
        unc_a = a.get("epistemic_uncertainty") or 0
        unc_b = b.get("epistemic_uncertainty") or 0
        diffs.append({
            "residue_id": rid,
            "residue_index": a["residue_index"],
            "chain_label": a["chain_label"],
            "depth_a": depth_a,
            "depth_b": depth_b,
            "depth_delta": abs(depth_b - depth_a),
            "uncertainty_a": unc_a,
            "uncertainty_b": unc_b,
            "uncertainty_delta": abs(unc_b - unc_a),
        })

    # Sort by largest depth change
    diffs.sort(key=lambda d: d["depth_delta"], reverse=True)

    # Highlight top movers
    top_movers = [d["residue_id"] for d in diffs[:15] if d["depth_delta"] > 0.1]

    directive = ViewportDirective(
        action=DirectiveAction.COMPARE_RUNS,
        structure_id=rows_a[0].get("structure_id") if rows_a else None,
        highlight_groups=[
            HighlightGroup(
                residue_ids=top_movers,
                color="#e74c3c",
                style=HighlightStyle.PULSE,
                label="High Δ between runs",
            )
        ] if top_movers else [],
        parameters={"run_a": run_id_a, "run_b": run_id_b},
        message=f"Comparing {run_id_a} vs {run_id_b}",
    )

    # Summary stats
    avg_depth_delta = sum(d["depth_delta"] for d in diffs) / len(diffs) if diffs else 0
    max_depth_delta = diffs[0]["depth_delta"] if diffs else 0

    return ToolResult(
        success=True,
        data={
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "model_a": rows_a[0].get("model_version") if rows_a else None,
            "model_b": rows_b[0].get("model_version") if rows_b else None,
            "matched_residues": len(diffs),
            "top_differences": diffs[:20],
            "avg_depth_delta": round(avg_depth_delta, 4),
            "max_depth_delta": round(max_depth_delta, 4),
        },
        message=(
            f"Compared {len(diffs)} residues between runs. "
            f"Avg Δdepth={avg_depth_delta:.3f}, max Δdepth={max_depth_delta:.3f}"
        ),
        viewport_directives=[directive],
    )
