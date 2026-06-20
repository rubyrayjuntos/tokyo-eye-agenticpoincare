"""Cryptic site query and validation tools for the agent.

Provides instant query-time retrieval of pre-computed binding sites
from the fact_cryptic_site table. Sites are pre-computed during the
scan phase (Phase 3.5) at ingestion time — queries are simple DB lookups.

Also provides on-demand MD validation: dispatch real steered MD for
specific pre-computed sites without re-running the full scan.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 8.1, 8.2, 8.3, 8.4
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from agent.models.viewport import (
    DirectiveAction,
    HighlightGroup,
    HighlightStyle,
    ViewportDirective,
)
from agent.tools.dtie.tools import ToolDB, ToolResult

logger = logging.getLogger(__name__)

# Mapping from site_type to the appropriate SMD protocol
SITE_TYPE_TO_PROTOCOL: dict[str, str] = {
    "cryptic_wedge": "SMD_three_phase",
    "structural_stent": "SMD_stent_stabilization",
    "dynamic_lid": "SMD_lid_restraint",
    "allosteric_clamp": "SMD_clamp_stabilization",
    "strain_relief_insert": "SMD_strain_relief",
    "surface_pocket": "SMD_three_phase",  # default for geometry-only pockets
}

# Valid MD validation status transitions
VALID_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"running"},
    "running": {"passed", "failed", "timeout"},
    "passed": {"pending"},
    "failed": {"pending"},
    "timeout": {"pending"},
}


async def query_binding_sites(
    structure_id: str,
    db: Any = None,
    site_type: str | None = None,
    min_druggability: float | None = None,
    md_status: str | None = None,
    limit: int = 50,
) -> ToolResult:
    """Query pre-computed binding sites for a structure.

    Returns instantly from fact_cryptic_site without triggering computation.
    Supports filtering by site_type, minimum druggability_score, and md_status.

    Args:
        structure_id: Structure to query binding sites for.
        db: Database adapter.
        site_type: Filter by site type (e.g., 'cryptic_wedge', 'surface_pocket').
        min_druggability: Minimum druggability_score threshold [0, 1].
        md_status: Filter by md_validation_status ('pending', 'running', 'passed', 'failed', 'timeout').
        limit: Maximum number of results to return.

    Returns:
        ToolResult with ranked site list or "pipeline not run" message.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # First check if a scan has been run for this structure
    scan_check = await tool_db.fetch_one(
        """
        SELECT scan_id, status, sites_found, heuristic_version, created_at
        FROM fact_binding_site_scan
        WHERE structure_id = :structure_id
        ORDER BY created_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )

    if not scan_check:
        return ToolResult(
            success=True,
            data={"structure_id": structure_id, "sites": [], "count": 0},
            message=(
                f"No binding site scan found for '{structure_id}'. "
                "Run the full DTIE pipeline first to generate pre-computed sites."
            ),
        )

    # Build dynamic query with optional filters
    conditions = ["structure_id = :structure_id"]
    params: dict[str, Any] = {"structure_id": structure_id, "limit": limit}

    if site_type is not None:
        conditions.append("site_type = :site_type")
        params["site_type"] = site_type

    if min_druggability is not None:
        conditions.append("druggability_score >= :min_druggability")
        params["min_druggability"] = min_druggability

    if md_status is not None:
        conditions.append("md_validation_status = :md_status")
        params["md_status"] = md_status

    where_clause = " AND ".join(conditions)

    rows = await tool_db.fetch_all(
        f"""
        SELECT site_id, site_type, residue_ids, druggability_score,
               site_rank, discovery_method, provenance_gate,
               md_validation_status, heuristic_version,
               composite_gnn_score, fpocket_druggability, volume_angstrom3
        FROM fact_cryptic_site
        WHERE {where_clause}
        ORDER BY site_rank ASC NULLS LAST
        LIMIT :limit
        """,
        params,
    )

    # Collect all residue IDs for viewport highlighting
    all_residue_ids: list[str] = []
    for row in rows:
        residue_ids = row.get("residue_ids")
        if isinstance(residue_ids, list):
            all_residue_ids.extend(residue_ids)

    # Build filter description for message
    filters = []
    if site_type is not None:
        filters.append(f"type={site_type}")
    if min_druggability is not None:
        filters.append(f"druggability≥{min_druggability}")
    if md_status is not None:
        filters.append(f"md_status={md_status}")
    filter_desc = ", ".join(filters) if filters else "none"

    # Build viewport directive to highlight binding site residues
    directive = ViewportDirective(
        action=DirectiveAction.HIGHLIGHT,
        structure_id=structure_id,
        highlight_groups=[
            HighlightGroup(
                residue_ids=all_residue_ids,
                color="#e056fd",
                style=HighlightStyle.GLOW,
                label="Binding Sites",
            )
        ] if all_residue_ids else [],
        message=f"Showing {len(rows)} binding sites for {structure_id}",
    )

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "sites": rows,
            "count": len(rows),
            "filters_applied": filter_desc,
            "scan_heuristic_version": scan_check.get("heuristic_version"),
        },
        message=(
            f"Found {len(rows)} binding sites for {structure_id} "
            f"(filters: {filter_desc})"
        ),
        viewport_directives=[directive],
    )


async def _update_md_status(
    site_id: str,
    new_status: str,
    db: Any,
) -> bool:
    """Update md_validation_status for a site with transition validation.

    Validates the transition against the allowed state machine:
    pending→running, running→passed/failed/timeout, passed/failed→pending.

    Returns True if update succeeded, False if transition is invalid.
    """
    tool_db = ToolDB(db)
    row = await tool_db.fetch_one(
        "SELECT md_validation_status FROM fact_cryptic_site WHERE site_id = :site_id",
        {"site_id": site_id},
    )
    if row is None:
        return False

    current_status = row.get("md_validation_status", "pending")
    allowed = VALID_STATUS_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        logger.warning(
            "Invalid MD status transition for site %s: %s → %s",
            site_id, current_status, new_status,
        )
        return False

    await db.execute(
        "UPDATE fact_cryptic_site SET md_validation_status = :status WHERE site_id = :site_id",
        {"status": new_status, "site_id": site_id},
    )
    return True


def _write_site_spec_json(
    site_id: str,
    structure_id: str,
    residue_ids: list[str],
    output_dir: Path,
) -> str:
    """Write a spec JSON file for dispatching SMD validation.

    Returns path to the created spec file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "structure_id": structure_id,
        "site_id": site_id,
        "residue_ids": residue_ids,
        "pdb_path": f"data/structures/{structure_id}.pdb",
        "pulling_direction": [1.0, 0.0, 0.0],
    }
    spec_path = output_dir / f"spec_{site_id}.json"
    with spec_path.open("w") as f:
        json.dump(spec, f, indent=2)
    return str(spec_path)


async def validate_site_md(
    site_id: str,
    db: Any = None,
    timeout_seconds: int = 1800,
) -> ToolResult:
    """Trigger MD validation for a specific pre-computed site.

    Dispatches real SMD to the science container for the site, updates
    md_validation_status through the state machine:
    pending → running → passed/failed/timeout.

    The appropriate SMD protocol is selected based on the site's site_type.

    Args:
        site_id: The site_id of the pre-computed binding site to validate.
        db: Database adapter.
        timeout_seconds: Max seconds to wait for SMD completion (default 1800).

    Returns:
        ToolResult with MD validation outcome and updated status.

    Requirements: 8.1, 8.2, 8.3
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Fetch the site record
    site = await tool_db.fetch_one(
        """
        SELECT site_id, structure_id, site_type, residue_ids,
               md_validation_status, druggability_score, site_rank
        FROM fact_cryptic_site
        WHERE site_id = :site_id
        """,
        {"site_id": site_id},
    )

    if site is None:
        return ToolResult(
            success=False,
            message=f"Site '{site_id}' not found in fact_cryptic_site.",
        )

    current_status = site.get("md_validation_status", "pending")
    if current_status == "running":
        return ToolResult(
            success=False,
            message=f"Site '{site_id}' already has MD validation in progress.",
        )

    # Determine protocol from site_type
    site_type = site.get("site_type", "surface_pocket")
    protocol = SITE_TYPE_TO_PROTOCOL.get(site_type, "SMD_three_phase")

    # Transition to "running"
    transitioned = await _update_md_status(site_id, "running", db)
    if not transitioned:
        return ToolResult(
            success=False,
            message=(
                f"Cannot start MD validation for site '{site_id}': "
                f"invalid status transition from '{current_status}' to 'running'."
            ),
        )

    # Write spec JSON
    structure_id = site["structure_id"]
    residue_ids = site.get("residue_ids", [])
    if isinstance(residue_ids, str):
        # Handle case where residue_ids stored as JSON string
        residue_ids = json.loads(residue_ids)

    tmp_dir = Path(tempfile.mkdtemp(prefix="md_validation_"))
    spec_path = _write_site_spec_json(site_id, structure_id, residue_ids, tmp_dir)

    # Dispatch SMD
    from agent.tools.cryptic.smd_dispatch import dispatch_smd_job

    try:
        result = await dispatch_smd_job(
            spec_json_path=spec_path,
            protocol=protocol,
            structure_id=structure_id,
            timeout_seconds=timeout_seconds,
        )
    except Exception as e:
        # On dispatch failure, transition to failed
        await _update_md_status(site_id, "failed", db)
        logger.error("SMD dispatch failed for site %s: %s", site_id, e)
        return ToolResult(
            success=False,
            message=f"MD validation dispatch failed for site '{site_id}': {e}",
            warnings=[str(e)],
        )

    # Determine final status from SMD result
    smd_status = result.get("status", "failed")
    if smd_status == "timeout":
        final_status = "timeout"
    elif result.get("success"):
        final_status = "passed"
    else:
        final_status = "failed"

    # Transition to final status
    await _update_md_status(site_id, final_status, db)

    # Build response data
    data = {
        "site_id": site_id,
        "structure_id": structure_id,
        "site_type": site_type,
        "protocol": protocol,
        "md_validation_status": final_status,
        "work_kcal_mol": result.get("work_kcal_mol", 0.0),
        "strain_delta": result.get("strain_delta", 0.0),
        "duration_ms": result.get("duration_ms", 0),
    }

    # Surface stub-mode warning so user knows results are synthetic
    from science.dtie.cryptic.stub_control import is_stub_mode

    warnings: list[str] = []
    if is_stub_mode():
        warnings.append(
            "⚠️ STUB MODE: MD results are SYNTHETIC — no real physics was performed."
        )

    return ToolResult(
        success=True,
        data=data,
        message=(
            f"MD validation for site '{site_id}' ({site_type}): {final_status}. "
            f"Protocol: {protocol}, work={result.get('work_kcal_mol', 0.0):.2f} kcal/mol."
        ),
        warnings=warnings if warnings else None,
    )


async def validate_top_sites_md(
    structure_id: str,
    top_n: int = 5,
    db: Any = None,
    timeout_seconds: int = 1800,
) -> ToolResult:
    """Batch MD validation for top N sites by rank.

    Dispatches real SMD for the top N ranked binding sites of a structure.
    Each site is validated independently using its site_type-appropriate protocol.

    Args:
        structure_id: Structure to validate top sites for.
        top_n: Number of top-ranked sites to validate (default 5).
        db: Database adapter.
        timeout_seconds: Max seconds to wait per site (default 1800).

    Returns:
        ToolResult with batch validation summary.

    Requirements: 8.4
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Fetch top N sites by rank that are eligible for validation (pending or previously failed)
    rows = await tool_db.fetch_all(
        """
        SELECT site_id, site_type, site_rank, md_validation_status
        FROM fact_cryptic_site
        WHERE structure_id = :structure_id
          AND md_validation_status IN ('pending', 'failed', 'timeout')
        ORDER BY site_rank ASC NULLS LAST
        LIMIT :limit
        """,
        {"structure_id": structure_id, "limit": top_n},
    )

    if not rows:
        return ToolResult(
            success=True,
            data={"structure_id": structure_id, "validated": 0, "results": []},
            message=(
                f"No sites eligible for MD validation in '{structure_id}'. "
                "All sites may already be validated or currently running."
            ),
        )

    # Validate each site sequentially
    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    timed_out = 0

    for row in rows:
        site_id = row["site_id"]
        site_result = await validate_site_md(
            site_id=site_id,
            db=db,
            timeout_seconds=timeout_seconds,
        )

        status = site_result.data.get("md_validation_status", "failed") if site_result.data else "failed"
        results.append({
            "site_id": site_id,
            "site_type": row.get("site_type"),
            "site_rank": row.get("site_rank"),
            "md_validation_status": status,
            "work_kcal_mol": site_result.data.get("work_kcal_mol", 0.0) if site_result.data else 0.0,
        })

        if status == "passed":
            passed += 1
        elif status == "timeout":
            timed_out += 1
        else:
            failed += 1

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "validated": len(results),
            "passed": passed,
            "failed": failed,
            "timed_out": timed_out,
            "results": results,
        },
        message=(
            f"Batch MD validation for top {len(results)} sites of '{structure_id}': "
            f"{passed} passed, {failed} failed, {timed_out} timed out."
        ),
    )
