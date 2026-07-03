"""Data Tools API router — REST endpoints for data access and export tools.

Exposes 8 endpoints for residue search, allosteric sites, provenance,
export, annotations, run summary, and run comparison.
Wraps the existing tool functions in agent/tools/data_tools.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.coordinator.deps import get_db
from science.dtie.common.keys import validate_structure_id
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


def _reject_invalid_structure_id(structure_id: str) -> JSONResponse | None:
    if not validate_structure_id(structure_id):
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_structure_id",
                "message": f"Invalid structure_id format: {structure_id}",
            },
        )
    return None


def _safe_export_path(export_dir: Path, filename: str) -> Path | None:
    """Resolve an export path and ensure it stays within export_dir."""
    file_path = (export_dir / filename).resolve()
    try:
        file_path.relative_to(export_dir.resolve())
    except ValueError:
        return None
    return file_path


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class ResidueSearchRequest(BaseModel):
    chain: str | None = Field(None, description="Filter by chain label (e.g., 'A')")
    residue_name: str | None = Field(None, description="Filter by residue name (e.g., 'ALA')")
    min_uncertainty: float | None = Field(None, description="Minimum uncertainty threshold")
    max_uncertainty: float | None = Field(None, description="Maximum uncertainty threshold")
    min_depth: float | None = Field(None, description="Minimum cone depth")
    max_depth: float | None = Field(None, description="Maximum cone depth")
    uncertainty_type: str = Field("epistemic", description="Uncertainty type: epistemic, aleatoric, total")
    limit: int = Field(100, description="Max results to return", ge=1, le=1000)


class ExportRequest(BaseModel):
    format: str = Field("csv", description="Export format: csv or json")
    include_fields: list[str] | None = Field(None, description="Specific fields to include (None = all)")


class AnnotationCreateRequest(BaseModel):
    residue_ids: list[str] | None = Field(None, description="Specific residues (None = structure-level)")
    annotation: str = Field(..., description="Annotation text", min_length=1)
    annotation_type: str = Field("finding", description="Type: finding, hypothesis, note, warning")


class RunCompareRequest(BaseModel):
    run_id_a: str = Field(..., description="First run ID")
    run_id_b: str = Field(..., description="Second run ID")


# ---------------------------------------------------------------------------
# POST /api/data/{structure_id}/search-residues
# ---------------------------------------------------------------------------


@router.post("/{structure_id}/search-residues")
async def search_residues_endpoint(structure_id: str, request: ResidueSearchRequest, db=Depends(get_db)):
    """Search and filter residues with flexible criteria.

    Supports filtering by chain, residue name, uncertainty range,
    and cone depth range.
    """
    from agent.tools.data_tools import search_residues

    result = await search_residues(
        structure_id=structure_id,
        chain=request.chain,
        residue_name=request.residue_name,
        min_uncertainty=request.min_uncertainty,
        max_uncertainty=request.max_uncertainty,
        min_depth=request.min_depth,
        max_depth=request.max_depth,
        uncertainty_type=request.uncertainty_type,
        limit=request.limit,
        db=db,
    )

    if not result.success:
        return JSONResponse(status_code=400, content={"error": "search_failed", "message": result.message})

    return {
        "structure_id": result.data.get("structure_id"),
        "residues": result.data.get("residues", []),
        "count": result.data.get("count", 0),
        "filters_applied": result.data.get("filters_applied"),
    }


# ---------------------------------------------------------------------------
# GET /api/data/{structure_id}/allosteric-sites
# ---------------------------------------------------------------------------


@router.get("/{structure_id}/allosteric-sites")
async def allosteric_sites_endpoint(structure_id: str, db=Depends(get_db)):
    """Retrieve allosteric site clusters for a structure.

    Returns site membership, confidence scores, and residue lists.
    """
    from agent.tools.data_tools import get_allosteric_sites

    result = await get_allosteric_sites(structure_id=structure_id, db=db)

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "no_data", "message": result.message})

    return {
        "structure_id": result.data.get("structure_id"),
        "sites": result.data.get("sites", []),
        "count": result.data.get("count", 0),
        "total_residues": result.data.get("total_residues", 0),
    }


# ---------------------------------------------------------------------------
# GET /api/data/{structure_id}/provenance
# ---------------------------------------------------------------------------


@router.get("/{structure_id}/provenance")
async def provenance_endpoint(structure_id: str, db=Depends(get_db)):
    """Provenance run history for a structure.

    Returns all pipeline runs associated with this structure.
    """
    from agent.tools.data_tools import get_provenance_lineage

    result = await get_provenance_lineage(structure_id=structure_id, db=db)

    if not result.success:
        return JSONResponse(status_code=400, content={"error": "query_failed", "message": result.message})

    return {
        "structure_id": structure_id,
        "runs": result.data.get("runs", []),
        "count": result.data.get("count", 0),
    }


# ---------------------------------------------------------------------------
# POST /api/data/{structure_id}/export
# ---------------------------------------------------------------------------


@router.post("/{structure_id}/export")
async def export_endpoint(structure_id: str, request: ExportRequest, db=Depends(get_db)):
    """Export pipeline results for a structure as CSV or JSON.

    Returns the file path and download URL for the generated export.
    """
    invalid = _reject_invalid_structure_id(structure_id)
    if invalid:
        return invalid

    from agent.tools.data_tools import export_structure_data

    result = await export_structure_data(
        structure_id=structure_id,
        format=request.format,
        include_fields=request.include_fields,
        db=db,
    )

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "no_data", "message": result.message})

    return {
        "structure_id": structure_id,
        "file_path": result.data.get("file_path"),
        "format": result.data.get("format"),
        "residue_count": result.data.get("residue_count", 0),
        "fields": result.data.get("fields", []),
        "download_url": f"/api/data/exports/{result.data.get('file_path', '').split('/')[-1]}",
    }


# ---------------------------------------------------------------------------
# GET /api/data/{structure_id}/annotations
# ---------------------------------------------------------------------------


@router.get("/{structure_id}/annotations")
async def list_annotations_endpoint(structure_id: str, db=Depends(get_db)):
    """List all annotations for a structure.

    Returns annotations in reverse chronological order.
    """
    from agent.tools.dtie.tools import ToolDB

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        """
        SELECT asset_id AS annotation_id, asset_type, structure_id,
               created_at, access_level
        FROM governed_asset
        WHERE structure_id = :structure_id
          AND asset_type LIKE 'annotation_%'
        ORDER BY created_at DESC
        """,
        {"structure_id": structure_id},
    )

    return {
        "structure_id": structure_id,
        "annotations": rows,
        "count": len(rows),
    }


# ---------------------------------------------------------------------------
# POST /api/data/{structure_id}/annotations
# ---------------------------------------------------------------------------


@router.post("/{structure_id}/annotations")
async def create_annotation_endpoint(structure_id: str, request: AnnotationCreateRequest, db=Depends(get_db)):
    """Add an annotation to a structure or specific residues.

    Persists through the governed asset catalog with provenance.
    """
    from agent.tools.data_tools import annotate_structure

    result = await annotate_structure(
        structure_id=structure_id,
        residue_ids=request.residue_ids,
        annotation=request.annotation,
        annotation_type=request.annotation_type,
        db=db,
    )

    if not result.success:
        return JSONResponse(status_code=400, content={"error": "annotation_failed", "message": result.message})

    return {
        "annotation_id": result.data.get("annotation_id"),
        "structure_id": result.data.get("structure_id"),
        "residue_ids": result.data.get("residue_ids"),
        "annotation": result.data.get("annotation"),
        "type": result.data.get("type"),
        "run_id": result.data.get("annotation_id"),  # provenance identifier
    }


# ---------------------------------------------------------------------------
# GET /api/data/runs/{run_id}/summary
# ---------------------------------------------------------------------------


@router.get("/runs/{run_id}/summary")
async def run_summary_endpoint(run_id: str, db=Depends(get_db)):
    """Summarize a pipeline run: phases, timing, assets, errors.

    Returns detailed information about what a run produced.
    """
    from agent.tools.data_tools import get_run_summary

    result = await get_run_summary(run_id=run_id, db=db)

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "run_not_found", "message": result.message})

    return {
        "run": result.data.get("run"),
        "assets_created": result.data.get("assets_created", 0),
        "audit_records": result.data.get("audit_records", []),
        "total_duration_ms": result.data.get("total_duration_ms", 0),
        "error_count": result.data.get("error_count", 0),
        "errors": result.data.get("errors", []),
    }


# ---------------------------------------------------------------------------
# GET /api/structures/{structure_id}/export — Comprehensive Data Inspector export
# ---------------------------------------------------------------------------


class ComprehensiveExportRequest(BaseModel):
    format: str = Field("csv", description="Export format: csv or json")


@router.get("/{structure_id}/export")
async def comprehensive_export(structure_id: str, format: str = "csv", db=Depends(get_db)):
    """Export ALL computed data for a structure, merged per-residue.

    Merges: embeddings + graph_metrics + source_leaks + resistance into
    a per-residue CSV/JSON. Includes separate sections for pockets and
    candidates. Includes metadata header.

    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    invalid = _reject_invalid_structure_id(structure_id)
    if invalid:
        return invalid

    from agent.tools.dtie.tools import ToolDB

    if format not in ("csv", "json"):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_format", "message": "Format must be 'csv' or 'json'"},
        )

    tool_db = ToolDB(db)

    # Fetch embeddings
    embeddings = await tool_db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, r.residue_name,
               c.chain_label,
               e.cone_depth, e.epistemic_uncertainty, e.aleatoric_uncertainty
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

    # Fetch graph metrics
    graph_metrics = await tool_db.fetch_all(
        """
        SELECT m.residue_id,
               MAX(CASE WHEN m.metric_type = 'betweenness' THEN m.metric_value END) AS betweenness,
               MAX(CASE WHEN m.metric_type = 'degree' THEN m.metric_value END) AS degree,
               MAX(CASE WHEN m.metric_type = 'clustering_coefficient' THEN m.metric_value END)
                   AS clustering_coefficient,
               MAX(CASE WHEN m.metric_type = 'closeness' THEN m.metric_value END) AS closeness,
               MAX(CASE WHEN m.metric_type = 'eigenvector_centrality' THEN m.metric_value END)
                   AS eigenvector_centrality,
               MAX(CASE WHEN m.metric_type = 'is_bridge' THEN m.metric_value END) AS is_bridge
        FROM fact_graph_metric m
        WHERE m.structure_id = :structure_id
        GROUP BY m.residue_id
        """,
        {"structure_id": structure_id},
    )

    # Fetch source leaks
    source_leaks = await tool_db.fetch_all(
        """
        SELECT residue_id, leak_score
        FROM fact_source_leak
        WHERE structure_id = :structure_id
        """,
        {"structure_id": structure_id},
    )

    # Fetch resistance (Phase 4 derived)
    resistance = await tool_db.fetch_all(
        """
        SELECT residue_id, sensitivity_score, classification, coupling_count, is_hinge
        FROM v_agent_resistance_sensitivity
        WHERE structure_id = :structure_id
        """,
        {"structure_id": structure_id},
    )

    # Fetch Phase 5 pharmacophore pockets
    pockets = await tool_db.fetch_all(
        """
        SELECT pocket_index, druggability_score, residue_count,
               volume_estimate, allosteric_coupling,
               center_x, center_y, center_z
        FROM fact_pharmacophore
        WHERE structure_id = :structure_id
        ORDER BY druggability_score DESC
        """,
        {"structure_id": structure_id},
    )

    # Fetch Phase 6 drug candidates
    candidates = await tool_db.fetch_all(
        """
        SELECT pocket_index, combined_druggability, accessibility_score,
               binding_potential, admet_pass, selectivity_ratio, is_state_selective
        FROM fact_drug_candidate
        WHERE structure_id = :structure_id
        ORDER BY combined_druggability DESC
        """,
        {"structure_id": structure_id},
    )

    if not embeddings and not graph_metrics and not source_leaks and not resistance:
        return JSONResponse(
            status_code=404,
            content={"error": "no_data", "message": f"No computed data for '{structure_id}'"},
        )

    # --- Merge per-residue ---
    merged: dict[str, dict[str, Any]] = {}

    for row in embeddings:
        rid = row["residue_id"]
        merged[rid] = {
            "residue_id": rid,
            "residue_index": row.get("residue_index"),
            "residue_name": row.get("residue_name"),
            "chain_label": row.get("chain_label"),
            "cone_depth": row.get("cone_depth"),
            "epistemic_uncertainty": row.get("epistemic_uncertainty"),
            "aleatoric_uncertainty": row.get("aleatoric_uncertainty"),
            "betweenness": None,
            "degree": None,
            "clustering_coefficient": None,
            "closeness": None,
            "eigenvector_centrality": None,
            "is_bridge": None,
            "leak_score": None,
            "sensitivity_score": None,
            "classification": None,
            "coupling_count": None,
            "is_hinge": None,
        }

    for row in graph_metrics:
        rid = row["residue_id"]
        if rid not in merged:
            merged[rid] = {"residue_id": rid}
        merged[rid]["betweenness"] = row.get("betweenness")
        merged[rid]["degree"] = row.get("degree")
        merged[rid]["clustering_coefficient"] = row.get("clustering_coefficient")
        merged[rid]["closeness"] = row.get("closeness")
        merged[rid]["eigenvector_centrality"] = row.get("eigenvector_centrality")
        merged[rid]["is_bridge"] = row.get("is_bridge")

    for row in source_leaks:
        rid = row["residue_id"]
        if rid not in merged:
            merged[rid] = {"residue_id": rid}
        merged[rid]["leak_score"] = row.get("leak_score")

    for row in resistance:
        rid = row["residue_id"]
        if rid not in merged:
            merged[rid] = {"residue_id": rid}
        merged[rid]["sensitivity_score"] = row.get("sensitivity_score")
        merged[rid]["classification"] = row.get("classification")
        merged[rid]["coupling_count"] = row.get("coupling_count")
        merged[rid]["is_hinge"] = row.get("is_hinge")

    # Sort by residue_index (nulls last)
    merged_rows = sorted(
        merged.values(),
        key=lambda r: (r.get("residue_index") is None, r.get("residue_index") or 0),
    )

    # --- Metadata ---
    metadata = {
        "structure_id": structure_id,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v5",
        "phases_included": {
            "embeddings": len(embeddings) > 0,
            "graph_metrics": len(graph_metrics) > 0,
            "source_leaks": len(source_leaks) > 0,
            "resistance": len(resistance) > 0,
            "phase5_pockets": len(pockets) > 0,
            "phase6_candidates": len(candidates) > 0,
        },
        "total_residues": len(merged_rows),
    }

    # --- Generate output ---
    import csv as csv_mod
    import io as io_mod
    import json as json_mod
    import uuid as uuid_mod

    export_dir = Path(os.getenv("EXPORT_OUTPUT_DIR", "./data/local_objects/exports"))
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{structure_id}_full_export_{uuid_mod.uuid4().hex[:8]}.{format}"
    file_path = _safe_export_path(export_dir, filename)
    if file_path is None:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_structure_id", "message": "Export path rejected"},
        )

    if format == "json":
        output = {
            "metadata": metadata,
            "residues": merged_rows,
            "pockets": pockets if pockets else [],
            "candidates": candidates if candidates else [],
        }
        with open(file_path, "w") as f:
            json_mod.dump(output, f, indent=2, default=str)
    else:
        # CSV: main residue file + separate pocket/candidate sections via multi-section
        with open(file_path, "w", newline="") as f:
            # Metadata comment header
            f.write(f"# structure_id: {structure_id}\n")
            f.write(f"# export_timestamp: {metadata['export_timestamp']}\n")
            f.write(f"# pipeline_version: v5\n")
            f.write(f"# total_residues: {len(merged_rows)}\n")
            f.write(f"# pockets: {len(pockets)}\n")
            f.write(f"# candidates: {len(candidates)}\n")
            f.write("#\n")

            # Residue table
            if merged_rows:
                fieldnames = list(merged_rows[0].keys())
                writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

            # Pockets section
            if pockets:
                f.write("\n# --- POCKETS (Phase 5) ---\n")
                pocket_fields = list(pockets[0].keys())
                writer = csv_mod.DictWriter(f, fieldnames=pocket_fields)
                writer.writeheader()
                writer.writerows(pockets)

            # Candidates section
            if candidates:
                f.write("\n# --- DRUG CANDIDATES (Phase 6) ---\n")
                candidate_fields = list(candidates[0].keys())
                writer = csv_mod.DictWriter(f, fieldnames=candidate_fields)
                writer.writeheader()
                writer.writerows(candidates)

    return {
        "structure_id": structure_id,
        "format": format,
        "file_path": str(file_path),
        "download_url": f"/api/data/exports/{filename}",
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# POST /api/data/runs/compare
# ---------------------------------------------------------------------------


@router.post("/runs/compare")
async def compare_runs_endpoint(request: RunCompareRequest, db=Depends(get_db)):
    """Compare two pipeline runs on the same structure.

    Returns per-residue deltas in cone depth and uncertainty.
    """
    from agent.tools.data_tools import compare_runs

    result = await compare_runs(
        run_id_a=request.run_id_a,
        run_id_b=request.run_id_b,
        db=db,
    )

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "comparison_failed", "message": result.message})

    return {
        "run_id_a": result.data.get("run_id_a"),
        "run_id_b": result.data.get("run_id_b"),
        "model_a": result.data.get("model_a"),
        "model_b": result.data.get("model_b"),
        "matched_residues": result.data.get("matched_residues", 0),
        "top_differences": result.data.get("top_differences", []),
        "avg_depth_delta": result.data.get("avg_depth_delta", 0.0),
        "max_depth_delta": result.data.get("max_depth_delta", 0.0),
    }
