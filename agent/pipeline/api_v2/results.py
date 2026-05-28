# backend/gosp/api_v2/results.py
"""Result read endpoints with source_type provenance filter (Task 20)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncConnection

from gosp.api_v2.structures import get_db_conn

router = APIRouter(prefix="/structures", tags=["results"])

SourceType = Literal["empirical", "deterministic", "probabilistic", "external"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _require_structure(conn: AsyncConnection, structure_id: str) -> None:
    """Raise 404 if structure_id does not exist in dim_structure.

    NOTE: For endpoints that return result rows, prefer checking if the
    main query returns empty and 404-ing on that — avoids a second round-trip.
    Use this helper only for endpoints that need the existence check before
    a write or side-effect.
    """
    exists = (
        await conn.execute(
            sa.text("SELECT 1 FROM dim_structure WHERE structure_id = :sid"),
            {"sid": structure_id},
        )
    ).fetchone()
    if exists is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")


def _source_clause(source_type: Optional[str]) -> str:
    """Return a SQL AND fragment for source_type filtering, or empty string."""
    if source_type is None:
        return ""
    return "AND source_type = :source_type"


def _parse_vector(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]

    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(item) for item in text.split(",")]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DehydronRow(BaseModel):
    dehydron_id: str
    donor_chain: str
    donor_residue_index: int
    acceptor_chain: str
    acceptor_residue_index: int
    wrapping_count: int
    is_dehydron: bool
    source_type: str


class VoidRow(BaseModel):
    void_id: str
    center_x: float
    center_y: float
    center_z: float
    volume: float
    point_count: int
    source_type: str


class EnergyRow(BaseModel):
    energy_id: str
    requested_force_field: str
    actual_method: str
    delta_g: float
    potential_energy: float
    solvation_term: float
    sasa: float
    converged: bool
    confidence: str
    source_type: str


class RedzoneRow(BaseModel):
    run_id: str
    run_type: str          # "immunogenicity" | "metabolism"
    flagged: bool
    warning: Optional[str]
    source_type: str


class ValidationRow(BaseModel):
    run_id: str
    status: str
    dehydron_count: int
    void_count: int
    glue_site_count: int


class GNNRow(BaseModel):
    gnn_run_id: str
    model_version: str
    num_nodes: int
    num_edges: int
    curvature_value: Optional[float]
    source_type: str


class GNNNodeRow(BaseModel):
    node_id: str
    residue_id: str
    input_rho: float
    input_tau_flag: float
    input_ss_type: float
    input_sasa: float
    projections: List[float]
    cone_depth: float
    cone_width: float
    expert_weights: List[float]
    epistemic_uncertainty: Optional[float]
    aleatoric_uncertainty: Optional[float]
    total_uncertainty: Optional[float]
    evidence_mu: Optional[float]
    evidence_nu: Optional[float]
    evidence_alpha: Optional[float]
    evidence_beta: Optional[float]
    projection_embedding: Optional[List[float]]
    routing_entropy: Optional[float]
    expert_winner: Optional[int]
    source_type: str


class FoldingRow(BaseModel):
    path_id: str
    method: str
    num_frames: int
    num_atoms: int
    gcs_uri: str
    source_type: str


class SynthesisRow(BaseModel):
    synthesis_id: str
    manufacturable: bool
    complexity_score: Optional[float]
    gc_content: Optional[float]
    aa_length: Optional[int]
    source_type: str


class AuditRow(BaseModel):
    event_id: str
    timestamp: str
    state_from: str
    state_to: str
    event_name: str
    user_id: str


class HdxRow(BaseModel):
    hdx_run_id: str
    r_squared: float
    spearman_rho: float
    p_value: float
    passed: bool
    num_residues: int
    source_type: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{structure_id}/dehydrons", response_model=Dict[str, Any])
async def get_dehydrons(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get dehydrons for a structure, optionally filtered by source_type."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT dehydron_id, donor_chain, donor_residue_index,
                       acceptor_chain, acceptor_residue_index,
                       wrapping_count, is_dehydron, source_type
                FROM   fact_dehydron
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY dehydron_id
                """
            ),
            params,
        )
    ).fetchall()

    return {
        "dehydrons": [
            DehydronRow(
                dehydron_id=r.dehydron_id,
                donor_chain=r.donor_chain,
                donor_residue_index=r.donor_residue_index,
                acceptor_chain=r.acceptor_chain,
                acceptor_residue_index=r.acceptor_residue_index,
                wrapping_count=r.wrapping_count,
                is_dehydron=r.is_dehydron,
                source_type=r.source_type,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/voids", response_model=Dict[str, Any])
async def get_voids(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get voids for a structure, optionally filtered by source_type."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT void_id, center_x, center_y, center_z,
                       volume, point_count, source_type
                FROM   fact_void
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY void_id
                """
            ),
            params,
        )
    ).fetchall()

    return {
        "voids": [
            VoidRow(
                void_id=r.void_id,
                center_x=r.center_x,
                center_y=r.center_y,
                center_z=r.center_z,
                volume=r.volume,
                point_count=r.point_count,
                source_type=r.source_type,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/energy", response_model=Dict[str, Any])
async def get_energy(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get energy calculations for a structure."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT energy_id, requested_force_field, actual_method,
                       delta_g, potential_energy, solvation_term, sasa,
                       converged, confidence, source_type
                FROM   fact_energy_calculation
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY computed_at DESC
                """
            ),
            params,
        )
    ).fetchall()

    return {
        "energy": [
            EnergyRow(
                energy_id=r.energy_id,
                requested_force_field=r.requested_force_field,
                actual_method=r.actual_method,
                delta_g=r.delta_g,
                potential_energy=r.potential_energy,
                solvation_term=r.solvation_term,
                sasa=r.sasa,
                converged=r.converged,
                confidence=r.confidence,
                source_type=r.source_type,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/redzone", response_model=Dict[str, Any])
async def get_redzone(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get red-zone screening results (immunogenicity + metabolism)."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    immuno_rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT immuno_run_id AS run_id, flagged_count > 0 AS flagged,
                       warning, source_type
                FROM   fact_immunogenicity_run
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY computed_at DESC
                """
            ),
            params,
        )
    ).fetchall()

    meta_rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT metabolism_run_id AS run_id, flagged, warning, source_type
                FROM   fact_metabolism_run
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY computed_at DESC
                """
            ),
            params,
        )
    ).fetchall()

    results = []
    for r in immuno_rows:
        results.append(
            RedzoneRow(
                run_id=r.run_id,
                run_type="immunogenicity",
                flagged=bool(r.flagged),
                warning=r.warning,
                source_type=r.source_type,
            ).model_dump()
        )
    for r in meta_rows:
        results.append(
            RedzoneRow(
                run_id=r.run_id,
                run_type="metabolism",
                flagged=bool(r.flagged),
                warning=r.warning,
                source_type=r.source_type,
            ).model_dump()
        )

    return {"redzone": results}


@router.get("/{structure_id}/validation", response_model=Dict[str, Any])
async def get_validation(
    structure_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get validation mining run summaries for a structure."""
    await _require_structure(conn, structure_id)

    rows = (
        await conn.execute(
            sa.text(
                """
                  SELECT vmr.run_id AS run_id, vmr.status,
                       COUNT(DISTINCT fd.dehydron_id)   AS dehydron_count,
                       COUNT(DISTINCT fv.void_id)       AS void_count,
                       COUNT(DISTINCT fgs.glue_site_id) AS glue_site_count
                FROM   fact_validation_mining_run vmr
                LEFT JOIN fact_dehydron   fd  ON vmr.structure_id = fd.structure_id
                LEFT JOIN fact_void        fv  ON vmr.structure_id = fv.structure_id
                LEFT JOIN fact_glue_site  fgs ON vmr.run_id       = fgs.run_id
                WHERE  vmr.structure_id = :sid
                GROUP  BY vmr.run_id, vmr.status
                ORDER  BY vmr.started_at DESC NULLS LAST
                """
            ),
            {"sid": structure_id},
        )
    ).fetchall()

    return {
        "validation": [
            ValidationRow(
                run_id=r.run_id,
                status=r.status,
                dehydron_count=r.dehydron_count or 0,
                void_count=r.void_count or 0,
                glue_site_count=r.glue_site_count or 0,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/gnn", response_model=Dict[str, Any])
async def get_gnn_results(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get GNN inference results for a structure."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT gnn_run_id, model_version, num_nodes, num_edges,
                       curvature_value, source_type
                FROM   fact_gnn_inference
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY computed_at DESC
                """
            ),
            params,
        )
    ).fetchall()

    return {
        "gnn": [
            GNNRow(
                gnn_run_id=r.gnn_run_id,
                model_version=r.model_version,
                num_nodes=r.num_nodes,
                num_edges=r.num_edges,
                curvature_value=r.curvature_value,
                source_type=r.source_type,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/gnn/{gnn_run_id}/nodes", response_model=Dict[str, Any])
async def get_gnn_node_results(
    structure_id: str,
    gnn_run_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get per-node GNN geometry, uncertainty, and embedding data for a run."""
    await _require_structure(conn, structure_id)

    rows = (
        await conn.execute(
            sa.text(
                """
                SELECT n.node_id,
                       n.residue_id,
                       n.input_rho,
                       n.input_tau_flag,
                       n.input_ss_type,
                       n.input_sasa,
                       n.projections,
                       n.cone_depth,
                       n.cone_width,
                       n.expert_weights,
                       n.epistemic_uncertainty,
                       n.aleatoric_uncertainty,
                       n.total_uncertainty,
                       n.evidence_mu,
                       n.evidence_nu,
                       n.evidence_alpha,
                       n.evidence_beta,
                       n.source_type,
                       e.projection_embedding::text AS projection_embedding,
                       e.routing_entropy,
                       e.expert_winner
                FROM fact_gnn_node_output n
                JOIN fact_gnn_inference i
                  ON i.gnn_run_id = n.gnn_run_id
                LEFT JOIN fact_gnn_node_embedding e
                  ON e.node_id = n.node_id
                WHERE i.structure_id = :structure_id
                  AND n.gnn_run_id = :gnn_run_id
                ORDER BY n.residue_id
                """
            ),
            {"structure_id": structure_id, "gnn_run_id": gnn_run_id},
        )
    ).fetchall()

    return {
        "nodes": [
            GNNNodeRow(
                node_id=row.node_id,
                residue_id=row.residue_id,
                input_rho=row.input_rho,
                input_tau_flag=row.input_tau_flag,
                input_ss_type=row.input_ss_type,
                input_sasa=row.input_sasa,
                projections=row.projections if isinstance(row.projections, list) else json.loads(row.projections),
                cone_depth=row.cone_depth,
                cone_width=row.cone_width,
                expert_weights=row.expert_weights if isinstance(row.expert_weights, list) else json.loads(row.expert_weights),
                epistemic_uncertainty=row.epistemic_uncertainty,
                aleatoric_uncertainty=row.aleatoric_uncertainty,
                total_uncertainty=row.total_uncertainty,
                evidence_mu=row.evidence_mu,
                evidence_nu=row.evidence_nu,
                evidence_alpha=row.evidence_alpha,
                evidence_beta=row.evidence_beta,
                projection_embedding=_parse_vector(row.projection_embedding),
                routing_entropy=row.routing_entropy,
                expert_winner=row.expert_winner,
                source_type=row.source_type,
            ).model_dump()
            for row in rows
        ]
    }


@router.get("/{structure_id}/folding", response_model=Dict[str, Any])
async def get_folding(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get folding path metadata (GCS URI only — no inline frame data)."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT path_id, method, num_frames, num_atoms, gcs_uri, source_type
                FROM   fact_folding_path
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY computed_at DESC
                """
            ),
            params,
        )
    ).fetchall()

    return {
        "folding": [
            FoldingRow(
                path_id=r.path_id,
                method=r.method,
                num_frames=r.num_frames,
                num_atoms=r.num_atoms,
                gcs_uri=r.gcs_uri,
                source_type=r.source_type,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/synthesis", response_model=Dict[str, Any])
async def get_synthesis(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get synthesis feasibility results for a structure."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT synthesis_id, manufacturable, complexity_score,
                       gc_content, aa_length, source_type
                FROM   fact_synthesis_feasibility
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY computed_at DESC
                """
            ),
            params,
        )
    ).fetchall()

    return {
        "synthesis": [
            SynthesisRow(
                synthesis_id=r.synthesis_id,
                manufacturable=r.manufacturable,
                complexity_score=r.complexity_score,
                gc_content=r.gc_content,
                aa_length=r.aa_length,
                source_type=r.source_type,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/audit", response_model=Dict[str, Any])
async def get_audit(
    structure_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get audit trail events for a structure."""
    await _require_structure(conn, structure_id)

    rows = (
        await conn.execute(
            sa.text(
                """
                SELECT event_id, timestamp::TEXT, state_from, state_to,
                       event_name, user_id
                FROM   fact_audit_event
                WHERE  structure_id = :sid
                ORDER  BY timestamp ASC
                """
            ),
            {"sid": structure_id},
        )
    ).fetchall()

    return {
        "audit": [
            AuditRow(
                event_id=r.event_id,
                timestamp=r[1],
                state_from=r.state_from,
                state_to=r.state_to,
                event_name=r.event_name,
                user_id=r.user_id,
            ).model_dump()
            for r in rows
        ]
    }


@router.get("/{structure_id}/hdx", response_model=Dict[str, Any])
async def get_hdx(
    structure_id: str,
    source_type: Optional[SourceType] = Query(None),
    conn: AsyncConnection = Depends(get_db_conn),
) -> Dict[str, Any]:
    """Get HDX-MS correlation results for a structure."""
    await _require_structure(conn, structure_id)

    params: Dict[str, Any] = {"sid": structure_id}
    if source_type is not None:
        params["source_type"] = source_type

    rows = (
        await conn.execute(
            sa.text(
                f"""
                SELECT hdx_run_id, r_squared, spearman_rho, p_value,
                       passed, num_residues, source_type
                FROM   fact_hdx_correlation
                WHERE  structure_id = :sid
                {_source_clause(source_type)}
                ORDER  BY computed_at DESC
                """
            ),
            params,
        )
    ).fetchall()

    return {
        "hdx": [
            HdxRow(
                hdx_run_id=r.hdx_run_id,
                r_squared=r.r_squared,
                spearman_rho=r.spearman_rho,
                p_value=r.p_value,
                passed=r.passed,
                num_residues=r.num_residues,
                source_type=r.source_type,
            ).model_dump()
            for r in rows
        ]
    }
