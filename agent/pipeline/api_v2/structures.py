# backend/gosp/api_v2/structures.py
"""Read endpoints for structure dimension data (Task 19)."""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncConnection

router = APIRouter(prefix="/structures", tags=["structures"])


async def get_db_conn() -> AsyncGenerator[AsyncConnection, None]:
    """FastAPI dependency: yields an AsyncConnection per request."""
    from gosp.db.connection import get_connection
    async for conn in get_connection():
        yield conn


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class StructureResponse(BaseModel):
    structure_id: str
    pdb_id: str
    resolution: Optional[float]
    total_residues: int
    total_atoms: int


class ResidueResponse(BaseModel):
    residue_id: str
    residue_index: int
    residue_name: str
    chain_label: str
    sse_code: Optional[str]
    sasa: Optional[float]


class AtomResponse(BaseModel):
    atom_id: str
    residue_id: str
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    b_factor: Optional[float]
    occupancy: Optional[float]
    sasa: Optional[float]  # per-residue SASA surfaced on each atom row


class JobStatusEntry(BaseModel):
    job_type: str
    status: str
    error_message: Optional[str]
    queued_at: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]


class StructureSummary(BaseModel):
    """Lightweight card payload — all metadata + statuses in one round-trip."""
    structure_id: str
    pdb_id: str
    resolution: Optional[float]
    total_residues: Optional[int]
    total_atoms: Optional[int]
    chain_count: int
    jobs: Dict[str, str]       # job_type -> latest status
    counts: Dict[str, int]     # dehydrons, voids, glue_sites
    assets: Dict[str, bool]    # has_gnn, has_folding, has_cdd, has_energy


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{structure_id}", response_model=StructureResponse)
async def get_structure(
    structure_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> StructureResponse:
    """Return structure metadata from dim_structure."""
    row = (
        await conn.execute(
            sa.text(
                """
                SELECT structure_id, pdb_id, resolution, total_residues, total_atoms
                FROM   dim_structure
                WHERE  structure_id = :sid
                """
            ),
            {"sid": structure_id},
        )
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")

    return StructureResponse(
        structure_id=row.structure_id,
        pdb_id=row.pdb_id,
        resolution=row.resolution,
        total_residues=row.total_residues,
        total_atoms=row.total_atoms,
    )


@router.get("/{structure_id}/status", response_model=List[JobStatusEntry])
async def get_structure_status(
    structure_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> List[JobStatusEntry]:
    """Return all job-status rows for a structure from fact_job_status."""
    exists = (
        await conn.execute(
            sa.text("SELECT 1 FROM dim_structure WHERE structure_id = :sid"),
            {"sid": structure_id},
        )
    ).fetchone()

    if exists is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")

    rows = (
        await conn.execute(
            sa.text(
                """
                SELECT job_type, status, error_message,
                       queued_at::TEXT, started_at::TEXT, finished_at::TEXT
                FROM   fact_job_status
                WHERE  structure_id = :sid
                ORDER  BY queued_at ASC NULLS LAST
                """
            ),
            {"sid": structure_id},
        )
    ).fetchall()

    return [
        JobStatusEntry(
            job_type=r.job_type,
            status=r.status,
            error_message=r.error_message,
            queued_at=r[3],
            started_at=r[4],
            finished_at=r[5],
        )
        for r in rows
    ]


@router.get("/{structure_id}/atoms", response_model=List[AtomResponse])
async def get_atoms(
    structure_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> List[AtomResponse]:
    """Return flat atom list with per-residue SASA surfaced on each atom row."""
    exists = (
        await conn.execute(
            sa.text("SELECT 1 FROM dim_structure WHERE structure_id = :sid"),
            {"sid": structure_id},
        )
    ).fetchone()

    if exists is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")

    rows = (
        await conn.execute(
            sa.text(
                """
                SELECT
                    a.atom_id,
                    a.residue_id,
                    a.atom_name,
                    a.element,
                    a.x,
                    a.y,
                    a.z,
                    a.b_factor,
                    a.occupancy,
                    r.sasa
                FROM   dim_atom     a
                JOIN   dim_residue  r  ON a.residue_id  = r.residue_id
                JOIN   dim_chain    c  ON r.chain_id     = c.chain_id
                WHERE  c.structure_id = :sid
                ORDER  BY a.atom_id
                """
            ),
            {"sid": structure_id},
        )
    ).fetchall()

    return [
        AtomResponse(
            atom_id=r.atom_id,
            residue_id=r.residue_id,
            atom_name=r.atom_name,
            element=r.element,
            x=r.x,
            y=r.y,
            z=r.z,
            b_factor=r.b_factor,
            occupancy=r.occupancy,
            sasa=r.sasa,
        )
        for r in rows
    ]


@router.get("/{structure_id}/summary", response_model=StructureSummary)
async def get_structure_summary(
    structure_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> StructureSummary:
    """Single-round-trip summary card for a structure.

    Returns structure metadata + chain count + latest job statuses +
    result counts + asset availability flags.  Designed so the client
    can render a complete card without any further API calls.
    """
    # Base metadata + chain count
    struct_row = (
        await conn.execute(
            sa.text("""
                SELECT ds.structure_id, ds.pdb_id, ds.resolution,
                       ds.total_residues, ds.total_atoms,
                       COUNT(dc.chain_id) AS chain_count
                FROM   dim_structure ds
                LEFT JOIN dim_chain dc ON dc.structure_id = ds.structure_id
                WHERE  ds.structure_id = :sid
                GROUP  BY ds.structure_id, ds.pdb_id, ds.resolution,
                          ds.total_residues, ds.total_atoms
            """),
            {"sid": structure_id},
        )
    ).fetchone()

    if struct_row is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found")

    # Latest status per job type
    job_rows = (
        await conn.execute(
            sa.text("""
                SELECT DISTINCT ON (job_type) job_type, status
                FROM   fact_job_status
                WHERE  structure_id = :sid
                ORDER  BY job_type, queued_at DESC NULLS LAST
            """),
            {"sid": structure_id},
        )
    ).fetchall()

    # Counts + asset flags — single query with LEFT JOINs instead of 7 subqueries
    counts_row = (
        await conn.execute(
            sa.text("""
                SELECT
                    COUNT(DISTINCT fd.dehydron_id)   AS dehydron_count,
                    COUNT(DISTINCT fv.void_id)       AS void_count,
                    COUNT(DISTINCT fgs.glue_site_id) AS glue_site_count,
                    COUNT(DISTINCT fg.gnn_run_id)    AS gnn_count,
                    COUNT(DISTINCT fp.path_id)       AS folding_count,
                    COUNT(DISTINCT fc.annotation_id) AS cdd_count,
                    COUNT(DISTINCT fe.energy_id)     AS energy_count
                FROM dim_structure ds
                LEFT JOIN fact_dehydron fd          ON fd.structure_id = ds.structure_id
                LEFT JOIN fact_void fv              ON fv.structure_id = ds.structure_id
                LEFT JOIN fact_validation_mining_run vmr ON vmr.structure_id = ds.structure_id
                LEFT JOIN fact_glue_site fgs        ON fgs.run_id = vmr.run_id
                LEFT JOIN fact_gnn_inference fg     ON fg.structure_id = ds.structure_id
                LEFT JOIN fact_folding_path fp      ON fp.structure_id = ds.structure_id
                LEFT JOIN fact_cdd_annotation fc    ON fc.structure_id = ds.structure_id
                LEFT JOIN fact_energy_calculation fe ON fe.structure_id = ds.structure_id
                WHERE ds.structure_id = :sid
            """),
            {"sid": structure_id},
        )
    ).fetchone()

    return StructureSummary(
        structure_id=struct_row.structure_id,
        pdb_id=struct_row.pdb_id,
        resolution=struct_row.resolution,
        total_residues=struct_row.total_residues,
        total_atoms=struct_row.total_atoms,
        chain_count=struct_row.chain_count or 0,
        jobs={r.job_type: r.status for r in job_rows},
        counts={
            "dehydrons":   counts_row.dehydron_count or 0,
            "voids":       counts_row.void_count or 0,
            "glue_sites":  counts_row.glue_site_count or 0,
        },
        assets={
            "has_gnn":     (counts_row.gnn_count or 0) > 0,
            "has_folding": (counts_row.folding_count or 0) > 0,
            "has_cdd":     (counts_row.cdd_count or 0) > 0,
            "has_energy":  (counts_row.energy_count or 0) > 0,
        },
    )
