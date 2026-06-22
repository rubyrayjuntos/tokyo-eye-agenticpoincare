"""Structures router — exposes the API surface the frontend expects.

Covers:
  GET  /api/structures                       list all structures with ingestion status
  GET  /api/structures/{id}/embeddings       poincaré disc data (per-residue)
  GET  /api/structures/{id}/metrics          graph metrics stub
  GET  /api/structures/{id}/hydrate          hydration / dehydron summary stub
  POST /api/ingest                           ingest a PDB ID
  POST /api/pipeline/run                     trigger DTIE pipeline
  GET  /api/pipeline/status/{job_id}         job status
  GET  /api/kpis                             dashboard KPIs
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["structures"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _get_conn():
    from data.db import get_connection
    return get_connection()


# ---------------------------------------------------------------------------
# GET /api/structures
# ---------------------------------------------------------------------------


@router.get("/structures")
async def list_structures():
    """List all structures with their ingestion phase status."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            rows = await db.fetch_all(
                """
                SELECT
                    s.structure_id,
                    s.pdb_id,
                    s.resolution,
                    s.total_residues,
                    s.total_atoms,
                    s.created_at,
                    COALESCE(
                        json_agg(
                            json_build_object(
                                'phase', i.phase_name,
                                'status', i.status,
                                'completed_at', i.completed_at,
                                'error', i.error_message
                            )
                        ) FILTER (WHERE i.phase_name IS NOT NULL),
                        '[]'
                    ) AS phases
                FROM dim_structure s
                LEFT JOIN dim_ingestion_status i ON i.structure_id = s.structure_id
                GROUP BY s.structure_id, s.pdb_id, s.resolution, s.total_residues,
                         s.total_atoms, s.created_at
                ORDER BY s.created_at DESC
                """,
                {},
            )

        structures = []
        for r in rows:
            phases = r.get("phases") or []
            if isinstance(phases, str):
                import json
                phases = json.loads(phases)

            # Derive overall status from phases
            statuses = {p["status"] for p in phases if p.get("status")}
            if not phases:
                overall = "ingested"
            elif "failed" in statuses:
                overall = "failed"
            elif all(p.get("status") == "complete" for p in phases):
                overall = "complete"
            elif "complete" in statuses:
                overall = "partial"
            else:
                overall = "pending"

            structures.append({
                "structure_id": r["structure_id"],
                "pdb_id": r["pdb_id"],
                "title": r.get("pdb_id", "").upper(),
                "resolution": r.get("resolution"),
                "total_residues": r.get("total_residues"),
                "total_atoms": r.get("total_atoms"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "status": overall,
                "phases": phases,
            })

        return {"structures": structures, "count": len(structures)}

    except Exception:
        logger.exception("list_structures failed")
        return JSONResponse(status_code=503, content={"error": "Database unavailable"})


# ---------------------------------------------------------------------------
# GET /api/structures/{structure_id}/embeddings
# ---------------------------------------------------------------------------


@router.get("/structures/{structure_id}/embeddings")
async def get_embeddings(structure_id: str):
    """Return per-residue Poincaré disc embeddings for a structure."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            rows = await db.fetch_all(
                """
                SELECT
                    r.residue_id,
                    r.residue_index,
                    r.residue_name,
                    c.chain_label,
                    e.projection_embedding,
                    e.cone_depth,
                    e.cone_width,
                    e.routing_entropy     AS epistemic_uncertainty,
                    e.expert_winner,
                    e.model_version
                FROM fact_gnn_node_embedding e
                JOIN dim_residue r ON r.residue_id = e.residue_id
                JOIN dim_chain   c ON c.chain_id   = r.chain_id
                WHERE e.structure_id = :sid
                ORDER BY r.residue_index
                """,
                {"sid": structure_id},
            )

        if not rows:
            return JSONResponse(
                status_code=404,
                content={"error": f"No GNN embeddings for '{structure_id}'. Run the pipeline first."},
            )

        residues = []
        for r in rows:
            proj = r.get("projection_embedding") or []
            if isinstance(proj, str):
                import json
                proj = json.loads(proj)
            x = float(proj[0]) if len(proj) > 0 else 0.0
            y = float(proj[1]) if len(proj) > 1 else 0.0

            residues.append({
                "residue_id": r["residue_id"],
                "residue_index": r["residue_index"],
                "residue_name": r["residue_name"],
                "chain_label": r["chain_label"],
                "x": x,
                "y": y,
                "cone_depth": r.get("cone_depth") or 0.0,
                "cone_width": r.get("cone_width"),
                "epistemic_uncertainty": r.get("epistemic_uncertainty") or 0.0,
                "aleatoric_uncertainty": None,
                "expert_winner": r.get("expert_winner"),
            })

        return {
            "structure_id": structure_id,
            "model_version": rows[0].get("model_version") or "GOSPConeMapper-v6",
            "residues": residues,
            "count": len(residues),
        }

    except Exception:
        logger.exception("get_embeddings failed for %s", structure_id)
        return JSONResponse(status_code=503, content={"error": "Database unavailable"})


# ---------------------------------------------------------------------------
# GET /api/structures/{structure_id}/metrics
# ---------------------------------------------------------------------------


@router.get("/structures/{structure_id}/metrics")
async def get_metrics(structure_id: str):
    """Return graph metrics for a structure (residue count, chain count, etc.)."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            row = await db.fetch_one(
                """
                SELECT s.total_residues, s.total_atoms, s.resolution,
                       COUNT(DISTINCT c.chain_id) AS chain_count
                FROM dim_structure s
                LEFT JOIN dim_chain c ON c.structure_id = s.structure_id
                WHERE s.structure_id = :sid
                GROUP BY s.structure_id, s.total_residues, s.total_atoms, s.resolution
                """,
                {"sid": structure_id},
            )

        if not row:
            return JSONResponse(status_code=404, content={"error": "Structure not found"})

        return {
            "structure_id": structure_id,
            "total_residues": row.get("total_residues") or 0,
            "total_atoms": row.get("total_atoms") or 0,
            "chain_count": row.get("chain_count") or 0,
            "resolution": row.get("resolution"),
        }

    except Exception:
        logger.exception("get_metrics failed for %s", structure_id)
        return JSONResponse(status_code=503, content={"error": "Database unavailable"})


# ---------------------------------------------------------------------------
# GET /api/structures/{structure_id}/hydrate
# ---------------------------------------------------------------------------


@router.get("/structures/{structure_id}/hydrate")
async def hydrate(structure_id: str):
    """Return dehydron / hydration summary for a structure."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            # Check structure exists
            row = await db.fetch_one(
                "SELECT structure_id, total_residues FROM dim_structure WHERE structure_id = :sid",
                {"sid": structure_id},
            )

        if not row:
            return JSONResponse(status_code=404, content={"error": "Structure not found"})

        return {
            "structure_id": structure_id,
            "total_residues": row.get("total_residues") or 0,
            "dehydrons": [],
            "hydration_sites": [],
            "summary": {"status": "not_computed"},
        }

    except Exception:
        logger.exception("hydrate failed for %s", structure_id)
        return JSONResponse(status_code=503, content={"error": "Database unavailable"})


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------

_ingest_jobs: dict[str, dict[str, Any]] = {}


@router.post("/ingest")
async def ingest(body: dict, background_tasks: BackgroundTasks):
    """Ingest a PDB ID — downloads from RCSB and loads into the DB."""
    pdb_id = (body.get("pdb_id") or "").strip().lower()
    if not pdb_id:
        return JSONResponse(status_code=422, content={"error": "pdb_id is required"})

    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "pdb_id": pdb_id,
        "status": "queued",
        "created_at": _now(),
        "message": "Queued for ingestion",
    }

    background_tasks.add_task(_run_ingest, job_id, pdb_id)

    return {"job_id": job_id, "pdb_id": pdb_id, "status": "queued"}


async def _run_ingest(job_id: str, pdb_id: str):
    """Background task: download PDB from RCSB and insert into dim_structure."""
    import httpx
    from data.db import DBAdapter, get_connection

    _ingest_jobs[job_id]["status"] = "running"
    _ingest_jobs[job_id]["message"] = f"Downloading {pdb_id.upper()} from RCSB…"

    try:
        # Fetch PDB metadata from RCSB
        async with httpx.AsyncClient(timeout=30) as client:
            rcsb_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
            resp = await client.get(rcsb_url)
            if resp.status_code != 200:
                raise RuntimeError(f"RCSB returned {resp.status_code} for {pdb_id}")
            meta = resp.json()

        structure_id = pdb_id.lower()
        resolution = None
        try:
            resolution = meta["rcsb_entry_info"].get("resolution_combined", [None])[0]
        except (KeyError, IndexError, TypeError):
            pass

        total_residues = None
        try:
            total_residues = meta["rcsb_entry_info"].get("deposited_polymer_monomer_count")
        except (KeyError, TypeError):
            pass

        total_atoms = None
        try:
            total_atoms = meta["rcsb_entry_info"].get("deposited_atom_count")
        except (KeyError, TypeError):
            pass

        _ingest_jobs[job_id]["message"] = f"Storing {pdb_id.upper()} in database…"

        async with get_connection() as conn:
            db = DBAdapter(conn)
            await db.execute(
                """
                INSERT INTO dim_structure (structure_id, pdb_id, resolution, total_residues, total_atoms, created_at)
                VALUES (:sid, :pdb_id, :res, :total_res, :total_atoms, NOW())
                ON CONFLICT (pdb_id) DO UPDATE
                  SET resolution    = EXCLUDED.resolution,
                      total_residues = EXCLUDED.total_residues,
                      total_atoms    = EXCLUDED.total_atoms
                """,
                {
                    "sid": structure_id,
                    "pdb_id": pdb_id.upper(),
                    "res": resolution,
                    "total_res": total_residues,
                    "total_atoms": total_atoms,
                },
            )
            await db.execute(
                """
                INSERT INTO dim_ingestion_status (structure_id, phase_name, status, completed_at)
                VALUES (:sid, 'ingestion', 'complete', NOW())
                ON CONFLICT (structure_id, phase_name) DO UPDATE
                  SET status = 'complete', completed_at = NOW()
                """,
                {"sid": structure_id},
            )

        _ingest_jobs[job_id]["status"] = "complete"
        _ingest_jobs[job_id]["structure_id"] = structure_id
        _ingest_jobs[job_id]["message"] = f"{pdb_id.upper()} ingested successfully"

    except Exception as exc:
        logger.exception("ingest failed for %s", pdb_id)
        _ingest_jobs[job_id]["status"] = "failed"
        _ingest_jobs[job_id]["message"] = str(exc)


# ---------------------------------------------------------------------------
# POST /api/pipeline/run
# ---------------------------------------------------------------------------

_pipeline_jobs: dict[str, dict[str, Any]] = {}


@router.post("/pipeline/run")
async def run_pipeline(body: dict, background_tasks: BackgroundTasks):
    """Trigger the DTIE pipeline for a structure."""
    structure_id = (body.get("structure_id") or body.get("pdb_id") or "").strip().lower()
    if not structure_id:
        return JSONResponse(status_code=422, content={"error": "structure_id is required"})

    job_id = str(uuid.uuid4())
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "structure_id": structure_id,
        "status": "queued",
        "created_at": _now(),
        "message": "Pipeline queued",
    }

    background_tasks.add_task(_run_pipeline_task, job_id, structure_id)
    return _pipeline_jobs[job_id]


async def _run_pipeline_task(job_id: str, structure_id: str):
    _pipeline_jobs[job_id]["status"] = "running"
    _pipeline_jobs[job_id]["message"] = "Pipeline running (science container required)"
    await asyncio.sleep(1)
    # Real pipeline runs in the science container via docker compose run.
    # This endpoint records the job so the frontend can poll status.
    _pipeline_jobs[job_id]["status"] = "pending_science"
    _pipeline_jobs[job_id]["message"] = (
        "Pipeline queued — start the science container to process: "
        f"docker compose run science python -m science.dtie.v5.orchestrator.pipeline {structure_id}"
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline/status/{job_id}
# ---------------------------------------------------------------------------


@router.get("/pipeline/status/{job_id}")
async def pipeline_status(job_id: str):
    job = _pipeline_jobs.get(job_id) or _ingest_jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job


# ---------------------------------------------------------------------------
# Agent chat alias — frontend api.ts calls /api/agent/chat
# ---------------------------------------------------------------------------


@router.post("/agent/chat")
async def agent_chat_alias(body: dict):
    """Proxy to /api/chat for callers using the /api/agent/chat path."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "http://localhost:8000/api/chat",
                json=body,
                headers={"Content-Type": "application/json"},
            )
            return resp.json()
    except Exception:
        logger.exception("agent_chat_alias failed")
        return JSONResponse(status_code=503, content={"error": "Chat service unavailable"})


@router.get("/agent/telemetry/{session_id}")
async def agent_telemetry(session_id: str):
    return {"session_id": session_id, "telemetry": {}}


# ---------------------------------------------------------------------------
# GET /api/kpis
# ---------------------------------------------------------------------------


@router.get("/kpis")
async def get_kpis():
    """Return dashboard KPI summary."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)

            struct_row = await db.fetch_one(
                "SELECT COUNT(*) AS n FROM dim_structure", {}
            )
            emb_row = await db.fetch_one(
                "SELECT COUNT(DISTINCT structure_id) AS n FROM fact_gnn_node_embedding", {}
            )
            job_row = await db.fetch_one(
                "SELECT COUNT(*) AS n FROM dim_ingestion_status WHERE status = 'complete'", {}
            )
            uncertainty_row = await db.fetch_one(
                """
                SELECT AVG(routing_entropy) AS mean_uncertainty
                FROM fact_gnn_node_embedding
                """,
                {},
            )

        return {
            "structure_count": struct_row["n"] if struct_row else 0,
            "structures_with_embeddings": emb_row["n"] if emb_row else 0,
            "completed_phases": job_row["n"] if job_row else 0,
            "mean_uncertainty": float(uncertainty_row["mean_uncertainty"] or 0.0) if uncertainty_row else 0.0,
            "active_jobs": 0,
            "model": "GOSPConeMapper-v6",
        }

    except Exception:
        logger.exception("get_kpis failed")
        return JSONResponse(status_code=503, content={"error": "Database unavailable"})
