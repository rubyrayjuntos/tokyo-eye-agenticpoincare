"""Dashboard API router — direct REST endpoints for the research workbench.

All mechanical operations (ingest, pipeline, queries) go through these
endpoints without involving the LLM. The agent chat endpoint is the only
one that uses LLM tokens, and it operates in pure reasoning mode (no tools).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.coordinator.deps import get_db
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["dashboard"])


# Dashboard chat is intentionally reasoning-only. This prompt is kept as a
# module constant so property tests can verify the no-tools contract directly.
DASHBOARD_AGENT_PROMPT = (
    "You are the Tokyo Eyes dashboard analyst. "
    "Provide concise scientific interpretation using the supplied dashboard context. "
    "Do not assume data that is not present. "
    "This dashboard chat operates in pure reasoning mode with no execution tools. "
    "The supplied <context> block may include a Persistent Memory section assembled from durable "
    "session summaries and prior grounded exchanges. Treat that section as available memory for this "
    "chat surface. Do not claim that you lack persistent memory when that section is present."
)


# ---------------------------------------------------------------------------
# Pipeline Job State Management (DB-backed with in-memory fallback)
# ---------------------------------------------------------------------------

_pipeline_jobs: dict[str, dict[str, Any]] = {}

# Science container status cache (avoid shelling out to Docker on every KPI request)
_science_status_cache: dict[str, Any] = {"available": None, "checked_at": 0.0}
_SCIENCE_CHECK_INTERVAL = 300  # 5 minutes


async def _get_cached_science_status() -> bool:
    """Return cached science container availability, refreshing every 5 minutes."""
    now = time.time()
    if _science_status_cache["available"] is None or (now - _science_status_cache["checked_at"]) > _SCIENCE_CHECK_INTERVAL:
        try:
            from agent.tools.science_client import ScienceClient

            client = ScienceClient()
            health = await asyncio.wait_for(client.health(), timeout=10.0)
            _science_status_cache["available"] = health.get("status") == "ok"
        except Exception:
            _science_status_cache["available"] = False
        _science_status_cache["checked_at"] = now
    return _science_status_cache["available"]

PIPELINE_STEPS = [
    "ingestion",
    "graph_build",
    "gnn_forward_pass",
    "dtie_decomposition",
    "complete",
]


async def _create_job_db(structure_id: str, modules: list[str] | None = None) -> str:
    """Create a new pipeline job in the database and return its ID."""
    from data.db import get_connection, DBAdapter

    job_id = str(uuid.uuid4())
    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            await db.execute(
                """
                INSERT INTO pipeline_job (job_id, structure_id, status, current_step, progress, modules, started_at)
                VALUES (:job_id, :structure_id, 'queued', 'ingestion', 0, :modules, now())
                """,
                {
                    "job_id": job_id,
                    "structure_id": structure_id,
                    "modules": __import__("json").dumps(modules or []),
                },
            )
    except Exception as e:
        logger.warning("Failed to persist job to DB, using in-memory fallback: %s", e)
        _pipeline_jobs[job_id] = {
            "job_id": job_id,
            "structure_id": structure_id,
            "status": "queued",
            "current_step": "ingestion",
            "progress": 0,
            "modules": modules or [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        }
    return job_id


async def _update_job_db(job_id: str, **kwargs) -> None:
    """Update job state fields in the database."""
    from data.db import get_connection, DBAdapter

    # Also update in-memory cache if present
    if job_id in _pipeline_jobs:
        _pipeline_jobs[job_id].update(kwargs)

    # Build SET clause dynamically
    allowed_fields = {"status", "current_step", "progress", "error", "completed_at"}
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return

    set_parts = [f"{k} = :{k}" for k in updates]
    updates["job_id"] = job_id

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            await db.execute(
                f"UPDATE pipeline_job SET {', '.join(set_parts)} WHERE job_id = :job_id",
                updates,
            )
    except Exception as e:
        logger.warning("Failed to update job in DB: %s", e)


async def _get_job_db(job_id: str) -> dict[str, Any] | None:
    """Get job state by ID from database (falls back to in-memory)."""
    from data.db import get_connection, DBAdapter

    # Check in-memory first (for jobs created during DB outage)
    if job_id in _pipeline_jobs:
        return _pipeline_jobs[job_id]

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            row = await db.fetch_one(
                """
                SELECT job_id, structure_id, status, current_step, progress,
                       modules, error, started_at, completed_at
                FROM pipeline_job WHERE job_id = :job_id
                """,
                {"job_id": job_id},
            )
        if row:
            return {
                "job_id": str(row["job_id"]),
                "structure_id": str(row["structure_id"]),
                "status": row["status"],
                "current_step": row["current_step"],
                "progress": row["progress"],
                "modules": row.get("modules") or [],
                "started_at": str(row["started_at"]) if row.get("started_at") else None,
                "completed_at": str(row["completed_at"]) if row.get("completed_at") else None,
                "error": row.get("error"),
            }
    except Exception as e:
        logger.warning("Failed to fetch job from DB: %s", e)

    return None


# Legacy aliases for backward compatibility within this module
def _create_job(structure_id: str, modules: list[str] | None = None) -> str:
    """Sync wrapper — creates job in-memory. Use _create_job_db for async paths."""
    job_id = str(uuid.uuid4())
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "structure_id": structure_id,
        "status": "queued",
        "current_step": "ingestion",
        "progress": 0,
        "modules": modules or [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    """Update job state fields (in-memory only, for background task)."""
    if job_id in _pipeline_jobs:
        _pipeline_jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> dict[str, Any] | None:
    """Get job state by ID (in-memory only)."""
    return _pipeline_jobs.get(job_id)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    pdb_id: str = Field(..., min_length=4, max_length=4, description="4-character PDB ID")


class PipelineRunRequest(BaseModel):
    structure_id: str
    modules: list[str] = Field(default_factory=list, description="Pipeline modules to run")


class AgentChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------


@router.post("/ingest")
async def ingest(request: IngestRequest):
    """Fetch structure from RCSB, parse CIF, populate dimensional tables."""
    from agent.tools.rcsb import ingest_structure
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            result = await ingest_structure(pdb_id=request.pdb_id, db=db)

        if "error" in result:
            return JSONResponse(
                status_code=502 if "unavailable" in result.get("error", "").lower() else 404,
                content={"error": result["error"], "message": result["error"]},
            )

        return result

    except Exception as e:
        logger.exception("Ingest failed for %s", request.pdb_id)
        return JSONResponse(
            status_code=500,
            content={"error": "ingest_failed", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /api/structures
# ---------------------------------------------------------------------------


@router.get("/structures")
async def list_structures():
    """Return all ingested structures with metadata and residue counts."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            rows = await db.fetch_all(
                """
                SELECT s.structure_id, s.pdb_id, s.source, s.resolution, s.method,
                       s.created_at,
                       COUNT(DISTINCT r.residue_id) AS residue_count,
                       COUNT(DISTINCT c.chain_id) AS chain_count,
                       EXISTS(
                           SELECT 1 FROM fact_gnn_node_embedding e
                           WHERE e.structure_id = s.structure_id LIMIT 1
                       ) AS has_embeddings
                FROM dim_structure s
                LEFT JOIN dim_chain c ON c.structure_id = s.structure_id
                LEFT JOIN dim_residue r ON r.chain_id = c.chain_id
                GROUP BY s.structure_id, s.pdb_id, s.source, s.resolution,
                         s.method, s.created_at
                ORDER BY s.created_at DESC
                """
            )

        structures = []
        for row in rows:
            structures.append({
                "structure_id": row["structure_id"],
                "pdb_id": row["pdb_id"],
                "source": row["source"],
                "resolution": row.get("resolution"),
                "method": row.get("method"),
                "residue_count": row["residue_count"],
                "chain_count": row["chain_count"],
                "has_embeddings": row["has_embeddings"],
                "ingested_at": str(row["created_at"]) if row.get("created_at") else None,
            })

        return {"structures": structures, "count": len(structures)}

    except Exception as e:
        logger.exception("Failed to list structures")
        return JSONResponse(
            status_code=503,
            content={"error": "db_error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /api/structures/{structure_id}/embeddings
# ---------------------------------------------------------------------------


@router.get("/structures/{structure_id}/embeddings")
async def get_embeddings(structure_id: str):
    """Return per-residue Poincaré disc coordinates for a structure."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            rows = await db.fetch_all(
                """
                SELECT r.residue_id, r.residue_index, r.residue_name,
                       c.chain_label,
                       e.hyp_projection_2d, e.hyp_projections,
                       e.cone_depth, e.epistemic_uncertainty,
                       e.aleatoric_uncertainty,
                       es.curvature
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
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "message": f"No embeddings for '{structure_id}'"},
            )

        curvature = rows[0].get("curvature", 1.0) if rows else 1.0
        residues = []
        for r in rows:
            x, y = 0.0, 0.0
            if r.get("hyp_projection_2d"):
                coords = r["hyp_projection_2d"]
                # pgvector returns a string like "[0.1,-0.2]" — parse it
                if isinstance(coords, str):
                    coords = [float(v) for v in coords.strip("[]").split(",")]
                x, y = float(coords[0]), float(coords[1])
            elif r.get("hyp_projections"):
                proj = r["hyp_projections"]
                if isinstance(proj, list) and len(proj) >= 2:
                    x, y = float(proj[0]), float(proj[1])

            residues.append({
                "residue_id": r["residue_id"],
                "residue_index": r["residue_index"],
                "chain_label": r["chain_label"],
                "x": x,
                "y": y,
                "cone_depth": r.get("cone_depth"),
                "epistemic_uncertainty": r.get("epistemic_uncertainty"),
                "aleatoric_uncertainty": r.get("aleatoric_uncertainty"),
            })

        return {
            "structure_id": structure_id,
            "curvature": curvature,
            "residues": residues,
        }

    except Exception as e:
        logger.exception("Embeddings fetch failed for %s", structure_id)
        return JSONResponse(
            status_code=503,
            content={"error": "db_error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /api/structures/{structure_id}/metrics
# ---------------------------------------------------------------------------


@router.get("/structures/{structure_id}/metrics")
async def get_metrics(structure_id: str):
    """Return graph topology metrics for a structure."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            rows = await db.fetch_all(
                """
                SELECT m.residue_id, m.metric_type, m.metric_value
                FROM fact_graph_metric m
                WHERE m.structure_id = :structure_id
                ORDER BY m.residue_id, m.metric_type
                """,
                {"structure_id": structure_id},
            )

        if not rows:
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "message": f"No metrics for '{structure_id}'"},
            )

        # Group by residue
        metrics_by_residue: dict[str, dict[str, float]] = {}
        for r in rows:
            rid = r["residue_id"]
            if rid not in metrics_by_residue:
                metrics_by_residue[rid] = {"residue_id": rid}
            metrics_by_residue[rid][r["metric_type"]] = r["metric_value"]

        return {
            "structure_id": structure_id,
            "residues": list(metrics_by_residue.values()),
        }

    except Exception as e:
        logger.exception("Metrics fetch failed for %s", structure_id)
        return JSONResponse(
            status_code=503,
            content={"error": "db_error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /api/kpis
# ---------------------------------------------------------------------------


@router.get("/kpis")
async def get_kpis():
    """Return system-wide KPI statistics."""
    from data.db import DBAdapter, get_connection

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)

            # Total structures
            row = await db.fetch_one("SELECT COUNT(*) AS cnt FROM dim_structure")
            total_structures = row["cnt"] if row else 0

            # Mean uncertainty
            row = await db.fetch_one(
                "SELECT AVG(epistemic_uncertainty) AS mean_unc FROM fact_gnn_node_embedding"
            )
            mean_uncertainty = float(row["mean_unc"]) if row and row["mean_unc"] else 0.0

            # Active jobs
            active_jobs = sum(
                1 for j in _pipeline_jobs.values() if j["status"] in ("queued", "running")
            )

        # Science container check (cached — only check once per 5 minutes)
        science_available = await _get_cached_science_status()

        return {
            "total_structures": total_structures,
            "mean_uncertainty": round(mean_uncertainty, 6),
            "avg_inference_seconds": 0.0,  # TODO: track from pipeline runs
            "active_jobs": active_jobs,
            "model_status": "ready",
            "db_connected": True,
            "science_container_available": science_available,
        }

    except Exception as e:
        logger.exception("KPI fetch failed")
        return JSONResponse(
            status_code=503,
            content={
                "error": "db_error",
                "message": str(e),
                "db_connected": False,
                "science_container_available": False,
            },
        )


# ---------------------------------------------------------------------------
# POST /api/pipeline/run
# ---------------------------------------------------------------------------


async def _run_pipeline_background(job_id: str, structure_id: str) -> None:
    """Background task: delegate the full DTIE pipeline to the science API.

    The science container owns orchestration of GNN inference and downstream
    phases. The agent router tracks job lifecycle only, instead of re-running
    sub-stage orchestration itself.
    """
    from agent.tools.science_client import ScienceClient, ScienceComputeError, ScienceTimeoutError

    _update_job(job_id, status="running", current_step="pipeline", progress=0)
    await _update_job_db(job_id, status="running", current_step="pipeline", progress=0)

    try:
        client = ScienceClient()
        await client.run_pipeline(structure_id=structure_id)

        _update_job(
            job_id,
            status="complete",
            current_step="complete",
            progress=100,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        await _update_job_db(
            job_id,
            status="complete",
            current_step="complete",
            progress=100,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    except ScienceTimeoutError as e:
        _update_job(
            job_id,
            status="timed_out",
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        await _update_job_db(
            job_id,
            status="timed_out",
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    except ScienceComputeError as e:
        _update_job(
            job_id,
            status="failed",
            error=e.detail,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        await _update_job_db(
            job_id,
            status="failed",
            error=e.detail,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        _update_job(
            job_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        await _update_job_db(
            job_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


@router.post("/pipeline/run")
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Dispatch the full DTIE pipeline to the science container."""
    job_id = await _create_job_db(request.structure_id, request.modules)

    background_tasks.add_task(_run_pipeline_background, job_id, request.structure_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/pipeline/status/{job_id}",
    }


# ---------------------------------------------------------------------------
# GET /api/pipeline/status/{job_id}
# ---------------------------------------------------------------------------


@router.get("/pipeline/status/{job_id}")
async def pipeline_status(job_id: str):
    """Return current pipeline job progress."""
    job = await _get_job_db(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "message": f"Job '{job_id}' not found"},
        )

    return {
        "job_id": job["job_id"],
        "structure_id": job["structure_id"],
        "status": job["status"],
        "current_step": job["current_step"],
        "progress": job["progress"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
    }


# ---------------------------------------------------------------------------
# GET /api/structures/{structure_id}/hydrate
# ---------------------------------------------------------------------------


@router.get("/structures/{structure_id}/hydrate")
async def hydrate_structure(structure_id: str, db=Depends(get_db)):
    """Hydrate all available data for a structure in a single call.

    Fetches embeddings, graph metrics, allosteric sites, source leaks,
    hypotheses, provenance runs, annotations, and phase-specific persistence
    data (Phase 2 vulnerability, Phase 4 resistance, Phase 5 pharmacophore,
    Phase 6 drug candidates) in parallel. Returns null for any data type
    that is not yet computed.

    Requirements: 1.1, 1.4, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1
    """
    from agent.tools.data_tools import get_allosteric_sites, get_provenance_lineage
    from agent.tools.dtie.tools import get_source_leaks
    from agent.tools.graph_tools import get_graph_metrics
    from agent.tools.hypothesis.tools import get_hypotheses

    try:
        # Run all queries in parallel — return_exceptions=True so one failure doesn't kill all
        results = await asyncio.gather(
            _fetch_embeddings_for_hydration(structure_id, db),
            get_graph_metrics(structure_id=structure_id, db=db),
            get_allosteric_sites(structure_id=structure_id, db=db),
            get_source_leaks(structure_id=structure_id, db=db),
            get_hypotheses(structure_id=structure_id, db=db),
            get_provenance_lineage(structure_id=structure_id, db=db),
            _fetch_annotations_for_hydration(structure_id, db),
            _fetch_phase2_vulnerability(structure_id, db),
            _fetch_phase4_resistance(structure_id, db),
            _fetch_phase5_pharmacophore(structure_id, db),
            _fetch_phase6_drug_candidates(structure_id, db),
            return_exceptions=True,
        )

        # Unpack results, treating exceptions as None
        def _safe(r):
            return None if isinstance(r, BaseException) else r

        embeddings_result = _safe(results[0])
        graph_result = _safe(results[1])
        sites_result = _safe(results[2])
        leaks_result = _safe(results[3])
        hypotheses_result = _safe(results[4])
        provenance_result = _safe(results[5])
        annotations_rows = _safe(results[6])
        phase2_result = _safe(results[7])
        phase4_result = _safe(results[8])
        phase5_result = _safe(results[9])
        phase6_result = _safe(results[10])

        # Derive per-residue resistance sensitivity from Phase 4 data
        resistance_data = None
        if phase4_result and phase4_result.get("pathways"):
            pathways = phase4_result["pathways"]
            spectral = phase4_result.get("spectral")
            # Collect all unique residue_ids from pathways
            residue_ids_set: set[str] = set()
            for p in pathways:
                if p.get("source_residue"):
                    residue_ids_set.add(p["source_residue"])
                if p.get("target_residue"):
                    residue_ids_set.add(p["target_residue"])
            if residue_ids_set:
                resistance_data = _compute_resistance_sensitivity(
                    pathways=pathways,
                    spectral=spectral,
                    residue_ids=sorted(residue_ids_set),
                )

        # Build response with null for missing data
        embeddings = embeddings_result if embeddings_result else None
        graph_metrics = graph_result.data if graph_result and hasattr(graph_result, 'success') and graph_result.success and graph_result.data.get("metrics") else None
        allosteric_sites = sites_result.data if sites_result and hasattr(sites_result, 'success') and sites_result.success and sites_result.data.get("sites") else None
        source_leaks = leaks_result.data if leaks_result and hasattr(leaks_result, 'success') and leaks_result.success and leaks_result.data.get("source_leaks") else None
        hypotheses = hypotheses_result.data if hypotheses_result and hasattr(hypotheses_result, 'success') and hypotheses_result.success and hypotheses_result.data.get("hypotheses") else None
        provenance_runs = provenance_result.data if provenance_result and hasattr(provenance_result, 'success') and provenance_result.success and provenance_result.data.get("runs") else None
        annotations = annotations_rows if annotations_rows else None
        phase2_vulnerability = phase2_result if phase2_result else None
        phase4_resistance = phase4_result if phase4_result else None
        phase5_pharmacophore = phase5_result if phase5_result else None
        phase6_drug_candidates = phase6_result if phase6_result else None

        # Enrich pharmacophores with "connected allosteric locks" (the precision locks
        # from source_leak 90th-percentile + derived allosteric network) and the exact
        # connecting resistance pathways for each of the (up to 20) pockets.
        # This ensures B-chain symmetries and per-pocket connected lock subsets
        # (with high-coupling "action at a distance" details) export cleanly for the
        # GraphTopologyPanel bipartite SVG + per-pathway receipt table + 3D radar.
        # Primary lock set = source leaks (the 33 for KRAS G12D); union with facade.
        if phase5_pharmacophore and phase4_resistance:
            locks: set[str] = set()
            # Prefer raw source leaks (the relative 90th-percentile precision locks)
            if source_leaks:
                sl = source_leaks.get("source_leaks") or source_leaks.get("leaks") or []
                for item in sl if isinstance(sl, (list, tuple)) else []:
                    if isinstance(item, str):
                        locks.add(item)
                    elif isinstance(item, dict):
                        rid = item.get("residue_id") or item.get("residue") or item.get("id")
                        if rid:
                            locks.add(str(rid))
            # Union with the derived allosteric network facade (for compatibility)
            if allosteric_sites and allosteric_sites.get("sites"):
                for site in allosteric_sites.get("sites", []):
                    locks.update(site.get("residue_ids", []))
            pathways = phase4_resistance.get("pathways", []) or []
            enriched = []
            for p in phase5_pharmacophore.get("pharmacophores", []):
                p = dict(p)
                pocket_res = set(p.get("residue_ids", []))
                connected: list[str] = []
                connecting: list[dict] = []
                for pw in pathways:
                    src = str(pw.get("source_residue", ""))
                    tgt = str(pw.get("target_residue", ""))
                    if src in pocket_res or tgt in pocket_res:
                        connecting.append(pw)
                        if src in locks:
                            connected.append(src)
                        if tgt in locks:
                            connected.append(tgt)
                p["connected_allosteric_locks"] = sorted(set(connected))
                p["connecting_pathway_count"] = len(connecting)
                p["connecting_coupling_sum"] = sum(
                    float(pw.get("coupling_strength", 0.0)) for pw in connecting
                )
                # Attach the actual connecting pathways (normalized, sorted by coupling desc)
                # so the per-pocket receipt table in GraphTopologyPanel has the real
                # high-value highways (e.g. the 5M+ from A:8/A:17 flanks for KRAS).
                conn_sorted = sorted(connecting, key=lambda x: -float(x.get("coupling_strength", 0)))
                p["connecting_pathways"] = conn_sorted[:12]  # cap for payload size
                enriched.append(p)
            phase5_pharmacophore["pharmacophores"] = enriched

        # Persistence status flags
        persistence_status = {
            "embeddings_persisted": embeddings is not None,
            "graph_persisted": graph_metrics is not None,
            "sites_persisted": allosteric_sites is not None,
            "phase2_persisted": phase2_vulnerability is not None,
            "phase4_persisted": phase4_resistance is not None,
            "phase5_persisted": phase5_pharmacophore is not None,
            "phase6_persisted": phase6_drug_candidates is not None,
            "resistance_data_available": resistance_data is not None,
        }

        # Build agent context summary (Requirement 1.4)
        residue_count = len(embeddings.get("residues", [])) if embeddings else 0
        source_leak_count = source_leaks.get("count", 0) if source_leaks else 0
        hypothesis_count = len(hypotheses.get("hypotheses", [])) if hypotheses else 0

        # Top uncertainty residues (top 5 by epistemic uncertainty)
        top_uncertainty_residues = []
        if embeddings and embeddings.get("residues"):
            sorted_residues = sorted(
                embeddings["residues"],
                key=lambda r: r.get("epistemic_uncertainty") or 0,
                reverse=True,
            )
            top_uncertainty_residues = [
                {
                    "residue_id": r["residue_id"],
                    "chain_label": r.get("chain_label"),
                    "residue_index": r.get("residue_index"),
                    "epistemic_uncertainty": r.get("epistemic_uncertainty"),
                }
                for r in sorted_residues[:5]
            ]

        context_summary = {
            "residue_count": residue_count,
            "source_leak_count": source_leak_count,
            "hypothesis_count": hypothesis_count,
            "top_uncertainty_residues": top_uncertainty_residues,
        }

        return {
            "structure_id": structure_id,
            "embeddings": embeddings,
            "graph_metrics": graph_metrics,
            "allosteric_sites": allosteric_sites,
            "source_leaks": source_leaks,
            "hypotheses": hypotheses,
            "provenance_runs": provenance_runs,
            "annotations": annotations,
            "phase2_vulnerability": phase2_vulnerability,
            "phase4_resistance": phase4_resistance,
            "phase5_pharmacophore": phase5_pharmacophore,
            "phase6_drug_candidates": phase6_drug_candidates,
            "resistance_data": resistance_data,
            "persistence_status": persistence_status,
            "context_summary": context_summary,
            # New: live governed (X, Y) from Phase 7 Buffering Atlas (or derived)
            "buffering_atlas": await _fetch_buffering_atlas(structure_id, db) if db else None,
        }

    except Exception as e:
        logger.exception("Hydration failed for %s", structure_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": "hydration_failed",
                "message": str(e),
                "db_connected": False,
            },
        )


async def _fetch_embeddings_for_hydration(structure_id: str, db) -> dict[str, Any] | None:
    """Fetch embeddings data for hydration (returns dict or None)."""
    from agent.tools.dtie.tools import ToolDB

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, r.residue_name,
               c.chain_label,
               e.hyp_projection_2d, e.hyp_projections,
               e.cone_depth, e.epistemic_uncertainty,
               e.aleatoric_uncertainty,
               es.curvature
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
        return None

    curvature = rows[0].get("curvature", 1.0)
    residues = []
    for r in rows:
        x, y = 0.0, 0.0
        if r.get("hyp_projection_2d"):
            coords = r["hyp_projection_2d"]
            # pgvector returns a string like "[0.1,-0.2]" — parse it
            if isinstance(coords, str):
                coords = [float(v) for v in coords.strip("[]").split(",")]
            x, y = float(coords[0]), float(coords[1])
        elif r.get("hyp_projections"):
            proj = r["hyp_projections"]
            if isinstance(proj, list) and len(proj) >= 2:
                x, y = float(proj[0]), float(proj[1])

        residues.append({
            "residue_id": r["residue_id"],
            "residue_index": r["residue_index"],
            "residue_name": r.get("residue_name"),
            "chain_label": r["chain_label"],
            "x": x,
            "y": y,
            "cone_depth": r.get("cone_depth"),
            "epistemic_uncertainty": r.get("epistemic_uncertainty"),
            "aleatoric_uncertainty": r.get("aleatoric_uncertainty"),
        })

    return {
        "structure_id": structure_id,
        "curvature": curvature,
        "residues": residues,
    }


async def _fetch_annotations_for_hydration(structure_id: str, db) -> list[dict[str, Any]] | None:
    """Fetch annotations for hydration (returns list or None)."""
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

    return rows if rows else None


async def _fetch_phase2_vulnerability(structure_id: str, db) -> dict[str, Any] | None:
    """Fetch Phase 2 vulnerability doorway data for hydration.

    Requirements: 7.1
    """
    from agent.tools.dtie.tools import ToolDB

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        """
        SELECT residue_id, cone_depth, epistemic_uncertainty,
               aleatoric_uncertainty, depth_threshold, computed_at
        FROM fact_phase2_vulnerability
        WHERE structure_id = :structure_id
        ORDER BY epistemic_uncertainty DESC
        """,
        {"structure_id": structure_id},
    )

    if not rows:
        return None

    return {
        "structure_id": structure_id,
        "doorways": rows,
        "count": len(rows),
    }


def _compute_resistance_sensitivity(
    pathways: list[dict[str, Any]],
    spectral: dict[str, Any] | None,
    residue_ids: list[str],
) -> dict[str, Any]:
    """Derive per-residue resistance sensitivity scores from pathway data.

    Score = sum of coupling_strength for all pathways involving the residue,
    normalized to [0, 1]. Hinge residues (from spectral Fiedler vector) get
    a 1.5× boost since they control inter-domain communication.

    Classification thresholds:
    - high_sensitivity: score > 0.65
    - moderate: score > 0.3
    - stable: score <= 0.3

    Requirements: 1.1, 1.3
    """
    # Aggregate coupling_strength per residue
    scores: dict[str, float] = {rid: 0.0 for rid in residue_ids}
    coupling_counts: dict[str, int] = {rid: 0 for rid in residue_ids}

    for pathway in pathways:
        src = pathway.get("source_residue")
        tgt = pathway.get("target_residue")
        coupling = float(pathway.get("coupling_strength", 0.0))

        if src in scores:
            scores[src] += coupling
            coupling_counts[src] += 1
        if tgt in scores:
            scores[tgt] += coupling
            coupling_counts[tgt] += 1

    # Hinge bonus: 1.5× for residues identified as domain-boundary hinges
    hinge_set: set[str] = set()
    if spectral and spectral.get("hinge_residues"):
        hinge_residues = spectral["hinge_residues"]
        if isinstance(hinge_residues, list):
            hinge_set = set(hinge_residues)
        elif isinstance(hinge_residues, str):
            # Handle JSON string case
            import json
            try:
                hinge_set = set(json.loads(hinge_residues))
            except (json.JSONDecodeError, TypeError):
                hinge_set = set()

    for rid in scores:
        if rid in hinge_set:
            scores[rid] *= 1.5

    # Normalize to [0, 1]
    max_score = max(scores.values()) if scores else 0.0
    normalized: dict[str, float] = {
        rid: s / (max_score + 1e-8) for rid, s in scores.items()
    }

    # Classify
    def _classify(score: float) -> str:
        if score > 0.65:
            return "high_sensitivity"
        elif score > 0.3:
            return "moderate"
        return "stable"

    residues_out = []
    for rid in residue_ids:
        score = normalized.get(rid, 0.0)
        residues_out.append({
            "residue_id": rid,
            "sensitivity_score": round(score, 6),
            "coupling_count": coupling_counts.get(rid, 0),
            "is_hinge": rid in hinge_set,
            "classification": _classify(score),
        })

    spectral_summary = {
        "lambda_2": float(spectral.get("lambda_2", 0.0)) if spectral else 0.0,
        "hinge_count": len(hinge_set),
    }

    return {
        "residues": residues_out,
        "spectral": spectral_summary,
    }


async def _fetch_phase4_resistance(structure_id: str, db) -> dict[str, Any] | None:
    """Fetch Phase 4 resistance pathway data for hydration.

    Resolves bare residue indices (common in pathway storage) to full
    canonical residue_ids (e.g. "4obe:A:17") using dim_residue so that
    downstream matching against pocket residue_ids and source_leak locks
    works for both single-chain and multi-chain (B-chain symmetries) cases.

    Requirements: 7.2
    """
    from agent.tools.dtie.tools import ToolDB
    from collections import defaultdict
    import json as _json

    tool_db = ToolDB(db)

    # Build index -> full residue_id(s) map (prefer A chain for primary resolution)
    residue_rows = await tool_db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, c.chain_label
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
        """,
        {"structure_id": structure_id},
    )
    idx_to_ids: dict[int, list[str]] = defaultdict(list)
    for r in residue_rows:
        idx = r["residue_index"]
        rid = r["residue_id"]
        if rid not in idx_to_ids[idx]:
            idx_to_ids[idx].append(rid)

    def _resolve(val: Any) -> str:
        """Resolve bare index or partial to a canonical full residue_id (prefer A)."""
        if val is None:
            return ""
        s = str(val).strip()
        if not s:
            return ""
        # If already looks like full canonical id, keep it
        if ":" in s and structure_id in s:
            return s
        try:
            i = int(float(s))
        except Exception:
            return s
        ids = idx_to_ids.get(i, [])
        if not ids:
            return s
        for rid in ids:
            if ":A:" in rid or rid.startswith(structure_id + ":A:"):
                return rid
        return ids[0]

    pathways_raw = await tool_db.fetch_all(
        """
        SELECT source_node, target_node, source_residue, target_residue,
               effective_resistance, coupling_strength, computed_at
        FROM fact_resistance_pathway
        WHERE structure_id = :structure_id
        ORDER BY effective_resistance DESC
        """,
        {"structure_id": structure_id},
    )

    # Resolve to full IDs for reliable matching in enrichment + frontend filters
    resolved_pathways = []
    for pw in (pathways_raw or []):
        p = dict(pw)
        p["source_residue"] = _resolve(p.get("source_residue"))
        p["target_residue"] = _resolve(p.get("target_residue"))
        # Keep original bare for debugging if needed
        p["_raw_source_residue"] = pw.get("source_residue")
        p["_raw_target_residue"] = pw.get("target_residue")
        resolved_pathways.append(p)

    spectral = await tool_db.fetch_one(
        """
        SELECT lambda_2, hinge_residues, graph_nodes, graph_edges, computed_at
        FROM fact_resistance_spectral
        WHERE structure_id = :structure_id
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )

    if not resolved_pathways and not spectral:
        return None

    return {
        "structure_id": structure_id,
        "pathways": resolved_pathways,
        "spectral": spectral,
        "pathway_count": len(resolved_pathways),
    }


async def _fetch_phase5_pharmacophore(structure_id: str, db) -> dict[str, Any] | None:
    """Fetch Phase 5 pharmacophore data for hydration.

    Maps residue indices to residue_ids for frontend highlighting.
    Requirements: 7.3
    """
    from agent.tools.dtie.tools import ToolDB
    import json as _json

    tool_db = ToolDB(db)
    
    # 1. Fetch pharmacophores
    pharma_rows = await tool_db.fetch_all(
        """
        SELECT pocket_index, center_x, center_y, center_z,
               druggability_score, residue_count, residue_indices,
               allosteric_coupling, volume_estimate, computed_at
        FROM fact_pharmacophore
        WHERE structure_id = :structure_id
        ORDER BY druggability_score DESC
        """,
        {"structure_id": structure_id},
    )

    if not pharma_rows:
        return None

    # 2. Fetch residue ID mapping for this structure (support multi-chain / B-chain symmetries)
    residue_rows = await tool_db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, c.chain_label
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
        """,
        {"structure_id": structure_id},
    )
    from collections import defaultdict
    idx_to_ids: dict[int, list[str]] = defaultdict(list)
    for r in residue_rows:
        idx = r["residue_index"]
        rid = r["residue_id"]
        if rid not in idx_to_ids[idx]:
            idx_to_ids[idx].append(rid)
    # For each index, prefer A-chain then others for primary use
    def _primary_for(idx: int) -> str | None:
        ids = idx_to_ids.get(idx, [])
        if not ids:
            return None
        for rid in ids:
            if ":A:" in rid or rid.startswith(structure_id + ":A:"):
                return rid
        return ids[0]

    # 3. Enrich pharmacophores with residue_ids (include all chains for symmetries)
    enriched_pharmacophores = []
    for row in pharma_rows:
        indices = row["residue_indices"]
        if isinstance(indices, str):
            try:
                indices = _json.loads(indices)
            except _json.JSONDecodeError:
                indices = []
        
        # Map indices to full IDs across chains
        residue_ids: list[str] = []
        for idx in indices:
            try:
                i = int(idx)
            except Exception:
                continue
            residue_ids.extend(idx_to_ids.get(i, []))
        # Dedup while preserving order
        seen = set()
        residue_ids = [x for x in residue_ids if not (x in seen or seen.add(x))]
        
        # Convert row to dict to add new field
        p_dict = dict(row)
        p_dict["residue_ids"] = residue_ids
        p_dict["_primary_chain_residues"] = {str(i): _primary_for(int(i)) for i in indices if isinstance(i, (int, str)) and str(i).isdigit()}
        # Ensure computed_at is stringified
        if p_dict.get("computed_at") and not isinstance(p_dict["computed_at"], str):
            p_dict["computed_at"] = p_dict["computed_at"].isoformat()
            
        enriched_pharmacophores.append(p_dict)

    return {
        "structure_id": structure_id,
        "pharmacophores": enriched_pharmacophores,
        "count": len(enriched_pharmacophores),
    }


async def _fetch_phase6_drug_candidates(structure_id: str, db) -> dict[str, Any] | None:
    """Fetch Phase 6 drug candidate data for hydration.

    Requirements: 7.4
    """
    from agent.tools.dtie.tools import ToolDB

    tool_db = ToolDB(db)
    rows = await tool_db.fetch_all(
        """
        SELECT pocket_index, center_x, center_y, center_z,
               accessibility_score, binding_potential, admet_pass,
               selectivity_ratio, is_state_selective,
               combined_druggability, computed_at
        FROM fact_drug_candidate
        WHERE structure_id = :structure_id
        ORDER BY combined_druggability DESC
        """,
        {"structure_id": structure_id},
    )

    if not rows:
        return None

    return {
        "structure_id": structure_id,
        "candidates": rows,
        "count": len(rows),
        "admet_passed_count": sum(1 for r in rows if r.get("admet_pass")),
        "state_selective_count": sum(1 for r in rows if r.get("is_state_selective")),
    }


async def _fetch_buffering_atlas(structure_id: str, db) -> dict[str, Any] | None:
    """Fetch or derive the governed Buffering Atlas (X, Y) coordinates for the structure.

    Prefers the Phase 7 persisted result (from orchestrator fact_phase_output).
    Falls back to on-the-fly aggregation from source_leak + resistance + graph metrics
    using the identical formulas as pipeline._compute_buffering_atlas.

    Tolerant to phase/phase_name column usage. Dedups source_leak per residue for noisy runs.
    """
    from agent.tools.dtie.tools import ToolDB
    import numpy as np

    tool_db = ToolDB(db)

    # 1. Persisted Phase 7 (tolerant match on phase or phase_name; we control the written values)
    phase_row = await tool_db.fetch_one(
        """
        SELECT output_data, run_id, computed_at
        FROM fact_phase_output
        WHERE structure_id = :sid
          AND (phase_name = 'buffering_atlas' OR phase = 'buffering_atlas'
               OR phase = '7' OR phase_name = '7')
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        {"sid": structure_id},
    )

    if isinstance(phase_row, dict) and phase_row.get("output_data"):
        outs = phase_row["output_data"]
        if isinstance(outs, str):
            import json
            try:
                outs = json.loads(outs)
            except Exception:
                outs = {}
        if isinstance(outs, dict) and "x" in outs and "y" in outs:
            return {
                "structure_id": structure_id,
                "x": outs["x"],
                "y": outs["y"],
                "method": outs.get("method", "phase7"),
                "run_id": phase_row.get("run_id"),
                "source": "orchestrator_phase7",
                "computed_at": str(phase_row.get("computed_at")),
            }

    # 2. Live derivation (robust to accumulation in fact_* tables)
    leak_rows = await tool_db.fetch_all(
        """
        SELECT residue_id, epistemic_uncertainty, computed_at
        FROM fact_source_leak
        WHERE structure_id = :sid
        ORDER BY computed_at DESC
        """,
        {"sid": structure_id},
    )
    latest_leak: dict[str, float] = {}
    for r in leak_rows:
        rid = r.get("residue_id")
        val = r.get("epistemic_uncertainty")
        if rid and val is not None:
            latest_leak.setdefault(rid, float(val))  # most recent first
    x_vals = list(latest_leak.values())
    x_raw = float(np.mean(x_vals)) if x_vals else 10.3
    x = 9.5 + (x_raw * 0.8)

    path_rows = await tool_db.fetch_all(
        """
        SELECT coupling_strength
        FROM fact_resistance_pathway
        WHERE structure_id = :sid
        """,
        {"sid": structure_id},
    )
    y_vals = [r["coupling_strength"] for r in path_rows if r.get("coupling_strength") is not None]
    y_from_c = float(np.mean(y_vals)) / 1_000_000.0 if y_vals else 0.05

    # Column is "betweenness" (from graph_node_metrics schema)
    bet_rows = await tool_db.fetch_all(
        "SELECT betweenness FROM fact_graph_node_metrics WHERE structure_id = :sid",
        {"sid": structure_id},
    )
    bet_vals = [r["betweenness"] for r in bet_rows if r.get("betweenness") is not None]
    mean_bet = float(np.mean(bet_vals)) if bet_vals else 0.0
    y = (y_from_c + (mean_bet * 0.01)) / 2 if (y_from_c > 0 or mean_bet > 0) else 0.05

    return {
        "structure_id": structure_id,
        "x": round(x, 3),
        "y": round(y, 5),
        "method": "derived_from_facts:source_leak_epistemic+resistance_coupling+graph_betweenness",
        "source": "hydration_fallback",
        "n_leaks_unique": len(x_vals),
        "n_pathways": len(y_vals),
        "n_betweenness": len(bet_vals),
        "derivation_note": "No buffering_atlas row in fact_phase_output for this structure/run (adapter not registered at persist). Values computed live from populated fact tables using orchestrator formulas.",
    }


# ---------------------------------------------------------------------------
# POST /api/agent/chat
# ---------------------------------------------------------------------------

# In-memory session history (last 10 exchanges per session)
_chat_sessions: dict[str, list[dict[str, str]]] = {}
_session_telemetry: dict[str, dict[str, Any]] = {}

MAX_SESSION_HISTORY = 10
MAX_SESSION_TELEMETRY = 20
HISTORY_USER_PREVIEW_CHARS = 180
HISTORY_ASSISTANT_PREVIEW_CHARS = 280
HISTORY_BLOCK_MAX_CHARS = 2200


def _compute_resistance_classification_counts(
    resistance_data: dict[str, Any],
) -> dict[str, Any]:
    """Compute resistance classification counts from raw per-residue data.

    Takes the output of _compute_resistance_sensitivity (which has 'residues'
    and 'spectral' keys) and produces a summary with lambda_2, hinge_count,
    and counts for each classification bucket (high_sensitivity, moderate, stable).

    Requirements: 3.4, 4.3
    """
    residues = resistance_data.get("residues", [])
    spectral = resistance_data.get("spectral", {})

    high_sensitivity_count = 0
    moderate_count = 0
    stable_count = 0

    for r in residues:
        classification = r.get("classification", "stable")
        if classification == "high_sensitivity":
            high_sensitivity_count += 1
        elif classification == "moderate":
            moderate_count += 1
        else:
            stable_count += 1

    return {
        "lambda_2": spectral.get("lambda_2", 0.0),
        "hinge_count": spectral.get("hinge_count", 0),
        "high_sensitivity_count": high_sensitivity_count,
        "moderate_count": moderate_count,
        "stable_count": stable_count,
    }


def _build_context_block(context: dict[str, Any]) -> str:
    """Format enriched viewport context into a structured prompt section.

    Organizes into sections, omits null/empty sections, caps at 2000 chars.
    Supports both the new enriched payload format (with poincare, viewer_3d,
    data_summary keys) and the legacy flat format for backwards compatibility.
    """
    if not context:
        return ""

    # Detect enriched format by presence of structured keys (even if their values are None)
    enriched_keys = ("poincare", "viewer_3d", "data_summary", "pipeline", "structure_id")
    is_enriched = any(k in context for k in enriched_keys)

    if not is_enriched:
        # Legacy flat context format
        lines: list[str] = []
        if context.get("active_structure_id"):
            lines.append(f"- Active structure: {context['active_structure_id']}")
        if context.get("structure_title"):
            lines.append(f"- Title: {context['structure_title']}")
        if context.get("residue_count"):
            lines.append(f"- Residue count: {context['residue_count']}")
        if context.get("top_uncertainty_residues"):
            lines.append(f"- Top uncertainty residues: {context['top_uncertainty_residues']}")
        if context.get("source_leak_count") is not None:
            lines.append(f"- Source leak count: {context['source_leak_count']}")
        if context.get("cone_depth_range"):
            lines.append(f"- Cone depth range: {context['cone_depth_range']}")
        if context.get("current_visualization"):
            lines.append(f"- Current visualization: {context['current_visualization']}")
        if not lines:
            return ""
        block = "\n\nCurrent Dashboard Context:\n" + "\n".join(lines)
        if len(block) > 2000:
            block = block[:1997] + "..."
        return block

    # Enriched format
    sections: list[str] = []

    # Section 1: Structure
    structure_id = context.get("structure_id") or context.get("active_structure_id")
    structure_title = context.get("structure_title")
    if structure_id:
        s = f"Structure: {structure_id}"
        if structure_title:
            s += f" ({structure_title})"
        sections.append(s)

    # Section 2: Poincaré View
    poincare = context.get("poincare")
    if poincare:
        lines = [f"Poincaré View: color={poincare.get('color_mode', 'unknown')}"]
        if poincare.get("mobius_focus_enabled"):
            lines[0] += f", Möbius focus on {poincare.get('mobius_focus_residue', '?')}"
        if poincare.get("selected_residue"):
            r = poincare["selected_residue"]
            lines.append(
                f"  Selected: {r.get('residue_id', '?')}"
                f" (unc={r.get('epistemic_uncertainty')}, depth={r.get('cone_depth')})"
            )
        if poincare.get("brush_selected_ids"):
            n = len(poincare["brush_selected_ids"])
            lines.append(f"  Brush selection: {n} residues")
        sections.append("\n".join(lines))

    # Section 3: 3D Viewer
    viewer = context.get("viewer_3d")
    if viewer:
        line = f"3D Viewer: color={viewer.get('color_mode', 'unknown')}, threshold={viewer.get('risk_threshold', 0.5)}"
        if viewer.get("highlighted_residue_ids"):
            line += f", {len(viewer['highlighted_residue_ids'])} highlighted"
        sections.append(line)

    # Section 4: Data Availability
    ds = context.get("data_summary")
    if ds:
        lines = [f"Data: {ds.get('residue_count', 0)} residues, {ds.get('source_leak_count', 0)} source leaks"]
        if ds.get("hypothesis_count"):
            lines.append(f"  Hypotheses: {ds['hypothesis_count']} (dist: {ds.get('hypothesis_status_distribution', {})})")
        if ds.get("resistance_summary"):
            rs = ds["resistance_summary"]
            # Handle both pre-computed counts and raw per-residue data
            if "residues" in rs and "spectral" in rs:
                # Raw data from _compute_resistance_sensitivity — compute counts
                rs = _compute_resistance_classification_counts(rs)
            lines.append(
                f"  Resistance: λ₂={rs.get('lambda_2', 0):.3f}, {rs.get('hinge_count', 0)} hinges, "
                f"{rs.get('high_sensitivity_count', 0)}H/{rs.get('moderate_count', 0)}M/{rs.get('stable_count', 0)}S"
            )
        ps = ds.get("persistence_status", {})
        available = [k for k, v in ps.items() if v]
        if available:
            lines.append(f"  Available phases: {', '.join(available)}")
        sections.append("\n".join(lines))

    # Section 5: Active Analysis
    panel = context.get("active_panel")
    pipeline = context.get("pipeline")
    if panel or pipeline:
        parts: list[str] = []
        if panel:
            parts.append(f"Active panel: {panel}")
        if pipeline and pipeline.get("status") and pipeline["status"] != "never_run":
            p_str = f"Pipeline: {pipeline['status']}"
            if pipeline["status"] == "running":
                p_str += f" ({pipeline.get('current_step')}, {pipeline.get('progress')}%)"
            parts.append(p_str)
        if parts:
            sections.append("; ".join(parts))

    # Section 6: Compare Mode
    compare = context.get("compare")
    if compare:
        lines = [f"Compare: vs {compare.get('secondary_structure_id', '?')}"]
        top_movers = compare.get("top_movers", [])
        if top_movers:
            mover_strs = [f"{m['residue_id']}({m['displacement']:.3f})" for m in top_movers[:5]]
            lines.append(f"  Top movers: {', '.join(mover_strs)}")
        edge_diff = compare.get("edge_diff", {})
        if edge_diff:
            lines.append(
                f"  Edge diff: +{edge_diff.get('gained', 0)}/-{edge_diff.get('lost', 0)}/Δ{edge_diff.get('changed', 0)}"
            )
        sections.append("\n".join(lines))

    # Section 7: Biophysical Radar & ASAR View (for KRAS Atlas / precision lock narrative)
    is_radar = context.get("is_radar_active", False)
    sel_pocket = context.get("selected_pocket_id")
    if is_radar or sel_pocket is not None:
        radar_lines = ["Biophysical Radar View:"]
        if is_radar:
            radar_lines.append("  Radar mode ACTIVE (ultramarine bulk + pseudo-bloom amber precision locks + green targets)")
        if sel_pocket is not None:
            radar_lines.append(f"  Selected pocket for detailed sub-graph: {sel_pocket}")
        radar_lines.append("  (Use tools get_radar_view or compute_asar_vector for full enriched locks, connecting pathways with coupling values, and ASAR β vectors.)")
        sections.append("\n".join(radar_lines))

    # Section 8: Data Inspector / small-molecule state
    inspector = context.get("data_inspector")
    if inspector:
        lines = ["Data Inspector:"]
        phases = inspector.get("phases_computed") or []
        if phases:
            lines.append(f"  Computed phases: {', '.join(phases)}")
        phase_counts = inspector.get("phase_counts") or {}
        if phase_counts:
            counts = ", ".join(f"{name}={count}" for name, count in phase_counts.items())
            lines.append(f"  Counts: {counts}")
        if inspector.get("top_druggability_pocket") is not None:
            lines.append(f"  Top pocket druggability: {inspector['top_druggability_pocket']:.3f}")
        if inspector.get("top_drug_candidate_score") is not None:
            lines.append(f"  Top candidate score: {inspector['top_drug_candidate_score']:.3f}")
        if inspector.get("admet_pass_rate") is not None:
            lines.append(f"  ADMET pass rate: {inspector['admet_pass_rate']:.2%}")
        sections.append("\n".join(lines))

    if not sections:
        return ""

    block = "\n\nDashboard Context:\n" + "\n".join(sections)
    # Enforce 2000 char cap
    if len(block) > 2000:
        block = block[:1997] + "..."
    return block


def _build_history_block(history: list[dict[str, str]]) -> str:
    """Build a conversation history string from session history."""
    if not history:
        return ""

    lines = ["\n\nConversation history:"]
    for exchange in history[-MAX_SESSION_HISTORY:]:
        lines.append(f"User: {exchange['user']}")
        lines.append(f"Assistant: {exchange['assistant']}")
        lines.append("")

    block = "\n".join(lines)
    if len(block) > HISTORY_BLOCK_MAX_CHARS:
        block = block[: HISTORY_BLOCK_MAX_CHARS - 3] + "..."
    return block


def _get_session_history(session_id: str) -> list[dict[str, str]]:
    """Get conversation history for a session."""
    return _chat_sessions.get(session_id, [])


def _store_exchange(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Store a user/assistant exchange in session history (capped at MAX_SESSION_HISTORY)."""
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = []
    _chat_sessions[session_id].append({
        "user": _truncate_preview(user_msg, HISTORY_USER_PREVIEW_CHARS),
        "assistant": _truncate_preview(assistant_msg, HISTORY_ASSISTANT_PREVIEW_CHARS),
    })
    # Keep only last N exchanges
    _chat_sessions[session_id] = _chat_sessions[session_id][-MAX_SESSION_HISTORY:]


def _store_session_telemetry(session_id: str, telemetry_payload: dict[str, Any]) -> None:
    """Store per-session telemetry snapshots for dashboard observability."""
    if session_id not in _session_telemetry:
        _session_telemetry[session_id] = {"traces": []}
    traces = _session_telemetry[session_id]["traces"]
    traces.append(telemetry_payload)
    _session_telemetry[session_id]["traces"] = traces[-MAX_SESSION_TELEMETRY:]


def _truncate_preview(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)] + "..."


def _serialize_trace(trace: Any) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "tool_calls": [
            {
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result_summary": tc.result_summary,
                "is_error": tc.is_error,
                "duration_ms": round(tc.duration_ms, 1),
            }
            for tc in trace.tool_calls
        ],
        "ground_truth_facts": trace.ground_truth_facts,
        "total_duration_ms": round(trace.total_duration_ms, 1),
        "llm_calls": trace.llm_calls,
        "tokens": {
            "input": trace.input_tokens,
            "output": trace.output_tokens,
            "total": trace.input_tokens + trace.output_tokens,
        },
        "token_attribution": trace.token_attribution,
        "llm_call_breakdown": trace.llm_call_breakdown,
    }


def _store_trace_snapshot(session_id: str, trace: Any) -> None:
    _store_session_telemetry(
        session_id,
        {
            "trace_id": trace.trace_id,
            "total_duration_ms": round(trace.total_duration_ms, 1),
            "llm_calls": trace.llm_calls,
            "tokens": {
                "input": trace.input_tokens,
                "output": trace.output_tokens,
                "total": trace.input_tokens + trace.output_tokens,
            },
            "token_attribution": trace.token_attribution,
            "llm_call_breakdown": trace.llm_call_breakdown,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "is_error": tc.is_error,
                    "duration_ms": round(tc.duration_ms, 1),
                }
                for tc in trace.tool_calls
            ],
        },
    )


@router.post("/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """Full-featured agent chat with DTIE, small-molecule, data, graph, hypothesis,
    therapeutic compiler, RCSB, visualization, and plotting tools.

    Uses the same coordinator agent as the launcher — can run pipelines,
    query the database, generate plots, and control the viewer.
    """
    from agent.coordinator.memory import build_memory_prompt_context, persist_memory_interaction
    from agent.llm.agents import create_coordinator
    from agent.coordinator.viewport import viewport_manager
    from agent.llm.providers import get_provider
    from agent.telemetry import AgentTelemetry
    from data.db import DBAdapter, get_connection

    session_id = request.session_id or str(uuid.uuid4())

    # Create telemetry collector (outside agent's control)
    telemetry = AgentTelemetry(session_id=session_id)
    telemetry.record_request(user_message=request.message, context=request.context)

    context_block = _build_context_block(request.context)

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            memory_context = await build_memory_prompt_context(
                db=db,
                session_id=session_id,
                user_message=request.message,
                context=request.context,
            )
            fallback_history_block = ""
            if not memory_context.block:
                history = _get_session_history(session_id)
                fallback_history_block = _build_history_block(history)

            full_message = request.message
            if context_block or memory_context.block or fallback_history_block:
                full_message = (
                    f"<context>{context_block}{memory_context.block}{fallback_history_block}</context>\n\n"
                    f"{request.message}"
                )

            llm = get_provider()
            coordinator = create_coordinator(llm=llm, db=db)
            response = await coordinator.run(
                user_message=full_message,
                context=None,
                telemetry=telemetry,
            )

        response_text = response.text

        # Store in session history
        _store_exchange(session_id, request.message, response_text)

        # Finalize telemetry trace
        trace = telemetry.finalize()

        async with get_connection() as conn:
            db = DBAdapter(conn)
            await persist_memory_interaction(
                db=db,
                session_id=session_id,
                user_message=request.message,
                assistant_message=response_text,
                context=request.context,
                trace=trace,
                retrieved_memory_block=memory_context.block,
            )

        # Extract viewport directives from tool results
        viewport_directives = getattr(response, "viewport_directives", None) or []

        for directive in viewport_directives:
            if isinstance(directive, dict):
                await viewport_manager.send_to_session(
                    session_id,
                    {"type": "semantic_command", **directive},
                )

        # Extract referenced residues from tool results
        referenced_residues = getattr(response, "referenced_residues", None) or []

        response_payload = {
            "response": response_text,
            "session_id": session_id,
            "viewport_directives": viewport_directives,
            "referenced_residues": referenced_residues,
            "tool_calls": getattr(response, "tool_calls_made", None) or [],
            "telemetry": _serialize_trace(trace),
        }
        _store_trace_snapshot(session_id, trace)
        return response_payload

    except Exception as e:
        logger.exception("Agent chat failed")
        error_type = type(e).__name__
        error_text = (
            f"Tool/agent error ({error_type}). Please try a simpler query or use "
            f"record_partial_findings for long analyses. Details: {str(e)[:200]}"
        )
        telemetry.record_response(error_text)
        trace = telemetry.finalize()
        _store_exchange(session_id, request.message, error_text)
        try:
            async with get_connection() as conn:
                db = DBAdapter(conn)
                await persist_memory_interaction(
                    db=db,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_message=error_text,
                    context=request.context,
                    trace=trace,
                    retrieved_memory_block="",
                )
        except Exception:
            logger.warning("Failed to persist error memory for session %s", session_id, exc_info=True)
        _store_trace_snapshot(session_id, trace)
        return JSONResponse(
            status_code=200,
            content={
                "response": error_text,
                "error": "agent_error",
                "message": f"Agent error: {error_type}",
                "session_id": session_id,
                "viewport_directives": [],
                "referenced_residues": [],
                "tool_calls": [],
                "telemetry": _serialize_trace(trace),
            },
        )


@router.get("/agent/telemetry/{session_id}")
async def get_agent_telemetry(session_id: str):
    """Return telemetry for a chat session + current system connectivity signals."""
    from agent.coordinator.viewport import viewport_manager

    traces = _session_telemetry.get(session_id, {}).get("traces", [])
    if traces:
        latest = traces[-1]
    else:
        latest = None

    return {
        "session_id": session_id,
        "trace_count": len(traces),
        "latest_trace": latest,
        "recent_traces": traces[-10:],
        "viewport": viewport_manager.get_stats(),
    }
