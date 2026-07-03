"""Compute endpoints for the Science API.

Exposes GNN inference, full pipeline, cryptic scan, motif analysis,
and MD validation as HTTP POST endpoints.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from data.db import DBAdapter, get_connection
from science.contracts.model_registry import (
    build_models_api_payload,
    compute_checkpoint_sha256,
    get_production_api_alias,
    get_production_checkpoint_path,
    resolve_checkpoint_file,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class GNNRequest(BaseModel):
    structure_id: str
    model_version: str = Field(default_factory=get_production_api_alias)
    device: str = "cpu"
    checkpoint_path: str = Field(default_factory=get_production_checkpoint_path)


class GNNResponse(BaseModel):
    run_id: str
    structure_id: str
    node_count: int
    checkpoint_version_hash: str
    duration_ms: float


class PipelineRequest(BaseModel):
    structure_id: str
    source_leak_only: bool = False
    detect_source_leaks: bool | None = None
    identify_allosteric_sites: bool | None = None
    run_buffering_atlas: bool | None = None
    run_binding_site_scan: bool | None = None
    run_graph_topology: bool | None = None
    run_phase2: bool | None = None
    run_phase35: bool | None = None
    run_phase4: bool | None = None
    run_phase5: bool | None = None
    run_phase6: bool | None = None
    run_gnn: bool | None = None
    device: str = "cpu"
    checkpoint_path: str = Field(default_factory=get_production_checkpoint_path)


class ComputeJobRequest(BaseModel):
    structure_id: str
    pathway: str = "discovery_story"
    computation_run_id: str | None = None
    parent_run_id: str | None = None
    pipeline_job_id: str | None = None
    device: str = "cpu"
    checkpoint_path: str = Field(default_factory=get_production_checkpoint_path)
    job_params: dict[str, Any] = Field(default_factory=dict)


class ComputeJobResponse(BaseModel):
    job_id: str
    run_id: str
    structure_id: str
    success: bool
    artifacts_produced: list[str]
    duration_ms: float
    outputs: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PipelineResponse(BaseModel):
    run_id: str
    structure_id: str
    phases_run: list[str]
    assets_created: int
    duration_ms: float
    warnings: list[str] = []


class CrypticScanRequest(BaseModel):
    structure_id: str
    candidate_percentile: float = 75.0
    cluster_distance_angstrom: float = 8.0
    min_cluster_size: int = 3
    max_pockets: int = 10


class CrypticScanResponse(BaseModel):
    run_id: str
    structure_id: str
    sites_found: int
    sites: list[dict[str, Any]]
    duration_ms: float


class MotifAnalysisRequest(BaseModel):
    structure_id: str
    min_cluster_size: int = 5
    min_samples: int = 3


class MotifAnalysisResponse(BaseModel):
    run_id: str
    structure_id: str
    motif_count: int
    motifs: list[dict[str, Any]]
    duration_ms: float


class MDValidateRequest(BaseModel):
    structure_id: str
    site_id: str
    duration_ns: float = 10.0
    temperature_k: float = 310.0
    force_field: str = "amber14-all"
    dry_run: bool = False


class MDValidateResponse(BaseModel):
    run_id: str
    structure_id: str
    site_id: str
    pocket_open_fraction: float | None
    confidence_delta: float | None
    duration_ms: float
    dry_run: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_status_for_job_failure(result: Any) -> int:
    missing = result.outputs.get("missing_artifacts") if hasattr(result, "outputs") else None
    if missing:
        return 422
    return 500


def _compute_job_response(result: Any, *, duration_ms: float) -> ComputeJobResponse:
    return ComputeJobResponse(
        job_id=result.job_id,
        run_id=result.run_id,
        structure_id=result.structure_id,
        success=result.success,
        artifacts_produced=result.artifacts_produced,
        duration_ms=round(duration_ms, 1),
        outputs=result.outputs,
        warnings=result.warnings,
    )


def _compute_checkpoint_hash(checkpoint_path: str) -> str:
    """Compute a SHA-256 hash of the checkpoint file for provenance tagging."""
    digest = compute_checkpoint_sha256(checkpoint_path)
    if digest is None:
        resolved = resolve_checkpoint_file(checkpoint_path)
        detail = (
            f"Checkpoint not found: {checkpoint_path}"
            if resolved is None
            else f"Checkpoint unreadable: {checkpoint_path}"
        )
        raise FileNotFoundError(detail)
    return digest


# ---------------------------------------------------------------------------
# GNN Inference Endpoint
# ---------------------------------------------------------------------------


@router.post("/gnn", response_model=GNNResponse)
async def run_gnn_inference(request: GNNRequest) -> GNNResponse:
    """Run GNN inference — alias for ``POST /compute/jobs/gnn_inference``."""
    start = time.monotonic()
    try:
        checkpoint_hash = _compute_checkpoint_hash(request.checkpoint_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    async with get_connection() as conn:
        db = DBAdapter(conn)
        from science.compute.dispatch_helpers import dispatch_job_in_process

        result = await dispatch_job_in_process(
            db,
            "gnn_inference",
            structure_id=request.structure_id,
            device=request.device,
            checkpoint_path=request.checkpoint_path,
        )

    duration_ms = (time.monotonic() - start) * 1000
    if not result.success:
        raise HTTPException(
            status_code=_http_status_for_job_failure(result),
            detail=result.outputs.get("error", "GNN inference failed"),
        )

    return GNNResponse(
        run_id=result.run_id,
        structure_id=request.structure_id,
        node_count=result.outputs.get("node_count", 0),
        checkpoint_version_hash=checkpoint_hash,
        duration_ms=round(duration_ms, 1),
    )


# ---------------------------------------------------------------------------
# Full Pipeline Endpoint
# ---------------------------------------------------------------------------


@router.post("/pipeline", response_model=PipelineResponse)
async def run_full_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Run the discovery pathway — canonical in-process scheduler dispatch."""
    from science.compute.pathway_executor import InProcessJobBackend, execute_pathway

    start = time.monotonic()
    structure_id = request.structure_id.strip().lower()

    async with get_connection() as conn:
        db = DBAdapter(conn)
        backend = InProcessJobBackend(db)
        try:
            pathway_result = await execute_pathway(
                structure_id,
                backend,
                source_leak_only=request.source_leak_only,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration_ms = (time.monotonic() - start) * 1000
    jobs_complete = pathway_result.get("jobs_complete", [])
    if not jobs_complete:
        raise HTTPException(status_code=500, detail="Pathway produced no completed jobs")

    return PipelineResponse(
        run_id=pathway_result.get("run_id") or f"onboard_{uuid.uuid4().hex[:12]}",
        structure_id=structure_id,
        phases_run=jobs_complete,
        assets_created=len(jobs_complete),
        duration_ms=round(duration_ms, 1),
        warnings=[],
    )


@router.get("/jobs/schema")
async def get_job_schema() -> dict[str, Any]:
    """Return the compute job catalog JSON schema (field metadata + destinations)."""
    from science.compute.job_schema import load_job_schema

    return load_job_schema()


@router.get("/models")
async def get_gnn_models() -> dict[str, Any]:
    """Return contract-driven GNN model and checkpoint registry."""
    return build_models_api_payload()


@router.post("/jobs/{job_id}", response_model=ComputeJobResponse)
async def run_compute_job(job_id: str, request: ComputeJobRequest) -> ComputeJobResponse:
    """Run a single atomic compute job from the discovery pathway registry."""
    from science.compute.dispatch_helpers import dispatch_job_in_process

    start = time.monotonic()
    job_id = job_id.strip()

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            result = await dispatch_job_in_process(
                db,
                job_id,
                structure_id=request.structure_id,
                pathway=request.pathway,
                computation_run_id=request.computation_run_id,
                parent_run_id=request.parent_run_id,
                pipeline_job_id=request.pipeline_job_id,
                device=request.device,
                checkpoint_path=request.checkpoint_path,
                job_params=request.job_params,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration_ms = (time.monotonic() - start) * 1000
    if not result.success:
        raise HTTPException(
            status_code=_http_status_for_job_failure(result),
            detail=result.outputs.get("error", "Compute job failed"),
        )

    return _compute_job_response(result, duration_ms=duration_ms)


class PostSourceLeakRequest(BaseModel):
    structure_id: str
    parent_run_id: str
    gnn_run_id: str | None = None
    identify_allosteric_sites: bool = True
    run_buffering_atlas: bool = True


class PostSourceLeakResponse(BaseModel):
    run_id: str
    structure_id: str
    phases_run: list[str]
    duration_ms: float


@router.post("/post-source-leak-phases", response_model=PostSourceLeakResponse)
async def run_post_source_leak_phases(request: PostSourceLeakRequest) -> PostSourceLeakResponse:
    """Run tail phases after peeled source_leak_detection."""
    from science.compute.jobs.post_source_leak import run_post_source_leak_tail

    start = time.monotonic()
    async with get_connection() as conn:
        db = DBAdapter(conn)
        phase_results = await run_post_source_leak_tail(
            db,
            structure_id=request.structure_id,
            parent_run_id=request.parent_run_id,
            gnn_run_id=request.gnn_run_id,
            identify_allosteric_sites=request.identify_allosteric_sites,
            run_buffering_atlas=request.run_buffering_atlas,
        )
        await db.commit()

    phases_run = [name for name, pr in phase_results.items() if pr.success]
    duration_ms = (time.monotonic() - start) * 1000
    return PostSourceLeakResponse(
        run_id=request.parent_run_id,
        structure_id=request.structure_id,
        phases_run=phases_run,
        duration_ms=round(duration_ms, 1),
    )


# ---------------------------------------------------------------------------
# Graph Topology Metrics Endpoint
# ---------------------------------------------------------------------------


class GraphTopologyRequest(BaseModel):
    structure_id: str
    contact_cutoff_angstrom: float = 8.0
    chain_filter: str | None = None
    include_hbonds: bool = True
    edge_types: list[str] = Field(default_factory=lambda: ["contact", "h_bond"])


class GraphTopologyResponse(BaseModel):
    run_id: str
    structure_id: str
    node_count: int
    edge_count: int
    metrics_computed: list[str]
    bridge_count: int
    duration_ms: float


@router.post("/graph-topology", response_model=GraphTopologyResponse)
async def run_graph_topology(request: GraphTopologyRequest) -> GraphTopologyResponse:
    """Alias for ``POST /compute/jobs/graph_topology`` with graph tuning params."""
    from science.compute.dispatch_helpers import dispatch_job_in_process

    start = time.monotonic()
    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await dispatch_job_in_process(
            db,
            "graph_topology",
            structure_id=request.structure_id,
            job_params={
                "contact_cutoff_angstrom": request.contact_cutoff_angstrom,
                "chain_filter": request.chain_filter,
            },
        )

    duration_ms = (time.monotonic() - start) * 1000
    if not result.success:
        raise HTTPException(
            status_code=_http_status_for_job_failure(result),
            detail=result.outputs.get("error", "Graph topology failed"),
        )

    outputs = result.outputs
    return GraphTopologyResponse(
        run_id=result.run_id,
        structure_id=request.structure_id.strip().lower(),
        node_count=outputs.get("node_count", 0),
        edge_count=outputs.get("edge_count", 0),
        metrics_computed=outputs.get("metrics_computed", []),
        bridge_count=outputs.get("bridge_count", 0),
        duration_ms=round(duration_ms, 1),
    )


@router.post("/cryptic-scan", response_model=CrypticScanResponse)
async def run_cryptic_scan(request: CrypticScanRequest) -> CrypticScanResponse:
    """Alias for ``POST /compute/jobs/binding_site_scan``."""
    from science.compute.dispatch_helpers import dispatch_job_in_process

    start = time.monotonic()
    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await dispatch_job_in_process(
            db,
            "binding_site_scan",
            structure_id=request.structure_id,
            job_params={
                "cluster_distance_angstrom": request.cluster_distance_angstrom,
                "min_cluster_size": request.min_cluster_size,
                "max_pockets": request.max_pockets,
            },
        )

    duration_ms = (time.monotonic() - start) * 1000
    if not result.success:
        raise HTTPException(
            status_code=_http_status_for_job_failure(result),
            detail=result.outputs.get("error", "Binding site scan failed"),
        )

    return CrypticScanResponse(
        run_id=result.outputs.get("scan_run_id", result.run_id),
        structure_id=request.structure_id,
        sites_found=result.outputs.get("sites_found", 0),
        sites=result.outputs.get("sites", []),
        duration_ms=round(duration_ms, 1),
    )


@router.post("/motif-analysis", response_model=MotifAnalysisResponse)
async def run_motif_analysis(request: MotifAnalysisRequest) -> MotifAnalysisResponse:
    """Alias for ``POST /compute/jobs/hyperbolic_motifs``."""
    from science.compute.dispatch_helpers import dispatch_job_in_process

    start = time.monotonic()
    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await dispatch_job_in_process(
            db,
            "hyperbolic_motifs",
            structure_id=request.structure_id,
            job_params={
                "min_cluster_size": request.min_cluster_size,
                "min_samples": request.min_samples,
            },
        )

    duration_ms = (time.monotonic() - start) * 1000
    if not result.success:
        raise HTTPException(
            status_code=_http_status_for_job_failure(result),
            detail=result.outputs.get("error", "Motif analysis failed"),
        )

    return MotifAnalysisResponse(
        run_id=result.outputs.get("run_id", result.run_id),
        structure_id=request.structure_id,
        motif_count=result.outputs.get("motif_count", 0),
        motifs=result.outputs.get("motifs", []),
        duration_ms=round(duration_ms, 1),
    )


@router.post("/md-validate", response_model=MDValidateResponse)
async def run_md_validate(request: MDValidateRequest) -> MDValidateResponse:
    """Alias for ``POST /compute/jobs/md_validate_top_n`` (single-site mode)."""
    from science.compute.dispatch_helpers import dispatch_job_in_process

    start = time.monotonic()
    async with get_connection() as conn:
        db = DBAdapter(conn)
        result = await dispatch_job_in_process(
            db,
            "md_validate_top_n",
            structure_id=request.structure_id,
            job_params={
                "site_id": request.site_id,
                "dry_run": request.dry_run,
                "duration_ns": request.duration_ns,
                "temperature_k": request.temperature_k,
                "force_field": request.force_field,
            },
        )

    duration_ms = (time.monotonic() - start) * 1000
    if not result.success:
        raise HTTPException(
            status_code=_http_status_for_job_failure(result),
            detail=result.outputs.get("error", "MD validation failed"),
        )

    site_result = (result.outputs.get("results") or [{}])[0]
    return MDValidateResponse(
        run_id=result.run_id,
        structure_id=request.structure_id,
        site_id=request.site_id,
        pocket_open_fraction=site_result.get("pocket_open_fraction"),
        confidence_delta=site_result.get("confidence_delta"),
        duration_ms=round(duration_ms, 1),
        dry_run=request.dry_run,
    )
