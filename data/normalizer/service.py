# Migrated from: new (Phase 2 implementation) on 2026-05-27
"""FastAPI service wrapper for the Normalizer.

This is the HTTP interface for remote callers (agent coordinator,
external tools, backfill scripts running in different environments).

The service is a thin wrapper around the same Normalizer library that
science code uses directly. No separate logic — just HTTP routing.

See: ADR-006 (Library-first, service-optional)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from data.normalizer.core import Normalizer, NormalizerError
from science.dtie.common.normalizer_payloads import (
    GNNOutputPayload,
    NormalizerResult,
    Phase3PersistencePayload,
)

router = APIRouter(prefix="/api/normalize", tags=["normalizer"])


# Dependency injection placeholder — in production this would come from
# app state or a connection pool manager.
async def get_normalizer() -> Normalizer:
    """Dependency that provides a configured Normalizer instance.

    In production, this would:
    1. Get a DB connection from the pool
    2. Instantiate the Normalizer with caller_identity from the request
    """
    raise NotImplementedError(
        "Configure get_normalizer dependency with your DB connection pool"
    )


class HealthResponse(BaseModel):
    status: str
    service: str = "normalizer"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/gnn-output", response_model=NormalizerResult)
async def normalize_gnn_output(
    payload: GNNOutputPayload,
    normalizer: Normalizer = Depends(get_normalizer),
) -> NormalizerResult:
    """Normalize and write GNN node embeddings.

    Accepts the same payload that science code passes to the library
    directly. Validates, enforces provenance, writes to governed layer.
    """
    try:
        return await normalizer.normalize_gnn_output(payload)
    except NormalizerError as e:
        raise HTTPException(
            status_code=422,
            detail={"message": str(e), "run_id": e.run_id, "details": e.details},
        )


@router.post("/phase3-persistence", response_model=NormalizerResult)
async def normalize_phase3_output(
    payload: Phase3PersistencePayload,
    normalizer: Normalizer = Depends(get_normalizer),
) -> NormalizerResult:
    """Normalize and write Phase 3 persistence results."""
    try:
        return await normalizer.normalize_phase3_output(payload)
    except NormalizerError as e:
        raise HTTPException(
            status_code=422,
            detail={"message": str(e), "run_id": e.run_id, "details": e.details},
        )
