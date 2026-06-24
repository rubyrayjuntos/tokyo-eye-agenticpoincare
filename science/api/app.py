"""Tokyo Eye Science API — FastAPI application.

Exposes compute endpoints (GNN, pipeline, cryptic scan, motif analysis, MD validation)
over HTTP on port 8001, internal Docker network only.

Start with: uvicorn science.api.app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from data.db import close_pool, open_pool
from science.api.routers import compute, health, ingest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Manage science API lifecycle: DB pool open/close."""
    logger.info("Science API starting — opening DB pool")
    try:
        await open_pool()
        logger.info("Science API ready")
    except Exception as e:
        logger.error("Science API failed to start: %s", e)
        raise SystemExit(1) from e
    yield
    logger.info("Science API shutting down — closing DB pool")
    await close_pool()


app = FastAPI(
    title="Tokyo Eye Science API",
    version="1.0.0",
    lifespan=_lifespan,
)

app.include_router(health.router)
app.include_router(compute.router, prefix="/compute")
app.include_router(ingest.router, prefix="/compute")
