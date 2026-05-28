"""FastAPI application entry point.

Uses lifespan context manager (replaces deprecated @app.on_event).
DB is AlloyDB-ready async engine — no more dual sync/async layers.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gosp.db.connection import get_engine, dispose_engine

# Legacy API routers (no v2 equivalents yet — kept for backward compat)
from gosp.api.physics import router as physics_router
from gosp.api.analysis import router as analysis_router
from gosp.api.redzone import router as redzone_router
from gosp.api.empirical import router as empirical_router
from gosp.api.synthesis import router as synthesis_router
from gosp.api.audit import router as audit_router
from gosp.api.benchmark import router as benchmark_router
from gosp.api.annotations import router as annotations_router

# Data-layer v2 routers
from gosp.api_v2.ingest import router as ingest_router
from gosp.api_v2.structures import router as structures_router
from gosp.api_v2.results import router as results_router
from gosp.api_v2.assets import router as assets_router
from gosp.api_v2.gnn_writeback import router as gnn_writeback_router
from gosp.api_v2.session import router as session_router
from gosp.api_v2.websocket import router as websocket_router
from gosp.api.tasks import router as tasks_router
from gosp.api.pubsub import router as pubsub_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm the connection pool
    get_engine()
    yield
    # Shutdown: dispose engine cleanly
    await dispose_engine()


app = FastAPI(
    title="GOSP Molecular Imager API",
    description="Physics-based molecular engineering platform",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Legacy routes
app.include_router(physics_router)
app.include_router(analysis_router)
app.include_router(redzone_router)
app.include_router(empirical_router)
app.include_router(synthesis_router)
app.include_router(audit_router)
app.include_router(benchmark_router)
app.include_router(annotations_router)

# Data-layer v2 routes
app.include_router(ingest_router)
app.include_router(structures_router)
app.include_router(results_router)
app.include_router(assets_router)
app.include_router(gnn_writeback_router)
app.include_router(session_router)
app.include_router(websocket_router)
app.include_router(tasks_router)
app.include_router(pubsub_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "GOSP Molecular Imager API", "version": "0.2.0"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}
