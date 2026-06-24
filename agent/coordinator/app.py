# Augmented from: SRC_AGENT/main.py + new governed infrastructure on 2026-05-27
"""Tokyo Eyes Agent Coordinator — Production FastAPI Application.

This is the augmented agent that combines:
- Legacy infrastructure: auth, sessions, viewport protocol, chat/LLM
- New governed data layer: Normalizer, production views, v5 pipeline

All data reads go through the governed views (v_agent_*, v_viz_*).
All data writes go through the Normalizer.
All computation uses the v5 GNN pipeline.
No legacy computation paths remain.

Start with: uvicorn agent.coordinator.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.coordinator.auth import get_current_user
from agent.coordinator.routers import chat, compare, dashboard, data, graph, hypotheses, ingest, plots, poincare, rcsb, sdrp, tools, therapeutic_compiler
from agent.coordinator.viewport import viewport_manager, websocket_viewport
from shared.logging import get_logger, setup_logging
from shared.middleware import RequestContextMiddleware

load_dotenv()

# Initialize structured logging (must be before any logger usage)
setup_logging()

logger = get_logger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")


# ---------------------------------------------------------------------------
# App Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Manage application lifecycle: pool startup and shutdown.

    Startup:
    - Open the database connection pool
    - Log startup complete

    Shutdown:
    - Drain active WebSocket connections
    - Close the database connection pool
    - Flush log handlers
    """
    from data.db import close_pool, open_pool

    await open_pool()
    # Run tool self-diagnostics
    from agent.tools.diagnostics import run_startup_diagnostics
    diagnostic_report = await run_startup_diagnostics()
    if not diagnostic_report.all_passed:
        logger.warning(
            "Startup diagnostics found issues — some tools may not work correctly"
        )
    logger.info("Application started", environment=ENVIRONMENT)
    yield
    # Graceful shutdown
    logger.info("Shutting down — draining connections")
    # Close all active WebSocket connections
    for conn in list(viewport_manager._connections.values()):
        try:
            await conn.websocket.close(code=1001)
        except Exception:
            pass
    viewport_manager._connections.clear()
    viewport_manager._by_session.clear()
    # Close DB pool
    await close_pool()
    # Flush log handlers
    for handler in logging.getLogger().handlers:
        handler.flush()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tokyo Eyes Agent Coordinator",
    description="Agentic orchestration for DTIE scientific workflows (v5)",
    version="1.0.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# CORS — environment-aware
# ---------------------------------------------------------------------------

def _get_cors_origins() -> list[str]:
    """Build CORS origins list based on environment."""
    origins: list[str] = []

    # Always allow the configured frontend origin
    frontend = os.getenv("FRONTEND_ORIGIN")
    if frontend:
        origins.append(frontend)

    # Only allow localhost in non-production environments
    if ENVIRONMENT != "prod":
        origins.extend([
            "http://localhost:5173",
            "http://localhost:3000",
        ])

    return origins or ["http://localhost:5173"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request context middleware (sets request_id, session_id for structured logging)
app.add_middleware(RequestContextMiddleware)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(chat.router)
app.include_router(compare.router)
app.include_router(tools.router)
app.include_router(poincare.router)
app.include_router(dashboard.router)
app.include_router(rcsb.router)
app.include_router(graph.router)
app.include_router(hypotheses.router)
app.include_router(data.router)
app.include_router(plots.router)
app.include_router(sdrp.router)
app.include_router(therapeutic_compiler.router)
app.include_router(ingest.router)


# ---------------------------------------------------------------------------
# WebSocket (can't use APIRouter for websockets easily)
# ---------------------------------------------------------------------------

app.websocket("/ws/viewport")(websocket_viewport)


# ---------------------------------------------------------------------------
# Viewport Directive Push (for external callers)
# ---------------------------------------------------------------------------


@app.post("/api/viewport/directive")
async def push_directive(directive: dict, user: dict = Depends(get_current_user)):
    """Push a viewport directive to all connected frontends."""
    await viewport_manager.broadcast_directive(directive)
    return {"status": "sent"}


# ---------------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "tokyo_eyes_agent", "model": "v5"}


@app.get("/health/tools")
async def health_tools():
    """Run tool diagnostics on demand and return the report."""
    from agent.tools.diagnostics import run_startup_diagnostics
    report = await run_startup_diagnostics()
    return {
        "status": "ok" if report.all_passed else "degraded",
        "summary": report.summary(),
        "results": [
            {"name": r.name, "passed": r.passed, "message": r.message, "severity": r.severity}
            for r in report.results
            if not r.passed or r.message  # Only include failures and items with messages
        ],
    }


@app.get("/")
async def root():
    return {"message": "Tokyo Eyes Agent Coordinator", "version": "1.0.0", "model": "GOSPConeMapper-v5"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
