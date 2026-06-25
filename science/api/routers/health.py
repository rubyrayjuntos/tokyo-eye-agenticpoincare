"""Health endpoint for the Science API.

Reports GPU availability, checkpoint listing, and DB connectivity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter

from data.db import get_connection

logger = logging.getLogger(__name__)

router = APIRouter()

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")


def _detect_gpu() -> dict:
    """Check if CUDA GPU is available via torch."""
    try:
        import torch

        available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if available else 0
        device_name = torch.cuda.get_device_name(0) if available and device_count > 0 else None
        return {
            "available": available,
            "device_count": device_count,
            "device_name": device_name,
        }
    except Exception as e:
        logger.warning("GPU detection failed: %s", e)
        return {"available": False, "device_count": 0, "device_name": None, "error": str(e)}


def _list_checkpoints() -> list[str]:
    """List .pt checkpoint files in the checkpoint directory."""
    checkpoint_path = Path(CHECKPOINT_DIR)
    if not checkpoint_path.exists():
        return []
    return sorted(str(p.relative_to(checkpoint_path)) for p in checkpoint_path.rglob("*.pt"))


async def _check_db() -> dict:
    """Test DB connectivity via the pool."""
    try:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                row = await cur.fetchone()
                return {"connected": row is not None and row[0] == 1}
    except Exception as e:
        logger.warning("DB health check failed: %s", e)
        return {"connected": False, "error": str(e)}


@router.get("/health")
async def health_check():
    """Return service status, GPU availability, checkpoints, and DB connectivity."""
    gpu = _detect_gpu()
    checkpoints = _list_checkpoints()
    db = await _check_db()

    status = "healthy" if db.get("connected") else "degraded"

    return {
        "status": status,
        "gpu": gpu,
        "checkpoints": checkpoints,
        "db": db,
    }
