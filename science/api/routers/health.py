"""Health endpoint for the Science API.

Reports GPU availability, checkpoint listing, contract-driven production GNN
metadata, job registry summary, and DB connectivity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from functools import lru_cache

from fastapi import APIRouter

from data.db import get_connection
from science.compute.registry import JOB_REGISTRY
from science.contracts.model_registry import (
    get_production_model,
    get_production_restore_summary,
)
from science.contracts.onboard_contract import geometric_enforcement_level, load_contract

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


def _job_registry_summary() -> dict[str, int]:
    counts = {"implemented": 0, "partial": 0, "planned": 0}
    for job in JOB_REGISTRY.values():
        if job.status in counts:
            counts[job.status] += 1
    return counts


def _gnn_production_summary() -> dict:
    model = get_production_model()
    restore = get_production_restore_summary(
        resolve_cache=os.environ.get("HEALTH_RESOLVE_MLFLOW_MODEL", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    summary = {
        "model_id": model.model_id,
        "model_version": model.model_version,
        "api_alias": model.api_alias,
        "runner_module": model.runner_module,
        "runner_class": model.runner_class,
        "restore": restore,
        "checkpoint_exists": bool(restore.get("cache_path")),
        "checkpoint_sha256_prefix": restore.get("checkpoint_sha256_16"),
    }
    if restore.get("cache_path"):
        summary["architecture_compat"] = _verify_production_checkpoint_cached()
    return summary


@lru_cache(maxsize=1)
def _verify_production_checkpoint_cached() -> dict:
    """Load production weights once per process — catches code/checkpoint drift."""
    try:
        from science.contracts.model_registry import get_production_model

        prod = get_production_model()
        if prod.model_id == "TokyoEye" or prod.model_version == "TokyoEye@champion":
            from science.tokyo_eye.v8.runner import TokyoEyeV8Runner

            runner = TokyoEyeV8Runner(device="cpu")
            runner._load_model_sync()
            return {
                "ok": True,
                "missing_keys": 0,
                "model_class": "TokyoEyeV8WithFrontend",
                "lineage": "equiformer-v3-moe",
            }

        if prod.model_id == "tokyo_eye_v7" or prod.model_version == "TokyoEye-v7":
            from science.tokyo_eye.TokyoEye import verify_tokyo_eye_checkpoint

            result = verify_tokyo_eye_checkpoint()
            return {
                "ok": result["load_ok"],
                "missing_keys": len(result["missing_keys"]),
                "model_class": "TokyoEye",
                "hyp_mp_primary": result.get("hyp_mp_primary", True),
                "deep_hyperbolic_gate": result["deep_hyperbolic_gate"],
                "gate_disc_scale": result["gate_disc_scale"],
                "hyperbolic_expert_mix": result["hyperbolic_expert_mix"],
                "lineage": "v7_archaeology",
            }

        from science.dtie.v6.gnn.model import verify_v6_checkpoint

        result = verify_v6_checkpoint()
        return {
            "ok": result["load_ok"],
            "missing_keys": len(result["missing_keys"]),
            "deep_hyperbolic_gate": result["deep_hyperbolic_gate"],
            "gate_disc_scale": result["gate_disc_scale"],
            "hyperbolic_expert_mix": result["hyperbolic_expert_mix"],
        }
    except Exception as exc:
        logger.warning("GNN checkpoint architecture verification failed: %s", exc)
        return {"ok": False, "error": str(exc)}


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
    contract = load_contract()
    gnn_production = _gnn_production_summary()

    status = "healthy" if db.get("connected") else "degraded"
    restore = gnn_production.get("restore") or {}
    if status == "healthy" and not restore.get("ok"):
        status = "degraded"
    compat = gnn_production.get("architecture_compat") or {}
    if status == "healthy" and compat and not compat.get("ok", True):
        status = "degraded"

    return {
        "status": status,
        "contract_version": str(contract.get("version", "")),
        "geometric_enforcement_level": geometric_enforcement_level(),
        "gnn_production": gnn_production,
        "job_registry": _job_registry_summary(),
        "gpu": gpu,
        "checkpoints": checkpoints,
        "db": db,
    }
