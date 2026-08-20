"""Resolve whether ingest should attach structural disc SSOT for a checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_checkpoint_structural_disc_frozen(checkpoint_path: str | Path) -> bool | None:
    """Return training_config.structural_disc_frozen from a checkpoint, or None if unknown."""
    path = Path(checkpoint_path)
    if not path.is_file():
        return None
    try:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Could not load checkpoint for SSOT resolve (%s): %s", path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    tc = raw.get("training_config")
    if not isinstance(tc, dict):
        return None
    if "structural_disc_frozen" in tc:
        return bool(tc["structural_disc_frozen"])
    if tc.get("slim_moe_structural_ssot"):
        return True
    if tc.get("master_cold_lineage"):
        return False
    return None


def resolve_structural_disc_frozen(
    checkpoint_path: str | Path | None,
    *,
    pipeline_config_value: bool = True,
    job_params: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Decide SSOT attach for gnn_inference.

    Priority:
      1. ``job_params["structural_disc_frozen"]`` (explicit A/B override)
      2. checkpoint ``training_config.structural_disc_frozen`` (or slim/master lineage hints)
      3. ``pipeline_config_value`` (legacy default True for unknown checkpoints)

    Returns ``(frozen, reason)``.
    """
    params = job_params or {}
    if "structural_disc_frozen" in params:
        val = params["structural_disc_frozen"]
        if isinstance(val, str):
            frozen = val.strip().lower() in ("1", "true", "yes", "on")
        else:
            frozen = bool(val)
        return frozen, "job_params"

    if checkpoint_path:
        from_ckpt = read_checkpoint_structural_disc_frozen(checkpoint_path)
        if from_ckpt is not None:
            return from_ckpt, "checkpoint_training_config"

    return bool(pipeline_config_value), "pipeline_config_or_legacy_default"
