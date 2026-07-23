"""TokyoEyeV8Runner — contract inference runner for Tokyo Eye v8.

Loads ``science.tokyo_eye.v8`` (Equiformer frontend + hyp spine). Full
``GNNInferenceResult`` parity with the legacy Hyp-MP runner is a tracked TODO;
load + forward smoke paths are active for verify / diagnostics.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from science.dtie.common.interfaces import GNNInferenceResult

logger = logging.getLogger(__name__)

MODEL_VERSION = "TokyoEye-v8"


@contextmanager
def _gpu_safe_inference(device: str):
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(
                "CUDA OOM during TokyoEye-v8 inference on device=%s.",
                device,
            )
            raise RuntimeError(
                f"GPU out of memory on {device}. "
                "Try a smaller structure or switch to CPU inference."
            ) from e
        raise
    finally:
        try:
            import torch

            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class TokyoEyeV8Runner:
    """Runner for TokyoEye-v8 inference (GNNRunner protocol — partial)."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        curvature_override: float | None = None,
        weight_map: str | Path | None = None,
    ):
        if checkpoint_path is None:
            from science.contracts.model_registry import get_production_checkpoint_path

            checkpoint_path = get_production_checkpoint_path()
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._curvature_override = curvature_override
        self._weight_map = Path(
            weight_map
            or "science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json"
        )
        self._system = None
        self._loaded = False
        self._load_lock = asyncio.Lock()

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    @property
    def space_type(self) -> str:
        return "hyperbolic"

    def _load_model_sync(self) -> None:
        import torch

        from science.contracts.model_registry import resolve_checkpoint_file
        from science.tokyo_eye.v8.equiformer_frontend import (
            StubEquiformerFrontend,
            TokyoEyeV8WithFrontend,
            load_weight_map,
        )
        from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

        resolved = resolve_checkpoint_file(self._checkpoint_path)
        checkpoint = resolved if resolved is not None else Path(self._checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"TokyoEye-v8 checkpoint not found: {self._checkpoint_path}"
            )

        cfg = load_weight_map(self._weight_map)
        frontend = StubEquiformerFrontend(
            in_dim=3,
            scalar_dim=int(cfg["scalar_dim"]),
            vector_dim=int(cfg["vector_dim"]),
            num_backbone_blocks=int(cfg.get("num_backbone_blocks", 7)),
            live_backbone=True,
        )
        spine = TokyoEyesHyperbolicV8(
            scalar_dim=int(cfg["scalar_dim"]),
            vector_dim=int(cfg["vector_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_attn_layers=int(cfg.get("num_attn_layers", 2)),
            num_relations=int(cfg.get("num_relations", 6)),
            num_sdrp_classes=int(cfg.get("num_sdrp_classes", 5)),
            gate_hidden=int(cfg.get("gate_hidden", 16)),
            c=float(cfg.get("curvature_c", cfg.get("c", 1.0))),
            eps=float(cfg.get("eps", 1e-5)),
            moe_temperature=float(cfg.get("gumbel_tau_start", 1.0)),
        )
        system = TokyoEyeV8WithFrontend(frontend, spine)
        blob = torch.load(checkpoint, map_location=self._device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = system.load_state_dict(state, strict=False)
        if missing:
            logger.info("TokyoEye-v8: %d missing keys", len(missing))
        if unexpected:
            logger.warning("TokyoEye-v8: %d unexpected keys ignored", len(unexpected))
        system.eval()
        system.to(self._device)
        self._system = system
        self._checkpoint_path = str(checkpoint)
        self._loaded = True
        logger.info(
            "TokyoEye-v8 loaded from %s (device=%s)",
            self._checkpoint_path,
            self._device,
        )

    async def _ensure_model_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            await asyncio.to_thread(self._load_model_sync)

    async def run_inference(
        self,
        structure_id: str,
        graph_data: Any,
        checkpoint_path: str | None = None,
    ) -> GNNInferenceResult:
        """TODO: map v8 ``z_hyp`` batch → governed ``GNNInferenceResult``.

        Affinity / training paths already run v8 forward. Onboard Normalizer
        adapter wiring is tracked under the v8 trunk cutover — do not fall back
        to the dead v7 Hyp-MP runner.
        """
        await self._ensure_model_loaded()
        raise NotImplementedError(
            "TokyoEyeV8Runner.run_inference: onboard GNNInferenceResult adapter "
            f"not yet wired for structure_id={structure_id!r}. "
            "Use experiments/training/v8 forward paths; do not call v7 runner."
        )


__all__ = ["MODEL_VERSION", "TokyoEyeV8Runner"]
