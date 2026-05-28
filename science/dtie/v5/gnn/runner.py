# Migrated from: new (Phase 3 integration) on 2026-05-27
"""V5 GNN Runner — wraps GOSPConeMapper-v5 for governed inference.

V5 is the most recent and most robust model. Key differences from v4:
- Decoupled radial/angular heads (no training conflict)
- Dual hyperbolic projections: 2D disc + 3D ball
- radial_features and angular_features exposed for loss routing

The runner produces GNNInferenceResult compatible with the same
adapter pipeline as v4 (the adapter handles the extra v5 outputs).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "checkpoints/v5/tokyo_eyes_v5.pt"


@contextmanager
def _gpu_safe_inference(device: str):
    """Context manager for GPU-safe inference.

    Handles CUDA OOM errors gracefully by clearing the cache and
    providing a clear error message. Also ensures GPU memory is
    released after inference completes (success or failure).
    """
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            import torch
            # Clear GPU cache to recover memory for subsequent requests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(
                "CUDA OOM during inference on device=%s. "
                "GPU cache cleared. Consider reducing batch size or using CPU.",
                device,
            )
            raise RuntimeError(
                f"GPU out of memory on {device}. "
                "Try a smaller structure or switch to CPU inference."
            ) from e
        raise
    finally:
        # Always release cached memory after inference
        try:
            import torch
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class V5GNNRunner:
    """Runner for GOSPConeMapper-v5 inference.

    Implements the GNNRunner protocol.

    Thread safety:
        Model loading is protected by an asyncio.Lock to prevent
        duplicate loads from concurrent requests. The model is loaded
        in a thread to avoid blocking the event loop.

    Usage:
        runner = V5GNNRunner(checkpoint_path="path/to/tokyo_eyes_v5.pt")
        result = await runner.run_inference(structure_id="4obe", graph_data=data)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        curvature_override: float | None = None,
    ):
        self._checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self._device = device
        self._curvature_override = curvature_override
        self._model = None
        self._loaded = False
        self._load_lock = asyncio.Lock()

    @property
    def model_version(self) -> str:
        return "GOSPConeMapper-v5"

    @property
    def space_type(self) -> str:
        return "hyperbolic"

    def _load_model_sync(self) -> None:
        """Load the v5 model checkpoint (synchronous, CPU-bound).

        This method is intended to be called via asyncio.to_thread()
        to avoid blocking the event loop.
        """
        try:
            import torch
            from torch_geometric.data import Data  # noqa: F401

            from science.dtie.v5.gnn.model import GOSPConeMapper

            checkpoint = Path(self._checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"V5 checkpoint not found: {self._checkpoint_path}"
                )

            self._model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
            state_dict = torch.load(checkpoint, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._model.to(self._device)

            logger.info("V5 model loaded from %s (device=%s)", self._checkpoint_path, self._device)
            self._loaded = True

        except ImportError as e:
            raise ImportError(
                "torch, torch_geometric, geoopt, and e3nn required for V5 GNN inference."
            ) from e

    async def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded, with async lock to prevent duplicate loads.

        Uses asyncio.Lock so that concurrent requests wait for the first
        load to complete rather than all loading simultaneously.
        """
        if self._loaded:
            return

        async with self._load_lock:
            # Double-check after acquiring lock (another coroutine may have loaded it)
            if self._loaded:
                return
            await asyncio.to_thread(self._load_model_sync)

    async def run_inference(
        self,
        structure_id: str,
        graph_data: Any,
        checkpoint_path: str | None = None,
    ) -> GNNInferenceResult:
        """Run v5 GNN inference on a prepared graph.

        Args:
            structure_id: Canonical structure_id.
            graph_data: PyG Data object with x, edge_index, edge_attr, clustering.
            checkpoint_path: Override checkpoint (triggers reload if different).

        Returns:
            GNNInferenceResult with per-node outputs (including v5 extras).

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            RuntimeError: If GPU OOM or other inference failure.
        """
        import torch

        # Handle checkpoint override (rare: for A/B testing different weights)
        if checkpoint_path and checkpoint_path != self._checkpoint_path:
            self._checkpoint_path = checkpoint_path
            self._loaded = False

        await self._ensure_model_loaded()

        # Ensure clustering is precomputed
        if not hasattr(graph_data, 'clustering') or graph_data.clustering is None:
            from science.dtie.v5.gnn.model import precompute_clustering
            graph_data = precompute_clustering(graph_data)

        # Run inference with GPU safety wrapper
        with _gpu_safe_inference(self._device):
            with torch.no_grad():
                output = await asyncio.to_thread(self._model, graph_data)

        return self._extract_results(output, graph_data, structure_id)

    def _extract_results(
        self,
        output: dict[str, Any],
        graph_data: Any,
        structure_id: str,
    ) -> GNNInferenceResult:
        """Convert raw v5 model output to GNNInferenceResult.

        V5 output format (superset of v4):
        - projections: [N, 64] Euclidean
        - hyp_projections_2d: [N, 2] native 2D disc
        - hyp_projections_3d: [N, 3] native 3D ball (NEW)
        - x_hyp: [N, hidden] full hyperbolic
        - x_routed_hyp: [N, hidden] post-MoE hyperbolic
        - cone_depth, cone_width: [N, 1]
        - epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty: [N]
        - expert_weights: [N, n_experts]
        - radial_features: [N, 1] (NEW)
        - angular_features: [N, hidden] (NEW)
        """
        num_nodes = output["projections"].shape[0]
        chain_ids = getattr(graph_data, 'chain_ids', ['A'] * num_nodes)
        residue_indices = getattr(graph_data, 'residue_indices', list(range(1, num_nodes + 1)))

        nodes = []
        for i in range(num_nodes):
            node = GNNNodeOutput(
                residue_index=int(residue_indices[i]),
                chain_label=str(chain_ids[i]),
                input_features=graph_data.x[i].cpu().numpy(),
                projections=output["projections"][i].cpu().numpy(),
                cone_depth=float(output["cone_depth"][i]),
                cone_width=float(output["cone_width"][i]),
                epistemic_uncertainty=float(output["uncertainty"]["epistemic"][i]),
                aleatoric_uncertainty=float(output["uncertainty"]["aleatoric"][i]),
                total_uncertainty=float(output["uncertainty"]["total"][i]),
                x_hyp=output["x_hyp"][i].cpu().numpy(),
                x_routed_hyp=output["x_routed_hyp"][i].cpu().numpy(),
                hyp_projections=output["hyp_projections_2d"][i].cpu().numpy(),
                expert_weights=output["expert_weights"][i].cpu().numpy(),
            )
            nodes.append(node)

        # Extract learned curvature
        curvature = self._curvature_override
        if curvature is None and hasattr(self._model, 'log_c'):
            import torch.nn.functional as F
            curvature = float(F.softplus(self._model.log_c).item())

        return GNNInferenceResult(
            structure_id=structure_id,
            model_version=self.model_version,
            checkpoint_path=self._checkpoint_path,
            nodes=nodes,
            curvature=curvature or 1.0,
            embedding_dim=output["x_hyp"].shape[1],
            space_type="hyperbolic",
            metadata={
                "device": self._device,
                "num_nodes": num_nodes,
                "architecture": "decoupled_radial_angular",
                "has_3d_projections": True,
            },
        )
