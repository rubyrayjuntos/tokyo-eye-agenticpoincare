"""V6 GNN Runner — wraps GOSPConeMapperV6 for governed inference.

V6 extends v5 with topologically-routed MoE specialization:
- Expanded gate input (hidden + 7 topological features)
- Asymmetric capacity regularization
- Expert dropout during training
- Capacity-aware soft routing

The runner produces GNNInferenceResult compatible with the same
adapter pipeline as v5, with additional v6 metadata (expert_load,
routing_entropy).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.curvature_values import require_learned_curvature
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput
from science.dtie.v6.visualization.interactive_viewer import disc_xy_from_model_output

logger = logging.getLogger(__name__)


@contextmanager
def _gpu_safe_inference(device: str):
    """Context manager for GPU-safe inference."""
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(
                "CUDA OOM during inference on device=%s. GPU cache cleared.",
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


class V6GNNRunner:
    """Runner for GOSPConeMapperV6 inference.

    Implements the GNNRunner protocol.

    Thread safety:
        Model loading is protected by an asyncio.Lock to prevent
        duplicate loads from concurrent requests.

    Usage:
        runner = V6GNNRunner(checkpoint_path="path/to/tokyo_eyes_v6.pt")
        result = await runner.run_inference(structure_id="4obe", graph_data=data)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        curvature_override: float | None = None,
        legacy_disc_projection: bool | None = None,
    ):
        if checkpoint_path is None:
            from science.contracts.model_registry import get_production_checkpoint_path

            checkpoint_path = get_production_checkpoint_path()
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._curvature_override = curvature_override
        self._legacy_disc_projection_override = legacy_disc_projection
        self._model = None
        self._loaded = False
        self._load_lock = asyncio.Lock()

    @property
    def model_version(self) -> str:
        return "GOSPConeMapper-v6"

    @property
    def space_type(self) -> str:
        return "hyperbolic"

    def _load_model_sync(self) -> None:
        """Load the v6 model checkpoint (synchronous, CPU-bound)."""
        try:
            import torch

            from science.contracts.model_registry import resolve_checkpoint_file
            from science.dtie.v6.gnn.model import (
                GOSPConeMapperV6,
                infer_legacy_disc_projection_from_checkpoint,
                infer_v6_model_kwargs,
                load_v6_state_dict,
            )

            resolved = resolve_checkpoint_file(self._checkpoint_path)
            checkpoint = resolved if resolved is not None else Path(self._checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"V6 checkpoint not found: {self._checkpoint_path}"
                )

            checkpoint_data = torch.load(
                checkpoint, map_location=self._device, weights_only=False
            )

            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data
                checkpoint_data = {}

            kwargs = infer_v6_model_kwargs(
                state_dict,
                checkpoint_data.get("architecture") if isinstance(checkpoint_data, dict) else None,
                checkpoint_data.get("training_config") if isinstance(checkpoint_data, dict) else None,
            )
            kwargs["legacy_disc_projection"] = infer_legacy_disc_projection_from_checkpoint(
                training_config=checkpoint_data.get("training_config")
                if isinstance(checkpoint_data, dict)
                else None,
                metrics=checkpoint_data.get("metrics") if isinstance(checkpoint_data, dict) else None,
                override=self._legacy_disc_projection_override,
            )

            self._model = GOSPConeMapperV6(node_dim=4, **kwargs)

            incompatible = load_v6_state_dict(self._model, state_dict)
            missing = getattr(incompatible, "missing_keys", incompatible[0] if isinstance(incompatible, tuple) else [])
            unexpected = getattr(incompatible, "unexpected_keys", incompatible[1] if isinstance(incompatible, tuple) else [])
            if missing:
                logger.info(
                    "V6 model: %d missing keys (new modules init from scratch)",
                    len(missing),
                )
            if unexpected:
                logger.warning("V6 model: %d unexpected keys ignored", len(unexpected))

            self._model.eval()
            self._model.to(self._device)
            self._checkpoint_path = str(checkpoint)

            logger.info(
                "V6 model loaded from %s (device=%s, deep_gate=%s, disc_scale=%.2f, legacy_disc=%s)",
                self._checkpoint_path,
                self._device,
                kwargs.get("deep_hyperbolic_gate", False),
                kwargs.get("gate_disc_scale", 1.0),
                kwargs.get("legacy_disc_projection", False),
            )
            self._loaded = True

        except ImportError as e:
            raise ImportError(
                "torch, torch_geometric, geoopt, and e3nn required for V6 GNN inference."
            ) from e

    async def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded, with async lock to prevent duplicate loads."""
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
        """Run v6 GNN inference on a prepared graph.

        Args:
            structure_id: Canonical structure_id.
            graph_data: PyG Data object with x, edge_index, edge_attr,
                        clustering, degree, ss_onehot, rho.
            checkpoint_path: Override checkpoint (triggers reload if different).

        Returns:
            GNNInferenceResult with per-node outputs including v6 extras.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            RuntimeError: If GPU OOM or other inference failure.
        """
        import torch

        if checkpoint_path and checkpoint_path != self._checkpoint_path:
            self._checkpoint_path = checkpoint_path
            self._loaded = False

        await self._ensure_model_loaded()

        # Ensure clustering is precomputed
        if not hasattr(graph_data, "clustering") or graph_data.clustering is None:
            from science.dtie.v5.gnn.model import precompute_clustering
            graph_data = precompute_clustering(graph_data)

        # Ensure v6-specific features exist
        graph_data = self._ensure_v6_features(graph_data)

        # Move data to same device as model
        graph_data = graph_data.to(self._device)

        with _gpu_safe_inference(self._device):
            with torch.no_grad():
                output = await asyncio.to_thread(self._model, graph_data)

        return self._extract_results(output, graph_data, structure_id)

    @staticmethod
    def _ensure_v6_features(graph_data: Any) -> Any:
        """Ensure v6-specific features (degree, ss_onehot, rho) are present."""
        import torch

        num_nodes = graph_data.x.shape[0]
        dev = graph_data.x.device

        # Compute degree from edge_index if missing
        if not hasattr(graph_data, "degree") or graph_data.degree is None:
            from torch_geometric.utils import degree as pyg_degree
            edge_index = graph_data.edge_index
            graph_data.degree = pyg_degree(
                edge_index[0], num_nodes=num_nodes
            ).long().to(dev)

        # Compute ss_onehot from x[:, 2] (ss_type) if missing
        if not hasattr(graph_data, "ss_onehot") or graph_data.ss_onehot is None:
            ss_type = graph_data.x[:, 2].long()
            graph_data.ss_onehot = torch.zeros(num_nodes, 3, device=dev)
            for i in range(num_nodes):
                idx = int(ss_type[i].item())
                if 0 <= idx <= 2:
                    graph_data.ss_onehot[i, idx] = 1.0
                else:
                    graph_data.ss_onehot[i, 2] = 1.0  # default to coil

        # Extract rho from x[:, 0] if missing
        if not hasattr(graph_data, "rho") or graph_data.rho is None:
            graph_data.rho = graph_data.x[:, 0].clone()

        return graph_data

    def _extract_results(
        self,
        output: dict[str, Any],
        graph_data: Any,
        structure_id: str,
    ) -> GNNInferenceResult:
        """Convert raw v6 model output to GNNInferenceResult.

        Uses radial_features directly as cone_depth (bypasses expmap coupling
        that causes angular mode collapse in the hyperbolic distance metric).
        The RadialHead output is the authoritative burial depth — it has
        perfect correlation with dehydron density from curriculum training.

        Normalized to [0, ~8] range for downstream threshold compatibility.
        """
        import torch

        num_nodes = output["projections"].shape[0]
        chain_ids = getattr(graph_data, "chain_ids", ["A"] * num_nodes)
        residue_indices = getattr(
            graph_data, "residue_indices", list(range(1, num_nodes + 1))
        )

        # Use radial_features as cone_depth (decoupled from angular expmap)
        # Normalize to [0, ~8] range for downstream phase threshold compat
        raw_depth = output["radial_features"].squeeze(-1).detach()
        depth_max = raw_depth.max() + 1e-8
        normalized_depth = (raw_depth / depth_max) * 8.0  # Scale to [0, 8]
        cone_width = torch.exp(-normalized_depth)

        nodes = []
        for i in range(num_nodes):
            node = GNNNodeOutput(
                residue_index=int(residue_indices[i]),
                chain_label=str(chain_ids[i]),
                input_features=graph_data.x[i].detach().cpu().numpy(),
                projections=output["projections"][i].detach().cpu().numpy(),
                cone_depth=float(normalized_depth[i]),
                cone_width=float(cone_width[i]),
                epistemic_uncertainty=float(
                    output["uncertainty"]["epistemic"][i].detach()
                ),
                aleatoric_uncertainty=float(
                    output["uncertainty"]["aleatoric"][i].detach()
                ),
                total_uncertainty=float(
                    output["uncertainty"]["total"][i].detach()
                ),
                # Force native float64 for hyperbolic embeddings to guarantee precision
                # for Lorentz inner product / arcosh near manifold boundary (avoids NaN).
                x_hyp=output["x_hyp"][i].detach().cpu().numpy().astype("float64"),
                x_routed_hyp=output["x_routed_hyp"][i].detach().cpu().numpy().astype("float64"),
                hyp_projections=disc_xy_from_model_output(output)[i].detach().cpu().numpy().astype("float64"),
                expert_weights=output["expert_weights"][i].detach().cpu().numpy(),
            )
            nodes.append(node)

        # Extract learned curvature (must match model.curvature: softplus(log_c)+eps)
        curvature = self._curvature_override
        if curvature is None and hasattr(self._model, "curvature"):
            curvature = float(self._model.curvature.detach().cpu().item())
        elif curvature is None and hasattr(self._model, "log_c"):
            import torch.nn.functional as F
            curvature = float(F.softplus(self._model.log_c).item() + 1e-4)

        return GNNInferenceResult(
            structure_id=structure_id,
            model_version=self.model_version,
            checkpoint_path=self._checkpoint_path,
            nodes=nodes,
            curvature=require_learned_curvature(curvature, context="v6 gnn_inference"),
            embedding_dim=output["x_hyp"].shape[1],
            space_type="hyperbolic",
            metadata={
                "device": self._device,
                "num_nodes": num_nodes,
                "architecture": output.get("audit_trail", {}).get(
                    "architecture",
                    "hyperbolic_prototype_moe" if getattr(self._model, "hyperbolic_gate", False) else "topological_moe_specialization",
                ),
                "deep_hyperbolic_gate": output.get("audit_trail", {}).get(
                    "deep_hyperbolic_gate",
                    getattr(self._model, "deep_hyperbolic_gate", False),
                ),
                "gate_disc_scale": getattr(self._model, "gate_disc_scale", 1.0),
                "hyp_projections_2d_source": output.get("audit_trail", {}).get(
                    "disc_projection_source", "pre_routing_x_hyp"
                ),
                "legacy_disc_projection": bool(
                    getattr(self._model, "legacy_disc_projection", False)
                ),
                "has_3d_projections": True,
                "expert_load": output["expert_load"].detach().cpu().numpy().tolist(),
                "routing_entropy": float(output["routing_entropy"].detach()),
                "gate_features_used": output["gate_features_used"],
            },
        )
