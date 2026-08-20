"""TokyoEyeRunner — contract inference runner for Tokyo Eye v7+.

Loads ``science.tokyo_eye.TokyoEye`` and emits ``GNNInferenceResult`` for the
governed adapter / Normalizer path. Euclidean graph construction stays upstream;
Hyp MP is the communication geometry inside the model.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from science.dtie.common.curvature_values import require_learned_curvature
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput

logger = logging.getLogger(__name__)

MODEL_VERSION = "TokyoEye-v7"


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
                "CUDA OOM during TokyoEye inference on device=%s. GPU cache cleared.",
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


def _disc_xy_from_model_output(output: dict[str, Any]):
    try:
        from science.dtie.v66.visualization.interactive_viewer import (
            disc_xy_from_model_output,
        )
    except ImportError:
        from science.dtie.v6.visualization.interactive_viewer import (
            disc_xy_from_model_output,
        )
    return disc_xy_from_model_output(output)


class TokyoEyeRunner:
    """Runner for TokyoEye inference (GNNRunner protocol)."""

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
        return MODEL_VERSION

    @property
    def space_type(self) -> str:
        return "hyperbolic"

    def _load_model_sync(self) -> None:
        try:
            import torch

            from science.contracts.model_registry import resolve_checkpoint_file
            from science.tokyo_eye.TokyoEye import (
                TokyoEye,
                infer_legacy_disc_projection_from_checkpoint,
                infer_tokyo_eye_kwargs,
                load_tokyo_eye_state_dict,
            )

            resolved = resolve_checkpoint_file(self._checkpoint_path)
            checkpoint = resolved if resolved is not None else Path(self._checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"TokyoEye checkpoint not found: {self._checkpoint_path}"
                )

            checkpoint_data = torch.load(
                checkpoint, map_location=self._device, weights_only=False
            )

            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data
                checkpoint_data = {}

            kwargs = infer_tokyo_eye_kwargs(
                state_dict,
                checkpoint_data.get("architecture")
                if isinstance(checkpoint_data, dict)
                else None,
                checkpoint_data.get("training_config")
                if isinstance(checkpoint_data, dict)
                else None,
            )
            kwargs["legacy_disc_projection"] = (
                infer_legacy_disc_projection_from_checkpoint(
                    training_config=checkpoint_data.get("training_config")
                    if isinstance(checkpoint_data, dict)
                    else None,
                    metrics=checkpoint_data.get("metrics")
                    if isinstance(checkpoint_data, dict)
                    else None,
                    override=self._legacy_disc_projection_override,
                )
            )

            node_dim = int(kwargs.pop("node_dim", 4))
            kwargs.setdefault("hyp_mp_primary", True)
            kwargs.setdefault("se3_aux", False)
            self._model = TokyoEye(node_dim=node_dim, **kwargs)

            incompatible = load_tokyo_eye_state_dict(self._model, state_dict)
            missing = getattr(
                incompatible,
                "missing_keys",
                incompatible[0] if isinstance(incompatible, tuple) else [],
            )
            unexpected = getattr(
                incompatible,
                "unexpected_keys",
                incompatible[1] if isinstance(incompatible, tuple) else [],
            )
            if missing:
                logger.info(
                    "TokyoEye: %d missing keys (new modules init from scratch)",
                    len(missing),
                )
            if unexpected:
                logger.warning("TokyoEye: %d unexpected keys ignored", len(unexpected))

            self._model.eval()
            self._model.to(self._device)
            self._checkpoint_path = str(checkpoint)

            logger.info(
                "TokyoEye loaded from %s (device=%s, hyp_mp_primary=%s)",
                self._checkpoint_path,
                self._device,
                getattr(self._model, "hyp_mp_primary", False),
            )
            self._loaded = True

        except ImportError as e:
            raise ImportError(
                "torch, torch_geometric, geoopt, and e3nn required for TokyoEye inference."
            ) from e

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
        import torch

        if checkpoint_path and checkpoint_path != self._checkpoint_path:
            self._checkpoint_path = checkpoint_path
            self._loaded = False

        await self._ensure_model_loaded()

        if not hasattr(graph_data, "clustering") or graph_data.clustering is None:
            from science.dtie.v5.gnn.model import precompute_clustering

            graph_data = precompute_clustering(graph_data)

        graph_data = self._ensure_features(graph_data)
        graph_data = graph_data.to(self._device)

        with _gpu_safe_inference(self._device):
            with torch.no_grad():
                output = await asyncio.to_thread(self._model, graph_data)

        return self._extract_results(output, graph_data, structure_id)

    @staticmethod
    def _ensure_features(graph_data: Any) -> Any:
        import torch

        num_nodes = graph_data.x.shape[0]
        dev = graph_data.x.device

        if not hasattr(graph_data, "degree") or graph_data.degree is None:
            from torch_geometric.utils import degree as pyg_degree

            edge_index = graph_data.edge_index
            graph_data.degree = (
                pyg_degree(edge_index[0], num_nodes=num_nodes).long().to(dev)
            )

        if not hasattr(graph_data, "ss_onehot") or graph_data.ss_onehot is None:
            ss_type = graph_data.x[:, 2].long()
            graph_data.ss_onehot = torch.zeros(num_nodes, 3, device=dev)
            for i in range(num_nodes):
                idx = int(ss_type[i].item())
                if 0 <= idx <= 2:
                    graph_data.ss_onehot[i, idx] = 1.0
                else:
                    graph_data.ss_onehot[i, 2] = 1.0

        if not hasattr(graph_data, "rho") or graph_data.rho is None:
            graph_data.rho = graph_data.x[:, 0].clone()

        return graph_data

    def _extract_results(
        self,
        output: dict[str, Any],
        graph_data: Any,
        structure_id: str,
    ) -> GNNInferenceResult:
        import torch

        num_nodes = output["projections"].shape[0]
        chain_ids = getattr(graph_data, "chain_ids", ["A"] * num_nodes)
        residue_indices = getattr(
            graph_data, "residue_indices", list(range(1, num_nodes + 1))
        )

        if bool(getattr(graph_data, "structural_z_disc_frozen", False)) and hasattr(
            graph_data, "structural_cone_depth"
        ):
            cone_depth_tensor = graph_data.structural_cone_depth.detach().float()
        else:
            raw_depth = output["radial_features"].squeeze(-1).detach()
            depth_max = raw_depth.max() + 1e-8
            cone_depth_tensor = (raw_depth / depth_max) * 8.0

        cone_width = torch.exp(-cone_depth_tensor)

        nodes = []
        for i in range(num_nodes):
            node = GNNNodeOutput(
                residue_index=int(residue_indices[i]),
                chain_label=str(chain_ids[i]),
                input_features=graph_data.x[i].detach().cpu().numpy(),
                projections=output["projections"][i].detach().cpu().numpy(),
                cone_depth=float(cone_depth_tensor[i]),
                cone_width=float(cone_width[i]),
                epistemic_uncertainty=float(
                    output["uncertainty"]["epistemic"][i].detach()
                ),
                aleatoric_uncertainty=float(
                    output["uncertainty"]["aleatoric"][i].detach()
                ),
                total_uncertainty=float(output["uncertainty"]["total"][i].detach()),
                x_hyp=output["x_hyp"][i].detach().cpu().numpy().astype("float64"),
                x_routed_hyp=output["x_routed_hyp"][i]
                .detach()
                .cpu()
                .numpy()
                .astype("float64"),
                hyp_projections=_disc_xy_from_model_output(output)[i]
                .detach()
                .cpu()
                .numpy()
                .astype("float64"),
                expert_weights=output["expert_weights"][i].detach().cpu().numpy(),
            )
            nodes.append(node)

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
            curvature=require_learned_curvature(
                curvature, context="tokyo_eye gnn_inference"
            ),
            embedding_dim=output["x_hyp"].shape[1],
            space_type="hyperbolic",
            metadata={
                "device": self._device,
                "num_nodes": num_nodes,
                "architecture": output.get("audit_trail", {}).get(
                    "architecture", "tokyo_eye_hyp_mp"
                ),
                "hyp_mp_primary": bool(getattr(self._model, "hyp_mp_primary", False)),
                "se3_aux": bool(getattr(self._model, "se3_aux", False)),
                "production_module": "science.tokyo_eye.TokyoEye",
                "deep_hyperbolic_gate": output.get("audit_trail", {}).get(
                    "deep_hyperbolic_gate",
                    getattr(self._model, "deep_hyperbolic_gate", False),
                ),
                "gate_disc_scale": getattr(self._model, "gate_disc_scale", 1.0),
                "hyp_projections_2d_source": output.get("audit_trail", {}).get(
                    "disc_projection_source", "pre_routing_x_hyp"
                ),
                "structural_disc_frozen": output.get("audit_trail", {}).get(
                    "structural_disc_frozen", False
                ),
                "structural_disc_layout": output.get("audit_trail", {}).get(
                    "structural_disc_layout"
                ),
            },
        )
