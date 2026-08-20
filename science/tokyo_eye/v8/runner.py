"""TokyoEyeV8Runner — contract inference runner for Tokyo Eye.

Loads ``science.tokyo_eye.v8`` (Equiformer frontend + hyp spine). Full
``GNNInferenceResult`` parity is owned here so onboard compute never falls back
to legacy Hyp-MP runners.
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

logger = logging.getLogger(__name__)

MODEL_VERSION = "TokyoEye@champion"


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
                "CUDA OOM during TokyoEye inference on device=%s.",
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
    """Runner for TokyoEye inference (GNNRunner protocol)."""

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
        self._restore_metadata: dict[str, Any] = {}

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    @property
    def space_type(self) -> str:
        return "hyperbolic"

    def _load_model_sync(self) -> None:
        import torch

        from science.tokyo_eye.v8.equiformer_frontend import (
            StubEquiformerFrontend,
            TokyoEyeV8WithFrontend,
            load_weight_map,
        )
        from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

        checkpoint = self._resolve_checkpoint_path()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"TokyoEye checkpoint not found: {self._checkpoint_path}"
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
            logger.info("TokyoEye: %d missing keys", len(missing))
        if unexpected:
            logger.warning("TokyoEye: %d unexpected keys ignored", len(unexpected))
        system.eval()
        system.to(self._device)
        self._system = system
        self._checkpoint_path = str(checkpoint)
        self._loaded = True
        logger.info("TokyoEye loaded from %s (device=%s)", self._checkpoint_path, self._device)

    def _resolve_checkpoint_path(self) -> Path:
        requested = str(self._checkpoint_path or "").strip()
        if requested.startswith("models:/TokyoEye@"):
            alias = requested.rsplit("@", 1)[-1]
            from science.contracts.model_registry import resolve_production_checkpoint

            resolved = resolve_production_checkpoint(alias=alias)
            self._restore_metadata = dict(resolved)
            return Path(resolved["cache_path"])

        from science.contracts.model_registry import resolve_checkpoint_file

        resolved = resolve_checkpoint_file(requested)
        return resolved if resolved is not None else Path(requested)

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
        """Run TokyoEye forward and map outputs into governed GNN results."""
        import torch

        if checkpoint_path and checkpoint_path != self._checkpoint_path:
            self._checkpoint_path = checkpoint_path
            self._loaded = False

        await self._ensure_model_loaded()
        graph_data = graph_data.to(self._device)
        coords = self._model_input_coords(graph_data).to(self._device)
        edge_index = graph_data.edge_index.to(self._device)
        edge_type = self._edge_type_from_graph(graph_data).to(self._device)

        with _gpu_safe_inference(self._device):
            with torch.no_grad():
                output = await asyncio.to_thread(
                    self._system,
                    coords,
                    edge_index,
                    edge_type,
                )

        return self._extract_results(output, graph_data, structure_id)

    def get_curvature(self) -> float:
        if self._curvature_override is not None:
            return require_learned_curvature(
                self._curvature_override,
                context="tokyoeye runner override",
            )
        if self._system is None:
            raise RuntimeError("TokyoEye model is not loaded")
        spine = getattr(self._system, "spine", None)
        raw = getattr(spine, "curvature", None)
        if raw is None:
            raw = getattr(spine, "c", None)
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().item()
        return require_learned_curvature(float(raw), context="tokyoeye runner")

    def _model_input_coords(self, graph_data: Any):
        import torch

        coords = getattr(graph_data, "ca_coords", None)
        source = "ca_coords"
        if coords is None:
            coords = getattr(graph_data, "pos", None)
            source = "pos"
        if coords is None:
            coords = graph_data.x[:, :3]
            source = "x_first3_fallback"
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"TokyoEye coordinates must be [N, 3], got {tuple(coords.shape)}")
        graph_data.coordinate_source = source
        return torch.nan_to_num(coords.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _edge_type_from_graph(self, graph_data: Any):
        import torch

        edge_index = graph_data.edge_index
        existing = getattr(graph_data, "edge_type", None)
        if existing is not None and int(existing.numel()) == int(edge_index.shape[1]):
            graph_data.edge_type_policy = "provided"
            return existing.long()

        from science.dtie.common.graph_builder import HBOND_CA_DISTANCE_CUTOFF
        from science.tokyo_eye.v8.r0_r5_graph import (
            R0_COVALENT,
            R1_HBOND,
            R5_LOCAL_NEIGHBORHOOD,
        )

        e = int(edge_index.shape[1])
        out = torch.full(
            (e,),
            int(R5_LOCAL_NEIGHBORHOOD),
            dtype=torch.long,
            device=edge_index.device,
        )
        residue_indices = list(getattr(graph_data, "residue_indices", []))
        edge_attr = getattr(graph_data, "edge_attr", None)
        for k in range(e):
            src = int(edge_index[0, k].item())
            dst = int(edge_index[1, k].item())
            seq_sep = (
                abs(int(residue_indices[src]) - int(residue_indices[dst]))
                if residue_indices and src < len(residue_indices) and dst < len(residue_indices)
                else abs(src - dst)
            )
            dist = (
                float(edge_attr[k, 3].detach().cpu().item())
                if edge_attr is not None and edge_attr.ndim == 2 and edge_attr.shape[1] >= 4
                else float("inf")
            )
            if seq_sep == 1:
                out[k] = int(R0_COVALENT)
            elif seq_sep <= 5 and dist <= float(HBOND_CA_DISTANCE_CUTOFF):
                out[k] = int(R1_HBOND)
        graph_data.edge_type_policy = "governed_ca_contact_r0_r1_r5_interim"
        return out

    def _input_features(self, graph_data: Any):
        import torch

        features = graph_data.x.detach().float().cpu()
        if features.ndim != 2:
            raise ValueError("graph_data.x must be [N, F]")
        if features.shape[1] >= 4:
            return features.numpy()
        sasa = getattr(graph_data, "sasa", None)
        if sasa is None:
            sasa = torch.zeros(features.shape[0], dtype=features.dtype)
        sasa = sasa.detach().float().cpu().reshape(features.shape[0], 1)
        return torch.cat([features, sasa], dim=1).numpy()

    def _extract_results(
        self,
        output: dict[str, Any],
        graph_data: Any,
        structure_id: str,
    ) -> GNNInferenceResult:
        import torch

        from science.tokyo_eye.v8.attention import poincare_dist

        z_hyp = output["z_hyp"].detach().float()
        h_euc = output["h_euc"].detach().float()
        num_nodes = int(z_hyp.shape[0])
        curvature = self.get_curvature()
        origin = torch.zeros_like(z_hyp)
        if bool(getattr(graph_data, "structural_z_disc_frozen", False)) and hasattr(
            graph_data, "structural_cone_depth"
        ):
            cone_depth_tensor = graph_data.structural_cone_depth.detach().float().to(z_hyp.device)
        else:
            raw_depth = poincare_dist(z_hyp, origin, c=curvature).detach()
            depth_max = raw_depth.max().clamp_min(1e-8)
            cone_depth_tensor = (raw_depth / depth_max) * 8.0
        cone_width = torch.exp(-cone_depth_tensor)

        evidence = output.get("evidence")
        if evidence is not None:
            evidence = evidence.detach().float()
            nu = evidence[:, 1].clamp_min(1e-4)
            alpha = evidence[:, 2].clamp_min(1.0001)
            beta = evidence[:, 3].clamp_min(1e-4)
            aleatoric = beta / (alpha - 1.0)
            epistemic = beta / (nu * (alpha - 1.0))
            total = aleatoric + epistemic
        else:
            aleatoric = torch.zeros(num_nodes, device=z_hyp.device)
            epistemic = torch.zeros(num_nodes, device=z_hyp.device)
            total = torch.zeros(num_nodes, device=z_hyp.device)
        aleatoric = torch.nan_to_num(aleatoric, nan=0.0, posinf=1e6, neginf=0.0)
        epistemic = torch.nan_to_num(epistemic, nan=0.0, posinf=1e6, neginf=0.0)
        total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)

        routing = (output.get("moe_aux") or {}).get("routing")
        routing = routing.detach().float() if routing is not None else None
        input_features = self._input_features(graph_data)
        chain_ids = getattr(graph_data, "chain_ids", ["A"] * num_nodes)
        residue_indices = getattr(graph_data, "residue_indices", list(range(1, num_nodes + 1)))
        hyp_xy = z_hyp[:, :2]
        if hyp_xy.shape[1] < 2:
            hyp_xy = torch.nn.functional.pad(hyp_xy, (0, 2 - hyp_xy.shape[1]))

        nodes: list[GNNNodeOutput] = []
        for i in range(num_nodes):
            nodes.append(
                GNNNodeOutput(
                    residue_index=int(residue_indices[i]),
                    chain_label=str(chain_ids[i]),
                    input_features=np.asarray(input_features[i], dtype=np.float32),
                    projections=h_euc[i].detach().cpu().numpy().astype("float32"),
                    cone_depth=float(cone_depth_tensor[i].detach().cpu().item()),
                    cone_width=float(cone_width[i].detach().cpu().item()),
                    epistemic_uncertainty=float(epistemic[i].detach().cpu().item()),
                    aleatoric_uncertainty=float(aleatoric[i].detach().cpu().item()),
                    total_uncertainty=float(total[i].detach().cpu().item()),
                    x_hyp=z_hyp[i].detach().cpu().numpy().astype("float64"),
                    x_routed_hyp=z_hyp[i].detach().cpu().numpy().astype("float64"),
                    hyp_projections=hyp_xy[i].detach().cpu().numpy().astype("float64"),
                    expert_weights=(
                        routing[i].detach().cpu().numpy().astype("float32")
                        if routing is not None
                        else None
                    ),
                )
            )

        moe_aux = output.get("moe_aux") or {}
        load = moe_aux.get("load")
        metadata = {
            "device": self._device,
            "num_nodes": num_nodes,
            "architecture": "tokyoeye_equiformer_v3_moe",
            "production_module": "science.tokyo_eye.v8",
            "restore": dict(self._restore_metadata),
            "curvature_source": "runner_override" if self._curvature_override is not None else "model_config",
            "edge_type_policy": getattr(graph_data, "edge_type_policy", "unknown"),
            "coordinate_source": getattr(graph_data, "coordinate_source", "unknown"),
            "hyp_projections_2d_source": "z_hyp_first2",
            "structural_disc_frozen": bool(getattr(graph_data, "structural_z_disc_frozen", False)),
            "moe_load": (
                load.detach().cpu().numpy().astype("float32").tolist()
                if hasattr(load, "detach")
                else None
            ),
            "moe_hard": moe_aux.get("hard"),
            "mechanism_score_mean": (
                float(output["mechanism_score"].detach().float().mean().cpu().item())
                if "mechanism_score" in output
                else None
            ),
        }

        return GNNInferenceResult(
            structure_id=structure_id,
            model_version=self.model_version,
            checkpoint_path=self._checkpoint_path,
            nodes=nodes,
            curvature=curvature,
            embedding_dim=int(z_hyp.shape[1]),
            space_type="hyperbolic",
            metadata=metadata,
        )


__all__ = ["MODEL_VERSION", "TokyoEyeV8Runner"]
