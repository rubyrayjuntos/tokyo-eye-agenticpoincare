"""PyFunc packaging for Tokyo Eye — real Equiformer + MoE load/predict.

Heavy weights load in ``load_context`` from logged artifacts (never pickled in
``__init__``). ``predict`` runs a forward pass and returns summary metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_MAP = "science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json"


def load_tokyoeye_system(
    checkpoint_path: Path | str,
    *,
    device: str = "cpu",
    weight_map: Path | str | None = None,
) -> Any:
    """Build Equiformer frontend + hyp spine + MoE and load checkpoint weights."""
    import torch

    from science.tokyo_eye.v8.equiformer_frontend import (
        StubEquiformerFrontend,
        TokyoEyeV8WithFrontend,
        load_weight_map,
    )
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"TokyoEye checkpoint not found: {path}")

    cfg = load_weight_map(weight_map or DEFAULT_WEIGHT_MAP)
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
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    missing, unexpected = system.load_state_dict(state, strict=False)
    if missing:
        logger.info("TokyoEye pyfunc: %d missing keys", len(missing))
    if unexpected:
        logger.warning("TokyoEye pyfunc: %d unexpected keys ignored", len(unexpected))
    system.eval()
    system.to(device)
    return system


def _synthetic_batch(*, n_nodes: int, device: str) -> tuple[Any, Any, Any]:
    import torch

    from science.tokyo_eye.v8.r0_r5_graph import R0_COVALENT, R5_LOCAL_NEIGHBORHOOD

    n = max(3, int(n_nodes))
    coords = torch.randn(n, 3, device=device)
    # Chain edges + a few skip edges
    src = list(range(n - 1)) + list(range(0, n - 2, 2))
    dst = list(range(1, n)) + list(range(2, n, 2))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long, device=device)
    e = edge_index.shape[1]
    edge_type = torch.full((e,), int(R5_LOCAL_NEIGHBORHOOD), dtype=torch.long, device=device)
    # Mark sequential neighbors as covalent when possible
    for k in range(min(e, 2 * (n - 1))):
        if abs(int(edge_index[0, k]) - int(edge_index[1, k])) == 1:
            edge_type[k] = int(R0_COVALENT)
    return coords, edge_index, edge_type


def forward_summary(system: Any, *, n_nodes: int = 8, device: str = "cpu") -> dict[str, Any]:
    """Run one no-grad forward; return JSON-serializable MoE/geometry summaries."""
    import torch

    coords, edge_index, edge_type = _synthetic_batch(n_nodes=n_nodes, device=device)
    with torch.no_grad():
        out = system(coords, edge_index, edge_type)
    z = out["z_hyp"]
    logits = out["sdrp_logits"]
    moe_aux = out.get("moe_aux") or {}
    spine = getattr(system, "spine", None)
    curv = getattr(spine, "c", None)
    if hasattr(curv, "detach"):
        curv = float(curv.detach().cpu().item())
    routing = moe_aux.get("routing")
    n_experts = None
    if routing is not None and hasattr(routing, "shape") and routing.ndim >= 2:
        n_experts = int(routing.shape[-1])
    return {
        "status": "ok",
        "n_nodes": float(z.shape[0]),
        "z_hyp_dim": float(z.shape[-1]),
        "z_hyp_norm_mean": float(torch.linalg.vector_norm(z, dim=-1).mean().cpu()),
        "sdrp_logits_dim": float(logits.shape[-1]),
        "curvature_c": float(curv) if curv is not None else float("nan"),
        "moe_n_experts": float(n_experts) if n_experts is not None else float("nan"),
        "has_moe_aux": 1.0 if moe_aux else 0.0,
    }


class TokyoEyePyFuncModel(mlflow.pyfunc.PythonModel):
    """MLflow ``PythonModel``: Equiformer + hyperbolic spine + MoE."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self._checkpoint_path: str | None = None
        self._system: Any | None = None
        self._device = str(self.config.get("device", "cpu"))
        self._loaded = False

    def load_context(self, context: Any) -> None:
        arts = getattr(context, "artifacts", {}) or {}
        ckpt = arts.get("checkpoint")
        if ckpt is None:
            raise FileNotFoundError("PyFunc context missing artifact 'checkpoint'")
        self._checkpoint_path = str(ckpt)
        weight_map = self.config.get("weight_map") or arts.get("weight_map")
        self._system = load_tokyoeye_system(
            self._checkpoint_path,
            device=self._device,
            weight_map=weight_map,
        )
        self._loaded = True
        logger.info(
            "TokyoEye PyFunc loaded system from %s (device=%s)",
            self._checkpoint_path,
            self._device,
        )

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame | dict[str, Any] | list[Any],
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context
        if not self._loaded or self._system is None:
            raise RuntimeError("TokyoEye PyFunc model is not loaded")
        params = dict(params or {})
        n_nodes = int(params.get("n_nodes", self.config.get("n_nodes", 8)))
        if isinstance(model_input, pd.DataFrame) and "n_nodes" in model_input.columns:
            n_nodes = int(model_input["n_nodes"].iloc[0])
        summary = forward_summary(self._system, n_nodes=n_nodes, device=self._device)
        summary["checkpoint"] = self._checkpoint_path
        return pd.DataFrame([summary])


def log_tokyoeye_pyfunc(
    *,
    checkpoint_path: Path | str,
    artifact_path: str = "tokyoeye_model",
    config: dict[str, Any] | None = None,
    pip_requirements: list[str] | None = None,
    code_paths: list[str] | None = None,
) -> str:
    """Log a PyFunc model bound to ``checkpoint`` artifact; return model URI."""
    import mlflow
    from mlflow.models import infer_signature

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    sample = pd.DataFrame({"n_nodes": [8]})
    output_example = pd.DataFrame(
        [
            {
                "status": "ok",
                "n_nodes": 8.0,
                "z_hyp_dim": 1.0,
                "z_hyp_norm_mean": 0.0,
                "sdrp_logits_dim": 1.0,
                "curvature_c": 1.0,
                "moe_n_experts": 4.0,
                "has_moe_aux": 1.0,
                "checkpoint": str(path),
            }
        ]
    )
    signature = infer_signature(sample, output_example)

    model = TokyoEyePyFuncModel(config=config)
    reqs = pip_requirements or ["mlflow", "pandas", "torch", "torch-geometric"]
    # Bundle science package so serving environments can import Tokyo Eye modules.
    paths = code_paths or ["science"]
    kwargs: dict[str, Any] = {
        "python_model": model,
        "artifacts": {"checkpoint": str(path)},
        "signature": signature,
        "input_example": sample,
        "pip_requirements": reqs,
        "code_paths": paths,
    }
    try:
        mlflow.pyfunc.log_model(name=artifact_path, **kwargs)
    except TypeError:
        mlflow.pyfunc.log_model(artifact_path=artifact_path, **kwargs)
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_tokyoeye_pyfunc requires an active MLflow run")
    return f"runs:/{run.info.run_id}/{artifact_path}"
