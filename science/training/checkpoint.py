"""Checkpoint save/load for v6 GNN training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointData:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    global_epoch: int
    phase: int
    phase_name: str
    metrics: dict[str, Any]
    training_config: dict[str, Any]
    architecture: dict[str, Any]
    score: float = -float("inf")


class CheckpointManager:
    """Manages training checkpoints with metadata."""

    def __init__(self, output_dir: Path, protein_count: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protein_count = protein_count

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        *,
        global_epoch: int,
        phase: int,
        phase_name: str,
        metrics: dict[str, Any],
        training_config: dict[str, Any],
        score: float,
    ) -> Path:
        path = self.output_dir / "v6_best.pt"
        curvature = float(model.curvature.item()) if hasattr(model, "curvature") else 0.0
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "global_epoch": global_epoch,
            "phase": phase,
            "phase_name": phase_name,
            "metrics": metrics,
            "training_config": training_config,
            "score": score,
            "curvature": curvature,
            "architecture": {
                "version": "v6",
                "node_dim": 4,
                "hidden": getattr(model, "hidden", 128),
                "num_experts": len(model.experts) if hasattr(model, "experts") else 4,
                "hyperbolic_gate": getattr(model, "hyperbolic_gate", False),
                "hyperbolic_expert_mix": getattr(model, "hyperbolic_expert_mix", False),
                "gate_disc_input": getattr(model, "gate_disc_input", False),
                "deep_hyperbolic_gate": getattr(model, "deep_hyperbolic_gate", False),
                "gate_gumbel": getattr(model, "gate_gumbel", False),
                "gate_disc_scale": getattr(model, "gate_disc_scale", 1.0),
                "disc_radial_source": getattr(model, "disc_radial_source", "mobius"),
                "topology_only_gate": getattr(getattr(model, "gate", None), "topology_only", False),
            },
        }
        torch.save(payload, path)
        return path

    def save_best_disc(
        self,
        model: torch.nn.Module,
        *,
        global_epoch: int,
        phase: int,
        phase_name: str,
        disc_sigma2_sigma1: float,
        metrics: dict[str, Any],
        training_config: dict[str, Any] | None = None,
    ) -> Path:
        """Best disc occupancy snapshot (eligible or not) for recovery warm-starts."""
        path = self.output_dir / "v6_best_disc.pt"
        curvature = float(model.curvature.item()) if hasattr(model, "curvature") else 0.0
        legacy = bool(getattr(model, "legacy_disc_projection", False))
        disc_source = (
            "legacy_post_routing_x_routed_hyp"
            if legacy
            else "pre_routing_x_hyp"
        )
        metrics_out = dict(metrics)
        metrics_out["disc_projection_source"] = disc_source
        metrics_out["legacy_disc_projection"] = legacy
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "global_epoch": global_epoch,
            "phase": phase,
            "phase_name": phase_name,
            "disc_sigma2_sigma1_mean": disc_sigma2_sigma1,
            "metrics": metrics_out,
            "curvature": curvature,
        }
        if training_config:
            payload["training_config"] = training_config
            payload["architecture"] = {
                "version": "v6",
                "node_dim": 4,
                "hidden": getattr(model, "hidden", 128),
                "num_experts": len(model.experts) if hasattr(model, "experts") else 4,
                "hyperbolic_gate": getattr(model, "hyperbolic_gate", False),
                "hyperbolic_expert_mix": getattr(model, "hyperbolic_expert_mix", False),
                "gate_disc_input": getattr(model, "gate_disc_input", False),
                "deep_hyperbolic_gate": getattr(model, "deep_hyperbolic_gate", False),
                "gate_gumbel": getattr(model, "gate_gumbel", False),
                "gate_disc_scale": getattr(model, "gate_disc_scale", 1.0),
                "topology_only_gate": getattr(getattr(model, "gate", None), "topology_only", False),
                "legacy_disc_projection": legacy,
                "disc_projection_path": getattr(
                    model, "disc_projection_path", "post_routing" if legacy else "pre_routing"
                ),
                "disc_projection_source": disc_source,
                "gate_mode": getattr(
                    model, "gate_mode", "hyperbolic" if getattr(model, "hyperbolic_gate", True) else "tangent_mlp"
                ),
                "radial_angular_recombine": getattr(model, "radial_angular_recombine", "multiply"),
                "disc_radial_source": getattr(model, "disc_radial_source", "mobius"),
            }
        torch.save(payload, path)
        return path

    def save_phase(
        self,
        model: torch.nn.Module,
        *,
        phase: int,
        phase_name: str,
        global_epoch: int,
    ) -> Path:
        slug = phase_name.split(":")[0].strip().lower().replace(" ", "_")
        slug = re.sub(r"[^\w.-]+", "_", slug).strip("_") or f"phase_{phase}"
        path = self.output_dir / f"v6_phase{phase}_{self.protein_count}prot.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "phase": phase,
                "phase_name": phase_name,
                "global_epoch": global_epoch,
            },
            path,
        )
        # Also write slug-named copy for human browsing
        torch.save(
            {"model_state_dict": model.state_dict(), "phase": phase, "global_epoch": global_epoch},
            self.output_dir / f"{slug}.pt",
        )
        return path

    def save_epoch(
        self,
        model: torch.nn.Module,
        *,
        global_epoch: int,
        phase: int,
        training_config: dict[str, Any] | None = None,
    ) -> Path:
        """Lightweight per-epoch snapshot for training filmstrip diagnostics."""
        epoch_dir = self.output_dir / "epochs"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        path = epoch_dir / f"epoch_{global_epoch:03d}.pt"
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "global_epoch": global_epoch,
            "phase": phase,
        }
        if training_config:
            payload["training_config"] = training_config
        torch.save(payload, path)
        return path

    def load(self, path: Path, device: str = "cpu") -> CheckpointData:
        raw = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError(f"Checkpoint at {path} is not a dict payload")
        return CheckpointData(
            model_state_dict=raw.get("model_state_dict", raw),
            optimizer_state_dict=raw.get("optimizer_state_dict"),
            global_epoch=int(raw.get("global_epoch", 0)),
            phase=int(raw.get("phase", 0)),
            phase_name=str(raw.get("phase_name", "")),
            metrics=dict(raw.get("metrics", {})),
            training_config=dict(raw.get("training_config", {})),
            architecture=dict(raw.get("architecture", {})),
            score=float(raw.get("score", raw.get("metrics", {}).get("score", -float("inf")))),
        )

    def write_metrics_log(self, entries: list[dict[str, Any]]) -> Path:
        path = self.output_dir / "metrics.json"
        path.write_text(json.dumps(entries, indent=2, default=str))
        return path
