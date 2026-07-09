"""GNN architecture lineage registry — versioned model packages + checkpoint naming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from science.training.config import TrainingConfig

if TYPE_CHECKING:
    import torch.nn as nn

GnnLineageId = Literal["v6", "v6.5"]


@dataclass(frozen=True)
class GnnLineageSpec:
    """One trainable GNN architecture line under ``science/dtie/``."""

    lineage_id: GnnLineageId
    package: str
    model_class_name: str
    model_version: str
    architecture_version: str
    checkpoint_prefix: str
    checkpoint_root: Path
    mlflow_experiment: str
    frozen_baseline: bool = False


LINEAGE_REGISTRY: dict[GnnLineageId, GnnLineageSpec] = {
    "v6": GnnLineageSpec(
        lineage_id="v6",
        package="science.dtie.v6.gnn.model",
        model_class_name="GOSPConeMapperV6",
        model_version="GOSPConeMapper-v6",
        architecture_version="v6",
        checkpoint_prefix="v6",
        checkpoint_root=Path("checkpoints/v6/runs"),
        mlflow_experiment="tokyo-eyes-v6",
        frozen_baseline=True,
    ),
    "v6.5": GnnLineageSpec(
        lineage_id="v6.5",
        package="science.dtie.v65.gnn.model",
        model_class_name="GOSPConeMapperV65",
        model_version="GOSPConeMapper-v6.5",
        architecture_version="v6.5",
        checkpoint_prefix="v65",
        checkpoint_root=Path("checkpoints/v65/runs"),
        mlflow_experiment="tokyo-eyes-v65",
        frozen_baseline=False,
    ),
}


def get_lineage(lineage_id: str | GnnLineageId) -> GnnLineageSpec:
    if lineage_id not in LINEAGE_REGISTRY:
        supported = ", ".join(sorted(LINEAGE_REGISTRY))
        raise ValueError(f"Unknown GNN lineage {lineage_id!r}; supported: {supported}")
    return LINEAGE_REGISTRY[lineage_id]  # type: ignore[index]


def default_output_dir(lineage_id: str | GnnLineageId, run_id: str = "default") -> Path:
    spec = get_lineage(lineage_id)
    return spec.checkpoint_root / run_id


def checkpoint_filename(prefix: str, kind: str, *, phase: int | None = None, protein_count: int | None = None) -> str:
    if kind == "best":
        return f"{prefix}_best.pt"
    if kind == "best_route":
        return f"{prefix}_best_route.pt"
    if kind == "best_disc":
        return f"{prefix}_best_disc.pt"
    if kind == "phase":
        if phase is None or protein_count is None:
            raise ValueError("phase checkpoint requires phase and protein_count")
        return f"{prefix}_phase{phase}_{protein_count}prot.pt"
    raise ValueError(f"Unknown checkpoint kind: {kind}")


def resolve_prior_checkpoint(
    output_dir: Path,
    *,
    checkpoint_prefix: str,
    phase: int,
    protein_count: int,
) -> Path | None:
    """Pick best or prior phase checkpoint for curriculum continuity."""
    best = output_dir / checkpoint_filename(checkpoint_prefix, "best")
    if best.is_file():
        return best
    if phase >= 2:
        prev = output_dir / checkpoint_filename(
            checkpoint_prefix, "phase", phase=phase - 1, protein_count=protein_count
        )
        if prev.is_file():
            return prev
    return None


def _import_model_module(spec: GnnLineageSpec) -> Any:
    import importlib

    return importlib.import_module(spec.package)


def build_model(config: TrainingConfig, node_dim: int | None = None) -> nn.Module:
    import torch.nn as nn  # noqa: F401 — return type; import kept local (agent has no torch)
    from science.dtie.common.residue_features import GnnInputMode, gnn_input_dim_for_barcode

    spec = get_lineage(config.gnn_lineage)
    module = _import_model_module(spec)
    model_cls = getattr(module, spec.model_class_name)
    input_mode = GnnInputMode.TOPOLOGY_THREE_VECTOR if config.use_dehydron_barcode else None
    resolved_node_dim = (
        int(node_dim)
        if node_dim is not None
        else gnn_input_dim_for_barcode(
            config.use_dehydron_barcode,
            config.use_binned_dehydron,
            mode=input_mode,
        )
    )
    model = model_cls(
        node_dim=resolved_node_dim,
        hidden=config.hidden,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        capacity_threshold=config.capacity_threshold,
        expert_dropout_p=0.0,
        min_usage=config.min_usage,
        topology_only_gate=config.topology_only_gate,
        hyperbolic_gate=config.hyperbolic_gate,
        hyperbolic_expert_mix=config.hyperbolic_expert_mix,
        gate_disc_scale=config.gate_disc_scale,
        gate_gumbel=config.gate_gumbel,
        deep_hyperbolic_gate=config.deep_hyperbolic_gate,
        legacy_disc_projection=config.legacy_disc_projection,
        radial_angular_recombine=config.radial_angular_recombine,
        disc_radial_source=config.disc_radial_source,
        decoupled_uncertainty_heads=config.decoupled_uncertainty_heads,
        expert_depth_decouple=config.expert_depth_decouple,
        structure_gate=config.structure_gate,
    )
    return model


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
    *,
    lineage_id: str | GnnLineageId | None = None,
    legacy_disc_projection: bool | None = None,
) -> nn.Module:
    """Load a checkpoint into the architecture line recorded in its metadata."""
    import torch

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw)
    arch = raw.get("architecture") if isinstance(raw, dict) else None
    training_config = raw.get("training_config") if isinstance(raw, dict) else None

    resolved_lineage = lineage_id
    if resolved_lineage is None and isinstance(arch, dict):
        version = str(arch.get("version", "v6"))
        resolved_lineage = "v6.5" if version in {"v6.5", "v65"} else "v6"
    if resolved_lineage is None and isinstance(training_config, dict):
        resolved_lineage = training_config.get("gnn_lineage", "v6")
    if resolved_lineage is None:
        resolved_lineage = "v6"

    spec = get_lineage(resolved_lineage)
    module = _import_model_module(spec)
    infer_kwargs = getattr(module, f"infer_{spec.checkpoint_prefix}_model_kwargs", None)
    if infer_kwargs is None:
        infer_kwargs = getattr(module, "infer_v6_model_kwargs", None)
    load_state = getattr(module, f"load_{spec.checkpoint_prefix}_state_dict", None)
    if load_state is None:
        load_state = getattr(module, "load_v6_state_dict", None)
    infer_legacy = getattr(module, "infer_legacy_disc_projection_from_checkpoint", None)

    kwargs = infer_kwargs(state, arch, training_config)
    if infer_legacy is not None:
        kwargs["legacy_disc_projection"] = infer_legacy(
            training_config=training_config if isinstance(raw, dict) else None,
            metrics=raw.get("metrics") if isinstance(raw, dict) else None,
            override=legacy_disc_projection,
        )
    model_cls = getattr(module, spec.model_class_name)
    model = model_cls(**kwargs)
    load_state(model, state)
    model.to(device)
    model.train(False)
    return model


def apply_lineage_defaults(config: TrainingConfig) -> TrainingConfig:
    """Fill lineage-derived fields when callers only set ``gnn_lineage``."""
    spec = get_lineage(config.gnn_lineage)
    updates: dict[str, Any] = {}
    if config.model_version == "GOSPConeMapper-v6" and spec.lineage_id != "v6":
        updates["model_version"] = spec.model_version
    if config.mlflow_experiment == "tokyo-eyes-v6" and spec.lineage_id != "v6":
        updates["mlflow_experiment"] = spec.mlflow_experiment
    out = config.output_dir.as_posix()
    if spec.lineage_id != "v6" and out.startswith("checkpoints/v6/runs"):
        suffix = out.removeprefix("checkpoints/v6/runs").lstrip("/")
        updates["output_dir"] = spec.checkpoint_root / suffix if suffix else spec.checkpoint_root
    if updates:
        return config.model_copy(update=updates)
    return config
