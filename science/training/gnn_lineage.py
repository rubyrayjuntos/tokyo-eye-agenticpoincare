"""GNN architecture lineage registry — versioned model packages + checkpoint naming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from science.training.config import TrainingConfig

if TYPE_CHECKING:
    import torch.nn as nn

GnnLineageId = Literal["v6", "v6.5", "v6.6"]


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
    "v6.6": GnnLineageSpec(
        lineage_id="v6.6",
        package="science.dtie.v66.gnn.model",
        model_class_name="GOSPConeMapperV66",
        model_version="GOSPConeMapper-v6.6",
        architecture_version="v6.6",
        checkpoint_prefix="v66",
        checkpoint_root=Path("checkpoints/v66/runs"),
        mlflow_experiment="tokyo-eyes-v66",
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


def checkpoint_filename(
    prefix: str,
    kind: str,
    *,
    phase: int | None = None,
    protein_count: int | None = None,
) -> str:
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

    from science.dtie.common.residue_features import (
        GnnInputMode,
        gnn_input_dim_for_barcode,
    )

    spec = get_lineage(config.gnn_lineage)
    module = _import_model_module(spec)
    model_cls = getattr(module, spec.model_class_name)
    input_mode = (
        GnnInputMode.TOPOLOGY_THREE_VECTOR if config.use_dehydron_barcode else None
    )
    resolved_node_dim = (
        int(node_dim)
        if node_dim is not None
        else gnn_input_dim_for_barcode(
            config.use_dehydron_barcode,
            config.use_binned_dehydron,
            mode=input_mode,
        )
    )
    model_kwargs = dict(
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
    if getattr(config, "init_seed", None) is not None:
        model_kwargs["init_seed"] = int(config.init_seed)
    if spec.lineage_id == "v6.6":
        model_kwargs["gate_include_sasa"] = bool(
            getattr(config, "gate_include_sasa", False)
        )
        role = bool(getattr(config, "role_edge_mp", False))
        multi_rel = bool(getattr(config, "multi_rel_edge_mp", False)) or role
        model_kwargs["role_edge_mp"] = role
        model_kwargs["multi_rel_edge_mp"] = multi_rel
        model_kwargs["dehydron_edge_barcode"] = bool(
            getattr(config, "dehydron_edge_barcode", False)
        )
        model_kwargs["role_coupling_edges"] = bool(
            getattr(config, "role_coupling_edges", False)
        )
        model_kwargs["chem_edge_mp"] = bool(getattr(config, "chem_edge_mp", False))
        model_kwargs["containment_edge_mp"] = bool(
            getattr(config, "containment_edge_mp", False)
        )
        model_kwargs["dehydron_exclusivity"] = bool(
            getattr(config, "dehydron_exclusivity", True)
        )
        model_kwargs["dehydron_angular_scale"] = float(
            getattr(config, "dehydron_angular_scale", 1.0)
        )
        model_kwargs["rim_fanout_forward"] = bool(
            getattr(config, "rim_fanout_forward", False)
        )
        model_kwargs["rim_fanout_strength"] = float(
            getattr(config, "rim_fanout_strength", 0.12)
        )
        model_kwargs["rim_fanout_min_r"] = float(
            getattr(config, "rim_fanout_min_r", 0.35)
        )
        model_kwargs["spoke_edge_scale"] = float(
            getattr(config, "spoke_edge_scale", 1.0)
        )
        model_kwargs["ribbon_edge_scale"] = float(
            getattr(config, "ribbon_edge_scale", 1.0)
        )
        model_kwargs["geometric_angular_prior"] = bool(
            getattr(config, "geometric_angular_prior", False)
        )
        model_kwargs["geometric_angular_kappa"] = float(
            getattr(config, "geometric_angular_kappa", 1.0)
        )
        model_kwargs["geometric_angular_alpha"] = float(
            getattr(config, "geometric_angular_alpha", 0.7853981633974483)
        )
        # Thermo attach still needed for type one-hots when multi-rel (non-role).
        model_kwargs["thermo_edge_message_gate"] = bool(
            getattr(config, "thermo_edge_features", False)
        ) and (not multi_rel)
    model = model_cls(**model_kwargs)
    # Marker for prepare_training_batch (not a nn.Parameter).
    model.hyperbolic_mp_graph = bool(getattr(config, "hyperbolic_mp_graph", False))
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
        if version in {"v6.6", "v66"}:
            resolved_lineage = "v6.6"
        elif version in {"v6.5", "v65"}:
            resolved_lineage = "v6.5"
        else:
            resolved_lineage = "v6"
    if resolved_lineage is None and isinstance(training_config, dict):
        resolved_lineage = training_config.get("gnn_lineage", "v6")
    if resolved_lineage is None and isinstance(state, dict):
        # Bare phase_*.pt often omit architecture/training_config. Detect lineage
        # from additive tensors so we don't build a v6 shell and then size-mismatch
        # on gate.topo_encoder / rim_fanout (seen regenerating controlled 3d viewers
        # from phase_12.pt while v66_best_disc.pt still held the metadata).
        state_keys = state.keys()
        if any(
            k.startswith("rim_fanout")
            or "disc_angular" in k
            or k.startswith("input_feature_zscore")
            for k in state_keys
        ):
            resolved_lineage = "v6.6"
        elif any(k.startswith("v65") or "angular_lift" in k for k in state_keys):
            resolved_lineage = "v6.5"
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

    # T1a input z-norm buffers are installed at train time via
    # fit_and_install_input_feature_norm (not constructor kwargs). Bare loads
    # treat them as unexpected keys and drop them — forward then skips z-score
    # and the Poincaré disc collapses to the origin (controlled 3d seed2:
    # train disc_r_mean≈0.31 vs export ≈0.005). Re-register before load_state.
    if isinstance(state, dict) and "input_feat_mean" in state and "input_feat_std" in state:
        import torch as _torch

        zscore_flag = True
        if isinstance(training_config, dict) and "input_feature_zscore" in training_config:
            zscore_flag = bool(training_config["input_feature_zscore"])
        model.input_feature_zscore = zscore_flag
        if isinstance(training_config, dict) and "replace_tau_with_abs_dist" in training_config:
            model.replace_tau_with_abs_dist = bool(
                training_config["replace_tau_with_abs_dist"]
            )
        for buf_name in ("input_feat_mean", "input_feat_std"):
            tensor = state[buf_name]
            if hasattr(model, buf_name):
                getattr(model, buf_name).resize_(tensor.shape).copy_(tensor)
            else:
                model.register_buffer(
                    buf_name, _torch.zeros_like(tensor) if buf_name.endswith("mean") else _torch.ones_like(tensor)
                )

    load_state(model, state)
    model.to(device)
    model.train(False)
    if isinstance(training_config, dict):
        model.hyperbolic_mp_graph = bool(training_config.get("hyperbolic_mp_graph", False))
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
        updates["output_dir"] = (
            spec.checkpoint_root / suffix if suffix else spec.checkpoint_root
        )
    if updates:
        return config.model_copy(update=updates)
    return config
