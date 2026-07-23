"""GNN architecture lineage registry — versioned model packages + checkpoint naming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from science.training.config import TrainingConfig

if TYPE_CHECKING:
    import torch.nn as nn

GnnLineageId = Literal["v6", "v6.5", "v6.6", "v7", "v8"]


@dataclass(frozen=True)
class GnnLineageSpec:
    """One trainable GNN architecture line.

    v8+ production models live under ``science/tokyo_eye/v8/``.
    v7 Hyp-MP under ``science/tokyo_eye/TokyoEye.py`` is frozen archaeology.
    """

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
        frozen_baseline=True,
    ),
    "v7": GnnLineageSpec(
        lineage_id="v7",
        package="science.tokyo_eye.TokyoEye",
        model_class_name="TokyoEye",
        model_version="TokyoEye-v7",
        architecture_version="v7",
        checkpoint_prefix="v7",
        checkpoint_root=Path("checkpoints/v7/runs"),
        mlflow_experiment="tokyo-eyes-v7",
        frozen_baseline=True,  # dead archaeology — do not open as trunk
    ),
    "v8": GnnLineageSpec(
        lineage_id="v8",
        package="science.tokyo_eye.v8.model",
        model_class_name="TokyoEyesHyperbolicV8",
        model_version="TokyoEye-v8",
        architecture_version="v8",
        checkpoint_prefix="v8",
        checkpoint_root=Path("checkpoints/v8/runs"),
        mlflow_experiment="tokyo-eyes-v8",
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
    if spec.lineage_id == "v8":
        # Prefer experiments/training/v8 harness for Equiformer+spine; this path
        # builds the hyp spine only for lineage registry / unit smoke.
        model = model_cls(
            scalar_dim=int(getattr(config, "scalar_dim", 128) or 128),
            vector_dim=int(getattr(config, "vector_dim", 3) or 3),
            hidden_dim=int(getattr(config, "hidden_dim", 64) or 64),
            num_attn_layers=int(getattr(config, "num_attn_layers", 2) or 2),
            num_relations=int(getattr(config, "num_relations", 6) or 6),
            num_sdrp_classes=int(getattr(config, "num_sdrp_classes", 5) or 5),
        )
        model.hyperbolic_mp_graph = False
        return model
    if spec.lineage_id in {"v6.6", "v7"}:
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
        model_kwargs["ha_edge_mp"] = bool(getattr(config, "ha_edge_mp", False))
        model_kwargs["containment_edge_mp"] = bool(
            getattr(config, "containment_edge_mp", False)
        )
        model_kwargs["euclidean_shortcut_mp"] = bool(
            getattr(config, "euclidean_shortcut_mp", False)
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
    if spec.lineage_id == "v7":
        model_kwargs["hyp_mp_primary"] = bool(
            getattr(config, "hyp_mp_primary", True)
        )
        model_kwargs["se3_aux"] = bool(getattr(config, "se3_aux", False))
        model_kwargs["hyp_mp_layers"] = int(getattr(config, "hyp_mp_layers", 3))
        model_kwargs["hyperbolic_gate"] = bool(
            getattr(config, "hyperbolic_gate", True)
        )
        model_kwargs["hyperbolic_expert_mix"] = bool(
            getattr(config, "hyperbolic_expert_mix", True)
        )
    model = model_cls(**model_kwargs)
    # Marker for prepare_training_batch (not a nn.Parameter).
    # v7: Hyp MP primary — do not use S4 "euc conv on hyp edges" flag.
    if spec.lineage_id == "v7":
        model.hyperbolic_mp_graph = False
    else:
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
        if version in {"v8", "TokyoEye-v8"}:
            resolved_lineage = "v8"
        if version in {"v7", "TokyoEye-v7"}:
            resolved_lineage = "v7"
        elif version in {"v6.6", "v66"}:
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
    default_versions = {
        "GOSPConeMapper-v6",
        "GOSPConeMapper-v6.5",
        "GOSPConeMapper-v6.6",
        "TokyoEye-v7",
        "TokyoEye-v8",
    }
    if (
        config.model_version in default_versions
        and config.model_version != spec.model_version
    ):
        updates["model_version"] = spec.model_version
    default_experiments = {
        "tokyo-eyes-v6",
        "tokyo-eyes-v65",
        "tokyo-eyes-v66",
        "tokyo-eyes-v7",
        "tokyo-eyes-v8",
    }
    if (
        config.mlflow_experiment in default_experiments
        and config.mlflow_experiment != spec.mlflow_experiment
    ):
        updates["mlflow_experiment"] = spec.mlflow_experiment
    out = config.output_dir.as_posix()
    default_roots = (
        "checkpoints/v6/runs",
        "checkpoints/v65/runs",
        "checkpoints/v66/runs",
        "checkpoints/v7/runs",
        "checkpoints/v8/runs",
    )
    if any(out == root or out.startswith(root + "/") for root in default_roots):
        if not out.startswith(spec.checkpoint_root.as_posix()):
            # Preserve run suffix after the lineage root.
            suffix = ""
            for root in default_roots:
                if out == root:
                    suffix = ""
                    break
                if out.startswith(root + "/"):
                    suffix = out[len(root) + 1 :]
                    break
            updates["output_dir"] = (
                spec.checkpoint_root / suffix if suffix else spec.checkpoint_root
            )
    if updates:
        return config.model_copy(update=updates)
    return config
