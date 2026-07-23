"""Tests for GNN lineage registry and checkpoint naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from science.training.config import TrainingConfig
from science.training.gnn_lineage import (
    LINEAGE_REGISTRY,
    apply_lineage_defaults,
    build_model,
    checkpoint_filename,
    default_output_dir,
    get_lineage,
    resolve_prior_checkpoint,
)
from science.training.checkpoint import CheckpointManager


def test_lineage_registry_has_v6_and_v65() -> None:
    v6 = get_lineage("v6")
    v65 = get_lineage("v6.5")
    assert v6.frozen_baseline is True
    assert v65.frozen_baseline is False
    assert v65.checkpoint_prefix == "v65"
    assert v65.mlflow_experiment == "tokyo-eyes-v65"
    assert "v6.5" in LINEAGE_REGISTRY


def test_lineage_registry_has_v66() -> None:
    v66 = get_lineage("v6.6")
    assert v66.frozen_baseline is True  # frozen compare-only in Tokyo Eye v7 era
    assert v66.checkpoint_prefix == "v66"
    assert v66.mlflow_experiment == "tokyo-eyes-v66"
    assert v66.package == "science.dtie.v66.gnn.model"
    assert "v6.6" in LINEAGE_REGISTRY


def test_lineage_registry_has_v7_archaeology() -> None:
    v7 = get_lineage("v7")
    assert v7.frozen_baseline is True
    assert v7.checkpoint_prefix == "v7"
    assert v7.package == "science.tokyo_eye.TokyoEye"
    assert v7.model_class_name == "TokyoEye"
    assert v7.model_version == "TokyoEye-v7"
    assert v7.mlflow_experiment == "tokyo-eyes-v7"
    assert v7.checkpoint_root == Path("checkpoints/v7/runs")


def test_lineage_registry_has_v8() -> None:
    v8 = get_lineage("v8")
    assert v8.frozen_baseline is False
    assert v8.checkpoint_prefix == "v8"
    assert v8.package == "science.tokyo_eye.v8.model"
    assert v8.model_class_name == "TokyoEyesHyperbolicV8"
    assert v8.model_version == "TokyoEye-v8"
    assert v8.mlflow_experiment == "tokyo-eyes-v8"
    assert v8.checkpoint_root == Path("checkpoints/v8/runs")


def test_checkpoint_filename_prefix() -> None:
    assert checkpoint_filename("v65", "best") == "v65_best.pt"
    assert checkpoint_filename("v65", "phase", phase=2, protein_count=12) == "v65_phase2_12prot.pt"
    assert checkpoint_filename("v66", "best") == "v66_best.pt"
    assert checkpoint_filename("v66", "phase", phase=1, protein_count=12) == "v66_phase1_12prot.pt"


def test_apply_lineage_defaults_rewrites_paths() -> None:
    config = TrainingConfig(gnn_lineage="v6.5")
    updated = apply_lineage_defaults(config)
    assert updated.model_version == "GOSPConeMapper-v6.5"
    assert updated.mlflow_experiment == "tokyo-eyes-v65"
    assert updated.output_dir == Path("checkpoints/v65/runs")


def test_apply_lineage_defaults_v66() -> None:
    config = TrainingConfig(gnn_lineage="v6.6")
    updated = apply_lineage_defaults(config)
    assert updated.model_version == "GOSPConeMapper-v6.6"
    assert updated.mlflow_experiment == "tokyo-eyes-v66"
    assert updated.output_dir == Path("checkpoints/v66/runs")


def test_apply_lineage_defaults_v7() -> None:
    config = TrainingConfig(gnn_lineage="v7")
    updated = apply_lineage_defaults(config)
    assert updated.model_version == "TokyoEye-v7"
    assert updated.mlflow_experiment == "tokyo-eyes-v7"


def test_apply_lineage_defaults_v8() -> None:
    config = TrainingConfig(gnn_lineage="v8")
    updated = apply_lineage_defaults(config)
    assert updated.model_version == "TokyoEye-v8"
    assert updated.mlflow_experiment == "tokyo-eyes-v8"
    assert updated.output_dir == Path("checkpoints/v8/runs")


def test_default_output_dir() -> None:
    assert default_output_dir("v6.5", "cold_v1") == Path("checkpoints/v65/runs/cold_v1")
    assert default_output_dir("v6.6", "feeler_p1_v1") == Path(
        "checkpoints/v66/runs/feeler_p1_v1"
    )


def test_build_model_v65_instantiates() -> None:
    config = TrainingConfig(gnn_lineage="v6.5", num_experts=2, hidden=16, num_layers=2)
    model = build_model(config)
    assert model.__class__.__name__ == "GOSPConeMapperV65"


def test_build_model_v66_instantiates() -> None:
    config = TrainingConfig(gnn_lineage="v6.6", num_experts=2, hidden=16, num_layers=2)
    model = build_model(config)
    assert model.__class__.__name__ == "GOSPConeMapperV66"


def test_checkpoint_manager_uses_lineage_prefix(tmp_path: Path) -> None:
    spec = get_lineage("v6.5")
    mgr = CheckpointManager(
        tmp_path,
        protein_count=8,
        checkpoint_prefix=spec.checkpoint_prefix,
        architecture_version=spec.architecture_version,
    )
    config = TrainingConfig(gnn_lineage="v6.5", num_experts=2, hidden=16, num_layers=2)
    model = build_model(config)
    path = mgr.save_phase(model, phase=1, phase_name="smoke", global_epoch=1)
    assert path.name == "v65_phase1_8prot.pt"


def test_resolve_prior_checkpoint_prefers_best(tmp_path: Path) -> None:
    best = tmp_path / "v65_best.pt"
    best.write_text("x")
    found = resolve_prior_checkpoint(
        tmp_path,
        checkpoint_prefix="v65",
        phase=2,
        protein_count=12,
    )
    assert found == best


def test_build_model_v7_instantiates() -> None:
    config = TrainingConfig(gnn_lineage="v7", num_experts=2, hidden=16, num_layers=2)
    model = build_model(config)
    assert model.__class__.__name__ == "TokyoEye"
    assert bool(getattr(model, "hyp_mp_primary", False)) is True


def test_unknown_lineage_raises() -> None:
    with pytest.raises(ValueError, match="Unknown GNN lineage"):
        get_lineage("v99")


def test_healthy_fix1_sealed_load_reconstructs_shell() -> None:
    """Sealed Fix-1 trunk must restore geom prior / hyp-MP / rim min_r / z-norm."""
    from experiments.training.v66.healthy_fix1 import HEALTHY_FIX1_CKPT
    from science.training.gnn_lineage import load_model_from_checkpoint

    if not HEALTHY_FIX1_CKPT.is_file():
        pytest.skip(f"missing sealed healthy ckpt: {HEALTHY_FIX1_CKPT}")

    model = load_model_from_checkpoint(HEALTHY_FIX1_CKPT, "cpu")
    assert type(model).__name__ == "GOSPConeMapperV66"
    assert getattr(model, "geometric_angular_prior", False) is True
    assert getattr(model, "disc_angular_residual", None) is not None
    assert abs(float(model.rim_fanout_min_r) - 0.2) < 1e-9
    assert getattr(model, "hyperbolic_mp_graph", None) is True
    assert model.input_feature_zscore is True
    assert hasattr(model, "input_feat_mean")
    assert tuple(model.input_feat_mean.shape) == (3,)
