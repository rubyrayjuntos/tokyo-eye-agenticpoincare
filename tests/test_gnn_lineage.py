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


def test_checkpoint_filename_prefix() -> None:
    assert checkpoint_filename("v65", "best") == "v65_best.pt"
    assert checkpoint_filename("v65", "phase", phase=2, protein_count=12) == "v65_phase2_12prot.pt"


def test_apply_lineage_defaults_rewrites_paths() -> None:
    config = TrainingConfig(gnn_lineage="v6.5")
    updated = apply_lineage_defaults(config)
    assert updated.model_version == "GOSPConeMapper-v6.5"
    assert updated.mlflow_experiment == "tokyo-eyes-v65"
    assert updated.output_dir == Path("checkpoints/v65/runs")


def test_default_output_dir() -> None:
    assert default_output_dir("v6.5", "cold_v1") == Path("checkpoints/v65/runs/cold_v1")


def test_build_model_v65_instantiates() -> None:
    config = TrainingConfig(gnn_lineage="v6.5", num_experts=2, hidden=16, num_layers=2)
    model = build_model(config)
    assert model.__class__.__name__ == "GOSPConeMapperV65"


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


def test_unknown_lineage_raises() -> None:
    with pytest.raises(ValueError, match="Unknown GNN lineage"):
        get_lineage("v7")
