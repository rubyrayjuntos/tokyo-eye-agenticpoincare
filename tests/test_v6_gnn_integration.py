"""Integration gates for production V6 GNN architecture + checkpoint pairing."""

from __future__ import annotations

import pytest

from science.contracts.model_registry import (
    checkpoint_exists,
    get_production_checkpoint_path,
    get_production_model,
)
from science.dtie.v6.gnn.hyperbolic_moe import HyperbolicPrototypeGate
from science.dtie.v6.gnn.model import verify_v6_checkpoint


def test_production_runner_points_at_v6_module() -> None:
    model = get_production_model()
    assert model.runner_module == "science.dtie.v6.gnn.runner"
    assert model.runner_class == "V6GNNRunner"
    assert model.model_version == "GOSPConeMapper-v6"


def test_hyperbolic_moe_importable() -> None:
    assert HyperbolicPrototypeGate is not None


@pytest.mark.skipif(
    not checkpoint_exists(get_production_checkpoint_path()),
    reason="production checkpoint not present in workspace",
)
def test_production_checkpoint_matches_v6_architecture() -> None:
    result = verify_v6_checkpoint()
    assert result["load_ok"], f"missing keys: {result['missing_keys']}"
    assert result["unexpected_keys"] == []
    assert result["hyperbolic_moe_ok"]
    assert result["mobius3_in_checkpoint"]
    assert result["deep_hyperbolic_gate"] is True
    assert result["gate_disc_scale"] == pytest.approx(2.5)
    assert result["hyperbolic_expert_mix"] is True


@pytest.mark.skipif(
    not checkpoint_exists(get_production_checkpoint_path()),
    reason="production checkpoint not present in workspace",
)
def test_gnn_inference_job_imports_v6_runner() -> None:
    import inspect

    from science.compute.jobs import gnn_inference as job

    source = inspect.getsource(job.run_gnn_inference)
    assert "V6GNNRunner" in source
    assert "science.dtie.v6.gnn.runner" in source
