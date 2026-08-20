"""Integration gates for production TokyoEye GNN + legacy v6 module smoke."""

from __future__ import annotations

import pytest

from science.contracts.model_registry import (
    checkpoint_exists,
    get_model_spec,
    get_production_checkpoint_path,
    get_production_model,
)
from science.dtie.v6.gnn.hyperbolic_moe import HyperbolicPrototypeGate


def test_production_runner_points_at_tokyo_eye() -> None:
    model = get_production_model()
    assert model.model_id == "TokyoEye"
    assert model.runner_module == "science.tokyo_eye.v8.runner"
    assert model.runner_class == "TokyoEyeV8Runner"
    assert model.model_version == "TokyoEye@champion"


def test_legacy_v6_still_registered() -> None:
    model = get_model_spec("gospc_v6")
    assert model.status == "legacy"
    assert model.runner_class == "V6GNNRunner"
    assert model.model_version == "GOSPConeMapper-v6"


def test_hyperbolic_moe_importable() -> None:
    assert HyperbolicPrototypeGate is not None


@pytest.mark.skipif(
    not checkpoint_exists(get_production_checkpoint_path()),
    reason="production checkpoint not present in workspace",
)
def test_production_checkpoint_loads_tokyo_eye() -> None:
    from science.tokyo_eye.v8.runner import TokyoEyeV8Runner

    runner = TokyoEyeV8Runner(checkpoint_path=get_production_checkpoint_path(), device="cpu")
    runner._load_model_sync()
    assert runner._loaded is True
    assert runner.model_version == "TokyoEye@champion"
