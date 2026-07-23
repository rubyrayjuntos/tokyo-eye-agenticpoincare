"""Tests for contract-driven GNN model and checkpoint registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from science.contracts.model_registry import (
    build_models_api_payload,
    get_checkpoint_catalog,
    get_gnn_model_catalog,
    get_production_api_alias,
    get_production_checkpoint_path,
    get_production_model,
    get_production_model_id,
    get_production_model_version,
    resolve_model_for_api_alias,
    resolve_model_version_for_checkpoint,
    validate_gnn_models_contract,
)
from science.contracts.onboard_contract import load_contract, validate_contract_against_registry


class TestGnnModelRegistry:
    def setup_method(self) -> None:
        get_gnn_model_catalog.cache_clear()
        get_checkpoint_catalog.cache_clear()

    def test_contract_includes_gnn_models(self) -> None:
        contract = load_contract()
        assert "gnn_models" in contract

    def test_validate_gnn_models_contract_clean(self) -> None:
        assert validate_gnn_models_contract() == []

    def test_validate_contract_against_registry_clean(self) -> None:
        assert validate_contract_against_registry() == []

    def test_production_model_is_tokyo_eye_v8(self) -> None:
        assert get_production_model_id() == "tokyo_eye_v8"
        model = get_production_model()
        assert model.model_version == "TokyoEye-v8"
        assert model.api_alias == "v8"
        assert model.status == "production"
        assert model.runner_class == "TokyoEyeV8Runner"
        assert model.runner_module == "science.tokyo_eye.v8.runner"

    def test_production_checkpoint_path_from_contract(self) -> None:
        assert (
            get_production_checkpoint_path()
            == "checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt"
        )
        assert get_production_api_alias() == "v8"
        assert get_production_model_version() == "TokyoEye-v8"

    def test_resolve_model_for_api_alias(self) -> None:
        model = resolve_model_for_api_alias("v5")
        assert model is not None
        assert model.model_id == "gospc_v5"
        v8 = resolve_model_for_api_alias("v8")
        assert v8 is not None
        assert v8.model_id == "tokyo_eye_v8"
        v7 = resolve_model_for_api_alias("v7")
        assert v7 is not None
        assert v7.status == "deprecated"
        assert resolve_model_for_api_alias("missing") is None

    def test_resolve_model_version_for_checkpoint(self) -> None:
        assert (
            resolve_model_version_for_checkpoint(
                "checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt"
            )
            == "TokyoEye-v8"
        )
        assert (
            resolve_model_version_for_checkpoint("checkpoints/v6/tokyo_eyes_v6.pt")
            == "GOSPConeMapper-v6"
        )
        assert resolve_model_version_for_checkpoint("unknown/path.pt") is None

    def test_catalogs_cover_legacy_and_production(self) -> None:
        models = get_gnn_model_catalog()
        checkpoints = get_checkpoint_catalog()
        assert "tokyo_eye_v8" in models
        assert "tokyo_eye_v7" in models
        assert models["tokyo_eye_v7"].status == "deprecated"
        assert "gospc_v6" in models
        assert "gospc_v5" in models
        assert models["gospc_v6"].status == "legacy"
        assert "tokyo_eye_v8_mode_c_moe_rebalance_s9" in checkpoints
        assert checkpoints["tokyo_eye_v8_mode_c_moe_rebalance_s9"].status == "production"

    def test_build_models_api_payload_shape(self) -> None:
        payload = build_models_api_payload()
        assert payload["production_model_id"] == "tokyo_eye_v8"
        assert (
            payload["production_checkpoint_path"]
            == "checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt"
        )
        production = next(
            m for m in payload["models"] if m["model_id"] == "tokyo_eye_v8"
        )
        assert (
            production["production_checkpoint"]["checkpoint_id"]
            == "tokyo_eye_v8_mode_c_moe_rebalance_s9"
        )
        assert "exists" in production["production_checkpoint"]


@pytest.mark.asyncio
async def test_models_endpoint_responds() -> None:
    with (
        patch("data.db.open_pool", new_callable=AsyncMock),
        patch("data.db.close_pool", new_callable=AsyncMock),
    ):
        from httpx import ASGITransport, AsyncClient

        from science.api.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/compute/models")

        assert resp.status_code == 200
        body = resp.json()
        assert body["production_model_id"] == "tokyo_eye_v8"
        assert any(m["api_alias"] == "v8" for m in body["models"])
        assert any(m["api_alias"] == "v6" for m in body["models"])
