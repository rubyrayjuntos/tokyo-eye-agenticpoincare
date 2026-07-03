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
    def test_contract_includes_gnn_models(self) -> None:
        contract = load_contract()
        assert contract["version"] == "1.6"
        assert "gnn_models" in contract

    def test_validate_gnn_models_contract_clean(self) -> None:
        assert validate_gnn_models_contract() == []

    def test_validate_contract_includes_gnn_models(self) -> None:
        assert validate_contract_against_registry() == []

    def test_production_model_is_v6(self) -> None:
        assert get_production_model_id() == "gospc_v6"
        model = get_production_model()
        assert model.model_version == "GOSPConeMapper-v6"
        assert model.api_alias == "v6"
        assert model.status == "production"

    def test_production_checkpoint_path_from_contract(self) -> None:
        assert get_production_checkpoint_path() == "checkpoints/v6/tokyo_eyes_v6.pt"
        assert get_production_api_alias() == "v6"
        assert get_production_model_version() == "GOSPConeMapper-v6"

    def test_resolve_model_for_api_alias(self) -> None:
        model = resolve_model_for_api_alias("v5")
        assert model is not None
        assert model.model_id == "gospc_v5"
        assert resolve_model_for_api_alias("missing") is None

    def test_resolve_model_version_for_checkpoint(self) -> None:
        assert (
            resolve_model_version_for_checkpoint("checkpoints/v6/tokyo_eyes_v6.pt")
            == "GOSPConeMapper-v6"
        )
        assert resolve_model_version_for_checkpoint("unknown/path.pt") is None

    def test_catalogs_cover_legacy_and_production(self) -> None:
        models = get_gnn_model_catalog()
        checkpoints = get_checkpoint_catalog()
        assert set(models) == {"gospc_v6", "gospc_v5"}
        assert set(checkpoints) == {
            "tokyo_eyes_v6",
            "tokyo_eyes_v6_candidate",
            "tokyo_eyes_v5",
        }

    def test_build_models_api_payload_shape(self) -> None:
        payload = build_models_api_payload()
        assert payload["contract_version"] == "1.6"
        assert payload["production_model_id"] == "gospc_v6"
        assert payload["production_checkpoint_path"] == "checkpoints/v6/tokyo_eyes_v6.pt"
        assert len(payload["models"]) == 2
        assert len(payload["checkpoints"]) == 3
        production = next(m for m in payload["models"] if m["model_id"] == "gospc_v6")
        assert production["production_checkpoint"]["checkpoint_id"] == "tokyo_eyes_v6"
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
        assert body["production_model_id"] == "gospc_v6"
        assert body["contract_version"] == "1.6"
        assert any(m["api_alias"] == "v6" for m in body["models"])
