"""Tests for GNN loader used in post-peel buffering atlas."""

from __future__ import annotations

import pytest

from science.compute.gnn_loader import load_gnn_inference_result


class TestGnnLoader:
    @pytest.mark.asyncio
    async def test_load_returns_none_when_empty(self):
        class _DB:
            async def fetch_all(self, *_args, **_kwargs):
                return []

        result = await load_gnn_inference_result(_DB(), "4obe")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_builds_minimal_result(self):
        class _DB:
            async def fetch_all(self, *_args, **_kwargs):
                return [
                    {
                        "residue_id": "4obe:A:12",
                        "epistemic_uncertainty": 0.42,
                        "cone_depth": 3.1,
                        "cone_width": 1.2,
                        "run_id": "run_gnn_abc",
                        "model_version": "GNN-v6",
                    }
                ]

        result = await load_gnn_inference_result(_DB(), "4obe")
        assert result is not None
        assert result.structure_id == "4obe"
        assert len(result.nodes) == 1
        assert result.nodes[0].epistemic_uncertainty == pytest.approx(0.42)
        assert result.metadata.get("run_id") == "run_gnn_abc"
