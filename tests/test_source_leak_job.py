"""Tests for peeled source_leak_detection job."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from science.compute.jobs.source_leak_detection import detect_source_leaks
from science.compute.runner_dispatch import PEELED_JOBS, dispatch_compute_job
from science.compute.runners.base import JobRunContext
from science.compute.runners.source_leak_detection import run_source_leak_detection
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

import numpy as np


class MockLeakDB:
    def __init__(self, rows: list[dict] | None = None):
        self.rows = rows or []
        self.queries: list[str] = []

    async def fetch_all(self, query: str, params: dict) -> list[dict]:
        self.queries.append(query)
        return self.rows

    async def fetch_one(self, query: str, params: dict) -> dict | None:
        self.queries.append(query)
        return None

    async def execute(self, query: str, params: dict | None = None) -> None:
        self.queries.append(query)

    async def commit(self) -> None:
        return None


class TestDetectSourceLeaks:
    @pytest.mark.asyncio
    async def test_db_backed_detection(self):
        db = MockLeakDB(
            [
                {
                    "residue_id": "4obe:A:12",
                    "cone_depth": 2.1,
                    "epistemic_uncertainty": 0.45,
                }
            ]
        )
        config = PipelineConfig(
            structure_id="4obe",
            uncertainty_threshold=0.3,
            depth_threshold=1.5,
            run_gnn=False,
        )
        result = await detect_source_leaks(db, config, gnn_result="legacy_db_query")
        assert result.success is True
        assert result.outputs["source_leak_count"] == 1
        assert "4obe:A:12" in result.outputs["source_leak_residues"]
        assert any("fact_gnn_node_embedding" in q for q in db.queries)
        assert any("hyperbolic" in q for q in db.queries)

    @pytest.mark.asyncio
    async def test_db_backed_detection_when_gnn_result_none(self):
        db = MockLeakDB(
            [
                {
                    "residue_id": "11qe:A:42",
                    "cone_depth": 2.5,
                    "epistemic_uncertainty": 0.5,
                }
            ]
        )
        config = PipelineConfig(
            structure_id="11qe",
            uncertainty_threshold=0.3,
            depth_threshold=1.5,
            run_gnn=False,
        )
        result = await detect_source_leaks(db, config, gnn_result=None)
        assert result.success is True
        assert result.outputs["source_leak_count"] == 1
        assert "11qe:A:42" in result.outputs["source_leak_residues"]

    @pytest.mark.asyncio
    async def test_in_memory_detection_with_gnn_nodes(self):
        db = MockLeakDB()
        config = PipelineConfig(
            structure_id="4obe",
            uncertainty_threshold=0.3,
            depth_threshold=1.5,
            run_gnn=False,
        )
        gnn_result = GNNInferenceResult(
            structure_id="4obe",
            model_version="test",
            checkpoint_path=None,
            nodes=[
                GNNNodeOutput(
                    residue_index=12,
                    chain_label="A",
                    input_features=np.zeros(4),
                    projections=np.zeros(64),
                    cone_depth=2.0,
                    cone_width=0.1,
                    epistemic_uncertainty=0.5,
                )
            ],
            space_type="hyperbolic",
            metadata={},
        )
        result = await detect_source_leaks(db, config, gnn_result=gnn_result)
        assert result.success is True
        assert result.outputs["source_leak_count"] == 1


class TestSourceLeakRunner:
    @pytest.mark.asyncio
    async def test_runner_loads_gnn_embeddings_before_detection(self):
        gnn_result = GNNInferenceResult(
            structure_id="11qe",
            model_version="test",
            checkpoint_path=None,
            nodes=[
                GNNNodeOutput(
                    residue_index=1,
                    chain_label="A",
                    input_features=np.zeros(4),
                    projections=np.zeros(64),
                    cone_depth=2.0,
                    cone_width=0.1,
                    epistemic_uncertainty=0.5,
                )
            ],
            space_type="hyperbolic",
            metadata={"run_id": "gnn-run-1"},
        )
        db = MockLeakDB()

        with (
            patch(
                "science.compute.runners.source_leak_detection._resolve_gnn_parent_run_id",
                new_callable=AsyncMock,
                return_value="gnn-run-1",
            ),
            patch(
                "science.compute.runners.source_leak_detection.load_gnn_inference_result",
                new_callable=AsyncMock,
                return_value=gnn_result,
            ) as mock_load,
            patch(
                "science.compute.runners.source_leak_detection.persist_phase_result",
                new_callable=AsyncMock,
            ),
        ):
            ctx = JobRunContext(structure_id="11qe", job_id="source_leak_detection")
            result = await run_source_leak_detection(db, ctx)

        mock_load.assert_awaited_once_with(db, "11qe", gnn_run_id="gnn-run-1")
        assert result.success is True
        assert result.outputs.get("source_leak_count", 0) >= 0


class TestRunnerDispatch:
    def test_source_leak_is_peeled(self):
        assert "source_leak_detection" in PEELED_JOBS

    @pytest.mark.asyncio
    async def test_unknown_job_raises(self):
        db = MockLeakDB()
        ctx = JobRunContext(structure_id="4obe", job_id="nonexistent_job_xyz")
        with pytest.raises(ValueError, match="Unknown job_id"):
            await dispatch_compute_job(db, "nonexistent_job_xyz", ctx)
