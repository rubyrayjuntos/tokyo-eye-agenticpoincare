"""Tests for science code adapters (v3/v4 → Normalizer bridge).

These tests validate the full flow:
  GNN inference → GNNOutputAdapter → Normalizer → governed data
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from data.normalizer.core import Normalizer
from science.dtie.common.adapters import GNNOutputAdapter, Phase3Adapter
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput, PhaseResult


# ---------------------------------------------------------------------------
# Mock DB (reused pattern)
# ---------------------------------------------------------------------------


class MockDB:
    def __init__(self):
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self._in_transaction = False

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        table = self._extract_table(query)
        if table:
            self.tables.setdefault(table, []).append(params)

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        for params in params_list:
            table = self._extract_table(query)
            if table:
                self.tables.setdefault(table, []).append(params)

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "provenance_run" in query:
            for row in self.tables.get("provenance_run", []):
                if row.get("run_id") == params.get("run_id"):
                    return row
        if "embedding_space" in query:
            for row in self.tables.get("embedding_space", []):
                if row.get("name") == params.get("name"):
                    return row
        return None

    async def begin(self) -> None:
        self._in_transaction = True

    async def commit(self) -> None:
        self._in_transaction = False

    async def rollback(self) -> None:
        self._in_transaction = False

    def _extract_table(self, query: str) -> str | None:
        q = query.strip().upper()
        if "INSERT INTO" in q:
            parts = q.split("INSERT INTO")[1].strip().split()
            if parts:
                return parts[0].lower()
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db() -> MockDB:
    return MockDB()


@pytest.fixture
def normalizer(mock_db: MockDB) -> Normalizer:
    return Normalizer(db=mock_db, caller_identity="test_adapter")  # type: ignore[arg-type]


@pytest.fixture
def v4_gnn_result() -> GNNInferenceResult:
    """Simulate v4 GNN inference output (5 residues, hyperbolic)."""
    nodes = []
    for i in range(10, 15):
        nodes.append(
            GNNNodeOutput(
                residue_index=i,
                chain_label="A",
                input_features=np.array([0.5 + i * 0.02, 1.0, float(i % 3), 40.0 + i]),
                projections=np.random.randn(64).astype(np.float32),
                cone_depth=float(i) / 20.0,
                cone_width=0.5 + i * 0.01,
                epistemic_uncertainty=0.1 + i * 0.005,
                aleatoric_uncertainty=0.05 + i * 0.003,
                total_uncertainty=0.15 + i * 0.008,
                x_hyp=np.random.randn(32).astype(np.float32),
                x_routed_hyp=np.random.randn(32).astype(np.float32),
                hyp_projections=np.array([0.1 * (i - 12), 0.1 * (i - 11)], dtype=np.float32),
                expert_weights=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
            )
        )

    return GNNInferenceResult(
        structure_id="4obe",
        model_version="GOSPConeMapper-v4",
        checkpoint_path="s3://checkpoints/tokyo_eyes_v4.pt",
        nodes=nodes,
        curvature=1.0,
        embedding_dim=32,
        space_type="hyperbolic",
    )


@pytest.fixture
def v3_gnn_result() -> GNNInferenceResult:
    """Simulate v3 GNN inference output (5 residues, Euclidean only)."""
    nodes = []
    for i in range(10, 15):
        nodes.append(
            GNNNodeOutput(
                residue_index=i,
                chain_label="A",
                input_features=np.array([0.5 + i * 0.02, 1.0, float(i % 3), 40.0 + i]),
                projections=np.random.randn(64).astype(np.float32),
                cone_depth=float(i) / 20.0,
                cone_width=0.5 + i * 0.01,
                epistemic_uncertainty=0.1 + i * 0.005,
                expert_weights=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
            )
        )

    return GNNInferenceResult(
        structure_id="4obe",
        model_version="GOSPConeMapper-v3",
        checkpoint_path="s3://checkpoints/robust_experts.pt",
        nodes=nodes,
        embedding_dim=64,
        space_type="euclidean",
    )


@pytest.fixture
def phase3_result() -> PhaseResult:
    """Simulate Phase 3 persistence output."""
    return PhaseResult(
        phase_name="phase3_persistence",
        structure_id="4obe",
        model_version="DTIE-v4-phase3",
        success=True,
        outputs={
            "barcodes": [
                {"birth": 0.0, "death": 1.5, "dimension": 0},
                {"birth": 0.2, "death": 0.8, "dimension": 1},
            ],
            "max_alpha": 3.0,
            "n_witnesses": 500,
            "n_landmarks": 50,
            "hyperbolic_distances_used": True,
            "curvature_c": 1.0,
        },
        residue_contributions={
            "4obe:A:12": 0.92,
            "4obe:A:13": 0.75,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGNNOutputAdapter:
    @pytest.mark.asyncio
    async def test_v4_produces_two_payloads(
        self, normalizer: Normalizer, mock_db: MockDB, v4_gnn_result: GNNInferenceResult
    ):
        """V4 adapter should produce both hyperbolic and Euclidean payloads."""
        adapter = GNNOutputAdapter(normalizer=normalizer)
        results = await adapter.normalize(v4_gnn_result, run_id="test_v4")

        assert len(results) == 2
        assert all(r.success for r in results)

        # Should have 10 total embeddings (5 hyp + 5 euc)
        assert len(mock_db.tables.get("fact_gnn_node_embedding", [])) == 10

        # Should have 2 embedding spaces registered
        assert len(mock_db.tables.get("embedding_space", [])) == 2

    @pytest.mark.asyncio
    async def test_v3_produces_one_payload(
        self, normalizer: Normalizer, mock_db: MockDB, v3_gnn_result: GNNInferenceResult
    ):
        """V3 adapter should produce only Euclidean payload."""
        adapter = GNNOutputAdapter(normalizer=normalizer)
        results = await adapter.normalize(v3_gnn_result, run_id="test_v3")

        assert len(results) == 1
        assert results[0].success

        # 5 embeddings (Euclidean only)
        assert len(mock_db.tables.get("fact_gnn_node_embedding", [])) == 5

        # 1 embedding space
        assert len(mock_db.tables.get("embedding_space", [])) == 1

    @pytest.mark.asyncio
    async def test_residue_ids_are_canonical(
        self, normalizer: Normalizer, mock_db: MockDB, v3_gnn_result: GNNInferenceResult
    ):
        """Adapter should generate canonical residue_ids."""
        adapter = GNNOutputAdapter(normalizer=normalizer)
        await adapter.normalize(v3_gnn_result, run_id="test_keys")

        embeddings = mock_db.tables["fact_gnn_node_embedding"]
        residue_ids = [e["residue_id"] for e in embeddings]

        # All should follow canonical format
        assert "4obe:A:10" in residue_ids
        assert "4obe:A:14" in residue_ids

    @pytest.mark.asyncio
    async def test_auto_generates_run_id(
        self, normalizer: Normalizer, mock_db: MockDB, v3_gnn_result: GNNInferenceResult
    ):
        """If no run_id provided, adapter auto-generates one."""
        adapter = GNNOutputAdapter(normalizer=normalizer)
        results = await adapter.normalize(v3_gnn_result)

        assert results[0].run_id.startswith("run_")

    @pytest.mark.asyncio
    async def test_numpy_arrays_converted(
        self, normalizer: Normalizer, mock_db: MockDB, v4_gnn_result: GNNInferenceResult
    ):
        """Numpy arrays should be converted to lists for serialization."""
        adapter = GNNOutputAdapter(normalizer=normalizer)
        await adapter.normalize(v4_gnn_result, run_id="test_numpy")

        embeddings = mock_db.tables["fact_gnn_node_embedding"]
        # The embedding field should be a list, not numpy array
        first = embeddings[0]
        assert isinstance(first["embedding"], list)


class TestPhase3Adapter:
    @pytest.mark.asyncio
    async def test_phase3_normalization(
        self, normalizer: Normalizer, mock_db: MockDB, phase3_result: PhaseResult
    ):
        """Phase 3 adapter should produce governed persistence data."""
        adapter = Phase3Adapter(normalizer=normalizer)
        result = await adapter.normalize(phase3_result, run_id="test_phase3")

        assert result.success
        # 1 structure-level + 2 residue contributions = 3
        assert result.assets_created == 3

    @pytest.mark.asyncio
    async def test_phase3_residue_contributions(
        self, normalizer: Normalizer, mock_db: MockDB, phase3_result: PhaseResult
    ):
        """Per-residue contributions should be written."""
        adapter = Phase3Adapter(normalizer=normalizer)
        await adapter.normalize(phase3_result, run_id="test_contribs")

        persistence_records = mock_db.tables.get("fact_phase3_persistence", [])
        # 1 structure-level + 2 residue-level
        assert len(persistence_records) == 3

    @pytest.mark.asyncio
    async def test_phase3_provenance_chain(
        self, normalizer: Normalizer, mock_db: MockDB, phase3_result: PhaseResult
    ):
        """Phase 3 should link to parent run for provenance."""
        adapter = Phase3Adapter(normalizer=normalizer)
        await adapter.normalize(
            phase3_result, run_id="test_chain", parent_run_id="parent_gnn_run"
        )

        prov_records = mock_db.tables.get("provenance_run", [])
        assert len(prov_records) == 1
        assert prov_records[0]["parent_run_id"] == "parent_gnn_run"
