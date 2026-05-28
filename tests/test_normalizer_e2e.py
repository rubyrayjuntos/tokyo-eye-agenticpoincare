"""End-to-end test: Ingest mock GNN output through the Normalizer.

This test validates the complete write path from science code output
through the Normalizer to the governed data layer, using an in-memory
mock database.

Priority 5: CI validation that the schema, payloads, and Normalizer
work together correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from data.normalizer.core import Normalizer, NormalizerError
from science.dtie.common.keys import make_residue_id, make_structure_id
from science.dtie.common.normalizer_payloads import (
    GNNNodeResult,
    GNNOutputPayload,
    PersistenceBarcode,
    Phase3PersistencePayload,
    Phase3ResidueContribution,
    ProvenanceContext,
    RunType,
    SourceType,
    SpaceType,
)


# ---------------------------------------------------------------------------
# Mock database for testing (no real PostgreSQL needed)
# ---------------------------------------------------------------------------


class MockDatabase:
    """In-memory mock that tracks all writes for assertion."""

    def __init__(self):
        self.tables: dict[str, list[dict[str, Any]]] = {
            "provenance_run": [],
            "embedding_space": [],
            "fact_gnn_node_embedding": [],
            "fact_phase3_persistence": [],
            "governed_asset": [],
        }
        self._in_transaction = False

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        table = self._extract_table(query)
        if table:
            # Simulate ON CONFLICT DO NOTHING for provenance_run
            if table == "provenance_run" and "ON CONFLICT" in query and "DO NOTHING" in query:
                for row in self.tables.get("provenance_run", []):
                    if row.get("run_id") == params.get("run_id"):
                        return  # Skip duplicate
            self.tables.setdefault(table, []).append(params)

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        table = self._extract_table(query)
        if table:
            for params in params_list:
                self.tables.setdefault(table, []).append(params)

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        # Simple lookup simulation
        if "provenance_run" in query and "run_id" in params:
            for row in self.tables.get("provenance_run", []):
                if row.get("run_id") == params["run_id"]:
                    return row
        if "embedding_space" in query and "name" in params:
            for row in self.tables.get("embedding_space", []):
                if row.get("name") == params["name"]:
                    return row
        return None

    async def begin(self) -> None:
        self._in_transaction = True

    async def commit(self) -> None:
        self._in_transaction = False

    async def rollback(self) -> None:
        self._in_transaction = False

    def _extract_table(self, query: str) -> str | None:
        """Extract table name from INSERT INTO statement."""
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
def mock_db() -> MockDatabase:
    return MockDatabase()


@pytest.fixture
def normalizer(mock_db: MockDatabase) -> Normalizer:
    return Normalizer(db=mock_db)  # type: ignore[arg-type]


@pytest.fixture
def structure_id() -> str:
    return make_structure_id(pdb_id="4OBE", source="rcsb")


@pytest.fixture
def v4_gnn_payload(structure_id: str) -> GNNOutputPayload:
    """Simulate realistic v4 GNN output for a small protein (5 residues)."""
    nodes = []
    for i in range(10, 15):
        residue_id = make_residue_id(structure_id, "A", i)
        nodes.append(
            GNNNodeResult(
                residue_id=residue_id,
                residue_index=i,
                chain_label="A",
                input_rho=0.5 + i * 0.02,
                input_tau_flag=1.0,
                input_ss_type=float(i % 3),
                input_sasa=40.0 + i * 0.5,
                embedding=[0.01 * i] * 32,  # 32-dim hyperbolic
                cone_depth=float(i) / 20.0,
                cone_width=0.5 + i * 0.01,
                epistemic_uncertainty=0.1 + i * 0.005,
                aleatoric_uncertainty=0.05 + i * 0.003,
                total_uncertainty=0.15 + i * 0.008,
                x_hyp=[0.02 * i] * 32,
                hyp_projections=[0.1 * (i - 12), 0.1 * (i - 11)],
                x_routed_hyp=[0.015 * i] * 32,
                expert_weights=[0.4, 0.3, 0.2, 0.1],
            )
        )

    return GNNOutputPayload(
        provenance=ProvenanceContext(
            run_id="run_v4_test_001",
            structure_id=structure_id,
            model_version="GOSPConeMapper-v4",
            pipeline_name="dtie_v4",
            run_type=RunType.INFERENCE,
            source_type=SourceType.PROBABILISTIC,
            checkpoint_uri="s3://checkpoints/tokyo_eyes_v4.pt",
            checkpoint_sha256="deadbeef" * 8,
            code_version="abc123",
        ),
        space_type=SpaceType.HYPERBOLIC,
        space_name="gospcone_v4_hyp32",
        dimensionality=32,
        curvature=1.0,
        nodes=nodes,
    )


@pytest.fixture
def v3_gnn_payload(structure_id: str) -> GNNOutputPayload:
    """Simulate v3 GNN output (Euclidean only, no hyperbolic fields)."""
    nodes = []
    for i in range(10, 15):
        residue_id = make_residue_id(structure_id, "A", i)
        nodes.append(
            GNNNodeResult(
                residue_id=residue_id,
                residue_index=i,
                chain_label="A",
                input_rho=0.5 + i * 0.02,
                input_tau_flag=1.0,
                input_ss_type=float(i % 3),
                input_sasa=40.0 + i * 0.5,
                embedding=[0.01 * i] * 64,  # 64-dim Euclidean
                cone_depth=float(i) / 20.0,
                cone_width=0.5 + i * 0.01,
                epistemic_uncertainty=0.1 + i * 0.005,
                # No aleatoric, no hyp fields (v3)
            )
        )

    return GNNOutputPayload(
        provenance=ProvenanceContext(
            run_id="run_v3_test_001",
            structure_id=structure_id,
            model_version="GOSPConeMapper-v3",
            pipeline_name="dtie_v3",
            run_type=RunType.INFERENCE,
            source_type=SourceType.PROBABILISTIC,
            checkpoint_uri="s3://checkpoints/robust_experts.pt",
            code_version="def456",
        ),
        space_type=SpaceType.EUCLIDEAN,
        space_name="gospcone_v3_euc64",
        dimensionality=64,
        nodes=nodes,
    )


@pytest.fixture
def phase3_payload(structure_id: str) -> Phase3PersistencePayload:
    """Simulate Phase 3 persistence output."""
    return Phase3PersistencePayload(
        provenance=ProvenanceContext(
            run_id="run_phase3_001",
            structure_id=structure_id,
            model_version="DTIE-v4-phase3",
            pipeline_name="dtie_v4",
            run_type=RunType.INFERENCE,
            source_type=SourceType.DETERMINISTIC,
            code_version="abc123",
        ),
        barcodes=[
            PersistenceBarcode(birth=0.0, death=1.5, dimension=0),
            PersistenceBarcode(birth=0.1, death=0.9, dimension=1),
            PersistenceBarcode(
                birth=0.0,
                death=2.3,
                dimension=0,
                generator_residues=[
                    make_residue_id(structure_id, "A", 12),
                    make_residue_id(structure_id, "A", 13),
                ],
            ),
        ],
        max_alpha=3.0,
        n_witnesses=500,
        n_landmarks=50,
        hyperbolic_distances_used=True,
        curvature_c=1.0,
        residue_contributions=[
            Phase3ResidueContribution(
                residue_id=make_residue_id(structure_id, "A", 12),
                persistence_score=0.92,
                max_barcode_length=2.3,
                topological_significance=0.88,
            ),
            Phase3ResidueContribution(
                residue_id=make_residue_id(structure_id, "A", 13),
                persistence_score=0.75,
                max_barcode_length=1.5,
            ),
        ],
        landmark_to_residue={
            "0": make_residue_id(structure_id, "A", 12),
            "1": make_residue_id(structure_id, "A", 13),
        },
    )


# ---------------------------------------------------------------------------
# End-to-End Tests
# ---------------------------------------------------------------------------


class TestNormalizerE2E_GNNOutput:
    """End-to-end: GNN output → Normalizer → governed data."""

    @pytest.mark.asyncio
    async def test_v4_gnn_full_flow(
        self, normalizer: Normalizer, mock_db: MockDatabase, v4_gnn_payload: GNNOutputPayload
    ):
        result = await normalizer.normalize_gnn_output(v4_gnn_payload)

        assert result.success is True
        assert result.run_id == "run_v4_test_001"
        assert result.assets_created == 5  # 5 residues

        # Provenance was created
        assert len(mock_db.tables["provenance_run"]) == 1
        prov_record = mock_db.tables["provenance_run"][0]
        assert prov_record["model_version"] == "GOSPConeMapper-v4"
        assert prov_record["run_type"] == "inference"

        # Embedding space was registered
        assert len(mock_db.tables["embedding_space"]) == 1
        space_record = mock_db.tables["embedding_space"][0]
        assert space_record["space_type"] == "hyperbolic"
        assert space_record["curvature"] == 1.0

        # All 5 nodes were written
        assert len(mock_db.tables["fact_gnn_node_embedding"]) == 5

        # Governed assets were registered
        assert len(mock_db.tables["governed_asset"]) == 5

    @pytest.mark.asyncio
    async def test_v3_gnn_full_flow(
        self, normalizer: Normalizer, mock_db: MockDatabase, v3_gnn_payload: GNNOutputPayload
    ):
        result = await normalizer.normalize_gnn_output(v3_gnn_payload)

        assert result.success is True
        assert result.assets_created == 5

        # Euclidean space registered
        space_record = mock_db.tables["embedding_space"][0]
        assert space_record["space_type"] == "euclidean"
        assert space_record["curvature"] is None

    @pytest.mark.asyncio
    async def test_idempotent_provenance(
        self, normalizer: Normalizer, mock_db: MockDatabase, v4_gnn_payload: GNNOutputPayload
    ):
        """Running the same payload twice should not duplicate provenance."""
        await normalizer.normalize_gnn_output(v4_gnn_payload)
        await normalizer.normalize_gnn_output(v4_gnn_payload)

        # Provenance should only be created once
        assert len(mock_db.tables["provenance_run"]) == 1

    @pytest.mark.asyncio
    async def test_invalid_residue_id_rejected(
        self, normalizer: Normalizer, structure_id: str
    ):
        """Normalizer rejects payloads with non-canonical residue_ids."""
        bad_node = GNNNodeResult(
            residue_id="NOT_A_VALID_ID",  # Bad format
            residue_index=12,
            chain_label="A",
            input_rho=0.5,
            input_tau_flag=1.0,
            input_ss_type=2.0,
            input_sasa=40.0,
            embedding=[0.1] * 32,
        )
        payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id="run_bad",
                structure_id=structure_id,
                model_version="v4",
                pipeline_name="test",
                source_type=SourceType.PROBABILISTIC,
            ),
            space_type=SpaceType.EUCLIDEAN,
            space_name="test",
            dimensionality=32,
            nodes=[bad_node],
        )

        with pytest.raises(NormalizerError, match="Invalid residue_id"):
            await normalizer.normalize_gnn_output(payload)


class TestNormalizerE2E_Phase3:
    """End-to-end: Phase 3 persistence → Normalizer → governed data."""

    @pytest.mark.asyncio
    async def test_phase3_full_flow(
        self,
        normalizer: Normalizer,
        mock_db: MockDatabase,
        phase3_payload: Phase3PersistencePayload,
    ):
        result = await normalizer.normalize_phase3_output(phase3_payload)

        assert result.success is True
        assert result.run_id == "run_phase3_001"
        # 1 structure-level + 2 residue contributions = 3 assets
        assert result.assets_created == 3

        # Provenance created
        assert len(mock_db.tables["provenance_run"]) == 1

        # Phase 3 records written (1 structure + 2 residue)
        assert len(mock_db.tables["fact_phase3_persistence"]) == 3

        # Governed assets registered
        assert len(mock_db.tables["governed_asset"]) == 3


class TestNormalizerE2E_CrossLineage:
    """Validate that v3 and v4 outputs coexist cleanly."""

    @pytest.mark.asyncio
    async def test_v3_and_v4_same_structure(
        self,
        normalizer: Normalizer,
        mock_db: MockDatabase,
        v3_gnn_payload: GNNOutputPayload,
        v4_gnn_payload: GNNOutputPayload,
    ):
        """Both v3 and v4 can write to the same structure without conflict."""
        result_v3 = await normalizer.normalize_gnn_output(v3_gnn_payload)
        result_v4 = await normalizer.normalize_gnn_output(v4_gnn_payload)

        assert result_v3.success is True
        assert result_v4.success is True

        # 10 total embeddings (5 v3 + 5 v4)
        assert len(mock_db.tables["fact_gnn_node_embedding"]) == 10

        # 2 provenance runs (different run_ids)
        assert len(mock_db.tables["provenance_run"]) == 2

        # 2 embedding spaces (Euclidean + Hyperbolic)
        assert len(mock_db.tables["embedding_space"]) == 2

        # All 10 governed assets registered
        assert len(mock_db.tables["governed_asset"]) == 10
