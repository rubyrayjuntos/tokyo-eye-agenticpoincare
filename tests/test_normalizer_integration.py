"""Integration tests for the Normalizer against a real database.

These tests require a running PostgreSQL instance with migrations applied.
Set TEST_DATABASE_URL to enable them. They are marked with @pytest.mark.integration
and skipped automatically if no DB is available.

Run: pytest tests/test_normalizer_integration.py -m integration -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from science.dtie.common.keys import make_residue_id, make_structure_id


@pytest.mark.integration
class TestNormalizerIntegration:
    """Test the Normalizer write path against a real database."""

    @pytest.fixture
    def structure_id(self) -> str:
        return make_structure_id(pdb_id="TEST1", source="rcsb")

    @pytest.fixture
    def run_id(self) -> str:
        return f"test_run_{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_normalize_gnn_output_writes_and_reads_back(
        self, integration_db_with_schema, structure_id, run_id
    ):
        """Full round-trip: write GNN output via Normalizer, read it back."""
        from data.normalizer.core import Normalizer
        from science.dtie.common.normalizer_payloads import (
            GNNNodeResult,
            GNNOutputPayload,
            ProvenanceContext,
            RunType,
            SourceType,
            SpaceType,
        )

        db = integration_db_with_schema
        normalizer = Normalizer(db=db, caller_identity="integration_test")

        # Build a minimal payload
        residue_id = make_residue_id(structure_id, "A", 12)
        payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id=run_id,
                structure_id=structure_id,
                model_version="test-v1",
                pipeline_name="test_pipeline",
                run_type=RunType.INFERENCE,
                source_type=SourceType.PROBABILISTIC,
            ),
            space_type=SpaceType.HYPERBOLIC,
            space_name="test_hyp32",
            dimensionality=32,
            curvature=1.0,
            nodes=[
                GNNNodeResult(
                    residue_id=residue_id,
                    residue_index=12,
                    chain_label="A",
                    input_rho=0.5,
                    input_tau_flag=1.0,
                    input_ss_type=2.0,
                    input_sasa=0.8,
                    embedding=[0.1] * 32,
                    cone_depth=2.5,
                    cone_width=0.3,
                    epistemic_uncertainty=0.4,
                    aleatoric_uncertainty=0.1,
                    total_uncertainty=0.5,
                ),
            ],
        )

        # Write
        result = await normalizer.normalize_gnn_output(payload)
        assert result.success is True
        assert result.assets_created == 1
        assert result.run_id == run_id

        # Read back
        row = await db.fetch_one(
            "SELECT * FROM fact_gnn_node_embedding WHERE run_id = :run_id AND residue_id = :residue_id",
            {"run_id": run_id, "residue_id": residue_id},
        )
        assert row is not None
        assert row["cone_depth"] == pytest.approx(2.5)
        assert row["epistemic_uncertainty"] == pytest.approx(0.4)
        assert row["model_version"] == "test-v1"

    @pytest.mark.asyncio
    async def test_normalize_gnn_output_is_idempotent(
        self, integration_db_with_schema, structure_id, run_id
    ):
        """Re-processing the same run updates data without duplicating rows."""
        from data.normalizer.core import Normalizer
        from science.dtie.common.normalizer_payloads import (
            GNNNodeResult,
            GNNOutputPayload,
            ProvenanceContext,
            RunType,
            SourceType,
            SpaceType,
        )

        db = integration_db_with_schema
        normalizer = Normalizer(db=db, caller_identity="integration_test")

        residue_id = make_residue_id(structure_id, "A", 15)
        payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id=run_id,
                structure_id=structure_id,
                model_version="test-v1",
                pipeline_name="test_pipeline",
                run_type=RunType.INFERENCE,
                source_type=SourceType.PROBABILISTIC,
            ),
            space_type=SpaceType.HYPERBOLIC,
            space_name="test_hyp32",
            dimensionality=32,
            curvature=1.0,
            nodes=[
                GNNNodeResult(
                    residue_id=residue_id,
                    residue_index=15,
                    chain_label="A",
                    input_rho=0.5,
                    input_tau_flag=1.0,
                    input_ss_type=2.0,
                    input_sasa=0.8,
                    embedding=[0.1] * 32,
                    cone_depth=1.0,
                    cone_width=0.2,
                    epistemic_uncertainty=0.3,
                ),
            ],
        )

        # Write twice
        result1 = await normalizer.normalize_gnn_output(payload)
        assert result1.success is True

        # Update cone_depth and re-write
        payload.nodes[0].cone_depth = 2.0
        result2 = await normalizer.normalize_gnn_output(payload)
        assert result2.success is True

        # Should have exactly one row (upserted, not duplicated)
        rows = await db.fetch_all(
            "SELECT cone_depth FROM fact_gnn_node_embedding WHERE run_id = :run_id AND residue_id = :residue_id",
            {"run_id": run_id, "residue_id": residue_id},
        )
        assert len(rows) == 1
        assert rows[0]["cone_depth"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_provenance_run_created(
        self, integration_db_with_schema, structure_id, run_id
    ):
        """Normalizer creates a provenance_run record."""
        from data.normalizer.core import Normalizer
        from science.dtie.common.normalizer_payloads import (
            GNNNodeResult,
            GNNOutputPayload,
            ProvenanceContext,
            RunType,
            SourceType,
            SpaceType,
        )

        db = integration_db_with_schema
        normalizer = Normalizer(db=db, caller_identity="integration_test")

        residue_id = make_residue_id(structure_id, "A", 20)
        payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id=run_id,
                structure_id=structure_id,
                model_version="test-v1",
                pipeline_name="test_pipeline",
                run_type=RunType.INFERENCE,
                source_type=SourceType.PROBABILISTIC,
                code_version="abc123",
            ),
            space_type=SpaceType.EUCLIDEAN,
            space_name="test_euc64",
            dimensionality=64,
            nodes=[
                GNNNodeResult(
                    residue_id=residue_id,
                    residue_index=20,
                    chain_label="A",
                    input_rho=0.5,
                    input_tau_flag=1.0,
                    input_ss_type=2.0,
                    input_sasa=0.8,
                    embedding=[0.1] * 64,
                    cone_depth=1.5,
                ),
            ],
        )

        await normalizer.normalize_gnn_output(payload)

        # Verify provenance record
        prov = await db.fetch_one(
            "SELECT * FROM provenance_run WHERE run_id = :run_id",
            {"run_id": run_id},
        )
        assert prov is not None
        assert prov["model_version"] == "test-v1"
        assert prov["code_version"] == "abc123"
        assert prov["pipeline_name"] == "test_pipeline"

    @pytest.mark.asyncio
    async def test_audit_trail_logged(
        self, integration_db_with_schema, structure_id, run_id
    ):
        """Normalizer logs an audit record on success."""
        from data.normalizer.core import Normalizer
        from science.dtie.common.normalizer_payloads import (
            GNNNodeResult,
            GNNOutputPayload,
            ProvenanceContext,
            RunType,
            SourceType,
            SpaceType,
        )

        db = integration_db_with_schema
        normalizer = Normalizer(db=db, caller_identity="audit_test")

        residue_id = make_residue_id(structure_id, "A", 25)
        payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id=run_id,
                structure_id=structure_id,
                model_version="test-v1",
                pipeline_name="test_pipeline",
                run_type=RunType.INFERENCE,
                source_type=SourceType.PROBABILISTIC,
            ),
            space_type=SpaceType.HYPERBOLIC,
            space_name="test_hyp32",
            dimensionality=32,
            curvature=1.0,
            nodes=[
                GNNNodeResult(
                    residue_id=residue_id,
                    residue_index=25,
                    chain_label="A",
                    input_rho=0.5,
                    input_tau_flag=1.0,
                    input_ss_type=2.0,
                    input_sasa=0.8,
                    embedding=[0.1] * 32,
                    cone_depth=1.0,
                ),
            ],
        )

        await normalizer.normalize_gnn_output(payload)

        audit = await db.fetch_one(
            "SELECT * FROM normalization_audit WHERE run_id = :run_id AND status = 'success'",
            {"run_id": run_id},
        )
        assert audit is not None
        assert audit["caller_identity"] == "audit_test"
        assert audit["assets_created"] == 1
