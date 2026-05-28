"""Tests for Normalizer audit trail logging."""

from __future__ import annotations

from typing import Any

import pytest

from data.normalizer.core import Normalizer, NormalizerError
from science.dtie.common.keys import make_residue_id
from science.dtie.common.normalizer_payloads import (
    GNNNodeResult,
    GNNOutputPayload,
    ProvenanceContext,
    RunType,
    SourceType,
    SpaceType,
)


class AuditTrackingDB:
    """Mock DB that tracks audit log writes separately."""

    def __init__(self):
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.audit_logs: list[dict[str, Any]] = []
        self._in_transaction = False

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        table = self._extract_table(query)
        if table == "normalization_audit":
            self.audit_logs.append(params)
        elif table:
            self.tables.setdefault(table, []).append(params)

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        table = self._extract_table(query)
        if table:
            for params in params_list:
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


@pytest.fixture
def audit_db() -> AuditTrackingDB:
    return AuditTrackingDB()


@pytest.fixture
def normalizer(audit_db: AuditTrackingDB) -> Normalizer:
    return Normalizer(db=audit_db, caller_identity="test_suite")  # type: ignore[arg-type]


@pytest.fixture
def valid_payload() -> GNNOutputPayload:
    structure_id = "4obe"
    return GNNOutputPayload(
        provenance=ProvenanceContext(
            run_id="run_audit_test",
            structure_id=structure_id,
            model_version="GOSPConeMapper-v4",
            pipeline_name="dtie_v4",
            source_type=SourceType.PROBABILISTIC,
        ),
        space_type=SpaceType.HYPERBOLIC,
        space_name="test_hyp",
        dimensionality=32,
        curvature=1.0,
        nodes=[
            GNNNodeResult(
                residue_id=make_residue_id(structure_id, "A", 12),
                residue_index=12,
                chain_label="A",
                input_rho=0.5,
                input_tau_flag=1.0,
                input_ss_type=2.0,
                input_sasa=40.0,
                embedding=[0.1] * 32,
                cone_depth=1.5,
                cone_width=0.8,
            ),
        ],
    )


class TestNormalizerAudit:
    @pytest.mark.asyncio
    async def test_success_creates_audit_record(
        self, normalizer: Normalizer, audit_db: AuditTrackingDB, valid_payload: GNNOutputPayload
    ):
        await normalizer.normalize_gnn_output(valid_payload)

        assert len(audit_db.audit_logs) == 1
        audit = audit_db.audit_logs[0]
        assert audit["status"] == "success"
        assert audit["run_id"] == "run_audit_test"
        assert audit["payload_type"] == "gnn_output"
        assert audit["assets_created"] == 1
        assert audit["caller_identity"] == "test_suite"
        assert audit["duration_ms"] is not None
        assert audit["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_validation_error_creates_audit_record(
        self, normalizer: Normalizer, audit_db: AuditTrackingDB
    ):
        bad_payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id="run_bad",
                structure_id="4obe",
                model_version="v4",
                pipeline_name="test",
                source_type=SourceType.PROBABILISTIC,
            ),
            space_type=SpaceType.EUCLIDEAN,
            space_name="test",
            dimensionality=32,
            nodes=[
                GNNNodeResult(
                    residue_id="INVALID_FORMAT",
                    residue_index=1,
                    chain_label="A",
                    input_rho=0.5,
                    input_tau_flag=1.0,
                    input_ss_type=2.0,
                    input_sasa=40.0,
                    embedding=[0.1] * 32,
                ),
            ],
        )

        with pytest.raises(NormalizerError):
            await normalizer.normalize_gnn_output(bad_payload)

        assert len(audit_db.audit_logs) == 1
        audit = audit_db.audit_logs[0]
        assert audit["status"] == "validation_error"
        assert "INVALID_FORMAT" in audit["error_message"]

    @pytest.mark.asyncio
    async def test_audit_includes_payload_summary(
        self, normalizer: Normalizer, audit_db: AuditTrackingDB, valid_payload: GNNOutputPayload
    ):
        await normalizer.normalize_gnn_output(valid_payload)

        audit = audit_db.audit_logs[0]
        summary = audit["payload_summary"]
        assert summary["node_count"] == 1
        assert summary["space"] == "test_hyp"
        assert summary["space_type"] == "hyperbolic"
        assert summary["dimensionality"] == 32
        assert summary["model_version"] == "GOSPConeMapper-v4"

    @pytest.mark.asyncio
    async def test_caller_identity_propagated(
        self, audit_db: AuditTrackingDB, valid_payload: GNNOutputPayload
    ):
        normalizer = Normalizer(db=audit_db, caller_identity="agent_coordinator")  # type: ignore[arg-type]
        await normalizer.normalize_gnn_output(valid_payload)

        audit = audit_db.audit_logs[0]
        assert audit["caller_identity"] == "agent_coordinator"
