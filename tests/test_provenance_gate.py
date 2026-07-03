"""Tests for governed-write provenance enforcement."""

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
from science.dtie.common.provenance_runtime import (
    is_provenance_strict,
    resolve_code_version,
    validate_provenance_fields,
)


class _StubDB:
    def __init__(self) -> None:
        self._provenance_runs: set[str] = set()
        self.execute_calls: list[str] = []

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        self.execute_calls.append(query)
        if "INSERT INTO provenance_run" in query:
            self._provenance_runs.add(str(params.get("run_id", "")))

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        self.execute_calls.append(query)

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "FROM provenance_run" in query and params.get("run_id") in self._provenance_runs:
            return {"ok": 1}
        return None

    async def begin(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


VALID_CHECKPOINT_SHA = "a" * 64


def _inference_payload(*, code_version: str | None, checkpoint_sha256: str | None) -> GNNOutputPayload:
    structure_id = "4uj1"
    return GNNOutputPayload(
        provenance=ProvenanceContext(
            run_id="run_provenance_gate",
            structure_id=structure_id,
            model_version="GOSPConeMapper-v6",
            pipeline_name="dtie_v5",
            run_type=RunType.INFERENCE,
            source_type=SourceType.PROBABILISTIC,
            code_version=code_version,
            checkpoint_sha256=checkpoint_sha256,
        ),
        space_type=SpaceType.HYPERBOLIC,
        space_name="test_hyp32",
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


class TestProvenanceValidation:
    def test_inference_requires_checkpoint_and_code_version_when_strict(self):
        errors = validate_provenance_fields(
            run_type=RunType.INFERENCE,
            code_version=None,
            checkpoint_sha256=None,
            strict=True,
        )
        assert "code_version" in errors[0]
        assert any("checkpoint_sha256" in err for err in errors)

    def test_analysis_requires_code_version_only_when_strict(self):
        errors = validate_provenance_fields(
            run_type=RunType.ANALYSIS,
            code_version=None,
            checkpoint_sha256=None,
            strict=True,
        )
        assert errors == ["code_version is required for governed writes"]

    def test_relaxed_mode_allows_incomplete_provenance(self):
        errors = validate_provenance_fields(
            run_type=RunType.INFERENCE,
            code_version=None,
            checkpoint_sha256=None,
            strict=False,
        )
        assert errors == []

    def test_invalid_checkpoint_format_rejected(self):
        errors = validate_provenance_fields(
            run_type=RunType.INFERENCE,
            code_version="abc123",
            checkpoint_sha256="not-a-valid-hash",
            strict=True,
        )
        assert any("64-character" in err for err in errors)

    def test_provenance_context_validate_for_governed_write(self):
        prov = ProvenanceContext(
            run_id="run_x",
            structure_id="4uj1",
            model_version="v6",
            pipeline_name="test",
            run_type=RunType.INFERENCE,
            code_version="abc",
            checkpoint_sha256=VALID_CHECKPOINT_SHA,
        )
        prov.validate_for_governed_write(strict=True)

    def test_is_provenance_strict_defaults_to_dev_relaxed(self, monkeypatch):
        monkeypatch.delenv("PROVENANCE_STRICT", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "dev")
        assert is_provenance_strict() is False

    def test_is_provenance_strict_true_in_prod(self, monkeypatch):
        monkeypatch.delenv("PROVENANCE_STRICT", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "prod")
        assert is_provenance_strict() is True

    def test_resolve_code_version_prefers_explicit(self):
        assert resolve_code_version("deadbeef") == "deadbeef"


class TestNormalizerProvenanceGate:
    @pytest.mark.asyncio
    async def test_strict_inference_rejected_by_normalizer(self, monkeypatch):
        monkeypatch.setenv("PROVENANCE_STRICT", "true")
        normalizer = Normalizer(db=_StubDB(), caller_identity="test")  # type: ignore[arg-type]
        payload = _inference_payload(code_version=None, checkpoint_sha256=None)

        with pytest.raises(NormalizerError, match="Incomplete provenance"):
            await normalizer.normalize_gnn_output(payload)

    @pytest.mark.asyncio
    async def test_complete_inference_passes_gate(self, monkeypatch):
        monkeypatch.setenv("PROVENANCE_STRICT", "true")
        normalizer = Normalizer(db=_StubDB(), caller_identity="test")  # type: ignore[arg-type]
        payload = _inference_payload(
            code_version="abc123def",
            checkpoint_sha256=VALID_CHECKPOINT_SHA,
        )

        result = await normalizer.normalize_gnn_output(payload)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_dev_mode_allows_incomplete_provenance(self, monkeypatch):
        monkeypatch.setenv("PROVENANCE_STRICT", "false")
        normalizer = Normalizer(db=_StubDB(), caller_identity="test")  # type: ignore[arg-type]
        payload = _inference_payload(code_version=None, checkpoint_sha256=None)

        result = await normalizer.normalize_gnn_output(payload)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_missing_run_id_rejected_before_governed_write(self, monkeypatch):
        monkeypatch.setenv("PROVENANCE_STRICT", "false")
        db = _StubDB()
        normalizer = Normalizer(db=db, caller_identity="test")  # type: ignore[arg-type]
        payload = _inference_payload(code_version="abc", checkpoint_sha256=VALID_CHECKPOINT_SHA)
        payload.provenance.run_id = "   "

        with pytest.raises(NormalizerError, match="provenance_run_id is required"):
            await normalizer.normalize_gnn_output(payload)

        assert not any("fact_gnn_node_embedding" in q for q in db.execute_calls)

    @pytest.mark.asyncio
    async def test_ingest_dimensions_creates_provenance_before_dim_writes(self, monkeypatch):
        from science.dtie.common.ingest_payloads import (
            IngestDimensionPayload,
            StructureDimension,
        )

        monkeypatch.setenv("PROVENANCE_STRICT", "false")
        db = _StubDB()
        normalizer = Normalizer(db=db, caller_identity="test")  # type: ignore[arg-type]
        payload = IngestDimensionPayload(
            provenance=ProvenanceContext(
                run_id="run_ingest_prov_gate",
                structure_id="4uj1",
                model_version="ingest-v1",
                pipeline_name="ingest",
                run_type=RunType.ANALYSIS,
                source_type=SourceType.EMPIRICAL,
            ),
            structure=StructureDimension(
                structure_id="4uj1",
                pdb_id="4UJ1",
                method="X-RAY",
                resolution=1.5,
                title="test",
                polymer_composition="protein",
            ),
            chains=[],
            residues=[],
            atoms=[],
            covalent_bonds=[],
            file_hash="abc123",
            biotite_version="1.0.0",
            rcsbapi_version="2.0.0",
        )

        result = await normalizer.normalize_ingest_dimensions(payload)
        assert result.success is True
        dim_idx = next(
            i for i, q in enumerate(db.execute_calls) if "INSERT INTO dim_structure" in q
        )
        prov_idx = next(
            i for i, q in enumerate(db.execute_calls) if "INSERT INTO provenance_run" in q
        )
        assert dim_idx < prov_idx
