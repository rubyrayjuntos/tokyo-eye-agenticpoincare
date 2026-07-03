"""Tests for the master onboard contract."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from science.compute.job_schema import load_job_schema, validate_schema_against_registry
from science.compute.registry import JOB_REGISTRY
from science.compute.runner_dispatch import validate_runners_against_registry
from science.contracts.onboard_contract import (
    get_artifact_geometric_space,
    get_hyperbolic_jobs,
    get_job_geometric_catalog,
    get_required_geometric_space,
    get_tier_artifact_keys,
    load_contract,
    requires_hyperbolic_readiness,
    validate_contract_against_registry,
    validate_geometric_contract,
    validate_normalizer_destinations,
)
from science.contracts.geometric_runtime import (
    apply_enforcement,
    check_onboard_hyperbolic_prerequisites,
    effective_enforcement_level,
    validate_learned_curvature_in_result,
)
from science.contracts.validation import (
    validate_job_geometric_constraints,
    validate_job_run_result,
)
from science.compute.runners.base import JobRunResult


class TestOnboardContract:
    def test_contract_loads(self) -> None:
        contract = load_contract()
        assert contract["version"] == "1.6"
        assert contract["pipeline"] == "onboard_v1"
        assert "artifacts" in contract
        assert "readiness" in contract
        assert "geometric_constraints" in contract

    def test_contract_matches_registry(self) -> None:
        assert validate_contract_against_registry() == []

    def test_registry_produces_subset_of_artifacts(self) -> None:
        from science.contracts.onboard_contract import get_artifact_catalog, get_artifact_aliases

        catalog = get_artifact_catalog()
        known = {spec["canonical_key"] for spec in catalog.values()} | set(
            get_artifact_aliases().keys()
        )
        produced: set[str] = set()
        for job in JOB_REGISTRY.values():
            produced |= set(job.produces)
        assert produced <= known

    def test_tier_artifacts_have_probes(self) -> None:
        from science.contracts.onboard_contract import get_probe_id

        for tier in (1, 2):
            for artifact in get_tier_artifact_keys(tier):
                assert get_probe_id(artifact), f"missing probe for tier artifact {artifact}"

    def test_job_schema_normalizer_destinations(self) -> None:
        schema = load_job_schema()
        errors = validate_normalizer_destinations(schema)
        assert errors == [], f"destination errors: {errors}"

    def test_job_schema_still_matches_registry(self) -> None:
        assert validate_schema_against_registry() == []

    def test_runner_registry_alignment(self) -> None:
        assert validate_runners_against_registry() == []

    def test_hyperbolic_jobs_authoritative(self) -> None:
        """Every job in geometric_constraints.hyperbolic_jobs must declare requires_hyperbolic."""
        catalog = get_job_geometric_catalog()
        for job_id in get_hyperbolic_jobs():
            assert job_id in catalog, f"{job_id} missing from jobs section"
            assert catalog[job_id].get("requires_hyperbolic") is True, (
                f"{job_id} in hyperbolic_jobs but requires_hyperbolic is not true"
            )
            assert catalog[job_id].get("geometric_space") == "hyperbolic"

    def test_geometric_contract_valid(self) -> None:
        assert validate_geometric_contract() == []

    def test_registry_job_keys_match_contract(self) -> None:
        from science.contracts.onboard_contract import validate_registry_job_keys_match_contract

        assert validate_registry_job_keys_match_contract() == []

    def test_all_artifacts_declare_geometric_space(self) -> None:
        from science.contracts.onboard_contract import get_artifact_catalog

        for key, spec in get_artifact_catalog().items():
            assert spec.get("geometric_space") is not None, (
                f"artifact {key!r} must declare geometric_space"
            )

    def test_hyperbolic_jobs_declared(self) -> None:
        assert "gnn_inference" in get_hyperbolic_jobs()
        assert get_required_geometric_space("gnn_inference") == "hyperbolic"

    def test_gnn_hyp_artifact_is_hyperbolic(self) -> None:
        assert get_artifact_geometric_space("gnn_hyp") == "hyperbolic"
        assert get_artifact_geometric_space("scope") == "none"

    def test_readiness_requires_hyperbolic(self) -> None:
        assert requires_hyperbolic_readiness() is True

    def test_hyperbolic_job_produces_hyperbolic_artifacts(self) -> None:
        from science.compute.registry import JOB_REGISTRY

        for job_id in get_hyperbolic_jobs():
            for artifact in JOB_REGISTRY[job_id].produces:
                if artifact == "gnn_euc":
                    continue
                space = get_artifact_geometric_space(artifact)
                assert space in ("hyperbolic", "mixed"), (
                    f"{job_id} -> {artifact} has space {space}"
                )


class TestGeometricValidation:
    def test_gnn_job_passes_geometric_constraints(self) -> None:
        assert validate_job_geometric_constraints("gnn_inference") == []

    def test_hyperbolic_result_validates(self) -> None:
        result = JobRunResult(
            job_id="gnn_inference",
            run_id="run_test",
            structure_id="kras",
            success=True,
            artifacts_produced=["gnn_hyp", "gnn_euc"],
            outputs={"curvature": 0.87},
        )
        assert validate_job_run_result(result) == []

    def test_gnn_missing_curvature_warns(self) -> None:
        result = JobRunResult(
            job_id="gnn_inference",
            run_id="run_test",
            structure_id="kras",
            success=True,
            artifacts_produced=["gnn_hyp", "gnn_euc"],
            outputs={},
        )
        messages = validate_job_run_result(result)
        assert any("curvature" in m for m in messages)

    def test_downstream_requires_learned_curvature_passthrough(self) -> None:
        messages = validate_learned_curvature_in_result(
            "source_leak_detection",
            success=True,
            outputs={},
            learned_curvature=None,
        )
        assert messages
        assert "passthrough" in messages[0]

    def test_downstream_curvature_mismatch_warns(self) -> None:
        messages = validate_learned_curvature_in_result(
            "source_leak_detection",
            success=True,
            outputs={"curvature": 1.5},
            learned_curvature=0.87,
        )
        assert messages
        assert "!=" in messages[0]

    def test_enforcement_error_jobs_env(self) -> None:
        with patch.dict(
            os.environ,
            {"GEOMETRIC_ENFORCEMENT_ERROR_JOBS": "gnn_inference,source_leak_detection"},
            clear=False,
        ):
            assert effective_enforcement_level("gnn_inference") == "error"
            assert effective_enforcement_level("graph_topology") == "warning"

    def test_apply_enforcement_fails_job_on_error_level(self) -> None:
        with patch.dict(os.environ, {"GEOMETRIC_ENFORCEMENT_LEVEL": "error"}, clear=False):
            warnings, fail = apply_enforcement(
                "source_leak_detection",
                ["hyperbolic job ran without learned curvature passthrough"],
            )
            assert warnings
            assert fail is True

    def test_apply_enforcement_warning_only_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            warnings, fail = apply_enforcement(
                "source_leak_detection",
                ["hyperbolic job ran without learned curvature passthrough"],
            )
            assert warnings
            assert fail is False


class TestOnboardGeometricPrerequisites:
    @pytest.mark.asyncio
    async def test_ingest_notes_include_hyperbolic_guidance(self) -> None:
        class _DB:
            async def fetch_one(self, query: str, params: dict):
                if "structure_computation_scope" in query:
                    return {"ok": 1}
                return None

        notes = await check_onboard_hyperbolic_prerequisites(
            _DB(),
            "kras",
            residue_count=100,
            primary_chain_ids=["A"],
        )
        assert any("learned" in n.lower() for n in notes)
        assert not any("No residues" in n for n in notes)

    @pytest.mark.asyncio
    async def test_ingest_notes_warn_empty_residues(self) -> None:
        class _DB:
            async def fetch_one(self, query: str, params: dict):
                return {"ok": 1}

        notes = await check_onboard_hyperbolic_prerequisites(
            _DB(),
            "kras",
            residue_count=0,
            primary_chain_ids=["A"],
        )
        assert any("No residues" in n for n in notes)


class TestJobRunValidation:
    def test_unknown_artifact_produces_warning(self) -> None:
        result = JobRunResult(
            job_id="gnn_inference",
            run_id="run_test",
            structure_id="kras",
            success=True,
            artifacts_produced=["not_a_real_artifact"],
        )
        errors = validate_job_run_result(result)
        assert errors
        assert "not_a_real_artifact" in errors[0]

    def test_known_artifact_is_valid(self) -> None:
        result = JobRunResult(
            job_id="gnn_inference",
            run_id="run_test",
            structure_id="kras",
            success=True,
            artifacts_produced=["gnn_hyp", "gnn_euc"],
            outputs={"curvature": 0.87},
        )
        assert validate_job_run_result(result) == []
