"""Tests for structure readiness probes and status derivation."""

from __future__ import annotations

from typing import Any

import pytest

from data.readiness import (
    TIER1_ARTIFACTS,
    TIER2_ARTIFACTS,
    assess_structure_readiness,
    derive_readiness_status,
    probe_binding_scan,
    probe_dims,
)


class MockReadinessDB:
    def __init__(self, responses: dict[str, Any]):
        self._responses = responses
        self.queries: list[str] = []

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        self.queries.append(query)
        key = self._match_key(query)
        return self._responses.get(key)

    def _match_key(self, query: str) -> str:
        q = query.lower()
        if "from dim_structure" in q and "select 1" in q:
            return "structure_exists"
        if "from dim_residue" in q:
            return "dims"
        if "structure_computation_scope" in q:
            return "scope"
        if "fact_gnn_node_embedding" in q:
            return "gnn_hyp"
        if "fact_graph_node_metrics" in q:
            return "graph"
        if "fact_source_leak" in q:
            return "source_leaks"
        if "from fact_binding_site_scan" in q and "sites_found" in q:
            return "binding_scan_detail"
        if "from fact_binding_site_scan" in q:
            return "binding_scan"
        if "fact_residue_alignment" in q:
            return "alignment"
        if "fact_phase2_vulnerability" in q:
            return "discovery_extended"
        if "fact_hyperbolic_motif" in q:
            return "motifs"
        if "fact_cryptic_site" in q and "md_validation_status" in q:
            return "md_validation"
        if "fact_resistance_pathway" in q:
            return "resistance_pathway"
        if "fact_pharmacophore" in q:
            return "pharmacophores"
        if "fact_drug_candidate" in q and "selectivity_ratio" in q:
            return "allele_selectivity"
        if "fact_drug_candidate" in q:
            return "drug_candidates"
        if "fact_allosteric_site" in q:
            return "allosteric_sites"
        if "fact_phase_output" in q and "buffering" in q:
            return "buffering_atlas"
        if "fact_phase_output" in q:
            return "witness_embedding"
        if "from pipeline_job" in q:
            return "pipeline_job"
        return "unknown"


class TestDeriveReadinessStatus:
    def test_running_when_job_active(self):
        tier1 = dict.fromkeys(TIER1_ARTIFACTS, False)
        tier2 = dict.fromkeys(TIER2_ARTIFACTS, False)
        status = derive_readiness_status(
            tier1,
            tier2,
            {"status": "running", "job_id": "j1"},
        )
        assert status == "running"

    def test_ready_when_all_tiers_complete(self):
        tier1 = dict.fromkeys(TIER1_ARTIFACTS, True)
        tier2 = dict.fromkeys(TIER2_ARTIFACTS, True)
        status = derive_readiness_status(tier1, tier2, {"status": "complete"})
        assert status == "ready"

    def test_degraded_when_tier2_incomplete(self):
        tier1 = dict.fromkeys(TIER1_ARTIFACTS, True)
        tier2 = dict.fromkeys(TIER2_ARTIFACTS, False)
        status = derive_readiness_status(tier1, tier2, None)
        assert status == "degraded"

    def test_failed_when_tier1_incomplete_and_no_job(self):
        tier1 = dict.fromkeys(TIER1_ARTIFACTS, False)
        tier2 = dict.fromkeys(TIER2_ARTIFACTS, False)
        status = derive_readiness_status(tier1, tier2, None)
        assert status == "failed"


class TestProbeDimsDbAdapter:
    """probe_dims must work with production DBAdapter (fetch_one only, no fetch_val)."""

    class FetchOneOnlyDB:
        def __init__(self, count: int):
            self.count = count
            self.queries: list[str] = []

        async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
            self.queries.append(query)
            return {"count": self.count}

    @pytest.mark.asyncio
    async def test_probe_dims_true_when_residue_count_positive(self):
        db = self.FetchOneOnlyDB(322)
        assert await probe_dims("11qe", db) is True
        assert "dim_residue" in db.queries[0]

    @pytest.mark.asyncio
    async def test_probe_dims_false_when_no_residues(self):
        db = self.FetchOneOnlyDB(0)
        assert await probe_dims("11qe", db) is False

    @pytest.mark.asyncio
    async def test_probe_dims_false_when_row_missing(self):
        class EmptyDB:
            async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
                return None

        assert await probe_dims("11qe", EmptyDB()) is False


class TestBindingScanProbe:
    @pytest.mark.asyncio
    async def test_complete_scan_status_passes(self):
        db = MockReadinessDB({"binding_scan": {"status": "complete"}})
        assert await probe_binding_scan("4uj1", db) is True

    @pytest.mark.asyncio
    async def test_no_gnn_data_status_fails(self):
        db = MockReadinessDB({"binding_scan": {"status": "no_gnn_data"}})
        assert await probe_binding_scan("4uj1", db) is False


class TestAssessStructureReadiness:
    @pytest.mark.asyncio
    async def test_unknown_structure(self):
        db = MockReadinessDB({"structure_exists": None})
        result = await assess_structure_readiness("missing", db)
        assert result.readiness_status == "failed"
        assert "dims" in result.missing_artifacts

    @pytest.mark.asyncio
    async def test_partial_tier1_is_failed_without_job(self):
        db = MockReadinessDB(
            {
                "structure_exists": {"ok": 1},
                "dims": {"count": 10},
                "scope": None,
                "gnn_hyp": None,
                "graph": None,
                "source_leaks": None,
                "binding_scan": None,
                "alignment": None,
                "discovery_extended": None,
                "motifs": None,
                "md_validation": None,
                "pipeline_job": None,
            }
        )
        result = await assess_structure_readiness("4uj1", db)
        assert result.readiness_status == "failed"
        assert result.tier1["dims"] is True
        assert result.tier1["scope"] is False

    @pytest.mark.asyncio
    async def test_tier1_complete_tier2_missing_is_degraded(self):
        db = MockReadinessDB(
            {
                "structure_exists": {"ok": 1},
                "dims": {"count": 10},
                "scope": {"ok": 1},
                "gnn_hyp": {"ok": 1},
                "graph": {"ok": 1},
                "source_leaks": {"ok": 1},
                "binding_scan": {"status": "complete"},
                "alignment": None,
                "discovery_extended": None,
                "motifs": None,
                "binding_scan_detail": {"sites_found": 2, "status": "complete"},
                "md_validation": None,
                "pipeline_job": {
                    "job_id": "j1",
                    "structure_id": "4uj1",
                    "status": "complete",
                    "current_step": "complete",
                    "progress": 100,
                    "modules": [],
                },
            }
        )
        result = await assess_structure_readiness("4uj1", db)
        assert result.readiness_status == "degraded"
        assert all(result.tier1.values())
        assert result.pathway == "discovery_story"
        assert result.current_act >= 1
        assert "signal" in result.acts
        assert result.acts["signal"]["title"] == "Signal"
        assert result.artifacts.get("source_leaks") is True
        assert result.geometric_readiness.get("hyperbolic_ready") is True
