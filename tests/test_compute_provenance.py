"""Tests for discovery pathway provenance helpers."""

from __future__ import annotations

from science.compute.provenance import (
    job_provenance_parameters,
    monolith_orchestrator_parameters,
    pathway_pipeline_name,
)


class TestComputeProvenance:
    def test_pathway_pipeline_name(self):
        assert pathway_pipeline_name() == "discovery_story"

    def test_job_parameters_include_discovery_act(self):
        params = job_provenance_parameters("binding_site_scan")
        assert params["pathway"] == "discovery_story"
        assert params["job_id"] == "binding_site_scan"
        assert params["discovery_act"] == "cryptic_pocket"
        assert params["resource_class"] == "cpu_heavy"

    def test_monolith_parameters_mark_facade(self):
        params = monolith_orchestrator_parameters(computation_run_id="job_123")
        assert params["pathway"] == "discovery_story"
        assert params["orchestrator_mode"] == "monolith_facade"
        assert params["computation_run_id"] == "job_123"
