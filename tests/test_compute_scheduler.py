"""Tests for pathway scheduler."""

from __future__ import annotations

from science.compute.pathway_executor import jobs_for_pathway, map_phases_to_jobs
from science.compute.runner_dispatch import JOB_RUNNERS, PEELED_JOBS


class TestPathwayScheduler:
    def test_viewport_explore_subset(self):
        jobs = jobs_for_pathway("discovery_story", source_leak_only=True)
        assert "source_leak_detection" in jobs
        assert "binding_site_scan" not in jobs
        assert jobs.index("gnn_inference") < jobs.index("source_leak_detection")

    def test_map_phases_to_jobs_includes_graph_with_gnn(self):
        completed = map_phases_to_jobs(["gnn_inference", "source_leak_detection", "phase2"])
        assert "gnn_inference" in completed
        assert "graph_topology" not in completed
        assert "source_leak_detection" in completed
        assert "strain_vulnerability_scan" in completed

    def test_act01_jobs_all_peeled(self):
        for job_id in (
            "gnn_inference",
            "graph_topology",
            "witness_embedding",
            "strain_vulnerability_scan",
        ):
            assert job_id in PEELED_JOBS
            assert job_id in JOB_RUNNERS

    def test_map_binding_scan_phase(self):
        completed = map_phases_to_jobs(["gnn_inference", "binding_site_scan"])
        assert "binding_site_scan" in completed

    def test_all_peeled_jobs_have_runners(self):
        for job_id in PEELED_JOBS:
            assert job_id in JOB_RUNNERS
