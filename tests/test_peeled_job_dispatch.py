"""Tests for peeled binding_site_scan runner dispatch."""

from __future__ import annotations

from science.compute.runner_dispatch import JOB_RUNNERS, PEELED_JOBS


class TestBindingSiteScanDispatch:
    def test_binding_scan_registered(self):
        assert "binding_site_scan" in PEELED_JOBS
        assert "binding_site_scan" in JOB_RUNNERS

    def test_md_validate_registered(self):
        assert "md_validate_top_n" in PEELED_JOBS
        assert "md_validate_top_n" in JOB_RUNNERS

    def test_pocket_pharmacophore_registered(self):
        assert "pocket_pharmacophore_map" in PEELED_JOBS

    def test_fragment_screen_has_runner(self):
        assert "fragment_screen" in JOB_RUNNERS

    def test_act04_jobs_registered(self):
        assert "pharmacophore_identification" in PEELED_JOBS
        assert "drug_candidate_ranking" in PEELED_JOBS

    def test_act01_act05_jobs_registered(self):
        for job_id in (
            "gnn_inference",
            "graph_topology",
            "witness_embedding",
            "strain_vulnerability_scan",
            "hyperbolic_motifs",
            "topological_lift",
            "resistance_pathway_map",
            "allosteric_site_detection",
        ):
            assert job_id in PEELED_JOBS
            assert job_id in JOB_RUNNERS
