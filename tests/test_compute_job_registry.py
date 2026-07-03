"""Tests for Discovery Story compute job registry."""

from __future__ import annotations

import pytest

from science.compute.registry import (
    ACT_JOB_MAP,
    DEFAULT_PATHWAY,
    JOB_REGISTRY,
    PATHWAY_JOBS,
    topological_order,
    validate_registry,
)


class TestComputeJobRegistry:
    def test_registry_validates_on_import(self) -> None:
        validate_registry()

    def test_dag_is_acyclic_and_complete(self) -> None:
        ordered = topological_order()
        assert len(ordered) == len(JOB_REGISTRY)
        assert ordered[0] == "ingest_dims"
        assert "gnn_inference" in ordered
        assert ordered.index("assign_computation_scope") < ordered.index("gnn_inference")
        assert ordered.index("gnn_inference") < ordered.index("binding_site_scan")
        assert ordered.index("graph_topology") < ordered.index("binding_site_scan")

    def test_scan_requires_graph_and_gnn(self) -> None:
        scan = JOB_REGISTRY["binding_site_scan"]
        assert scan.requires == frozenset({"gnn_inference", "graph_topology"})

    def test_source_leaks_legacy_alias(self) -> None:
        leak = JOB_REGISTRY["source_leak_detection"]
        assert leak.produces == frozenset({"source_leaks"})
        assert leak.legacy_alias == "dtie_core"

    def test_act_job_map_covers_registry_jobs(self) -> None:
        act_jobs = {jid for jobs in ACT_JOB_MAP.values() for jid in jobs}
        foundation = {
            jid
            for jid, job in JOB_REGISTRY.items()
            if job.discovery_act in ("foundation", "sidecar")
        }
        assert act_jobs | foundation == set(JOB_REGISTRY.keys())

    def test_viewport_explore_subset(self) -> None:
        explore = PATHWAY_JOBS["viewport_explore"]
        full = PATHWAY_JOBS[DEFAULT_PATHWAY]
        assert explore < full
        assert "binding_site_scan" not in explore
        assert "gnn_inference" in explore
        assert "source_leak_detection" in explore

    def test_unknown_job_id_raises(self) -> None:
        bad_ids = frozenset({"ingest_dims", "nonexistent_job"})
        with pytest.raises(ValueError, match="Unknown job IDs"):
            topological_order(bad_ids)
