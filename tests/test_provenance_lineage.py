"""Tests for provenance lineage traversal and querying."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from data.provenance.lineage import ProvenanceTracer, ProvenanceNode, AssetRecord, LineageResult


# ---------------------------------------------------------------------------
# Mock database with realistic provenance data
# ---------------------------------------------------------------------------


class MockProvenanceDB:
    """In-memory mock with a realistic provenance DAG for testing.

    DAG structure:
        run_root (ingestion)
          └── run_v3_gnn (v3 GNN inference)
          └── run_v4_gnn (v4 GNN inference)
                └── run_v4_phase3 (v4 Phase 3 persistence)
    """

    def __init__(self):
        self.runs = [
            {
                "run_id": "run_root",
                "model_version": "ingestion-v1",
                "pipeline_name": "pdb_ingestion",
                "run_type": "analysis",
                "source_type": "external",
                "structure_id": "4obe",
                "started_at": datetime(2026, 5, 20, tzinfo=timezone.utc),
                "completed_at": datetime(2026, 5, 20, 0, 1, tzinfo=timezone.utc),
                "parent_run_id": None,
                "parameters": {"source": "rcsb"},
            },
            {
                "run_id": "run_v3_gnn",
                "model_version": "GOSPConeMapper-v3",
                "pipeline_name": "dtie_v3",
                "run_type": "inference",
                "source_type": "probabilistic",
                "structure_id": "4obe",
                "started_at": datetime(2026, 5, 21, tzinfo=timezone.utc),
                "completed_at": datetime(2026, 5, 21, 0, 5, tzinfo=timezone.utc),
                "parent_run_id": "run_root",
                "parameters": {"checkpoint": "robust_experts.pt"},
            },
            {
                "run_id": "run_v4_gnn",
                "model_version": "GOSPConeMapper-v4",
                "pipeline_name": "dtie_v4",
                "run_type": "inference",
                "source_type": "probabilistic",
                "structure_id": "4obe",
                "started_at": datetime(2026, 5, 22, tzinfo=timezone.utc),
                "completed_at": datetime(2026, 5, 22, 0, 3, tzinfo=timezone.utc),
                "parent_run_id": "run_root",
                "parameters": {"checkpoint": "tokyo_eyes_v4.pt"},
            },
            {
                "run_id": "run_v4_phase3",
                "model_version": "DTIE-v4-phase3",
                "pipeline_name": "dtie_v4",
                "run_type": "inference",
                "source_type": "deterministic",
                "structure_id": "4obe",
                "started_at": datetime(2026, 5, 22, 1, 0, tzinfo=timezone.utc),
                "completed_at": datetime(2026, 5, 22, 1, 2, tzinfo=timezone.utc),
                "parent_run_id": "run_v4_gnn",
                "parameters": {"n_landmarks": 50},
            },
        ]

        self.assets = [
            {"asset_id": "emb_v3_1", "asset_type": "gnn_node_embedding", "structure_id": "4obe", "residue_id": "4obe:A:12", "run_id": "run_v3_gnn", "created_at": datetime(2026, 5, 21, tzinfo=timezone.utc)},
            {"asset_id": "emb_v4_1", "asset_type": "gnn_node_embedding", "structure_id": "4obe", "residue_id": "4obe:A:12", "run_id": "run_v4_gnn", "created_at": datetime(2026, 5, 22, tzinfo=timezone.utc)},
            {"asset_id": "phase3_1", "asset_type": "dtie_phase3_persistence", "structure_id": "4obe", "residue_id": "4obe:A:12", "run_id": "run_v4_phase3", "created_at": datetime(2026, 5, 22, 1, 2, tzinfo=timezone.utc)},
        ]

        self.embeddings = [
            {"run_id": "run_v3_gnn", "residue_id": "4obe:A:12"},
            {"run_id": "run_v4_gnn", "residue_id": "4obe:A:12"},
        ]

        self.phase3 = [
            {"run_id": "run_v4_phase3", "residue_id": "4obe:A:12"},
        ]

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "governed_asset" in query and "asset_id" in params:
            for a in self.assets:
                if a["asset_id"] == params["asset_id"]:
                    return a
        return None

    async def fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        # Handle the recursive lineage query (ancestors)
        if "WITH RECURSIVE lineage" in query and "parent_run_id" in query:
            run_id = params["run_id"]
            max_depth = params.get("max_depth", 10)

            # Simple iterative ancestor traversal
            if "c.parent_run_id = l.run_id" in query:
                # Descendants query
                results = []
                queue = [run_id]
                depth = 0
                while queue and depth <= max_depth:
                    next_queue = []
                    for rid in queue:
                        for r in self.runs:
                            if r["run_id"] == rid:
                                results.append({**r, "depth": depth})
                            if r.get("parent_run_id") == rid and r["run_id"] != rid:
                                next_queue.append(r["run_id"])
                    queue = next_queue
                    depth += 1
                return results
            else:
                # Ancestors query
                results = []
                current = run_id
                depth = 0
                while current and depth <= max_depth:
                    for r in self.runs:
                        if r["run_id"] == current:
                            results.append({**r, "depth": depth})
                            current = r.get("parent_run_id")
                            break
                    else:
                        break
                    depth += 1
                return results

        # Handle residue contributing runs query
        if "fact_gnn_node_embedding" in query and "residue_id" in params:
            residue_id = params["residue_id"]
            run_ids = set()
            for e in self.embeddings:
                if e["residue_id"] == residue_id:
                    run_ids.add(e["run_id"])
            for p in self.phase3:
                if p["residue_id"] == residue_id:
                    run_ids.add(p["run_id"])
            return [{"run_id": rid} for rid in run_ids]

        # Handle governed_asset queries
        if "governed_asset" in query:
            if "residue_id" in params:
                return [a for a in self.assets if a.get("residue_id") == params["residue_id"]]
            if "run_id" in params:
                return [a for a in self.assets if a["run_id"] == params["run_id"]]

        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db() -> MockProvenanceDB:
    return MockProvenanceDB()


@pytest.fixture
def tracer(mock_db: MockProvenanceDB) -> ProvenanceTracer:
    return ProvenanceTracer(db=mock_db)  # type: ignore[arg-type]


class TestProvenanceTracer:
    @pytest.mark.asyncio
    async def test_get_ancestors_from_leaf(self, tracer: ProvenanceTracer):
        """Phase 3 run should trace back through v4 GNN to root."""
        ancestors = await tracer.get_ancestors("run_v4_phase3")

        assert len(ancestors) == 3  # phase3 -> v4_gnn -> root
        assert ancestors[0].run_id == "run_v4_phase3"
        assert ancestors[1].run_id == "run_v4_gnn"
        assert ancestors[2].run_id == "run_root"

    @pytest.mark.asyncio
    async def test_get_ancestors_from_root(self, tracer: ProvenanceTracer):
        """Root run has no ancestors beyond itself."""
        ancestors = await tracer.get_ancestors("run_root")

        assert len(ancestors) == 1
        assert ancestors[0].run_id == "run_root"
        assert ancestors[0].parent_run_id is None

    @pytest.mark.asyncio
    async def test_get_descendants_from_root(self, tracer: ProvenanceTracer):
        """Root should have v3_gnn, v4_gnn, and v4_phase3 as descendants."""
        descendants = await tracer.get_descendants("run_root")

        run_ids = {d.run_id for d in descendants}
        assert "run_root" in run_ids
        assert "run_v3_gnn" in run_ids
        assert "run_v4_gnn" in run_ids
        assert "run_v4_phase3" in run_ids

    @pytest.mark.asyncio
    async def test_trace_residue(self, tracer: ProvenanceTracer):
        """Tracing a residue should find all contributing runs."""
        lineage = await tracer.trace_residue("4obe:A:12")

        assert lineage.target_id == "4obe:A:12"
        assert lineage.target_type == "residue"
        assert lineage.total_runs == 3  # v3_gnn, v4_gnn, v4_phase3
        assert lineage.total_assets == 3

    @pytest.mark.asyncio
    async def test_trace_asset(self, tracer: ProvenanceTracer):
        """Tracing an asset should find its producing run's lineage."""
        lineage = await tracer.trace_asset("emb_v4_1")

        assert lineage.target_id == "emb_v4_1"
        assert lineage.target_type == "asset"
        assert len(lineage.ancestors) >= 1

    @pytest.mark.asyncio
    async def test_trace_nonexistent_asset(self, tracer: ProvenanceTracer):
        """Tracing a non-existent asset returns empty lineage."""
        lineage = await tracer.trace_asset("does_not_exist")

        assert lineage.total_runs == 0
        assert lineage.total_assets == 0

    @pytest.mark.asyncio
    async def test_get_run_assets(self, tracer: ProvenanceTracer):
        """Should return all assets produced by a specific run."""
        assets = await tracer.get_run_assets("run_v4_gnn")

        assert len(assets) == 1
        assert assets[0].asset_id == "emb_v4_1"
        assert assets[0].asset_type == "gnn_node_embedding"

    @pytest.mark.asyncio
    async def test_compare_lineages_v3_vs_v4(self, tracer: ProvenanceTracer):
        """Comparing v3 and v4 runs should show shared root ancestor."""
        comparison = await tracer.compare_lineages("run_v3_gnn", "run_v4_gnn")

        assert comparison["run_a"] == "run_v3_gnn"
        assert comparison["run_b"] == "run_v4_gnn"
        assert "run_root" in comparison["shared_ancestors"]

    @pytest.mark.asyncio
    async def test_max_depth_limits_traversal(self, tracer: ProvenanceTracer):
        """max_depth=0 should only return the starting node."""
        ancestors = await tracer.get_ancestors("run_v4_phase3", max_depth=0)

        assert len(ancestors) == 1
        assert ancestors[0].run_id == "run_v4_phase3"
