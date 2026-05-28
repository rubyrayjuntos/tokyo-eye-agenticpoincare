"""Property-based tests for the graph topology normalizer path.

Tests Properties 1-4 and 11 from the design document using Hypothesis.

Feature: graph-topology-compare
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from data.normalizer.core import Normalizer, NormalizerError, _compute_graph_metrics
from science.dtie.common.keys import make_residue_id, make_structure_id, validate_residue_id
from science.dtie.common.normalizer_payloads import (
    GraphEdge,
    GraphTopologyPayload,
    ProvenanceContext,
    RunType,
    SourceType,
)


# ---------------------------------------------------------------------------
# Mock database that tracks writes for property verification
# ---------------------------------------------------------------------------


class GraphTopologyMockDB:
    """In-memory mock DB that supports graph topology assertions."""

    def __init__(self):
        self.tables: dict[str, list[dict[str, Any]]] = {
            "provenance_run": [],
            "fact_graph_edge": [],
            "fact_graph_node_metrics": [],
            "governed_asset": [],
            "normalization_audit": [],
        }
        self._in_transaction = False

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        table = self._extract_table(query)
        if table:
            if table == "provenance_run" and "ON CONFLICT" in query and "DO NOTHING" in query:
                for row in self.tables.get("provenance_run", []):
                    if row.get("run_id") == params.get("run_id"):
                        return
            if table == "normalization_audit":
                self.tables["normalization_audit"].append(params)
                return
            self.tables.setdefault(table, []).append(params)

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        table = self._extract_table(query)
        if table:
            is_upsert = "ON CONFLICT" in query and "DO UPDATE" in query
            for params in params_list:
                if is_upsert:
                    self._upsert(table, query, params)
                else:
                    self.tables.setdefault(table, []).append(params)

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "provenance_run" in query and "run_id" in params:
            for row in self.tables.get("provenance_run", []):
                if row.get("run_id") == params["run_id"]:
                    return row
        return None

    async def begin(self) -> None:
        self._in_transaction = True

    async def commit(self) -> None:
        self._in_transaction = False

    async def rollback(self) -> None:
        # On rollback, remove uncommitted data
        self._in_transaction = False

    def _extract_table(self, query: str) -> str | None:
        q = query.strip().upper()
        if "INSERT INTO" in q:
            parts = q.split("INSERT INTO")[1].strip().split()
            if parts:
                return parts[0].lower()
        return None

    def _upsert(self, table: str, query: str, params: dict[str, Any]) -> None:
        """Handle ON CONFLICT DO UPDATE by replacing matching rows."""
        rows = self.tables.setdefault(table, [])

        if table == "fact_graph_edge":
            key_fields = ("run_id", "source_residue_id", "target_residue_id", "edge_type")
        elif table == "fact_graph_node_metrics":
            key_fields = ("run_id", "residue_id")
        elif table == "governed_asset":
            # ON CONFLICT (asset_id) DO NOTHING
            for row in rows:
                if row.get("asset_id") == params.get("asset_id"):
                    return
            rows.append(params)
            return
        else:
            rows.append(params)
            return

        # Find existing row with same natural key
        for i, row in enumerate(rows):
            if all(row.get(k) == params.get(k) for k in key_fields):
                rows[i] = params
                return
        rows.append(params)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

STRUCTURE_ID = "4obe"
VALID_EDGE_TYPES = ["h_bond", "contact", "covalent", "disulfide", "salt_bridge"]


@st.composite
def valid_residue_id_strategy(draw):
    """Generate a valid canonical residue_id."""
    chain = draw(st.sampled_from(["A", "B", "C"]))
    index = draw(st.integers(min_value=1, max_value=500))
    return make_residue_id(STRUCTURE_ID, chain, index)


@st.composite
def valid_graph_edge_strategy(draw, residue_ids=None):
    """Generate a valid GraphEdge."""
    if residue_ids is None or len(residue_ids) < 2:
        src = draw(valid_residue_id_strategy())
        tgt = draw(valid_residue_id_strategy())
        assume(src != tgt)
    else:
        src, tgt = draw(st.sampled_from(
            [(a, b) for a in residue_ids for b in residue_ids if a != b]
        ))

    edge_type = draw(st.sampled_from(VALID_EDGE_TYPES))
    distance = draw(st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False))
    weight = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))

    return GraphEdge(
        source_residue_id=src,
        target_residue_id=tgt,
        edge_type=edge_type,
        distance_angstrom=distance,
        weight=weight,
    )


@st.composite
def valid_graph_topology_payload_strategy(draw):
    """Generate a valid GraphTopologyPayload with 3-15 edges."""
    # Generate a pool of residue_ids first to ensure connectivity
    n_residues = draw(st.integers(min_value=3, max_value=10))
    residue_ids = [
        make_residue_id(STRUCTURE_ID, "A", i)
        for i in range(1, n_residues + 1)
    ]

    n_edges = draw(st.integers(min_value=3, max_value=min(15, n_residues * (n_residues - 1) // 2)))
    edges = []
    seen_keys = set()

    for _ in range(n_edges * 3):  # Try more times to get unique edges
        if len(edges) >= n_edges:
            break
        edge = draw(valid_graph_edge_strategy(residue_ids=residue_ids))
        key = (edge.source_residue_id, edge.target_residue_id, edge.edge_type)
        if key not in seen_keys:
            seen_keys.add(key)
            edges.append(edge)

    assume(len(edges) >= 1)

    run_id = f"run_test_{draw(st.integers(min_value=1, max_value=99999))}"

    return GraphTopologyPayload(
        provenance=ProvenanceContext(
            run_id=run_id,
            structure_id=STRUCTURE_ID,
            model_version="test-v1",
            pipeline_name="test_pipeline",
            run_type=RunType.ANALYSIS,
            source_type=SourceType.DETERMINISTIC,
        ),
        structure_id=STRUCTURE_ID,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestGraphTopologyProperties:
    """Property-based tests for graph topology normalizer."""

    # Feature: graph-topology-compare, Property 1: Edge persistence round trip
    @settings(max_examples=100)
    @given(payload=valid_graph_topology_payload_strategy())
    def test_property_1_edge_persistence_round_trip(self, payload: GraphTopologyPayload):
        """For any valid GraphTopologyPayload with N edges, after calling
        normalize_graph_topology, querying fact_graph_edge for that run_id
        should return exactly N edges with matching source, target, edge_type,
        and distance values.

        **Validates: Requirements 1.1**
        """
        db = GraphTopologyMockDB()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        result = asyncio.run(normalizer.normalize_graph_topology(payload))

        assert result.success is True
        assert result.run_id == payload.provenance.run_id

        # Verify edges persisted
        persisted_edges = db.tables["fact_graph_edge"]
        assert len(persisted_edges) == len(payload.edges)

        # Verify each edge matches
        for original_edge in payload.edges:
            matching = [
                e for e in persisted_edges
                if e["source_residue_id"] == original_edge.source_residue_id
                and e["target_residue_id"] == original_edge.target_residue_id
                and e["edge_type"] == original_edge.edge_type
            ]
            assert len(matching) == 1, (
                f"Expected exactly 1 match for edge "
                f"{original_edge.source_residue_id} -> {original_edge.target_residue_id} "
                f"({original_edge.edge_type})"
            )
            assert matching[0]["distance_angstrom"] == original_edge.distance_angstrom

    # Feature: graph-topology-compare, Property 2: Idempotent graph topology writes
    @settings(max_examples=100)
    @given(payload=valid_graph_topology_payload_strategy())
    def test_property_2_idempotent_writes(self, payload: GraphTopologyPayload):
        """For any valid GraphTopologyPayload, calling normalize_graph_topology
        twice with identical data should produce the same database state as
        calling it once — no duplicate rows.

        **Validates: Requirements 1.2, 2.2**
        """
        db = GraphTopologyMockDB()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        # Write once
        asyncio.run(normalizer.normalize_graph_topology(payload))
        edges_after_first = len(db.tables["fact_graph_edge"])
        metrics_after_first = len(db.tables["fact_graph_node_metrics"])

        # Write again (same payload)
        asyncio.run(normalizer.normalize_graph_topology(payload))
        edges_after_second = len(db.tables["fact_graph_edge"])
        metrics_after_second = len(db.tables["fact_graph_node_metrics"])

        # No duplicates
        assert edges_after_second == edges_after_first
        assert metrics_after_second == metrics_after_first

    # Feature: graph-topology-compare, Property 3: Invalid residue_id rejection
    @settings(max_examples=100)
    @given(
        valid_payload=valid_graph_topology_payload_strategy(),
        bad_id=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=10,
        ),
    )
    def test_property_3_invalid_residue_id_rejection(
        self, valid_payload: GraphTopologyPayload, bad_id: str
    ):
        """For any GraphTopologyPayload containing at least one edge with a
        residue_id that does not match the canonical format,
        normalize_graph_topology should raise NormalizerError and leave no
        partial data in fact_graph_edge.

        **Validates: Requirements 1.3**
        """
        assume(not validate_residue_id(bad_id))

        # Inject a bad residue_id into the first edge
        # We bypass Pydantic validation by constructing the payload manually
        bad_edge_data = valid_payload.edges[0].model_dump()
        bad_edge_data["source_residue_id"] = bad_id

        # Create payload with raw edge data (bypass field validator)
        edges_data = [bad_edge_data] + [e.model_dump() for e in valid_payload.edges[1:]]

        # Build payload without Pydantic validation on residue_ids
        # by directly constructing with model_construct
        bad_edges = []
        for ed in edges_data:
            edge = GraphEdge.model_construct(**ed)
            bad_edges.append(edge)

        bad_payload = GraphTopologyPayload.model_construct(
            provenance=valid_payload.provenance,
            structure_id=valid_payload.structure_id,
            edges=bad_edges,
            computed_at=valid_payload.computed_at,
        )

        db = GraphTopologyMockDB()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        with pytest.raises(NormalizerError):
            asyncio.run(normalizer.normalize_graph_topology(bad_payload))

        # No partial data should remain
        assert len(db.tables["fact_graph_edge"]) == 0

    # Feature: graph-topology-compare, Property 4: Node metrics computation correctness
    @settings(max_examples=100)
    @given(payload=valid_graph_topology_payload_strategy())
    def test_property_4_node_metrics_correctness(self, payload: GraphTopologyPayload):
        """For any valid graph, after normalize_graph_topology completes, the
        persisted metrics should match independently-computed networkx metrics.

        **Validates: Requirements 2.1**
        """
        import networkx as nx

        db = GraphTopologyMockDB()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        asyncio.run(normalizer.normalize_graph_topology(payload))

        # Independently compute metrics
        G = nx.Graph()
        for edge in payload.edges:
            w = edge.distance_angstrom if edge.distance_angstrom is not None else edge.weight
            G.add_edge(edge.source_residue_id, edge.target_residue_id, weight=w)

        expected_degree = dict(G.degree())
        expected_betweenness = nx.betweenness_centrality(G)
        expected_clustering = nx.clustering(G)
        expected_closeness = nx.closeness_centrality(G)
        expected_bridges = set(nx.articulation_points(G))

        # Compare persisted metrics
        persisted_metrics = db.tables["fact_graph_node_metrics"]
        persisted_by_residue = {m["residue_id"]: m for m in persisted_metrics}

        assert set(persisted_by_residue.keys()) == set(G.nodes())

        for node in G.nodes():
            m = persisted_by_residue[node]
            assert m["degree"] == expected_degree[node]
            assert abs(m["betweenness"] - expected_betweenness[node]) < 1e-10
            assert abs(m["clustering_coefficient"] - expected_clustering[node]) < 1e-10
            assert abs(m["closeness"] - expected_closeness[node]) < 1e-10
            assert m["is_bridge"] == (node in expected_bridges)

    # Feature: graph-topology-compare, Property 11: Governance invariants
    @settings(max_examples=100)
    @given(payload=valid_graph_topology_payload_strategy())
    def test_property_11_governance_invariants(self, payload: GraphTopologyPayload):
        """For any successful normalize_graph_topology call, the system should
        have: (a) a provenance_run record, (b) governed_asset records for all
        created assets, and (c) a normalization_audit record with status='success'.

        **Validates: Requirements 9.2, 9.3, 9.4**
        """
        db = GraphTopologyMockDB()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        result = asyncio.run(normalizer.normalize_graph_topology(payload))

        # (a) provenance_run record exists
        prov_records = db.tables["provenance_run"]
        assert len(prov_records) == 1
        assert prov_records[0]["run_id"] == payload.provenance.run_id

        # (b) governed_asset records for all created assets
        governed_assets = db.tables["governed_asset"]
        assert len(governed_assets) == result.assets_created

        # (c) normalization_audit with status='success'
        audit_records = db.tables["normalization_audit"]
        success_audits = [a for a in audit_records if a.get("status") == "success"]
        assert len(success_audits) == 1
        assert success_audits[0]["run_id"] == payload.provenance.run_id
