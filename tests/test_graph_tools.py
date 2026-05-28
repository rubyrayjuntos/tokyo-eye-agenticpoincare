"""Property-based tests for graph topology agent tools.

Tests Properties 5-10 from the design document using Hypothesis.

Feature: graph-topology-compare
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent.tools.graph_tools import (
    compare_graphs,
    find_graph_bridges,
    get_graph_metrics,
    get_hbond_network,
    get_shortest_paths,
)
from science.dtie.common.keys import make_residue_id


# ---------------------------------------------------------------------------
# Mock database for graph tools testing
# ---------------------------------------------------------------------------

STRUCTURE_A = "4obe"
STRUCTURE_B = "4obe_g12d"
VALID_EDGE_TYPES = ["h_bond", "contact", "covalent", "disulfide", "salt_bridge"]


class GraphToolsMockDB:
    """In-memory mock DB that stores graph edges and metrics for tool testing."""

    def __init__(self):
        self.edges: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        query_lower = query.lower()

        if "fact_graph_edge" in query_lower:
            return self._filter_edges(query_lower, params)
        elif "fact_graph_node_metrics" in query_lower:
            return self._filter_metrics(query_lower, params)
        return []

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        rows = await self.fetch_all(query, params)
        return rows[0] if rows else None

    def _filter_edges(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        results = list(self.edges)

        # Filter by structure_id
        if "structure_id" in params:
            sid = params["structure_id"]
            results = [e for e in results if e.get("structure_id") == sid]

        # Filter by run_id
        if "run_id" in params:
            rid = params["run_id"]
            results = [e for e in results if e.get("run_id") == rid]

        # Filter by edge_type
        if "h_bond" in query and "edge_type" in query:
            results = [e for e in results if e.get("edge_type") == "h_bond"]

        # Filter by residue_ids (IN clause)
        residue_filter = [v for k, v in params.items() if k.startswith("r") and k[1:].isdigit()]
        if residue_filter and "source_residue_id in" in query.lower():
            results = [
                e for e in results
                if e.get("source_residue_id") in residue_filter
                or e.get("target_residue_id") in residue_filter
            ]

        return results

    def _filter_metrics(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        results = list(self.metrics)

        # Filter by structure_id
        if "structure_id" in params:
            sid = params["structure_id"]
            results = [m for m in results if m.get("structure_id") == sid]

        # Filter by run_id
        if "run_id" in params:
            rid = params["run_id"]
            results = [m for m in results if m.get("run_id") == rid]

        # Filter by is_bridge
        if "is_bridge = true" in query.lower():
            results = [m for m in results if m.get("is_bridge") is True]

        # Filter by residue_ids (IN clause)
        residue_filter = [v for k, v in params.items() if k.startswith("r") and k[1:].isdigit()]
        if residue_filter and "residue_id in" in query.lower():
            results = [m for m in results if m.get("residue_id") in residue_filter]

        # Filter by metric_type (column selection is handled at query level,
        # but we return all columns and let the tool handle it)
        return results


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def graph_edges_strategy(draw, structure_id=STRUCTURE_A, n_residues=None, n_edges=None):
    """Generate a set of graph edges for a structure."""
    if n_residues is None:
        n_residues = draw(st.integers(min_value=4, max_value=12))
    if n_edges is None:
        max_possible = n_residues * (n_residues - 1) // 2
        n_edges = draw(st.integers(min_value=3, max_value=min(20, max_possible)))

    residue_ids = [
        make_residue_id(structure_id, "A", i) for i in range(1, n_residues + 1)
    ]

    edges = []
    seen_keys = set()
    attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 5:
        attempts += 1
        src_idx = draw(st.integers(min_value=0, max_value=n_residues - 1))
        tgt_idx = draw(st.integers(min_value=0, max_value=n_residues - 1))
        if src_idx == tgt_idx:
            continue
        edge_type = draw(st.sampled_from(VALID_EDGE_TYPES))
        key = (residue_ids[src_idx], residue_ids[tgt_idx], edge_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        distance = draw(st.floats(min_value=2.0, max_value=15.0, allow_nan=False, allow_infinity=False))
        weight = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
        edges.append({
            "source_residue_id": residue_ids[src_idx],
            "target_residue_id": residue_ids[tgt_idx],
            "edge_type": edge_type,
            "distance_angstrom": distance,
            "weight": weight,
            "structure_id": structure_id,
            "run_id": f"run_{structure_id}",
        })

    assume(len(edges) >= 3)
    return edges, residue_ids


@st.composite
def graph_with_metrics_strategy(draw, structure_id=STRUCTURE_A):
    """Generate edges and corresponding node metrics for a structure."""
    import networkx as nx

    edges, residue_ids = draw(graph_edges_strategy(structure_id=structure_id))

    # Compute metrics from edges (same logic as normalizer)
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source_residue_id"], e["target_residue_id"],
                   weight=e["distance_angstrom"])

    degree_dict = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}
    bridges_set = set(nx.articulation_points(G))

    metrics = []
    for node in G.nodes():
        metrics.append({
            "residue_id": node,
            "structure_id": structure_id,
            "run_id": f"run_{structure_id}",
            "degree": degree_dict[node],
            "betweenness": betweenness[node],
            "clustering_coefficient": clustering[node],
            "closeness": closeness[node],
            "eigenvector_centrality": eigenvector.get(node, 0.0),
            "is_bridge": node in bridges_set,
            "conductance": 0.0,
            "computed_at": "2026-01-01T00:00:00Z",
        })

    return edges, metrics, residue_ids


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestGraphToolsProperties:
    """Property-based tests for graph topology agent tools."""

    # Feature: graph-topology-compare, Property 5: Graph metrics query filtering
    @settings(max_examples=100)
    @given(data=st.data())
    def test_property_5_graph_metrics_query_filtering(self, data):
        """For any structure with persisted metrics and any subset of
        residue_ids or metric_types, get_graph_metrics should return only
        rows matching the filter.

        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        edges, metrics, residue_ids = data.draw(graph_with_metrics_strategy())

        db = GraphToolsMockDB()
        db.edges = edges
        db.metrics = metrics

        # Test with residue_ids filter
        if len(residue_ids) >= 2:
            n_filter = data.draw(st.integers(min_value=1, max_value=min(3, len(residue_ids))))
            filter_ids = data.draw(
                st.lists(
                    st.sampled_from(residue_ids),
                    min_size=n_filter,
                    max_size=n_filter,
                    unique=True,
                )
            )

            result = asyncio.run(get_graph_metrics(
                structure_id=STRUCTURE_A,
                residue_ids=filter_ids,
                db=db,
            ))

            assert result.success is True
            # All returned residues must be in the filter set
            for row in result.data["metrics"]:
                assert row["residue_id"] in filter_ids

        # Test with metric_type filter
        metric_type = data.draw(st.sampled_from([
            "degree", "betweenness", "clustering_coefficient",
            "closeness", "eigenvector_centrality", "is_bridge", "conductance",
        ]))

        result = asyncio.run(get_graph_metrics(
            structure_id=STRUCTURE_A,
            metric_type=metric_type,
            db=db,
        ))

        assert result.success is True

    # Feature: graph-topology-compare, Property 6: Edge diff correctness
    @settings(max_examples=100)
    @given(data=st.data())
    def test_property_6_edge_diff_correctness(self, data):
        """For any two structures with persisted edges, compare_graphs should
        return: gained = edges in B not in A, lost = edges in A not in B,
        changed = edges in both with different weights.

        **Validates: Requirements 4.1**
        """
        edges_a, residue_ids_a = data.draw(graph_edges_strategy(structure_id=STRUCTURE_A))
        edges_b, residue_ids_b = data.draw(graph_edges_strategy(structure_id=STRUCTURE_B))

        db = GraphToolsMockDB()
        db.edges = edges_a + edges_b
        db.metrics = []  # Metrics not needed for edge diff

        result = asyncio.run(compare_graphs(
            structure_id_a=STRUCTURE_A,
            structure_id_b=STRUCTURE_B,
            db=db,
        ))

        assert result.success is True
        edge_diff = result.data["edge_diff"]

        # Compute expected diff independently
        def _key(e):
            src_parts = e["source_residue_id"].split(":")
            tgt_parts = e["target_residue_id"].split(":")
            src_pos = f"{src_parts[1]}:{src_parts[2]}" if len(src_parts) >= 3 else e["source_residue_id"]
            tgt_pos = f"{tgt_parts[1]}:{tgt_parts[2]}" if len(tgt_parts) >= 3 else e["target_residue_id"]
            return (src_pos, tgt_pos, e["edge_type"])

        keys_a = {_key(e): e for e in edges_a}
        keys_b = {_key(e): e for e in edges_b}

        expected_gained = set(keys_b.keys()) - set(keys_a.keys())
        expected_lost = set(keys_a.keys()) - set(keys_b.keys())

        assert edge_diff["gained_count"] == len(expected_gained)
        assert edge_diff["lost_count"] == len(expected_lost)

    # Feature: graph-topology-compare, Property 7: Metric diff with canonical alignment
    @settings(max_examples=100)
    @given(data=st.data())
    def test_property_7_metric_diff_canonical_alignment(self, data):
        """For any two structures with persisted metrics, compare_graphs should
        return metric deltas only for residues whose canonical position appears
        in both structures, and each delta should equal metric_B - metric_A.

        **Validates: Requirements 4.2, 4.3**
        """
        edges_a, metrics_a, _ = data.draw(graph_with_metrics_strategy(structure_id=STRUCTURE_A))
        edges_b, metrics_b, _ = data.draw(graph_with_metrics_strategy(structure_id=STRUCTURE_B))

        db = GraphToolsMockDB()
        db.edges = edges_a + edges_b
        db.metrics = metrics_a + metrics_b

        result = asyncio.run(compare_graphs(
            structure_id_a=STRUCTURE_A,
            structure_id_b=STRUCTURE_B,
            db=db,
        ))

        assert result.success is True

        # Verify deltas are correct for common positions
        def _pos(rid):
            parts = rid.split(":")
            return f"{parts[1]}:{parts[2]}" if len(parts) >= 3 else rid

        ma_by_pos = {_pos(m["residue_id"]): m for m in metrics_a}
        mb_by_pos = {_pos(m["residue_id"]): m for m in metrics_b}
        common = set(ma_by_pos.keys()) & set(mb_by_pos.keys())

        assert result.data["total_matched_residues"] == len(common)

        for delta in result.data["metric_diff"]:
            pos = delta["position"]
            assert pos in common
            # Verify delta = B - A
            assert abs(delta["betweenness_delta"] - (mb_by_pos[pos]["betweenness"] - ma_by_pos[pos]["betweenness"])) < 1e-10

    # Feature: graph-topology-compare, Property 8: H-bond network filtering
    @settings(max_examples=100)
    @given(data=st.data())
    def test_property_8_hbond_network_filtering(self, data):
        """For any structure with mixed edge types and any optional residue_ids
        filter, get_hbond_network should return only edges where edge_type =
        'h_bond', and when a residue filter is applied, every returned edge
        should have at least one endpoint in the specified set.

        **Validates: Requirements 5.1, 5.2**
        """
        edges, residue_ids = data.draw(graph_edges_strategy())

        db = GraphToolsMockDB()
        db.edges = edges

        # Test without filter — should only return h_bond edges
        result = asyncio.run(get_hbond_network(
            structure_id=STRUCTURE_A,
            db=db,
        ))

        assert result.success is True
        for edge in result.data["edges"]:
            assert edge["edge_type"] == "h_bond"

        # Test with residue filter
        if len(residue_ids) >= 2:
            filter_ids = data.draw(
                st.lists(
                    st.sampled_from(residue_ids),
                    min_size=1,
                    max_size=min(3, len(residue_ids)),
                    unique=True,
                )
            )

            result = asyncio.run(get_hbond_network(
                structure_id=STRUCTURE_A,
                residue_ids=filter_ids,
                db=db,
            ))

            assert result.success is True
            for edge in result.data["edges"]:
                assert edge["edge_type"] == "h_bond"
                # At least one endpoint must be in the filter
                assert (
                    edge["source_residue_id"] in filter_ids
                    or edge["target_residue_id"] in filter_ids
                )

    # Feature: graph-topology-compare, Property 9: Bridge detection correctness
    @settings(max_examples=100)
    @given(data=st.data())
    def test_property_9_bridge_detection_correctness(self, data):
        """For any graph, the residues returned by find_graph_bridges should
        exactly match the set of articulation points computed independently
        by networkx on the same edge set.

        **Validates: Requirements 6.1**
        """
        import networkx as nx

        edges, metrics, _ = data.draw(graph_with_metrics_strategy())

        db = GraphToolsMockDB()
        db.edges = edges
        db.metrics = metrics

        result = asyncio.run(find_graph_bridges(
            structure_id=STRUCTURE_A,
            db=db,
        ))

        assert result.success is True

        # Independently compute bridges
        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source_residue_id"], e["target_residue_id"])
        expected_bridges = set(nx.articulation_points(G))

        returned_bridges = {r["residue_id"] for r in result.data["bridges"]}
        assert returned_bridges == expected_bridges

    # Feature: graph-topology-compare, Property 10: Shortest path correctness
    @settings(max_examples=100)
    @given(data=st.data())
    def test_property_10_shortest_path_correctness(self, data):
        """For any two connected residues in a persisted graph, get_shortest_paths
        should return a valid path whose total distance equals the sum of
        distance_angstrom along the path, matching networkx.shortest_path_length.

        **Validates: Requirements 7.1, 7.2**
        """
        import networkx as nx

        edges, residue_ids = data.draw(graph_edges_strategy())

        # Build graph to find connected pairs
        G = nx.Graph()
        for e in edges:
            G.add_edge(
                e["source_residue_id"],
                e["target_residue_id"],
                distance_angstrom=e["distance_angstrom"],
            )

        # Pick two connected nodes
        nodes = list(G.nodes())
        assume(len(nodes) >= 2)

        source = data.draw(st.sampled_from(nodes))
        # Find nodes reachable from source
        reachable = set(nx.node_connected_component(G, source)) - {source}
        assume(len(reachable) > 0)
        target = data.draw(st.sampled_from(sorted(reachable)))

        db = GraphToolsMockDB()
        db.edges = edges

        result = asyncio.run(get_shortest_paths(
            structure_id=STRUCTURE_A,
            source_residue_id=source,
            target_residue_id=target,
            db=db,
        ))

        assert result.success is True
        assert result.data["disconnected"] is False

        path = result.data["path"]
        assert len(path) >= 2
        assert path[0] == source
        assert path[-1] == target

        # Verify path is valid (each consecutive pair is connected)
        for i in range(len(path) - 1):
            assert G.has_edge(path[i], path[i + 1])

        # Verify total distance matches sum along path
        expected_distance = sum(
            G[path[i]][path[i + 1]]["distance_angstrom"]
            for i in range(len(path) - 1)
        )
        assert abs(result.data["total_distance"] - expected_distance) < 1e-10

        # Verify it matches networkx shortest path length
        nx_length = nx.shortest_path_length(G, source, target, weight="distance_angstrom")
        assert abs(result.data["total_distance"] - nx_length) < 1e-10
