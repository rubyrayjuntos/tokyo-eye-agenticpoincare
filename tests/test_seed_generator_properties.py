"""Property-based tests for seed generator (binding site scan at ingestion).

Feature: binding-site-scan-at-ingestion
Properties: 1, 2, 3
Validates: Requirements 1.1, 1.3, 1.4, 1.5
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent.tools.cryptic.seed_generator import (
    GNNNodeOutput,
    cluster_residues_dbscan,
    filter_qualifying_residues,
    generate_seeds_from_gnn,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def gnn_node_strategy(draw, residue_id=None):
    """Generate a single GNNNodeOutput with realistic ranges."""
    rid = residue_id or draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=3, max_size=15,
    ))
    return GNNNodeOutput(
        residue_id=rid,
        epistemic_uncertainty=draw(
            st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False)
        ),
        cone_depth=draw(
            st.floats(min_value=0.0, max_value=25.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def gnn_node_list_strategy(draw, min_size=1, max_size=50):
    """Generate a list of GNNNodeOutput with unique residue_ids."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    nodes = []
    for i in range(count):
        rid = f"res_{i:04d}"
        node = GNNNodeOutput(
            residue_id=rid,
            epistemic_uncertainty=draw(
                st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False)
            ),
            cone_depth=draw(
                st.floats(min_value=0.0, max_value=25.0, allow_nan=False, allow_infinity=False)
            ),
        )
        nodes.append(node)
    return nodes


@st.composite
def threshold_pair_strategy(draw):
    """Generate a pair of thresholds for uncertainty and cone_depth."""
    uncertainty = draw(
        st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False)
    )
    cone_depth = draw(
        st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False)
    )
    return uncertainty, cone_depth


def make_ca_coords_for_nodes(
    nodes: list[GNNNodeOutput], spread: float = 5.0
) -> dict[str, tuple[float, float, float]]:
    """Generate Cα coordinates for a list of nodes in a tight spatial cluster."""
    rng = np.random.default_rng(42)
    coords = {}
    for node in nodes:
        coords[node.residue_id] = (
            float(rng.uniform(0, spread)),
            float(rng.uniform(0, spread)),
            float(rng.uniform(0, spread)),
        )
    return coords


# ---------------------------------------------------------------------------
# Property 1: Qualifying Residue Filter Correctness
# Feature: binding-site-scan-at-ingestion, Property 1
# ---------------------------------------------------------------------------


class TestProperty1QualifyingResidueFilter:
    """Property 1: Qualifying Residue Filter Correctness.

    For any set of GNN node outputs and any pair of thresholds, the qualifying
    residue set SHALL contain exactly those residues where
    epistemic_uncertainty >= uncertainty_threshold AND
    cone_depth >= cone_depth_threshold.
    """

    @settings(max_examples=100)
    @given(
        nodes=gnn_node_list_strategy(min_size=1, max_size=40),
        thresholds=threshold_pair_strategy(),
    )
    def test_filter_correctness(self, nodes, thresholds):
        """Feature: binding-site-scan-at-ingestion, Property 1: Qualifying Residue Filter Correctness

        **Validates: Requirements 1.1**
        """
        uncertainty_threshold, cone_depth_threshold = thresholds

        result = filter_qualifying_residues(nodes, uncertainty_threshold, cone_depth_threshold)

        # Compute expected set manually
        expected_ids = {
            n.residue_id
            for n in nodes
            if n.epistemic_uncertainty >= uncertainty_threshold
            and n.cone_depth >= cone_depth_threshold
        }

        result_ids = {n.residue_id for n in result}

        # No qualifying residue excluded
        assert result_ids == expected_ids, (
            f"Filter mismatch: got {result_ids}, expected {expected_ids}"
        )

        # Every returned node actually meets both thresholds
        for node in result:
            assert node.epistemic_uncertainty >= uncertainty_threshold
            assert node.cone_depth >= cone_depth_threshold


# ---------------------------------------------------------------------------
# Property 2: Minimum Cluster Size Enforcement
# Feature: binding-site-scan-at-ingestion, Property 2
# ---------------------------------------------------------------------------


class TestProperty2MinClusterSize:
    """Property 2: Minimum Cluster Size Enforcement.

    For any output of the scan phase, every returned CandidateCluster SHALL
    contain at least min_cluster_size residues.
    """

    @settings(max_examples=100)
    @given(
        min_cluster_size=st.integers(min_value=2, max_value=6),
        nodes=gnn_node_list_strategy(min_size=5, max_size=50),
    )
    def test_minimum_cluster_size_enforced(self, min_cluster_size, nodes):
        """Feature: binding-site-scan-at-ingestion, Property 2: Minimum Cluster Size Enforcement

        **Validates: Requirements 1.3**
        """
        # Use low thresholds so most nodes qualify
        qualifying = filter_qualifying_residues(nodes, 0.0, 0.0)
        assume(len(qualifying) >= min_cluster_size)

        # Place all qualifying residues close together so DBSCAN forms clusters
        ca_coords = make_ca_coords_for_nodes(qualifying, spread=5.0)

        clusters = cluster_residues_dbscan(
            qualifying, ca_coords, eps_angstrom=10.0, min_cluster_size=min_cluster_size
        )

        # Every cluster must have at least min_cluster_size members
        for cluster in clusters:
            assert cluster.member_count >= min_cluster_size, (
                f"Cluster {cluster.cluster_id} has {cluster.member_count} members, "
                f"expected >= {min_cluster_size}"
            )
            assert len(cluster.residue_ids) >= min_cluster_size
            assert cluster.member_count == len(cluster.residue_ids)


# ---------------------------------------------------------------------------
# Property 3: Output Cap and Score Ordering
# Feature: binding-site-scan-at-ingestion, Property 3
# ---------------------------------------------------------------------------


class TestProperty3OutputCapAndOrdering:
    """Property 3: Output Cap and Score Ordering.

    For any structure producing more than max_clusters candidates, the scan
    SHALL return exactly max_clusters candidates, and for every retained
    candidate the composite_score >= any discarded candidate's score.
    The output is sorted by composite_score descending.
    """

    @settings(max_examples=100)
    @given(
        max_clusters=st.integers(min_value=1, max_value=10),
        nodes=gnn_node_list_strategy(min_size=20, max_size=50),
    )
    def test_output_cap_and_score_ordering(self, max_clusters, nodes):
        """Feature: binding-site-scan-at-ingestion, Property 3: Output Cap and Score Ordering

        **Validates: Requirements 1.4, 1.5**
        """
        # Place nodes in well-separated spatial groups to force many clusters
        # Create groups of 3 residues each at distinct locations
        ca_coords = {}
        group_size = 3
        for i, node in enumerate(nodes):
            group_idx = i // group_size
            base_x = group_idx * 100.0  # Far apart so each group is a cluster
            ca_coords[node.residue_id] = (
                base_x + float(i % group_size),
                float(i % group_size),
                0.0,
            )

        result = generate_seeds_from_gnn(
            gnn_nodes=nodes,
            ca_coords=ca_coords,
            uncertainty_threshold=0.0,  # All nodes qualify
            cone_depth_threshold=0.0,
            eps_angstrom=8.0,
            min_cluster_size=group_size,
            max_clusters=max_clusters,
        )

        # Cap enforcement: never more than max_clusters
        assert len(result) <= max_clusters

        # Score ordering: descending by composite_score
        for i in range(len(result) - 1):
            assert result[i].composite_score >= result[i + 1].composite_score, (
                f"Score ordering violated at index {i}: "
                f"{result[i].composite_score} < {result[i + 1].composite_score}"
            )

        # If we have results, verify the cap was applied correctly when there
        # are more potential clusters than max_clusters
        if len(result) == max_clusters:
            # Run without cap to see all clusters
            all_clusters = generate_seeds_from_gnn(
                gnn_nodes=nodes,
                ca_coords=ca_coords,
                uncertainty_threshold=0.0,
                cone_depth_threshold=0.0,
                eps_angstrom=8.0,
                min_cluster_size=group_size,
                max_clusters=999,
            )
            if len(all_clusters) > max_clusters:
                # The retained clusters should have scores >= discarded ones
                min_retained_score = result[-1].composite_score
                discarded = all_clusters[max_clusters:]
                for d in discarded:
                    assert min_retained_score >= d.composite_score, (
                        f"Retained cluster score {min_retained_score} < "
                        f"discarded cluster score {d.composite_score}"
                    )
