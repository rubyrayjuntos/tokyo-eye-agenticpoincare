"""Property-based tests for Poincaré disc topology.

Feature: poincare-visual-context
Properties: 1, 7
Validates: Requirements 1.1, 1.2, 4.1
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from agent.tools.disc_topology import (
    DiscTopologyResult,
    compute_disc_topology,
    poincare_distance,
)
from tests.strategies.disc_strategies import (
    positive_curvature,
    valid_disc_point,
    valid_disc_point_pair,
    valid_disc_point_triple,
    valid_disc_points,
)


# ---------------------------------------------------------------------------
# Property 7: Poincaré distance correctness
# Feature: poincare-visual-context, Property 7: Poincaré distance correctness
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------


class TestProperty7PoincareDistanceCorrectness:
    """Property 7: Poincaré distance correctness.

    For any two points on the Poincaré disc (within the boundary defined by
    curvature c), the computed distance SHALL satisfy:
    - d(p, p) = 0
    - d(p, q) = d(q, p)  (symmetry)
    - d(p, q) >= 0  (non-negativity)
    - d(p, r) <= d(p, q) + d(q, r)  (triangle inequality)
    """

    @settings(max_examples=200)
    @given(data=st.data(), curvature=positive_curvature())
    def test_identity(self, data: st.DataObject, curvature: float):
        """Feature: poincare-visual-context, Property 7: d(p, p) = 0

        **Validates: Requirements 4.1**
        """
        p = data.draw(valid_disc_point(curvature=curvature))
        d = poincare_distance(p, p, curvature)
        assert abs(d) < 1e-10, f"d(p, p) should be 0, got {d} for p={p}, c={curvature}"

    @settings(max_examples=200)
    @given(data=st.data(), curvature=positive_curvature())
    def test_symmetry(self, data: st.DataObject, curvature: float):
        """Feature: poincare-visual-context, Property 7: d(p, q) = d(q, p)

        **Validates: Requirements 4.1**
        """
        p = data.draw(valid_disc_point(curvature=curvature))
        q = data.draw(valid_disc_point(curvature=curvature))
        d_pq = poincare_distance(p, q, curvature)
        d_qp = poincare_distance(q, p, curvature)
        assert abs(d_pq - d_qp) < 1e-10, (
            f"Symmetry violated: d(p,q)={d_pq}, d(q,p)={d_qp}"
        )

    @settings(max_examples=200)
    @given(data=st.data(), curvature=positive_curvature())
    def test_non_negativity(self, data: st.DataObject, curvature: float):
        """Feature: poincare-visual-context, Property 7: d(p, q) >= 0

        **Validates: Requirements 4.1**
        """
        p = data.draw(valid_disc_point(curvature=curvature))
        q = data.draw(valid_disc_point(curvature=curvature))
        d = poincare_distance(p, q, curvature)
        assert d >= 0, f"Distance should be non-negative, got {d}"

    @settings(max_examples=200)
    @given(data=st.data(), curvature=positive_curvature())
    def test_triangle_inequality(self, data: st.DataObject, curvature: float):
        """Feature: poincare-visual-context, Property 7: d(p, r) <= d(p, q) + d(q, r)

        **Validates: Requirements 4.1**
        """
        p = data.draw(valid_disc_point(curvature=curvature))
        q = data.draw(valid_disc_point(curvature=curvature))
        r = data.draw(valid_disc_point(curvature=curvature))

        d_pr = poincare_distance(p, r, curvature)
        d_pq = poincare_distance(p, q, curvature)
        d_qr = poincare_distance(q, r, curvature)

        # Allow small numerical tolerance
        assert d_pr <= d_pq + d_qr + 1e-9, (
            f"Triangle inequality violated: d(p,r)={d_pr} > d(p,q)+d(q,r)={d_pq + d_qr}"
        )


# ---------------------------------------------------------------------------
# Property 1: Topology completeness
# Feature: poincare-visual-context, Property 1: Topology completeness
# Validates: Requirements 1.1, 1.2
# ---------------------------------------------------------------------------


class TestProperty1TopologyCompleteness:
    """Property 1: Topology completeness.

    For any set of valid 2D Poincaré disc coordinates (all within the disc
    boundary), the computed topology result SHALL contain:
    - A cluster assignment for every input residue
    - At least one cluster
    - Centroid angle and radius for each cluster
    - A hub residue for each cluster
    - Counts matching the input size
    """

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_topology_completeness(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 1: Topology completeness

        **Validates: Requirements 1.1, 1.2**
        """
        coordinates, curvature = data.draw(valid_disc_points(min_points=10, max_points=80))

        result = compute_disc_topology(
            coordinates,
            curvature_c=curvature,
            min_cluster_size=5,
            structure_id="test",
            run_id="run1",
        )

        n = len(coordinates)
        residue_ids_input = {coord[0] for coord in coordinates}

        # Total residues must match input
        assert result.total_residues == n, (
            f"total_residues={result.total_residues}, expected {n}"
        )

        # At least one cluster
        assert result.cluster_count >= 1, "Must have at least one cluster"
        assert len(result.clusters) == result.cluster_count

        # Every input residue must be assigned to exactly one cluster
        all_assigned: list[str] = []
        for cluster in result.clusters:
            all_assigned.extend(cluster.residue_ids)

        assert len(all_assigned) == n, (
            f"Assigned {len(all_assigned)} residues, expected {n}"
        )
        assert set(all_assigned) == residue_ids_input, "Not all residues assigned to clusters"

        # Each cluster has valid attributes
        for cluster in result.clusters:
            assert cluster.size == len(cluster.residue_ids)
            assert cluster.size > 0
            assert 0.0 <= cluster.centroid_angle_deg < 360.0
            assert cluster.centroid_radius >= 0.0
            assert cluster.hub_residue_id in cluster.residue_ids
            assert cluster.angular_sector in (
                "N", "NE", "E", "SE", "S", "SW", "W", "NW"
            )

        # Radial density counts must sum to total
        density_sum = sum(result.radial_density.values())
        assert density_sum == n, (
            f"Radial density sum={density_sum}, expected {n}"
        )
        assert "core" in result.radial_density
        assert "mid" in result.radial_density
        assert "periphery" in result.radial_density


# ---------------------------------------------------------------------------
# Property 4: Neighborhood query completeness
# Feature: poincare-visual-context, Property 4: Neighborhood query completeness
# Validates: Requirements 2.1, 2.2
# ---------------------------------------------------------------------------


class TestProperty4NeighborhoodQueryCompleteness:
    """Property 4: Neighborhood query completeness.

    For any residue in a set of disc embeddings with at least k+1 residues,
    the neighborhood query SHALL return exactly k neighbors, each with a valid
    hyperbolic distance > 0, cluster membership, cone_depth, and
    epistemic_uncertainty values.
    """

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_neighborhood_completeness(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 4: Neighborhood query completeness

        **Validates: Requirements 2.1, 2.2**
        """
        from agent.tools.disc_topology import compute_disc_neighborhood

        k = data.draw(st.integers(min_value=3, max_value=8))
        # Need at least k+1 points so the target has k neighbors
        coordinates, curvature = data.draw(
            valid_disc_points(min_points=max(10, k + 2), max_points=60)
        )

        # Compute topology first
        topology = compute_disc_topology(
            coordinates, curvature_c=curvature, min_cluster_size=3,
            structure_id="test", run_id="run1",
        )

        # Pick a random target residue
        target_idx = data.draw(st.integers(min_value=0, max_value=len(coordinates) - 1))
        target_id = coordinates[target_idx][0]

        result = compute_disc_neighborhood(
            target_id, coordinates, topology, curvature_c=curvature, k=k,
        )

        # Must return exactly k neighbors (since n > k)
        assert len(result.neighbors) == k, (
            f"Expected {k} neighbors, got {len(result.neighbors)}"
        )

        # Target residue ID must match
        assert result.target_residue_id == target_id

        # Each neighbor must have valid fields
        for neighbor in result.neighbors:
            # Distance must be non-negative (0 possible for coincident points)
            assert neighbor.hyperbolic_distance >= 0, (
                f"Neighbor {neighbor.residue_id} has distance {neighbor.hyperbolic_distance}"
            )
            # Cluster ID is int or None
            assert neighbor.cluster_id is None or isinstance(neighbor.cluster_id, int)
            # Cone depth must be non-negative and <= 1.0 (normalized)
            assert 0.0 <= neighbor.cone_depth <= 1.0 + 1e-9, (
                f"cone_depth={neighbor.cone_depth} out of [0, 1] range"
            )
            # Epistemic uncertainty must be non-negative
            assert neighbor.epistemic_uncertainty >= 0.0

        # Neighbors must be sorted by distance (nearest first)
        distances = [n.hyperbolic_distance for n in result.neighbors]
        assert distances == sorted(distances), "Neighbors not sorted by distance"

        # No neighbor should be the target itself
        neighbor_ids = [n.residue_id for n in result.neighbors]
        assert target_id not in neighbor_ids, "Target should not appear in its own neighbors"

        # All neighbor IDs should be from the input coordinates
        input_ids = {coord[0] for coord in coordinates}
        for nid in neighbor_ids:
            assert nid in input_ids, f"Neighbor {nid} not in input coordinates"


# ---------------------------------------------------------------------------
# Property 5: Topology annotation correctness
# Feature: poincare-visual-context, Property 5: Topology annotation correctness
# Validates: Requirements 2.3, 2.4
# ---------------------------------------------------------------------------


class TestProperty5TopologyAnnotationCorrectness:
    """Property 5: Topology annotation correctness.

    For any residue that qualifies as a hub (highest disc-degree in its cluster)
    or peripheral (radial distance > threshold), the neighborhood result SHALL
    include the appropriate annotation (is_hub=True or is_peripheral=True) with
    the correct cluster or nearest-cluster reference.
    """

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_hub_annotation(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 5: Hub annotation correctness

        **Validates: Requirements 2.3**
        """
        from agent.tools.disc_topology import compute_disc_neighborhood

        coordinates, curvature = data.draw(
            valid_disc_points(min_points=15, max_points=60)
        )

        topology = compute_disc_topology(
            coordinates, curvature_c=curvature, min_cluster_size=3,
            structure_id="test", run_id="run1",
        )

        # Test that hub residues are correctly annotated
        for cluster in topology.clusters:
            hub_id = cluster.hub_residue_id
            result = compute_disc_neighborhood(
                hub_id, coordinates, topology, curvature_c=curvature, k=5,
            )
            # The hub of a cluster must be annotated as is_hub=True
            assert result.is_hub is True, (
                f"Hub residue {hub_id} of cluster {cluster.cluster_id} "
                f"not annotated as hub"
            )
            # Hub must belong to its cluster
            assert result.target_cluster_id == cluster.cluster_id

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_peripheral_annotation(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 5: Peripheral annotation correctness

        **Validates: Requirements 2.4**
        """
        from agent.tools.disc_topology import compute_disc_neighborhood

        coordinates, curvature = data.draw(
            valid_disc_points(min_points=15, max_points=60)
        )

        topology = compute_disc_topology(
            coordinates, curvature_c=curvature, min_cluster_size=3,
            structure_id="test", run_id="run1",
        )

        disc_boundary = 1.0 / math.sqrt(curvature)
        peripheral_threshold = 0.85 * disc_boundary

        # Check every residue: if radius > threshold → is_peripheral must be True
        for coord in coordinates:
            rid, x, y = coord
            radius = math.sqrt(x * x + y * y)
            result = compute_disc_neighborhood(
                rid, coordinates, topology, curvature_c=curvature, k=5,
            )
            if radius > peripheral_threshold:
                assert result.is_peripheral is True, (
                    f"Residue {rid} at radius {radius:.3f} > threshold "
                    f"{peripheral_threshold:.3f} not marked peripheral"
                )
            else:
                assert result.is_peripheral is False, (
                    f"Residue {rid} at radius {radius:.3f} <= threshold "
                    f"{peripheral_threshold:.3f} incorrectly marked peripheral"
                )


# ---------------------------------------------------------------------------
# Property 9: Topology cache idempotence
# Feature: poincare-visual-context, Property 9: Topology cache idempotence
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------


class TestProperty9TopologyCacheIdempotence:
    """Property 9: Topology cache idempotence.

    For any (structure_id, run_id) pair, computing disc topology twice SHALL
    return identical results, and the second call SHALL be served from cache
    (not recomputed).
    """

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_cache_idempotence(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 9: Topology cache idempotence

        **Validates: Requirements 4.4**
        """
        from agent.tools.disc_topology_cache import TopologyCache

        coordinates, curvature = data.draw(valid_disc_points(min_points=10, max_points=60))

        structure_id = data.draw(st.text(min_size=1, max_size=10, alphabet="abcdefghijk0123456789"))
        run_id = data.draw(st.text(min_size=1, max_size=10, alphabet="abcdefghijk0123456789"))

        # Compute topology twice with same inputs
        result1 = compute_disc_topology(
            coordinates,
            curvature_c=curvature,
            min_cluster_size=5,
            structure_id=structure_id,
            run_id=run_id,
        )
        result2 = compute_disc_topology(
            coordinates,
            curvature_c=curvature,
            min_cluster_size=5,
            structure_id=structure_id,
            run_id=run_id,
        )

        # Results must be identical
        assert result1.structure_id == result2.structure_id
        assert result1.run_id == result2.run_id
        assert result1.curvature_c == result2.curvature_c
        assert result1.total_residues == result2.total_residues
        assert result1.cluster_count == result2.cluster_count
        assert len(result1.clusters) == len(result2.clusters)

        for c1, c2 in zip(result1.clusters, result2.clusters):
            assert c1.cluster_id == c2.cluster_id
            assert set(c1.residue_ids) == set(c2.residue_ids)
            assert c1.hub_residue_id == c2.hub_residue_id
            assert c1.angular_sector == c2.angular_sector
            assert abs(c1.centroid_angle_deg - c2.centroid_angle_deg) < 1e-10
            assert abs(c1.centroid_radius - c2.centroid_radius) < 1e-10

        assert set(result1.bridge_residues) == set(result2.bridge_residues)
        assert set(result1.peripheral_residues) == set(result2.peripheral_residues)
        assert result1.radial_density == result2.radial_density

        # Verify cache behavior: put first result, get should return it
        cache = TopologyCache(max_size=10)
        cache.put(result1)
        cached = cache.get(structure_id, run_id)

        assert cached is not None
        assert cached.structure_id == result1.structure_id
        assert cached.run_id == result1.run_id
        assert cached.cluster_count == result1.cluster_count
        assert cached.total_residues == result1.total_residues

        # Second put of same key should not change behavior
        cache.put(result2)
        cached_again = cache.get(structure_id, run_id)
        assert cached_again is not None
        assert cached_again.cluster_count == result2.cluster_count

    @settings(max_examples=100)
    @given(data=st.data())
    def test_cache_invalidation(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 9: Cache invalidation

        **Validates: Requirements 4.5**
        """
        from agent.tools.disc_topology_cache import TopologyCache

        coordinates, curvature = data.draw(valid_disc_points(min_points=10, max_points=40))

        structure_id = data.draw(st.text(min_size=1, max_size=8, alphabet="abcdefg"))
        run_id_1 = "run_001"
        run_id_2 = "run_002"

        result1 = compute_disc_topology(
            coordinates, curvature_c=curvature, min_cluster_size=5,
            structure_id=structure_id, run_id=run_id_1,
        )
        result2 = compute_disc_topology(
            coordinates, curvature_c=curvature, min_cluster_size=5,
            structure_id=structure_id, run_id=run_id_2,
        )

        cache = TopologyCache(max_size=10)
        cache.put(result1)
        cache.put(result2)

        # Both should be present
        assert cache.get(structure_id, run_id_1) is not None
        assert cache.get(structure_id, run_id_2) is not None

        # Invalidate the structure
        cache.invalidate(structure_id)

        # Both should be gone
        assert cache.get(structure_id, run_id_1) is None
        assert cache.get(structure_id, run_id_2) is None

    @settings(max_examples=50)
    @given(data=st.data())
    def test_cache_lru_eviction(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 9: LRU eviction

        **Validates: Requirements 4.4**
        """
        from agent.tools.disc_topology_cache import TopologyCache

        max_size = data.draw(st.integers(min_value=2, max_value=5))
        cache = TopologyCache(max_size=max_size)

        coordinates, curvature = data.draw(valid_disc_points(min_points=10, max_points=30))

        # Fill cache beyond capacity
        results = []
        for i in range(max_size + 2):
            result = compute_disc_topology(
                coordinates, curvature_c=curvature, min_cluster_size=5,
                structure_id=f"struct_{i}", run_id=f"run_{i}",
            )
            results.append(result)
            cache.put(result)

        # Cache should not exceed max_size
        assert len(cache) <= max_size

        # Most recent entries should be present
        assert cache.get(f"struct_{max_size + 1}", f"run_{max_size + 1}") is not None

        # Oldest entry should have been evicted
        assert cache.get("struct_0", "run_0") is None


# ---------------------------------------------------------------------------
# Property 8: Curvature parameterization
# Feature: poincare-visual-context, Property 8: Curvature parameterization
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------


class TestProperty8CurvatureParameterization:
    """Property 8: Curvature parameterization.

    For any pair of disc points and two different curvature values c1 != c2,
    the Poincaré distances computed with c1 and c2 SHALL differ (demonstrating
    that curvature affects computation).
    """

    @settings(max_examples=200)
    @given(data=st.data())
    def test_curvature_affects_distance(self, data: st.DataObject):
        """Feature: poincare-visual-context, Property 8: Curvature parameterization

        **Validates: Requirements 4.3**
        """
        # Generate two distinct curvature values
        c1 = data.draw(st.floats(min_value=0.1, max_value=2.0).filter(lambda x: x != 1.0))
        c2 = data.draw(
            st.floats(min_value=0.1, max_value=2.0).filter(lambda x: abs(x - c1) > 0.05)
        )

        # Use the smaller curvature's boundary to ensure both points are valid
        # for BOTH curvature values (points must be inside the disc for both).
        max_c = max(c1, c2)
        max_radius = 1.0 / math.sqrt(max_c) - 0.02  # stay well inside boundary

        # Generate two distinct non-origin points (if both are origin, distance is 0
        # regardless of curvature)
        angle_p = data.draw(st.floats(min_value=0.0, max_value=2 * math.pi))
        # Ensure the point is not too close to origin (distance would be ~0 for any c)
        radius_p = data.draw(st.floats(min_value=0.05, max_value=max_radius))
        p = (radius_p * math.cos(angle_p), radius_p * math.sin(angle_p))

        angle_q = data.draw(st.floats(min_value=0.0, max_value=2 * math.pi))
        radius_q = data.draw(st.floats(min_value=0.05, max_value=max_radius))
        q = (radius_q * math.cos(angle_q), radius_q * math.sin(angle_q))

        # Skip if points are essentially the same (distance 0 for any curvature)
        diff_sq = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
        if diff_sq < 1e-8:
            return  # trivially true, skip

        d1 = poincare_distance(p, q, c1)
        d2 = poincare_distance(p, q, c2)

        # Distances must differ when curvatures differ and points are distinct
        assert abs(d1 - d2) > 1e-12, (
            f"Curvature should affect distance: d(c1={c1})={d1}, d(c2={c2})={d2}, "
            f"p={p}, q={q}"
        )
