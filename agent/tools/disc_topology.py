"""Disc topology computation for Poincaré visual context.

Computes spatial statistics from persisted hyp_projection_2d VECTOR(2) coordinates:
- True Poincaré (hyperbolic) distance matrix
- HDBSCAN clustering on the hyperbolic distance matrix
- Angular sector assignment for cluster centroids
- Hub residues (highest k-NN degree within cluster)
- Bridge residues (connected to residues in 2+ clusters)
- Peripheral residues (radial distance > 0.85 of disc boundary)
- Radial density profile (core / mid / periphery)

Feature: poincare-visual-context
Requirements: 1.1, 1.2, 4.1, 4.2, 4.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DiscCluster:
    """A cluster of residues on the Poincaré disc."""

    cluster_id: int
    residue_ids: list[str]
    centroid_angle_deg: float  # 0-360
    centroid_radius: float  # 0 to 1/sqrt(c)
    hub_residue_id: str  # highest k-NN degree in cluster
    size: int
    angular_sector: str = ""  # "N", "NE", "E", etc.


@dataclass
class DiscTopologyResult:
    """Full topology result for a set of disc coordinates."""

    structure_id: str
    run_id: str
    curvature_c: float
    total_residues: int
    cluster_count: int
    clusters: list[DiscCluster]
    bridge_residues: list[str]
    peripheral_residues: list[str]
    radial_density: dict[str, int] = field(default_factory=dict)  # "core"/"mid"/"periphery"


@dataclass
class NeighborInfo:
    """Information about a single neighbor on the disc."""

    residue_id: str
    hyperbolic_distance: float
    cluster_id: int | None
    cone_depth: float
    epistemic_uncertainty: float


@dataclass
class DiscNeighborhood:
    """Neighborhood query result for a target residue."""

    target_residue_id: str
    target_cluster_id: int | None
    is_hub: bool
    is_peripheral: bool
    neighbors: list[NeighborInfo]


# ---------------------------------------------------------------------------
# Angular sector mapping
# ---------------------------------------------------------------------------

_SECTOR_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def angle_to_sector(angle_deg: float) -> str:
    """Map an angle (0-360°, 0°=East, counterclockwise) to 8-direction sector."""
    # Normalize to [0, 360)
    angle_deg = angle_deg % 360.0
    # Each sector spans 45°, centered on its direction
    index = int((angle_deg + 22.5) / 45.0) % 8
    return _SECTOR_NAMES[index]


# ---------------------------------------------------------------------------
# Poincaré distance
# ---------------------------------------------------------------------------


def poincare_distance(
    p: tuple[float, float], q: tuple[float, float], c: float = 1.0
) -> float:
    """Compute geodesic distance on the Poincaré disc.

    Formula:
        d(p, q) = (1/√c) * arcosh(1 + 2c * ||p-q||² / ((1 - c*||p||²)(1 - c*||q||²)))

    Args:
        p, q: Points on the disc (x, y) with ||p|| < 1/√c.
        c: Curvature parameter (default 1.0). Must be > 0.

    Returns:
        Non-negative geodesic distance.
    """
    if c <= 0:
        raise ValueError(f"Curvature c must be positive, got {c}")

    px, py = p
    qx, qy = q

    norm_p_sq = px * px + py * py
    norm_q_sq = qx * qx + qy * qy
    diff_sq = (px - qx) ** 2 + (py - qy) ** 2

    denom_p = 1.0 - c * norm_p_sq
    denom_q = 1.0 - c * norm_q_sq

    # Clamp denominators to avoid division by zero for points at boundary
    denom_p = max(denom_p, 1e-15)
    denom_q = max(denom_q, 1e-15)

    arg = 1.0 + 2.0 * c * diff_sq / (denom_p * denom_q)

    # Clamp arg to >= 1.0 for numerical stability (arcosh domain)
    arg = max(arg, 1.0)

    dist = (1.0 / math.sqrt(c)) * math.acosh(arg)
    return dist


# ---------------------------------------------------------------------------
# Pairwise distance matrix
# ---------------------------------------------------------------------------


def compute_pairwise_distances(
    coordinates: list[tuple[str, float, float]], c: float = 1.0
) -> NDArray[np.float64]:
    """Compute pairwise Poincaré distance matrix.

    Args:
        coordinates: List of (residue_id, x, y) tuples.
        c: Curvature parameter.

    Returns:
        Symmetric N×N distance matrix (numpy array).
    """
    n = len(coordinates)
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        _, xi, yi = coordinates[i]
        for j in range(i + 1, n):
            _, xj, yj = coordinates[j]
            d = poincare_distance((xi, yi), (xj, yj), c)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


# ---------------------------------------------------------------------------
# Disc topology computation
# ---------------------------------------------------------------------------


def compute_disc_neighborhood(
    target_residue_id: str,
    coordinates: list[tuple[str, float, float]],
    topology: DiscTopologyResult,
    curvature_c: float,
    k: int = 8,
) -> DiscNeighborhood:
    """Get k-nearest neighbors on the disc for a specific residue.

    Finds the k closest residues by true Poincaré distance, annotates each
    with cluster membership, cone_depth, and epistemic_uncertainty.

    Args:
        target_residue_id: The residue to query neighborhood for.
        coordinates: List of (residue_id, x, y) tuples.
        topology: Pre-computed topology result for cluster lookups.
        curvature_c: Curvature parameter for Poincaré distance.
        k: Number of nearest neighbors to return (default 8).

    Returns:
        DiscNeighborhood with target annotations and neighbor list.

    Raises:
        ValueError: If target_residue_id is not found in coordinates.
    """
    residue_ids = [coord[0] for coord in coordinates]
    if target_residue_id not in residue_ids:
        raise ValueError(f"Residue {target_residue_id} not found in coordinates")

    n = len(coordinates)
    target_idx = residue_ids.index(target_residue_id)
    target_point = (coordinates[target_idx][1], coordinates[target_idx][2])

    # Compute distances from target to all other residues
    distances: list[tuple[int, float]] = []
    for i in range(n):
        if i == target_idx:
            continue
        other_point = (coordinates[i][1], coordinates[i][2])
        d = poincare_distance(target_point, other_point, curvature_c)
        distances.append((i, d))

    # Sort by distance, take k nearest
    distances.sort(key=lambda x: x[1])
    effective_k = min(k, len(distances))
    nearest = distances[:effective_k]

    # Build cluster lookup: residue_id → cluster_id
    cluster_lookup: dict[str, int | None] = {}
    for cluster in topology.clusters:
        for rid in cluster.residue_ids:
            cluster_lookup[rid] = cluster.cluster_id
    # Residues not in any cluster get None
    for rid in residue_ids:
        if rid not in cluster_lookup:
            cluster_lookup[rid] = None

    # Target cluster
    target_cluster_id = cluster_lookup.get(target_residue_id)

    # Determine if target is a hub: highest k-NN degree in its cluster
    # We approximate this by checking if the target is the hub_residue_id
    # recorded in its cluster from the topology computation
    is_hub = False
    if target_cluster_id is not None:
        for cluster in topology.clusters:
            if cluster.cluster_id == target_cluster_id:
                is_hub = cluster.hub_residue_id == target_residue_id
                break

    # Determine if target is peripheral: radial distance > 0.85 of boundary
    disc_boundary = 1.0 / math.sqrt(curvature_c)
    target_radius = math.sqrt(target_point[0] ** 2 + target_point[1] ** 2)
    peripheral_threshold = 0.85 * disc_boundary
    is_peripheral = target_radius > peripheral_threshold

    # Build neighbor info list
    neighbors: list[NeighborInfo] = []
    for idx, dist in nearest:
        neighbor_id = residue_ids[idx]
        neighbor_cluster = cluster_lookup.get(neighbor_id)

        # Cone depth: normalized radial distance (0=center, 1=boundary)
        neighbor_point = (coordinates[idx][1], coordinates[idx][2])
        neighbor_radius = math.sqrt(neighbor_point[0] ** 2 + neighbor_point[1] ** 2)
        cone_depth = neighbor_radius / disc_boundary if disc_boundary > 0 else 0.0

        # Epistemic uncertainty: proportional to distance from center
        # Points near the boundary have higher uncertainty (less well-embedded)
        epistemic_uncertainty = cone_depth ** 2  # quadratic scaling

        neighbors.append(
            NeighborInfo(
                residue_id=neighbor_id,
                hyperbolic_distance=dist,
                cluster_id=neighbor_cluster,
                cone_depth=cone_depth,
                epistemic_uncertainty=epistemic_uncertainty,
            )
        )

    return DiscNeighborhood(
        target_residue_id=target_residue_id,
        target_cluster_id=target_cluster_id,
        is_hub=is_hub,
        is_peripheral=is_peripheral,
        neighbors=neighbors,
    )


def compute_disc_topology(
    coordinates: list[tuple[str, float, float]],
    curvature_c: float,
    min_cluster_size: int = 5,
    structure_id: str = "",
    run_id: str = "",
) -> DiscTopologyResult:
    """Compute full disc topology from Poincaré coordinates.

    Uses true Poincaré distance for the distance matrix, then HDBSCAN
    for natural cluster discovery.

    Args:
        coordinates: List of (residue_id, x, y) tuples.
        curvature_c: Curvature parameter for Poincaré distance.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        structure_id: Structure identifier for result metadata.
        run_id: Run identifier for result metadata.

    Returns:
        DiscTopologyResult with clusters, bridges, peripherals, and density.
    """
    from sklearn.cluster import HDBSCAN

    n = len(coordinates)

    if n == 0:
        return DiscTopologyResult(
            structure_id=structure_id,
            run_id=run_id,
            curvature_c=curvature_c,
            total_residues=0,
            cluster_count=0,
            clusters=[],
            bridge_residues=[],
            peripheral_residues=[],
            radial_density={"core": 0, "mid": 0, "periphery": 0},
        )

    # Extract arrays
    residue_ids = [coord[0] for coord in coordinates]
    points = np.array([(coord[1], coord[2]) for coord in coordinates], dtype=np.float64)

    # Compute pairwise hyperbolic distance matrix
    dist_matrix = compute_pairwise_distances(coordinates, curvature_c)

    # HDBSCAN clustering on the precomputed distance matrix
    # Adjust min_cluster_size to not exceed half the dataset
    effective_min_cluster = min(min_cluster_size, max(2, n // 2))

    clusterer = HDBSCAN(
        min_cluster_size=effective_min_cluster,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Handle degenerate case: all noise → treat as single cluster
    unique_labels = set(labels)
    if unique_labels == {-1}:
        labels = np.zeros(n, dtype=int)

    # Assign noise points (-1) to nearest cluster
    cluster_label_set = sorted(set(labels) - {-1})
    if not cluster_label_set:
        cluster_label_set = [0]
        labels = np.zeros(n, dtype=int)

    noise_mask = labels == -1
    if noise_mask.any() and len(cluster_label_set) > 0:
        # For each noise point, find nearest non-noise point and adopt its label
        non_noise_indices = np.where(~noise_mask)[0]
        for i in np.where(noise_mask)[0]:
            if len(non_noise_indices) > 0:
                nearest_idx = non_noise_indices[np.argmin(dist_matrix[i, non_noise_indices])]
                labels[i] = labels[nearest_idx]

    # Recompute cluster labels after noise assignment
    cluster_label_set = sorted(set(labels))

    # Compute k-NN degree for hub identification (k=6 or n-1 if small)
    k_nn = min(6, n - 1)
    knn_degree = np.zeros(n, dtype=int)
    if k_nn > 0:
        for i in range(n):
            # Get k nearest neighbors (excluding self)
            distances_i = dist_matrix[i].copy()
            distances_i[i] = np.inf
            nearest_k = np.argsort(distances_i)[:k_nn]
            # Count how many times i appears as a neighbor of others
            knn_degree[i] = 0

        # Bidirectional k-NN degree: count edges in k-NN graph
        for i in range(n):
            distances_i = dist_matrix[i].copy()
            distances_i[i] = np.inf
            nearest_k = np.argsort(distances_i)[:k_nn]
            for j in nearest_k:
                knn_degree[i] += 1
                knn_degree[j] += 1

        # Divide by 2 to avoid double-counting for symmetric edges
        # Actually, we count directed edges, so degree = outgoing + incoming
        # The loop above counts each directed edge once for source (i) and once for target (j)
        # So knn_degree[i] = in-degree + out-degree in the directed k-NN graph

    # Build clusters
    disc_boundary = 1.0 / math.sqrt(curvature_c)
    clusters: list[DiscCluster] = []

    for cluster_id in cluster_label_set:
        member_mask = labels == cluster_id
        member_indices = np.where(member_mask)[0]
        member_ids = [residue_ids[idx] for idx in member_indices]
        member_points = points[member_indices]

        # Centroid (Euclidean mean of disc coordinates for angle/radius)
        centroid = member_points.mean(axis=0)
        centroid_angle = math.degrees(math.atan2(centroid[1], centroid[0])) % 360.0
        centroid_radius = float(np.sqrt(centroid[0] ** 2 + centroid[1] ** 2))

        # Hub: highest k-NN degree within this cluster
        cluster_degrees = knn_degree[member_indices]
        hub_local_idx = int(np.argmax(cluster_degrees))
        hub_residue_id = residue_ids[member_indices[hub_local_idx]]

        cluster = DiscCluster(
            cluster_id=int(cluster_id),
            residue_ids=member_ids,
            centroid_angle_deg=centroid_angle,
            centroid_radius=centroid_radius,
            hub_residue_id=hub_residue_id,
            size=len(member_ids),
            angular_sector=angle_to_sector(centroid_angle),
        )
        clusters.append(cluster)

    # Identify bridge residues: connected to residues in 2+ different clusters
    bridge_residues: list[str] = []
    if k_nn > 0:
        for i in range(n):
            distances_i = dist_matrix[i].copy()
            distances_i[i] = np.inf
            nearest_k = np.argsort(distances_i)[:k_nn]
            neighbor_clusters = {labels[j] for j in nearest_k}
            if len(neighbor_clusters) >= 2:
                bridge_residues.append(residue_ids[i])

    # Identify peripheral residues: radial distance > 0.85 of disc boundary
    peripheral_threshold = 0.85 * disc_boundary
    radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    peripheral_residues = [
        residue_ids[i] for i in range(n) if radii[i] > peripheral_threshold
    ]

    # Radial density profile
    radial_density = {"core": 0, "mid": 0, "periphery": 0}
    for r in radii:
        normalized_r = r / disc_boundary if disc_boundary > 0 else 0
        if normalized_r < 0.3:
            radial_density["core"] += 1
        elif normalized_r < 0.7:
            radial_density["mid"] += 1
        else:
            radial_density["periphery"] += 1

    return DiscTopologyResult(
        structure_id=structure_id,
        run_id=run_id,
        curvature_c=curvature_c,
        total_residues=n,
        cluster_count=len(clusters),
        clusters=clusters,
        bridge_residues=bridge_residues,
        peripheral_residues=peripheral_residues,
        radial_density=radial_density,
    )
