"""Hyperbolic disc graph — edges and features from Poincaré geometry (not Cα Euclidean).

Uses ``hyperbolic_utils`` / ``hyperbolic_lorentz_ops`` for distances and tangent
log maps. The contact topology for structural-SSOT GNN passes is the k-NN graph
in hyperbolic distance on the frozen disc layout.
"""

from __future__ import annotations

import numpy as np

from science.dtie.common.hyperbolic_lorentz_ops import log_p, poincare_distance
from science.dtie.common.hyperbolic_utils import hyperbolic_dist0, hyperbolic_pairwise_distance


def cone_depth_from_disc(
    z_disc: np.ndarray,
    c: float,
) -> np.ndarray:
    """Hyperbolic geodesic distance from origin — authoritative burial depth on disc.

    Center = deep (wrapped core); rim = surface. Matches ``hyperbolic_dist0`` and
    the frontend ``hyperbolicRadiusFromOrigin`` convention.
    """
    return hyperbolic_dist0(z_disc, c=c)


def build_hyperbolic_disc_graph(
    z_disc: np.ndarray,
    c: float,
    *,
    k_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build bidirectional k-NN graph in hyperbolic distance on disc coordinates.

    Returns:
        edge_index: [2, E] int64
        edge_attr: [E, 4] float32 — (tangent_x, tangent_y, 0, hyperbolic_dist)
            Tangent components are ``log_p(z_j, z_i)`` in the tangent space at z_i
            (hyperbolic displacement direction for message passing).
    """
    z = np.asarray(z_disc, dtype=np.float64)
    n = z.shape[0]
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    k = min(int(k_neighbors), n - 1)
    dist_mat = hyperbolic_pairwise_distance(z, c=c)

    sources: list[int] = []
    targets: list[int] = []
    attrs: list[list[float]] = []

    for i in range(n):
        order = np.argsort(dist_mat[i])
        neighbors = [int(j) for j in order if j != i][:k]
        for j in neighbors:
            hyp_d = float(dist_mat[i, j])
            tangent = log_p(z[j], z[i], c)
            tx, ty = float(tangent[0]), float(tangent[1])
            sources.extend([i, j])
            targets.extend([j, i])
            attrs.append([tx, ty, 0.0, hyp_d])
            # Reverse edge: tangent at j toward i
            tangent_ji = log_p(z[i], z[j], c)
            attrs.append([float(tangent_ji[0]), float(tangent_ji[1]), 0.0, hyp_d])

    edge_index = np.array([sources, targets], dtype=np.int64)
    edge_attr = np.array(attrs, dtype=np.float32)
    return edge_index, edge_attr


def assert_hyperbolic_ops_consistent(z_disc: np.ndarray, c: float, *, atol: float = 1e-8) -> None:
    """Validate hyperbolic_utils and hyperbolic_lorentz_ops agree on dist0 / pairwise."""
    z = np.asarray(z_disc, dtype=np.float64)
    d_utils = hyperbolic_dist0(z, c=c)
    for i in range(min(len(z), 20)):
        d_ops = poincare_distance(np.zeros_like(z[i]), z[i], c)
        if abs(d_utils[i] - d_ops) > atol:
            raise AssertionError(
                f"dist0 mismatch at {i}: utils={d_utils[i]}, lorentz_ops={d_ops}"
            )
    d_pair = hyperbolic_pairwise_distance(z, c=c)
    for i in range(min(n := len(z), 5)):
        for j in range(i + 1, min(n, i + 4)):
            d_ij = poincare_distance(z[i], z[j], c)
            if abs(d_pair[i, j] - d_ij) > atol:
                raise AssertionError(
                    f"pairwise mismatch ({i},{j}): utils={d_pair[i,j]}, lorentz={d_ij}"
                )
