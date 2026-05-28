# Migrated from: SRC_DEM/backend/hyperbolic_graph_utils.py on 2026-05-27
"""
hyperbolic_graph_utils.py
=========================
Poincaré ball distance utilities used by the DTIE pipeline
and property-based tests.
"""

from __future__ import annotations

import numpy as np


def _as_2d(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def hyperbolic_dist0(points: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Hyperbolic distance from origin in the Poincare ball with curvature -c."""
    if c <= 0:
        raise ValueError("c must be positive")
    x = _as_2d(points)
    norms = np.linalg.norm(x, axis=1)
    clipped = np.clip(np.sqrt(c) * norms, 0.0, 1.0 - 1e-12)
    return (2.0 / np.sqrt(c)) * np.arctanh(clipped)


def hyperbolic_pairwise_distance(points: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Pairwise Poincare-ball distances for points with norm < 1/sqrt(c)."""
    if c <= 0:
        raise ValueError("c must be positive")
    x = _as_2d(points)
    sq_norm = np.sum(x * x, axis=1)
    max_sq = (1.0 / c) - 1e-10
    if np.any(sq_norm >= max_sq):
        raise ValueError("All points must lie strictly inside the Poincare ball")
    diff = x[:, None, :] - x[None, :, :]
    sq_diff = np.sum(diff * diff, axis=-1)
    denom = (1.0 - c * sq_norm)[:, None] * (1.0 - c * sq_norm)[None, :]
    z = 1.0 + (2.0 * c * sq_diff) / np.clip(denom, 1e-12, None)
    z = np.clip(z, 1.0, None)
    dist = (1.0 / np.sqrt(c)) * np.arccosh(z)
    np.fill_diagonal(dist, 0.0)
    return dist


from typing import Optional


def outer_shell_mask(depth: np.ndarray, threshold: float) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    return d >= float(threshold)


def build_rips_adjacency(
    dist_matrix: np.ndarray,
    epsilon: float,
    node_mask: Optional[np.ndarray] = None,
    max_neighbors: Optional[int] = None,
) -> np.ndarray:
    """Binary adjacency for a Rips-type graph with optional mask and degree cap."""
    d = np.asarray(dist_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("dist_matrix must be square")

    n = d.shape[0]
    if node_mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(node_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != n:
            raise ValueError("node_mask length must match dist_matrix size")

    adj = (d <= float(epsilon)).astype(np.int8)
    np.fill_diagonal(adj, 0)

    # Remove nodes outside shell.
    off = ~mask
    adj[off, :] = 0
    adj[:, off] = 0

    # Optional symmetric degree cap by nearest neighbors.
    if max_neighbors is not None and max_neighbors > 0:
        keep = np.zeros_like(adj, dtype=np.int8)
        for i in range(n):
            if not mask[i]:
                continue
            nbr_idx = np.where(adj[i] == 1)[0]
            if nbr_idx.size == 0:
                continue
            order = np.argsort(d[i, nbr_idx])
            chosen = nbr_idx[order[:max_neighbors]]
            keep[i, chosen] = 1
        adj = np.maximum(keep, keep.T)

    return adj
