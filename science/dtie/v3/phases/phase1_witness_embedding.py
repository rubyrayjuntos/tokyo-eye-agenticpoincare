"""
phase1_witness_embedding.py
===========================
Eidetix Bio — DTIE Pipeline v3.0

Phase 1: Hyperbolic State Ingestion & Witness Selection.

Selects landmark points via k-means density sampling on dehydron positions
(NO_Midpoints) and embeds them into Poincaré Ball space to preserve
allosteric hierarchy.

## CONTRACT

### Reads
- ingestion_gdp.npz             <- written by ingestion.py
  - no_midpoints   : [N, 3]     dehydron positions (k-means input)
  - residue_ids    : [N]        residue identifiers
- gnn_output.npz                <- written by gnn_runner.py
  - curvature_c    : scalar     learned Poincaré ball curvature

### Writes
- phase1_output (returned as Phase1Output dataclass)
  - witnesses              : [N, 3]        all NO_Midpoints
  - landmarks              : [K, 3]        Poincaré-embedded landmark centroids
  - landmarks_euclidean    : [K, 3]        Euclidean landmark centroids
  - landmark_indices       : [N]           cluster assignment per residue
  - landmark_to_residue_map: Dict[int, str] landmark -> nearest residue id
  - mapping_function       : str           "exp_map_origin_poincare"
  - curvature_c            : float         from gnn_output.npz
  - residue_ids            : List[str]     from ingestion

### GNN fields used directly
  - curvature_c

### What this phase adds
  - k-means landmark selection on dehydron positions (NO_Midpoints)
  - Poincaré ball embedding via exponential map at origin
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

try:
    from .contracts import Phase1Output
except ImportError:
    from contracts import Phase1Output

# DEPRECATED: CALIBRATED_C is no longer used. curvature_c is sourced from
# gnn_output.npz (extracted from the GNN checkpoint via softplus(log_c) + 1e-4).
# Kept only for backward compatibility with any external callers.
CALIBRATED_C = 1.0

logger_p1 = logging.getLogger("DTIE_Phase1")


def map_to_poincare(x: np.ndarray, c: float) -> np.ndarray:
    """
    Exponential map at the origin: Euclidean tangent space → Poincaré Ball.
    Implements: exp_0(v) = tanh(√c · ‖v‖) / (√c · ‖v‖) · v

    Must match the curvature c used during GOSP-GNN training.
    """
    if c <= 0:
        logger_p1.warning("c <= 0 detected — falling back to Euclidean (c→0 limit).")
        return x.copy()

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("map_to_poincare expects shape (N, D)")

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    mask = norm > 1e-15

    scale = np.zeros_like(norm)
    scale[mask] = np.tanh(np.sqrt(c) * norm[mask]) / (np.sqrt(c) * norm[mask])

    poincare = scale * x
    poincare[~mask.flatten()] = x[~mask.flatten()]

    final_norm = np.linalg.norm(poincare, axis=1)
    if np.any(final_norm >= 0.999):
        logger_p1.warning("Points near Poincaré ball boundary — consider increasing c.")

    logger_p1.info(f"map_to_poincare: {x.shape} → {poincare.shape} | c={c}")
    return poincare


def compute_nearest(landmarks: np.ndarray, witnesses: np.ndarray) -> list:
    """
    Full sorted nearest landmark table for gudhi.WitnessComplex.

    Returns list of length n_witnesses. Each entry is a sorted list of
    (landmark_id: int, distance: float) tuples, ascending by distance.
    GUDHI requires ALL landmarks ranked per witness.
    """
    if len(landmarks) == 0 or len(witnesses) == 0:
        raise ValueError("Landmarks and witnesses cannot be empty")

    logger_p1.info(
        f"compute_nearest: {len(witnesses)} witnesses × {len(landmarks)} landmarks"
    )

    dist_matrix = cdist(witnesses, landmarks)  # (n_witnesses, n_landmarks)

    nearest_table = []
    for i in range(len(witnesses)):
        sorted_indices = np.argsort(dist_matrix[i])
        row = [(int(j), float(dist_matrix[i, j])) for j in sorted_indices]
        nearest_table.append(row)

    logger_p1.info(
        f"compute_nearest complete — ({len(nearest_table)}, {len(landmarks)})"
    )
    return nearest_table


def execute_phase_1_witness_embedding(
    ingestion_data: dict,
    gnn_output: dict,
    n_landmarks: int = 30,
) -> Phase1Output:
    """
    Phase 1: Hyperbolic State Ingestion & Witness Selection.

    Selects landmark witness points via k-means density sampling on
    NO_Midpoints (dehydron positions) and embeds them into Poincaré Ball
    space using curvature_c from the GNN checkpoint.

    Parameters
    ----------
    ingestion_data : dict
        Output from ingest_pdb() for the GDP condition. Must contain
        'no_midpoints' [N, 3] and 'residue_ids' [N].
    gnn_output : dict
        Output from run_gnn(). Must contain 'curvature_c' (scalar).
    n_landmarks : int
        Number of k-means clusters (landmark count). Default 30.

    Returns
    -------
    Phase1Output with witnesses as NO_Midpoints, landmarks in Poincaré space,
    and curvature_c from the GNN checkpoint.
    """
    no_midpoints = np.asarray(ingestion_data["no_midpoints"], dtype=np.float64)
    residue_ids = list(ingestion_data["residue_ids"])
    curvature_c = float(gnn_output["curvature_c"])

    n = len(no_midpoints)
    logger_p1.info(
        f"Phase 1: {n} dehydron positions → {n_landmarks} landmarks | c={curvature_c}"
    )

    # Clamp n_landmarks to available points
    actual_landmarks = min(n_landmarks, n)
    if actual_landmarks < n_landmarks:
        logger_p1.warning(
            f"Only {n} points available, reducing landmarks from {n_landmarks} to {actual_landmarks}"
        )

    # k-means on NO_Midpoints (dehydron positions, NOT Cα or all-atom)
    kmeans = KMeans(
        n_clusters=actual_landmarks, random_state=42, n_init="auto"
    ).fit(no_midpoints)
    landmarks_euclidean = kmeans.cluster_centers_  # [K, 3]

    # Embed landmarks into Poincaré ball using GNN-learned curvature
    landmarks_poincare = map_to_poincare(landmarks_euclidean, c=curvature_c)

    # Map each point to its nearest landmark's residue
    landmark_to_residue_map = {}
    for point_idx, landmark_idx in enumerate(kmeans.labels_):
        landmark_to_residue_map[int(landmark_idx)] = residue_ids[point_idx]

    logger_p1.info("Phase 1 complete.")
    return Phase1Output(
        witnesses=no_midpoints,
        landmarks=landmarks_poincare,
        landmarks_euclidean=landmarks_euclidean,
        landmark_indices=kmeans.labels_,
        landmark_to_residue_map=landmark_to_residue_map,
        mapping_function="exp_map_origin_poincare",
        curvature_c=curvature_c,
        residue_ids=residue_ids,
    )
