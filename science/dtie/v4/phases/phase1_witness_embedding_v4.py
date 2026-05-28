"""
phase1_witness_embedding_v4.py
==============================
Eidetix Bio — DTIE Pipeline v4

Phase 1: Hyperbolic Witness Selection.

v4 CHANGE vs v3:
  v3 selected landmarks by k-means on NO_Midpoint Cartesian coordinates
  (Ångström space) and incorrectly applied the exponential map to raw
  3D coordinates, producing |p|=1.192 for all points (ball boundary).

  v4 selects landmarks directly from the GNN's Poincaré ball output
  (x_routed_hyp, shape [N, hidden]), gated by the uncertainty
  decomposition: landmarks are residues where
    epistemic_uncertainty < median  (model is confident about this position)
    aleatoric_uncertainty > median  (the biology is genuinely open here)

  This intersection identifies sites that are:
    - Well-represented in the GNN's training distribution
    - Genuinely under-constrained biologically
  These are the sites worth running TDA on.

  Witness distances in Phase 3 are now hyperbolic (not Euclidean),
  computed from the same x_routed_hyp positions.

## CONTRACT

### Reads
- gnn_output (dict from gnn_runner.py v4):
  - gdp_x_routed_hyp : [N, hidden]  Poincaré ball positions (NEW in v4)
  - gdp_epistemic    : [N]          epistemic uncertainty per residue
  - gdp_aleatoric    : [N]          aleatoric uncertainty per residue
  - gdp_residue_ids  : [N]          residue identifiers (passthrough)
  - gdp_no_midpoints : [N, 3]       physical coordinates (for Phase 3.5 lift)
  - curvature_c      : scalar       learned Poincaré ball curvature

### Writes
- Phase1Output dataclass:
  - witnesses              : [N, hidden]   x_routed_hyp (all residues in ball)
  - witnesses_physical     : [N, 3]        NO_Midpoints (for Phase 3.5 lift back to 3D)
  - landmarks              : [K, hidden]   selected landmark positions in ball
  - landmarks_physical     : [K, 3]        physical coordinates of landmark residues
  - landmark_indices       : [K]           indices into witnesses array
  - landmark_to_residue_map: Dict[int, str]
  - curvature_c            : float
  - residue_ids            : List[str]
  - mapping_function       : str           "gnn_hyperbolic_uncertainty_gate"
  - n_witnesses            : int
  - n_landmarks            : int
  - landmark_selection_stats: dict         audit of uncertainty thresholds used

### Backward compatibility note:
  The Phase1Output dataclass field 'landmarks' now contains [K, hidden]
  hyperbolic positions, NOT [K, 3] Poincaré-embedded Cartesian centroids.
  Phase 3 must use hyperbolic distances from these positions.
  Phase 3.5 uses 'landmarks_physical' for the 3D lift (same as before).
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist

logger = logging.getLogger("DTIE_Phase1_v4")

try:
    from .contracts import Phase1Output
except ImportError:
    from contracts import Phase1Output


# ──────────────────────────────────────────────────────────────────────────────
# Hyperbolic distance utilities
# ──────────────────────────────────────────────────────────────────────────────

def hyperbolic_cross_distances(
    witnesses: np.ndarray,    # [N, D] — points in Poincaré ball
    landmarks: np.ndarray,    # [K, D] — points in Poincaré ball
    c: float,
) -> np.ndarray:
    """
    Compute [N, K] matrix of hyperbolic distances between witnesses and landmarks.

    Uses the Poincaré ball distance formula:
        d(x, y) = (1/√c) * arccosh(1 + 2c|x-y|² / ((1-c|x|²)(1-c|y|²)))

    All inputs must satisfy |p|² < 1/c. Points near the boundary are
    clamped to avoid numerical overflow in arccosh.

    This is the vectorized form — O(N*K*D) memory, which is manageable
    for typical protein sizes (N≤200, K≤50, D≤128).
    """
    if c <= 0:
        raise ValueError(f"Curvature c must be positive, got {c}")

    sq_w = np.sum(witnesses ** 2, axis=1)   # [N]
    sq_l = np.sum(landmarks ** 2, axis=1)   # [K]

    # Clamp to strictly inside the ball — prevents division by zero
    max_sq = 1.0 / c - 1e-6
    sq_w = np.clip(sq_w, 0.0, max_sq)
    sq_l = np.clip(sq_l, 0.0, max_sq)

    # |x - y|² for all (witness, landmark) pairs
    diff = witnesses[:, None, :] - landmarks[None, :, :]  # [N, K, D]
    sq_diff = np.sum(diff ** 2, axis=-1)                   # [N, K]

    # Denominators: (1 - c|x|²)(1 - c|y|²)
    denom_w = (1.0 - c * sq_w)[:, None]   # [N, 1]
    denom_l = (1.0 - c * sq_l)[None, :]   # [1, K]
    denom = denom_w * denom_l              # [N, K]
    denom = np.clip(denom, 1e-12, None)

    z = 1.0 + 2.0 * c * sq_diff / denom   # [N, K]
    z = np.clip(z, 1.0 + 1e-10, None)     # arccosh requires z >= 1

    dist = (1.0 / np.sqrt(c)) * np.arccosh(z)  # [N, K]
    return dist


def compute_nearest_hyperbolic(
    witnesses: np.ndarray,   # [N, D] all residue positions in ball
    landmarks: np.ndarray,   # [K, D] selected landmark positions in ball
    c: float,
) -> list:
    """
    Full sorted nearest landmark table for gudhi.WitnessComplex,
    using hyperbolic distances instead of Euclidean.

    Returns list of length N. Each entry is a sorted list of
    (landmark_id: int, distance: float) tuples, ascending by hyperbolic distance.
    GUDHI requires ALL landmarks ranked per witness.

    This replaces compute_nearest() from phase1_witness_embedding.py (v3),
    which used cdist (Euclidean) on NO_Midpoint Cartesian coordinates.
    Now distances reflect position in the GNN's learned conformational
    hierarchy, not physical Ångström distance.
    """
    if len(landmarks) == 0 or len(witnesses) == 0:
        raise ValueError("Landmarks and witnesses cannot be empty")

    logger.info(
        f"compute_nearest_hyperbolic: {len(witnesses)} witnesses × "
        f"{len(landmarks)} landmarks | c={c:.4f}"
    )

    dist_matrix = hyperbolic_cross_distances(witnesses, landmarks, c=c)  # [N, K]

    nearest_table = []
    for i in range(len(witnesses)):
        sorted_idx = np.argsort(dist_matrix[i])
        row = [(int(j), float(dist_matrix[i, j])) for j in sorted_idx]
        nearest_table.append(row)

    logger.info(
        f"compute_nearest_hyperbolic complete — "
        f"dist range [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]"
    )
    return nearest_table


# ──────────────────────────────────────────────────────────────────────────────
# Landmark selection
# ──────────────────────────────────────────────────────────────────────────────

def _select_landmarks_by_uncertainty(
    x_hyp: np.ndarray,          # [N, D] in Poincaré ball
    epistemic: np.ndarray,       # [N]
    aleatoric: np.ndarray,       # [N]
    n_landmarks: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Select landmark indices by uncertainty gating.

    Primary criterion: low_epistemic AND high_aleatoric
      (model confident AND biology genuinely open)

    Fallback: if fewer than n_landmarks residues meet the criterion,
    supplement with top-k by aleatoric uncertainty (the model at least
    knows these are genuine voids, even if it's less certain about position).

    Returns:
        landmark_indices: [K] indices into x_hyp
        landmark_positions: [K, D] ball positions of selected landmarks
        stats: audit dict with threshold values and selection counts
    """
    N = len(epistemic)
    epi_median = float(np.median(epistemic))
    ale_median = float(np.median(aleatoric))

    primary_mask = (epistemic < epi_median) & (aleatoric > ale_median)
    primary_indices = np.where(primary_mask)[0]

    stats = {
        "epistemic_median": epi_median,
        "aleatoric_median": ale_median,
        "primary_candidates": int(primary_mask.sum()),
        "n_requested": n_landmarks,
        "selection_method": "primary",
    }

    if len(primary_indices) >= n_landmarks:
        # Enough primary candidates — rank by aleatoric descending, take top-K
        ale_primary = aleatoric[primary_indices]
        rank_order = np.argsort(-ale_primary)
        selected = primary_indices[rank_order[:n_landmarks]]
        stats["selection_method"] = "primary_ranked_by_aleatoric"
    else:
        # Fallback: use all primary + supplement from remaining by aleatoric
        fallback_mask = ~primary_mask
        fallback_indices = np.where(fallback_mask)[0]
        n_needed = n_landmarks - len(primary_indices)

        if len(fallback_indices) > 0:
            ale_fallback = aleatoric[fallback_indices]
            fallback_rank = np.argsort(-ale_fallback)
            supplement = fallback_indices[fallback_rank[:n_needed]]
            selected = np.concatenate([primary_indices, supplement])
        else:
            selected = primary_indices

        stats["selection_method"] = "primary_plus_aleatoric_supplement"
        stats["supplement_count"] = max(0, len(selected) - len(primary_indices))

    stats["n_selected"] = int(len(selected))
    logger.info(
        f"Landmark selection: {stats['primary_candidates']} primary candidates → "
        f"{stats['n_selected']} selected ({stats['selection_method']})"
    )

    landmark_positions = x_hyp[selected]
    return selected, landmark_positions, stats


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def execute_phase_1_witness_embedding(
    ingestion_data: dict,
    gnn_output: dict,
    n_landmarks: int = 30,
    condition_prefix: str = "gdp_",
) -> "Phase1Output":
    """
    Phase 1 v4: Hyperbolic Witness Selection.

    Selects landmark residues from the GNN's Poincaré ball output
    using the uncertainty decomposition as a gate, then builds the
    hyperbolic nearest-landmark table for Phase 3.

    Parameters
    ----------
    ingestion_data : dict
        Output from ingest_pdb(). Must contain:
        'no_midpoints' [N, 3], 'residue_ids' [N].
    gnn_output : dict
        Output from gnn_runner.py v4. Must contain:
        '{prefix}x_routed_hyp' [N, hidden],
        '{prefix}epistemic' [N],
        '{prefix}aleatoric' [N],
        '{prefix}residue_ids' [N],
        'curvature_c' scalar.
    n_landmarks : int
        Number of landmark residues to select. Default 30.
    condition_prefix : str
        Key prefix for the condition ('gdp_' or 'gtp_').

    Returns
    -------
    Phase1Output with:
      witnesses: [N, hidden] — all residues in Poincaré ball
      landmarks: [K, hidden] — selected landmark positions in ball
      witnesses_physical: [N, 3] — NO_Midpoints (for Phase 3.5 lift)
      landmarks_physical: [K, 3] — physical coords of landmark residues
    """
    p = condition_prefix

    # Retrieve GNN hyperbolic embeddings
    x_routed_hyp_key = f"{p}x_routed_hyp"
    if x_routed_hyp_key not in gnn_output:
        raise KeyError(
            f"gnn_output missing '{x_routed_hyp_key}'. "
            f"Run gnn_runner.py with Gnnv4 (which exposes x_routed_hyp). "
            f"Available keys: {list(gnn_output.keys())}"
        )

    x_hyp = np.asarray(gnn_output[x_routed_hyp_key], dtype=np.float64)  # [N, hidden]
    epistemic = np.asarray(gnn_output[f"{p}epistemic"], dtype=np.float64).squeeze()
    aleatoric = np.asarray(gnn_output[f"{p}aleatoric"], dtype=np.float64).squeeze()
    residue_ids = list(gnn_output[f"{p}residue_ids"])
    curvature_c = float(gnn_output["curvature_c"])

    no_midpoints = np.asarray(ingestion_data["no_midpoints"], dtype=np.float64)

    N = len(x_hyp)
    n_landmarks_actual = min(n_landmarks, N)
    if n_landmarks_actual < n_landmarks:
        logger.warning(
            f"Only {N} residues available, reducing landmarks to {n_landmarks_actual}"
        )

    logger.info(
        f"Phase 1 v4: {N} residues in Poincaré ball | "
        f"c={curvature_c:.4f} | requesting {n_landmarks_actual} landmarks"
    )

    # Validate ball constraint
    norms = np.linalg.norm(x_hyp, axis=1)
    max_allowed = 1.0 / np.sqrt(curvature_c) - 1e-4
    if np.any(norms >= max_allowed):
        logger.warning(
            f"{(norms >= max_allowed).sum()} residues near/outside ball boundary. "
            f"Clamping to {max_allowed:.4f}."
        )
        # Clamp boundary violators radially
        violators = norms >= max_allowed
        x_hyp[violators] = (
            x_hyp[violators] / norms[violators, None] * (max_allowed - 1e-6)
        )

    # Select landmarks by uncertainty gate
    landmark_indices, landmark_positions, selection_stats = _select_landmarks_by_uncertainty(
        x_hyp, epistemic, aleatoric, n_landmarks=n_landmarks_actual
    )

    # Physical coordinates for Phase 3.5 lift
    # no_midpoints are indexed parallel to residue_ids / x_hyp
    landmarks_physical = no_midpoints[landmark_indices]

    # Landmark → residue_id map (for Phase 3.5)
    landmark_to_residue_map: Dict[int, str] = {
        int(i): str(residue_ids[landmark_indices[i]])
        for i in range(len(landmark_indices))
    }

    logger.info(
        f"Phase 1 v4 complete. "
        f"Landmarks: {len(landmark_indices)} | "
        f"Selection method: {selection_stats['selection_method']}"
    )

    # Build landmark_indices array parallel to witnesses (all residues)
    # Each residue → index of its nearest landmark in ball space
    # (used by Phase 3.5 for barycenter computation)
    dist_all_to_landmarks = hyperbolic_cross_distances(
        x_hyp, landmark_positions, c=curvature_c
    )  # [N, K]
    cluster_assignment = np.argmin(dist_all_to_landmarks, axis=1)  # [N]

    return Phase1Output(
        # Hyperbolic witnesses = all residues in Poincaré ball
        witnesses=x_hyp,                           # [N, hidden] — v4 semantic change
        landmarks=landmark_positions,              # [K, hidden] — in ball
        landmarks_euclidean=landmarks_physical,    # [K, 3] — physical for Phase 3.5
        landmark_indices=cluster_assignment,       # [N] nearest landmark assignment
        landmark_to_residue_map=landmark_to_residue_map,
        mapping_function="gnn_hyperbolic_uncertainty_gate",
        curvature_c=curvature_c,
        residue_ids=residue_ids,
        # Phase1Output may not have these fields in v3 — extend if needed:
        # witnesses_physical=no_midpoints,
        # landmarks_physical=landmarks_physical,
        # landmark_selection_stats=selection_stats,
        # landmark_original_indices=landmark_indices,
    )
