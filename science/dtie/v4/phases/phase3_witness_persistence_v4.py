"""
phase3_witness_persistence_v4.py
=================================
Eidetix Bio — DTIE Pipeline v4

Phase 3: Witness Complex & Persistence in Hyperbolic Space.

v4 CHANGE vs v3:
  v3 computed the witness complex using Euclidean distances between
  NO_Midpoint Cartesian coordinates (Ångström space). max_alpha=20.0
  was calibrated for Ångström-scale inter-dehydron distances.

  v4 computes the witness complex using hyperbolic distances between
  GNN Poincaré ball positions (x_routed_hyp). The filtration parameter
  max_alpha is now in hyperbolic distance units.

  Why this matters:
    Euclidean TDA on NO_Midpoints finds topological features in physical
    space — geometric loops in the protein structure.

    Hyperbolic TDA on GNN ball positions finds topological features in
    the learned conformational hierarchy — loops in the wrapping landscape.
    A persistent H1 feature in this space means: a set of residues that
    are conformationally coupled at the same hierarchical depth, forming
    a cycle in the allostery graph that the GNN has learned.

    This is the allosteric channel identification that the original
    Tokyo Eyes design specified.

  max_alpha calibration:
    In hyperbolic space with learned curvature c ≈ 0.7, typical inter-residue
    hyperbolic distances range from 0.5 (nearby in hierarchy) to 6.0
    (opposite ends of the conformational funnel). A max_alpha of 3.0
    hyperbolic units captures the intermediate-range coupling that
    corresponds to allosteric communication distances.

    Compare with v3 max_alpha=20.0 Å, which captured the same range
    in physical space.

## CONTRACT

### Reads
- phase1_output (Phase1Output from phase1_witness_embedding_v4.py)
  - witnesses      : [N, hidden]  x_routed_hyp (all residues in ball)
  - landmarks      : [K, hidden]  selected landmark positions in ball
  - curvature_c    : float        learned curvature
- phase2_output (unchanged interface)
  - doorways       : List[Doorway]

### Writes
- Phase3Output (same interface as v3):
  - barcode        : List[Tuple]   full persistence barcode
  - terminal_leaks : List[Tuple]   infinite H1 bars
  - simplex_tree   : gudhi.SimplexTree
  - h1_edges       : List[List[int]] birth edges for infinite H1
  - generators     : None
  - witness_mask   : List[int]     all witness indices (no scoping)
  - landmarks      : [K, hidden]   hyperbolic landmark positions

### GNN fields used (via Phase 1 output)
  - x_routed_hyp → witnesses (hyperbolic positions)
  - curvature_c  → distance computation

### What this phase adds
  - Witness complex on hyperbolic distances (not Euclidean)
  - Persistent H1 features = allosteric channels in conformational hierarchy
  - max_alpha in hyperbolic distance units (not Ångströms)
"""

from __future__ import annotations

import logging
from typing import Dict

import gudhi
from gudhi import WitnessComplex

logger_p3 = logging.getLogger("DTIE_Phase3_v4")

try:
    from .contracts import Phase1Output, Phase2Output, Phase3Output
    from .phase1_witness_embedding_v4 import compute_nearest_hyperbolic
except ImportError:
    from contracts import Phase1Output, Phase2Output, Phase3Output
    from phase1_witness_embedding_v4 import compute_nearest_hyperbolic

# Default max_alpha in hyperbolic distance units.
#
# Calibration: with c ≈ 0.7, typical distances between residues in
# the same structural domain are ~0.5–2.0 hyperbolic units. Allosteric
# communication distances (domain-crossing coupling) are ~2.0–5.0.
# max_alpha = 4.0 captures domain-crossing topology without including
# the entire conformational funnel (which would fill with short bars).
#
# This is analogous to the 20.0 Å cutoff in v3, which captured
# inter-dehydron distances at the structural domain scale.
DEFAULT_MAX_ALPHA_HYPERBOLIC = 4.0


def execute_phase_3_witness_persistence(
    phase_input: "Phase3Input",
) -> "Phase3Output":
    """
    Phase 3 v4: Witness Complex & Persistence in Hyperbolic Space.

    Uses GNN Poincaré ball positions (from Phase 1 v4) as both witnesses
    and landmarks. Distances are hyperbolic geodesic distances, not
    Euclidean. The filtration is over hyperbolic distance scales.

    Persistent H1 features (infinite bars) identify cycles in the
    conformational hierarchy — allosteric channels at the hierarchical
    level where the GNN places them.

    Parameters
    ----------
    phase_input : Phase3Input
        - phase1_result: Phase1Output from phase1_witness_embedding_v4
          * witnesses: [N, hidden] hyperbolic positions
          * landmarks: [K, hidden] hyperbolic positions
          * curvature_c: float
        - phase2_result: Phase2Output (unchanged)
        - max_alpha: float (hyperbolic distance units, default 4.0)

    Returns
    -------
    Phase3Output — same interface as v3.
      landmarks field contains [K, hidden] hyperbolic positions.
      Phase 3.5 uses landmark_to_residue_map for the 3D lift, not
      the hyperbolic coordinates directly.

    Requires GUDHI >= 3.6.
    """
    assert tuple(int(x) for x in gudhi.__version__.split(".")[:2]) >= (3, 6), \
        "Phase 3 requires GUDHI >= 3.6"

    phase1_result = phase_input.phase1_result
    phase2_result = phase_input.phase2_result
    max_alpha = getattr(phase_input, "max_alpha", DEFAULT_MAX_ALPHA_HYPERBOLIC)

    # Witnesses = all residues in hyperbolic space
    witnesses = phase1_result.witnesses       # [N, hidden] in ball
    # Landmarks = uncertainty-selected subset
    landmarks = phase1_result.landmarks       # [K, hidden] in ball
    curvature_c = phase1_result.curvature_c

    # Validate that we have hyperbolic positions
    # v3 had witnesses [N, 3] (NO_Midpoints), v4 has [N, hidden]
    if witnesses.shape[1] == 3:
        raise ValueError(
            "Phase 3 v4 received [N, 3] witnesses — this is the v3 format (NO_Midpoints). "
            "Ensure Phase 1 v4 ran and produced hyperbolic ball positions. "
            "witnesses should be [N, hidden] with hidden >> 3."
        )

    logger_p3.info(
        f"Phase 3 v4: {len(witnesses)} witnesses [N={witnesses.shape[0]}, "
        f"D={witnesses.shape[1]}] | "
        f"{len(landmarks)} landmarks | "
        f"c={curvature_c:.4f} | max_alpha={max_alpha:.2f} (hyperbolic units)"
    )

    # Build hyperbolic nearest-landmark table.
    # This replaces compute_nearest() (Euclidean cdist on NO_Midpoints).
    # Each row: sorted list of (landmark_id, hyperbolic_distance) for one witness.
    nearest_table = compute_nearest_hyperbolic(
        witnesses=witnesses,
        landmarks=landmarks,
        c=curvature_c,
    )

    # Use all witnesses — no doorway scoping.
    # With hyperbolic geometry, doorway scoping would exclude residues
    # that are geometrically central in the hierarchy but not classified
    # as doorways. The hyperbolic filtration naturally de-emphasizes
    # wrapped residues (they appear at high distance from the landmark set).
    witness_mask = list(range(len(witnesses)))

    logger_p3.info(
        f"Building WitnessComplex: {len(witnesses)} witnesses, "
        f"{len(landmarks)} landmarks, "
        f"max_alpha_square={max_alpha} (hyperbolic units)"
    )

    w_complex = WitnessComplex(nearest_landmark_table=nearest_table)

    # limit_dimension=2: sufficient for H0 + H1 homology.
    # Avoids exponential blowup for high-dimensional point clouds.
    # max_alpha_square in hyperbolic distance units — note GUDHI's
    # WitnessComplex uses squared distances internally; the max_alpha
    # parameter passed here is the filtration value, not squared.
    simplex_tree = w_complex.create_simplex_tree(
        max_alpha_square=max_alpha,
        limit_dimension=2,
    )

    simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    persistence = simplex_tree.persistence()

    # Infinite H1 bars = topological loops that persist across all scales.
    # In the conformational hierarchy, these are allosteric channels:
    # cycles of residues that remain coupled regardless of how fine-grained
    # the hyperbolic filtration becomes.
    terminal_leaks = [
        p for p in persistence if p[0] == 1 and p[1][1] == float("inf")
    ]

    # Birth edges of persistent H1 bars.
    pairs = simplex_tree.persistence_pairs()
    h1_pairs = [(b, d) for b, d in pairs if len(b) == 2 and len(d) == 0]
    h1_edges = [list(birth_simplex) for birth_simplex, _ in h1_pairs]

    logger_p3.info(
        f"Phase 3 v4 complete: "
        f"{len(terminal_leaks)} terminal leaks (allosteric channels) | "
        f"barcode length={len(persistence)} | "
        f"H1 pairs={len(h1_pairs)}"
    )

    if len(terminal_leaks) == 0:
        logger_p3.warning(
            "No terminal leaks found. Consider reducing max_alpha "
            f"(current: {max_alpha:.2f} hyperbolic units) or increasing "
            "the number of landmarks."
        )

    return Phase3Output(
        barcode=persistence,
        terminal_leaks=terminal_leaks,
        simplex_tree=simplex_tree,
        h1_edges=h1_edges,
        generators=None,
        witness_mask=witness_mask,
        landmarks=landmarks,   # [K, hidden] hyperbolic positions
                               # Phase 3.5 uses landmark_to_residue_map for 3D lift
    )
