"""
phase3_witness_persistence.py
=============================
Eidetix Bio — DTIE Pipeline v3.0

Phase 3: Witness Complex & Persistence Extraction.

## CONTRACT

### Reads
- phase1_output (Phase1Output dataclass)  <- returned by phase1_witness_embedding.py
  - witnesses            : [N, 3]   NO_Midpoints (dehydron positions)
  - landmarks_euclidean  : [K, 3]   Euclidean k-means centroids
- phase2_output (Phase2Output dataclass)  <- returned by phase2_vulnerability_scan.py
  - doorways             : List[Doorway]

### Writes
- Phase3Output dataclass
  - barcode              : List[Tuple]   full persistence barcode
  - terminal_leaks       : List[Tuple]   infinite H1 bars
  - simplex_tree         : gudhi.SimplexTree
  - h1_edges             : List[List[int]]  birth edges for infinite H1
  - generators           : None
  - witness_mask         : List[int]     indices of witnesses used
  - landmarks            : [K, 3]       Euclidean landmarks (passthrough)

### GNN fields used directly
  - (none — reads Phase 1 output which already consumed curvature_c)

### What this phase adds
  - Witness complex construction on NO_Midpoint dehydron positions
  - H1 persistence extraction for allosteric leak detection
  - max_alpha=20.0 (Angstrom scale for dehydron inter-distances)
"""

from __future__ import annotations

import logging
from typing import Dict

import gudhi
from gudhi import WitnessComplex

try:
    from .contracts import Phase1Output, Phase2Output, Phase3Output
    from .phase1_witness_embedding import compute_nearest
except ImportError:
    from contracts import Phase1Output, Phase2Output, Phase3Output
    from phase1_witness_embedding import compute_nearest

logger_p3 = logging.getLogger("DTIE_Phase3")

# Default max_alpha for witness complex construction.
# 20.0 Å is appropriate for dehydron NO_Midpoint inter-distances
# (typical range 3–15 Å between backbone H-bond midpoints).
DEFAULT_MAX_ALPHA = 20.0


def execute_phase_3_witness_persistence(
    phase_input: "Phase3Input",
) -> "Phase3Output":
    """
    Phase 3: Witness Complex & Persistence Extraction.

    Uses NO_Midpoints (dehydron positions) from Phase 1 as witnesses and
    Euclidean landmark centroids for the witness complex. max_alpha is set
    to 20.0 Å (Angstrom scale for dehydron distances).

    The core TDA concepts used here are:
    1.  Witness Complex: A computationally efficient approximation of the
        true shape of the protein's point cloud data. It is built from a
        small subset of "landmark" points (from Phase 1) and a larger set
        of "witness" points (the NO_Midpoints).
    2.  Persistence: A method for identifying topological features (like
        holes) that are "real" and not just artifacts of noise. It measures
        the range of distance scales (alpha values) over which a feature
        persists. Long-lived features (like the "terminal leaks" we are
        looking for) are considered significant.
    3.  limit_dimension=2: This parameter is critical for performance. It
        tells GUDHI to only build simplices up to dimension 2 (triangles),
        which is sufficient for finding 1-dimensional holes (H1 homology).
    4.  max_alpha=20.0: Angstrom-scale cutoff for dehydron distances.
        Previous value of 1.0 was calibrated for Poincaré-ball coordinates;
        NO_Midpoints live in Angstrom space where inter-dehydron distances
        are typically 3–15 Å.

    Requires GUDHI >= 3.6.
    """
    phase1_result = phase_input.phase1_result
    phase2_result = phase_input.phase2_result
    max_alpha = phase_input.max_alpha
    assert tuple(int(x) for x in gudhi.__version__.split(".")[:2]) >= (
        3,
        6,
    ), "Phase 3 requires GUDHI >= 3.6"

    # Witnesses are NO_Midpoints (dehydron positions) from Phase 1.
    # These are Angstrom-scale coordinates, NOT all-atom or Cα coords.
    lm_euclidean = phase1_result.landmarks_euclidean
    witnesses = phase1_result.witnesses  # NO_Midpoints from ingestion

    # Use ALL witnesses: with limit_dimension=2 the full set is fast
    # and scoping to doorways breaks cycle formation.
    doorway_witness_mask = list(range(len(witnesses)))
    scoped_witnesses = witnesses

    logger_p3.info(
        f"Phase 3: {len(scoped_witnesses)} witnesses (NO_Midpoints) | "
        f"{len(lm_euclidean)} landmarks | max_alpha_sq={max_alpha}"
    )

    w_complex = WitnessComplex(
        nearest_landmark_table=compute_nearest(lm_euclidean, scoped_witnesses)
    )
    # limit_dimension=2 keeps only 0/1/2-simplices — sufficient for H0+H1 and
    # avoids exponential blowup. flag_persistence_generators() segfaults with
    # limit_dimension=2; use persistence_pairs() instead.
    simplex_tree = w_complex.create_simplex_tree(
        max_alpha_square=max_alpha, limit_dimension=2
    )

    simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    persistence = simplex_tree.persistence()

    terminal_leaks = [p for p in persistence if p[0] == 1 and p[1][1] == float("inf")]

    # persistence_pairs() gives (birth_simplex, death_simplex) tuples.
    # Infinite H1: birth = 1-simplex [v0, v1], death = [] (empty).
    pairs = simplex_tree.persistence_pairs()
    h1_pairs = [(b, d) for b, d in pairs if len(b) == 2 and len(d) == 0]

    # Indexed edge list — one entry per infinite H1 bar, parallel to terminal_leaks.
    h1_edges = [list(birth_simplex) for birth_simplex, _ in h1_pairs]

    logger_p3.info(
        f"Phase 3: {len(terminal_leaks)} terminal leaks | "
        f"barcode length={len(persistence)}"
    )
    return Phase3Output(
        barcode=persistence,
        terminal_leaks=terminal_leaks,
        simplex_tree=simplex_tree,
        h1_edges=h1_edges,
        generators=None,
        witness_mask=doorway_witness_mask,
        landmarks=lm_euclidean,
    )
