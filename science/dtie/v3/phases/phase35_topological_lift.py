"""
phase35_topological_lift.py
===========================
Eidetix Bio — DTIE Pipeline v3.0

Phase 3.5: Topological Lift.

Bridges GUDHI persistence tuples to real 3D Euclidean coordinates by mapping
each terminal H1 leak's birth-edge landmarks back to NO_Midpoint centroids.

## CONTRACT

### Reads
- phase1_output (Phase1Output dataclass)  <- written by phase1_witness_embedding.py
  - witnesses          : [N, 3]   NO_Midpoints (dehydron positions)
  - landmark_indices   : [N]      cluster assignment per residue
- phase3_output (Phase3Output dataclass)  <- written by phase3_witness_persistence.py
  - terminal_leaks     : List     H1 persistence bars (dim, (birth, death))
  - h1_edges           : List     birth-simplex edge vertex pairs

### Writes
- Phase35Output (returned as dataclass)
  - lifted_sites       : List[LiftedSite]  3D-located persistence features
  - method             : str               "persistence_pairs_birth_edge"

### GNN fields used directly
  - (none — uses NO_Midpoints passed through Phase 1)

### What this phase adds
  - Maps abstract topological features (terminal H1 leaks) to 3D coordinates
  - Barycenters computed from NO_Midpoint landmark centroids (not Cα)
  - Each barycenter is the midpoint of the two landmark centroids identified
    by the birth-simplex edge of the persistence bar
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict

try:
    from .contracts import Phase1Output, Phase3Output, Phase35Output, LiftedSite
except ImportError:
    from contracts import Phase1Output, Phase3Output, Phase35Output, LiftedSite

logger_p35 = logging.getLogger("DTIE_Phase3.5")


def execute_phase_35_topological_lift(
    phase_input: "Phase35Input",
) -> "Phase35Output":
    """
    Phase 3.5: Topological Lift.

    Maps terminal H1 persistence leaks back to 3D coordinates using
    NO_Midpoint landmark centroids. Each leak's birth-simplex edge
    identifies two landmarks; the barycenter is the midpoint of their
    NO_Midpoint centroids.

    Coordinate anchor: barycenters are computed from NO_Midpoint landmark
    centroids (dehydron positions), NOT Cα coordinates.
    """
    phase1_result = phase_input.phase1_result
    phase3_result = phase_input.phase3_result
    terminal_leaks = phase3_result.terminal_leaks
    h1_edges = phase3_result.h1_edges
    witnesses = phase1_result.witnesses          # NO_Midpoints from Phase 1
    landmark_indices = phase1_result.landmark_indices

    # Build landmark → NO_Midpoint centroid map.
    # witnesses are NO_Midpoints (dehydron positions), so these centroids
    # are in dehydron coordinate space, not Cα space.
    landmark_coords: Dict[int, np.ndarray] = {}
    for k in np.unique(landmark_indices):
        mask = landmark_indices == k
        landmark_coords[k] = witnesses[mask].mean(axis=0)

    lifted_sites = []
    h1_idx = 0  # tracks position in h1_edges across H1 leaks

    for leak in terminal_leaks:
        dim, (birth, death) = leak
        if dim != 1:
            continue

        # Get the two landmark vertices of the birth edge
        edge_verts = h1_edges[h1_idx] if h1_idx < len(h1_edges) else None
        h1_idx += 1

        if edge_verts is None:
            logger_p35.warning(
                f"No generator edge for leak at birth={birth:.4f} — skipping."
            )
            continue

        coords_3d = np.array(
            [landmark_coords[v] for v in edge_verts if v in landmark_coords]
        )

        if len(coords_3d) == 0:
            continue

        # Barycenter: midpoint of NO_Midpoint landmark centroids
        barycenter = coords_3d.mean(axis=0)

        lifted_sites.append(
            LiftedSite(
                leak=leak,
                barycenter_xyz=barycenter,
                cycle_vertices=list(edge_verts),
                birth=birth,
                num_vertices=len(edge_verts),
                generator_simplices=[edge_verts],
            )
        )

    logger_p35.info(
        f"Phase 3.5: {len(lifted_sites)} persistence bars lifted to 3D coordinates"
    )
    return Phase35Output(
        lifted_sites=lifted_sites,
        method="persistence_pairs_birth_edge",
    )
