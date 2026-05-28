from __future__ import annotations

import logging
import numpy as np
from typing import Dict

from .contracts import Phase1Output, Phase3Output, Phase35Output, LiftedSite

logger_p35 = logging.getLogger("DTIE_Phase3.5")


def execute_phase_35_topological_lift(
    phase_input: "Phase35Input",
) -> "Phase35Output":
    """
    Phase 3.5: Topological Lift.

    This phase performs the "topological lift," which is the process of
    mapping the abstract topological features (the "terminal leaks" or
    infinite persistence bars) from Phase 3 back to concrete 3D coordinates
    in the protein structure.

    The key idea is that each terminal leak, which represents a persistent
    1-dimensional hole, has a "birth simplex" that is a 1-simplex (an edge)
    in the Witness Complex. This edge connects two landmark points. The
    midpoint of these two landmarks in 3D Euclidean space gives us a
    concrete location for the topological feature. This location is then
    used as the center of the pharmacophore in the subsequent phases of the
    pipeline.

    Bridges GUDHI persistence tuples to real 3D Euclidean coordinates.
    Uses persistence_pairs() birth-simplex edges (not flag_persistence_generators)
    to locate each terminal leak in 3D space.

    Each infinite H1 bar's birth edge [v0, v1] identifies two landmark
    centroids whose midpoint approximates the topological loop's location.
    """
    phase1_result = phase_input.phase1_result
    phase3_result = phase_input.phase3_result
    terminal_leaks = phase3_result.terminal_leaks
    h1_edges = phase3_result.h1_edges
    witnesses = phase1_result.witnesses
    landmark_indices = phase1_result.landmark_indices

    # Build landmark → Euclidean centroid map (from Euclidean atom coordinates)
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

        # Get the two landmark vertices of the birth edge (indexed, not birth-value keyed).
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
