"""V5-native Phase 3.5: Topological Lift.

Maps persistent H1 features from Phase 3 back to 3D coordinates
using the Cα positions stored in the governed layer.

Instead of requiring Phase 1 witnesses (NO_Midpoints), this uses
the graph builder's Cα coordinates directly. The Phase 3 barcodes
identify which landmark indices form persistent cycles — we map
those back to real 3D positions via the residue coordinate data.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from science.dtie.common.interfaces import GNNInferenceResult, PhaseResult

logger = logging.getLogger(__name__)


async def run_phase35_topological_lift(
    db: Any,
    gnn_result: GNNInferenceResult,
    structure_id: str,
    phase3_result: PhaseResult | None = None,
) -> PhaseResult:
    """Lift topological features to 3D coordinates.

    Uses Cα coordinates from the governed layer to place persistent
    H1 features (allosteric channels) in physical space.

    If Phase 3 didn't produce terminal leaks, this phase identifies
    spatially clustered high-uncertainty residues as candidate sites.
    """
    nodes = gnn_result.nodes
    if not nodes:
        return PhaseResult(
            phase_name="phase35_topological_lift",
            structure_id=structure_id,
            model_version="DTIE-v5-phase35",
            success=False,
            outputs={"error": "No nodes in GNN result"},
        )

    # Fetch Cα coordinates from governed layer
    rows = await db.fetch_all(
        """
        SELECT r.residue_index, c.chain_label, a.x, a.y, a.z
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN dim_structure s ON s.structure_id = c.structure_id
        LEFT JOIN dim_atom a ON a.residue_id = r.residue_id AND a.atom_name = 'CA'
        WHERE s.structure_id = :structure_id AND a.x IS NOT NULL
        ORDER BY c.chain_label, r.residue_index
        """,
        {"structure_id": structure_id},
    )

    # Build residue_index → 3D coordinate map (NaN-safe)
    coord_map: dict[tuple[str, int], np.ndarray] = {}
    for row in rows:
        x, y, z = row["x"], row["y"], row["z"]
        if x is None or y is None or z is None:
            continue
        key = (row["chain_label"], row["residue_index"])
        coord = np.array([x, y, z], dtype=np.float64)
        if not np.isnan(coord).any():
            coord_map[key] = coord

    # If Phase 3 produced h1_edges (persistent cycles), lift those
    lifted_sites = []
    if phase3_result and phase3_result.success:
        h1_edges = phase3_result.outputs.get("h1_edges", [])
        for edge_idx, edge in enumerate(h1_edges):
            coords = []
            for vertex_idx in edge:
                if vertex_idx < len(nodes):
                    node = nodes[vertex_idx]
                    key = (node.chain_label, node.residue_index)
                    if key in coord_map:
                        coords.append(coord_map[key])
            if coords:
                barycenter = np.mean(coords, axis=0)
                lifted_sites.append({
                    "site_index": edge_idx,
                    "barycenter_xyz": barycenter.tolist(),
                    "vertex_count": len(edge),
                    "source": "phase3_h1_persistence",
                })

    # Fallback: cluster high-uncertainty residues spatially
    if not lifted_sites:
        # Use residues with above-median epistemic uncertainty
        epistemics = np.array([n.epistemic_uncertainty for n in nodes])
        threshold = float(np.percentile(epistemics, 75))

        candidate_coords = []
        candidate_indices = []
        for i, node in enumerate(nodes):
            if node.epistemic_uncertainty >= threshold:
                key = (node.chain_label, node.residue_index)
                if key in coord_map:
                    candidate_coords.append(coord_map[key])
                    candidate_indices.append(i)

        if candidate_coords:
            coords_arr = np.array(candidate_coords)
            # Simple spatial clustering: group within 10Å
            from scipy.spatial.distance import pdist, squareform
            if len(coords_arr) > 1:
                dists = squareform(pdist(coords_arr))
                visited = set()
                clusters = []
                for i in range(len(coords_arr)):
                    if i in visited:
                        continue
                    cluster = [i]
                    visited.add(i)
                    for j in range(i + 1, len(coords_arr)):
                        if j not in visited and dists[i, j] < 10.0:
                            cluster.append(j)
                            visited.add(j)
                    if len(cluster) >= 2:
                        clusters.append(cluster)

                for ci, cluster in enumerate(clusters):
                    cluster_coords = coords_arr[cluster]
                    barycenter = cluster_coords.mean(axis=0)
                    lifted_sites.append({
                        "site_index": ci,
                        "barycenter_xyz": barycenter.tolist(),
                        "vertex_count": len(cluster),
                        "source": "uncertainty_spatial_cluster",
                    })
            elif len(coords_arr) == 1:
                lifted_sites.append({
                    "site_index": 0,
                    "barycenter_xyz": coords_arr[0].tolist(),
                    "vertex_count": 1,
                    "source": "single_high_uncertainty",
                })

    logger.info(
        "Phase 3.5 v5: %d lifted sites for %s", len(lifted_sites), structure_id
    )

    return PhaseResult(
        phase_name="phase35_topological_lift",
        structure_id=structure_id,
        model_version="DTIE-v5-phase35",
        success=True,
        outputs={
            "lifted_site_count": len(lifted_sites),
            "lifted_sites": lifted_sites,
            "method": "v5_native_ca_lift",
            "coord_source": "dim_atom_ca",
        },
    )
