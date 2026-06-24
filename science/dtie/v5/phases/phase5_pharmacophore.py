"""V5-native Phase 5: Pharmacophore Identification.

Identifies druggable pockets from GNN embeddings + spatial data.
Instead of requiring fpocket binary + protein PDB on disk, this
uses the GNN's aleatoric uncertainty as a druggability proxy and
spatial clustering of vulnerable residues to define pocket centers.

Druggability signal: high aleatoric uncertainty at shallow depth
indicates a region where the protein's conformational flexibility
creates a potential binding pocket.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

from science.dtie.common.interfaces import GNNInferenceResult, PhaseResult

logger = logging.getLogger(__name__)


async def run_phase5_pharmacophore(
    db: Any,
    gnn_result: GNNInferenceResult,
    structure_id: str,
    phase4_result: PhaseResult | None = None,
    phase35_result: PhaseResult | None = None,
    candidate_percentile: float = 75.0,
    cluster_distance_angstrom: float = 8.0,
    min_cluster_size: int = 3,
    max_pockets: int | None = None,
) -> PhaseResult:
    """Identify pharmacophore centers from GNN embeddings.

    Druggability = aleatoric_uncertainty * (1 / cone_depth).
    High aleatoric at shallow depth = flexible exposed pocket.
    """
    nodes = gnn_result.nodes
    if not nodes:
        return PhaseResult(
            phase_name="phase5_pharmacophore",
            structure_id=structure_id,
            model_version="DTIE-v5-phase5",
            success=False,
            outputs={"error": "No nodes in GNN result"},
        )

    # Fetch Cα coordinates
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

    node_key_to_idx = {
        (n.chain_label, n.residue_index): i for i, n in enumerate(nodes)
    }
    ca_coords = np.full((len(nodes), 3), np.nan)
    for row in rows:
        key = (row["chain_label"], row["residue_index"])
        if key in node_key_to_idx:
            idx = node_key_to_idx[key]
            x, y, z = row["x"], row["y"], row["z"]
            if x is not None and y is not None and z is not None:
                ca_coords[idx] = [x, y, z]

    candidate_percentile = min(max(float(candidate_percentile), 50.0), 99.0)
    cluster_distance_angstrom = max(float(cluster_distance_angstrom), 1.0)
    min_cluster_size = max(int(min_cluster_size), 1)
    if max_pockets is not None:
        max_pockets = max(int(max_pockets), 1)

    # Compute druggability score per residue
    druggability = np.array([
        (n.aleatoric_uncertainty or 0.0) / max(n.cone_depth, 0.1)
        for n in nodes
    ])

    # Select top druggable residues above the requested percentile cutoff.
    threshold = float(np.percentile(druggability, candidate_percentile))
    candidate_mask = druggability >= threshold
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) < 2:
        return PhaseResult(
            phase_name="phase5_pharmacophore",
            structure_id=structure_id,
            model_version="DTIE-v5-phase5",
            success=True,
            outputs={
                "pharmacophore_count": 0,
                "pharmacophores": [],
                "note": "Insufficient druggable residues for clustering",
            },
        )

    # Filter candidates to only those with valid coordinates
    valid_candidates = [i for i in candidate_indices if not np.isnan(ca_coords[i]).any()]
    if len(valid_candidates) < 2:
        return PhaseResult(
            phase_name="phase5_pharmacophore",
            structure_id=structure_id,
            model_version="DTIE-v5-phase5",
            success=True,
            outputs={
                "pharmacophore_count": 0,
                "pharmacophores": [],
                "note": "Insufficient druggable residues with valid coordinates",
            },
        )
    candidate_indices = np.array(valid_candidates)

    # Spatial clustering of druggable residues.
    candidate_coords = ca_coords[candidate_indices]
    dists = squareform(pdist(candidate_coords))

    visited = set()
    clusters = []
    for i in range(len(candidate_coords)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, len(candidate_coords)):
            if j not in visited and dists[i, j] < cluster_distance_angstrom:
                cluster.append(j)
                visited.add(j)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    # Build pharmacophore records
    pharmacophores = []
    for ci, cluster in enumerate(clusters):
        global_indices = candidate_indices[cluster]
        cluster_coords = ca_coords[global_indices]
        center = cluster_coords.mean(axis=0)

        # Pocket druggability = mean druggability of constituent residues
        pocket_druggability = float(druggability[global_indices].mean())

        # Coupling strength from Phase 4 (if available)
        coupling = 0.0
        if phase4_result and phase4_result.success:
            pathways = phase4_result.outputs.get("pathways", [])
            for pw in pathways:
                if pw["source_residue"] in [nodes[gi].residue_index for gi in global_indices]:
                    coupling = max(coupling, pw["coupling_strength"])

        pharmacophores.append({
            "pocket_index": ci,
            "center_xyz": center.tolist(),
            "druggability_score": pocket_druggability,
            "residue_count": len(cluster),
            "residue_indices": [int(nodes[gi].residue_index) for gi in global_indices],
            "allosteric_coupling": coupling,
            "volume_estimate_A3": float(len(cluster)) * 15.0,  # ~15 ų per residue
        })

    pharmacophores.sort(key=lambda p: -p["druggability_score"])
    if max_pockets is not None:
        pharmacophores = pharmacophores[:max_pockets]

    logger.info(
        "Phase 5 v5: %d pharmacophores from %d candidates",
        len(pharmacophores), len(candidate_indices),
    )

    return PhaseResult(
        phase_name="phase5_pharmacophore",
        structure_id=structure_id,
        model_version="DTIE-v5-phase5",
        success=True,
        outputs={
            "pharmacophore_count": len(pharmacophores),
            "pharmacophores": pharmacophores,
            "druggability_threshold": threshold,
            "candidate_residues": len(candidate_indices),
            "parameters": {
                "candidate_percentile": candidate_percentile,
                "cluster_distance_angstrom": cluster_distance_angstrom,
                "min_cluster_size": min_cluster_size,
                "max_pockets": max_pockets,
            },
        },
    )
