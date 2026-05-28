"""
Glueable Site Detection Service

Clusters dehydrons into candidate glue sites.
"""

from __future__ import annotations

from typing import List
import numpy as np
from scipy.spatial import cKDTree

from gosp.models.data_models import Dehydron, GlueableSite


def detect_glueable_sites(
    dehydrons: List[Dehydron],
    rho_threshold: int = 19,
    min_dehydrons: int = 3,
    cluster_radius: float = 10.0
) -> List[GlueableSite]:
    """Cluster dehydrons into glueable sites and rank them."""
    if not dehydrons:
        return []

    positions = np.array([d.midpoint for d in dehydrons], dtype=float)
    kdtree = cKDTree(positions)

    visited = set()
    sites: List[GlueableSite] = []
    site_id = 1

    for i, dehydron in enumerate(dehydrons):
        if i in visited:
            continue
        indices = kdtree.query_ball_point(dehydron.midpoint, cluster_radius)
        cluster = [idx for idx in indices if idx not in visited]
        if len(cluster) < min_dehydrons:
            visited.update(cluster)
            continue

        cluster_dehydrons = [dehydrons[idx] for idx in cluster]
        avg_rho = float(np.mean([d.wrapping_count for d in cluster_dehydrons]))
        void_volume = float(len(cluster_dehydrons) * 20.0)
        score = float((rho_threshold - avg_rho) * len(cluster_dehydrons))

        representative = cluster_dehydrons[0]
        representative_label = (
            f"{representative.donor_chain_id}{representative.donor_res_id}-"
            f"{representative.acceptor_chain_id}{representative.acceptor_res_id}"
        )

        sites.append(
            GlueableSite(
                site_id=site_id,
                dehydron_ids=cluster,
                avg_rho=avg_rho,
                void_volume=void_volume,
                score=score,
                representative_dehydron=representative_label,
            )
        )
        site_id += 1
        visited.update(cluster)

    return sorted(sites, key=lambda s: s.score, reverse=True)
