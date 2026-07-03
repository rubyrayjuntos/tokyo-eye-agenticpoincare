"""Hyperbolic motif clustering and persistence."""

from __future__ import annotations

import json
import math
import uuid
from typing import Any

import numpy as np

from data.normalizer.core import Normalizer
from science.dtie.common.normalizer_payloads import (
    HyperbolicMotifPayload,
    HyperbolicMotifRecord,
    ProvenanceContext,
    RunType,
    SourceType,
)


def compute_poincare_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Poincaré disc distances for 2D points."""
    n = len(points)
    norms_sq = np.sum(points**2, axis=1)
    mask = norms_sq >= 1.0
    if mask.any():
        scale = 0.99 / np.sqrt(norms_sq[mask])
        points[mask] = points[mask] * scale[:, None]
        norms_sq = np.sum(points**2, axis=1)

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff_sq = np.sum((points[i] - points[j]) ** 2)
            denom = (1.0 - norms_sq[i]) * (1.0 - norms_sq[j])
            if denom <= 0:
                denom = 1e-10
            arg = 1.0 + 2.0 * diff_sq / denom
            if arg < 1.0:
                arg = 1.0
            dist_matrix[i, j] = float(np.arccosh(arg))
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


def classify_angular_sector(angle_deg: float, radius: float) -> str:
    """Classify a point's position on the Poincaré disc into a sector."""
    if radius < 0.15:
        return "core"

    angle = angle_deg % 360.0
    if angle < 22.5 or angle >= 337.5:
        return "E"
    if angle < 67.5:
        return "NE"
    if angle < 112.5:
        return "N"
    if angle < 157.5:
        return "NW"
    if angle < 202.5:
        return "W"
    if angle < 247.5:
        return "SW"
    if angle < 292.5:
        return "S"
    return "SE"


def cluster_motifs(
    residue_ids: list[str],
    points: np.ndarray,
    *,
    run_id: str,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> list[dict[str, Any]]:
    """Cluster embedding points and return motif records."""
    dist_matrix = compute_poincare_distance_matrix(points)

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(dist_matrix)
    except ImportError:
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(
            eps=0.5,
            min_samples=min_samples,
            metric="precomputed",
        ).fit(dist_matrix)
        labels = clustering.labels_

    unique_labels = sorted(set(labels) - {-1})
    motifs: list[dict[str, Any]] = []

    for cluster_id in unique_labels:
        member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        member_residue_ids = [residue_ids[i] for i in member_indices]
        member_coords = points[member_indices]
        cluster_dists = dist_matrix[np.ix_(member_indices, member_indices)]
        medoid_local_idx = int(np.argmin(cluster_dists.sum(axis=1)))
        medoid_residue_id = member_residue_ids[medoid_local_idx]
        centroid_xy = member_coords.mean(axis=0)
        centroid_radius = float(np.linalg.norm(centroid_xy))
        centroid_angle_deg = float(math.degrees(math.atan2(centroid_xy[1], centroid_xy[0])))
        angular_sector = classify_angular_sector(centroid_angle_deg, centroid_radius)
        motifs.append(
            {
                "motif_id": f"motif_{run_id}_{cluster_id}",
                "cluster_id": cluster_id,
                "residue_ids": member_residue_ids,
                "medoid_residue_id": medoid_residue_id,
                "centroid_angle_deg": round(centroid_angle_deg, 2),
                "centroid_radius": round(centroid_radius, 4),
                "motif_size": len(member_residue_ids),
                "angular_sector": angular_sector,
                "classification": None,
            }
        )
    return motifs


async def persist_motifs(
    normalizer: Normalizer,
    db: Any,
    run_id: str,
    structure_id: str,
    motifs: list[dict[str, Any]],
    *,
    pipeline_name: str = "motif_analysis",
    parent_run_id: str | None = None,
) -> None:
    """Persist motif results to fact_hyperbolic_motif via Normalizer."""
    prov = ProvenanceContext(
        run_id=run_id,
        structure_id=structure_id,
        model_version="motif_analyzer_v1",
        checkpoint_uri=None,
        checkpoint_sha256=None,
        code_version=None,
        pipeline_name=pipeline_name,
        run_type=RunType.ANALYSIS,
        source_type=SourceType.DETERMINISTIC,
        parameters={"motif_count": len(motifs)},
        parent_run_id=parent_run_id,
    )
    payload = HyperbolicMotifPayload(
        provenance=prov,
        structure_id=structure_id,
        motifs=[
            HyperbolicMotifRecord(
                motif_id=motif["motif_id"],
                cluster_id=motif["cluster_id"],
                residue_ids=motif["residue_ids"],
                medoid_residue_id=motif["medoid_residue_id"],
                centroid_angle_deg=motif["centroid_angle_deg"],
                centroid_radius=motif["centroid_radius"],
                motif_size=motif["motif_size"],
                angular_sector=motif["angular_sector"],
                classification=motif.get("classification"),
            )
            for motif in motifs
        ],
    )
    await normalizer.normalize_hyperbolic_motifs(payload)


def parse_embedding_rows(embedding_rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    """Extract residue IDs and 2D coordinates from embedding query rows."""
    residue_ids: list[str] = []
    coords: list[list[float]] = []
    for row in embedding_rows:
        residue_ids.append(row["residue_id"])
        hyp = row["hyp_projections"]
        if isinstance(hyp, (list, tuple)) and len(hyp) >= 2:
            coords.append([float(hyp[0]), float(hyp[1])])
        elif isinstance(hyp, str):
            parsed = json.loads(hyp)
            coords.append([float(parsed[0]), float(parsed[1])])
        else:
            coords.append([0.0, 0.0])
    return residue_ids, np.array(coords, dtype=np.float64)


async def discover_and_persist_motifs(
    db: Any,
    structure_id: str,
    *,
    run_id: str | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    caller_identity: str = "compute_motifs",
    pipeline_name: str = "motif_analysis",
    parent_run_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Load embeddings, cluster motifs, persist, and return (run_id, motifs)."""
    embedding_rows = await db.fetch_all(
        """
        SELECT residue_id, hyp_projections
        FROM fact_gnn_node_embedding
        WHERE structure_id = :structure_id
          AND hyp_projections IS NOT NULL
        """,
        {"structure_id": structure_id},
    )
    if not embedding_rows:
        raise ValueError("No hyperbolic embeddings found")

    residue_ids, points = parse_embedding_rows(embedding_rows)
    effective_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    motifs = cluster_motifs(
        residue_ids,
        points,
        run_id=effective_run_id,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    normalizer = Normalizer(db=db, caller_identity=caller_identity)
    await persist_motifs(
        normalizer,
        db,
        effective_run_id,
        structure_id,
        motifs,
        pipeline_name=pipeline_name,
        parent_run_id=parent_run_id,
    )
    return effective_run_id, motifs
