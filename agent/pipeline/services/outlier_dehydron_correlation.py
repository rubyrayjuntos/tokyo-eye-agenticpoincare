"""
Outlier-Dehydron Correlation Service

Maps validation outliers to nearby dehydrons using spatial search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial import cKDTree

from gosp.models.data_models import BondOutlier, Dehydron, OutlierDehydronCorrelation, StructureData


@dataclass
class CorrelationResult:
    """Correlation result and warnings."""
    correlations: List[OutlierDehydronCorrelation]
    warnings: List[str]


def build_atom_coord_map(structure: StructureData) -> Dict[str, np.ndarray]:
    """Build atom coordinate map keyed by chain:res_id:atom_name."""
    coords: Dict[str, np.ndarray] = {}
    for chain in structure.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                key = f"{atom.chain_id}:{atom.residue_id}:{atom.atom_name}"
                coords[key] = np.array([atom.x, atom.y, atom.z], dtype=float)
    return coords


def outlier_midpoint(outlier: BondOutlier, coords: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute midpoint for an outlier using atom coordinates."""
    atom1_key = f"{outlier.chain}:{outlier.residue_id}:{outlier.atoms[0]}"
    atom2_key = f"{outlier.chain}:{outlier.residue_id}:{outlier.atoms[1]}"
    if atom1_key not in coords or atom2_key not in coords:
        raise KeyError(f"Coordinates not found for outlier atoms {outlier.atoms}")
    return (coords[atom1_key] + coords[atom2_key]) / 2.0


def compute_correlation_score(
    outlier: BondOutlier,
    dehydron: Dehydron,
    distance: float,
    rho_threshold: int
) -> float:
    """Compute correlation score for outlier and dehydron."""
    if distance <= 0:
        distance = 0.1
    wrapping_deficit = max(0, rho_threshold - dehydron.wrapping_count)
    return (1.0 / distance) * wrapping_deficit * abs(outlier.z_score)


def correlate_outliers_to_dehydrons(
    outliers: List[BondOutlier],
    dehydrons: List[Dehydron],
    structure: StructureData,
    radius: float = 5.0,
    rho_threshold: int = 19,
    top_n: int = 3
) -> CorrelationResult:
    """Compute outlier-dehydron correlations within radius."""
    warnings: List[str] = []
    if not dehydrons:
        return CorrelationResult(correlations=[], warnings=["No dehydrons available for correlation"]) 

    coords = build_atom_coord_map(structure)
    dehydron_positions = np.array([d.midpoint for d in dehydrons], dtype=float)
    kdtree = cKDTree(dehydron_positions)

    all_correlations: List[OutlierDehydronCorrelation] = []

    for outlier in outliers:
        try:
            midpoint = outlier_midpoint(outlier, coords)
        except KeyError as exc:
            warnings.append(str(exc))
            continue

        indices = kdtree.query_ball_point(midpoint, radius)
        for idx in indices:
            dehydron = dehydrons[idx]
            distance = float(np.linalg.norm(midpoint - dehydron_positions[idx]))
            score = compute_correlation_score(outlier, dehydron, distance, rho_threshold)
            all_correlations.append(
                OutlierDehydronCorrelation(
                    outlier=outlier,
                    dehydron_id=idx,
                    distance=distance,
                    correlation_score=score,
                )
            )

    # Sort and keep top N per outlier
    all_correlations.sort(key=lambda c: c.correlation_score, reverse=True)
    filtered: List[OutlierDehydronCorrelation] = []
    counts: Dict[Tuple[str, int, Tuple[str, str]], int] = {}

    for corr in all_correlations:
        key = (corr.outlier.chain, corr.outlier.residue_id, corr.outlier.atoms)
        counts.setdefault(key, 0)
        if counts[key] < top_n:
            filtered.append(corr)
            counts[key] += 1

    return CorrelationResult(correlations=filtered, warnings=warnings)
