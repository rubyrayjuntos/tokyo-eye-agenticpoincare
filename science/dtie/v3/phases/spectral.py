"""
Tokyo Eyes Stage 5 — Spectral Stabilization via Algebraic Connectivity
========================================================================

Computes the Laplacian matrix of the protein interaction graph and measures
how fragment binding shifts the algebraic connectivity (λ₂). A positive Δλ₂
means the fragment stabilizes the network — it couples a dynamic pocket to
the stable core, arresting conformational shifts.

The Fiedler vector (eigenvector of λ₂) identifies the optimal bisection of
the graph — the hinge regions separating protein domains.

Pipeline:
  1. Build weighted adjacency from Cα distances
  2. Compute Laplacian L = D - A
  3. Compute λ₂ (algebraic connectivity) and Fiedler vector
  4. Simulate fragment insertion at H1 pocket boundaries
  5. Recompute λ₂ → Δλ₂ = λ₂_after - λ₂_before
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist


@dataclass
class SpectralResult:
    """Spectral analysis result for a protein."""
    protein_id: str
    n_residues: int
    lambda_2: float                    # algebraic connectivity (baseline)
    fiedler_vector: np.ndarray         # eigenvector of λ₂
    hinge_residues: List[int]          # residues near the Fiedler zero-crossing
    # Top-spectrum fields for Purple Cage Hypothesis testing
    lambda_n_minus1: float = 0.0       # second-largest Laplacian eigenvalue
    lambda_n: float = 0.0              # largest Laplacian eigenvalue (spectral radius)
    spectral_gap_full: float = 0.0     # λ_{n-1} - λ₂ (total network stress range)
    # [N, 2] spectral embedding via eigenvectors 2 & 3 of Laplacian (Poincaré disk proxy)
    spectral_embedding: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    fragment_scores: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FragmentScore:
    """Score for a simulated fragment insertion at a pocket."""
    pocket_residues: List[int]
    persistence: float
    lambda_2_before: float
    lambda_2_after: float
    delta_lambda_2: float
    binding_affinity_proxy: float      # proxy ΔG from pocket geometry
    joint_score: float                 # 0.6 * (-ΔG) + 0.4 * Δλ₂
    fiedler_displacement: float        # how much the fragment shifts the Fiedler vector


def compute_top_spectrum(
    adj: np.ndarray,
    n_top: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the *largest* eigenvalues and eigenvectors of the graph Laplacian.

    Used to track λ_{n-1} drift — the Purple Cage Hypothesis predicts that
    the second-largest Laplacian eigenvalue drifts (drops) in G12D as wrapping
    erosion destabilises the most tightly coupled hub residues, *before* the
    conductance network visibly reorganises.

    Returns:
        eigenvalues: [n_top] largest eigenvalues, sorted descending
        eigenvectors: [N, n_top] corresponding eigenvectors
    """
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    laplacian = np.diag(degree) - adj
    L_sparse = csr_matrix(laplacian)
    k = min(n_top, n - 1)
    eigenvalues, eigenvectors = eigsh(L_sparse, k=k, which="LM")
    order = np.argsort(eigenvalues)[::-1]   # descending
    return eigenvalues[order], eigenvectors[:, order]


def compute_spectral_embedding(
    adj: np.ndarray,
    dims: int = 2,
) -> np.ndarray:
    """
    Build a 2D spectral embedding (eigenvectors 2 & 3 of the Laplacian),
    then map it onto the unit disk as a Poincaré proxy.

    Residues near the disk boundary (radius → 1) are *peripheral* in the
    network — weakly coupled to the core.  In a healthy WT structure these
    correspond to flexible loops; in G12D the Purple Cage Hypothesis predicts
    that dehydron residues drift outward as wrapping erodes.

    Returns:
        embedding: [N, 2] coordinates on the open unit disk
    """
    n = adj.shape[0]
    if n < dims + 2:
        return np.zeros((n, 2))

    degree = adj.sum(axis=1)
    laplacian = np.diag(degree) - adj
    L_sparse = csr_matrix(laplacian)
    k = min(dims + 2, n - 1)          # +2 to skip λ₁ = 0
    eigenvalues, eigenvectors = eigsh(L_sparse, k=k, which="SM")
    order = np.argsort(eigenvalues)
    # columns 1 and 2 are eigenvectors of λ₂ and λ₃
    emb = eigenvectors[:, order[1 : 1 + dims]]

    # Project onto unit disk: normalise so max radius = 0.99
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    max_norm = float(np.max(norms)) + 1e-12
    return emb / max_norm * 0.99


def build_protein_adjacency(
    ca_coords: np.ndarray,
    cutoff: float = 10.0,
    weight_decay: float = 2.0,
) -> np.ndarray:
    """
    Build weighted adjacency matrix from Cα coordinates.

    Edge weight = exp(-distance / weight_decay) for distances < cutoff.
    This gives stronger connections for closer residues.
    """
    n = len(ca_coords)
    dist = cdist(ca_coords, ca_coords)

    adj = np.zeros((n, n))
    mask = (dist < cutoff) & (dist > 0)
    adj[mask] = np.exp(-dist[mask] / weight_decay)

    return adj


def compute_laplacian_spectrum(
    adj: np.ndarray,
    n_eigenvalues: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the smallest eigenvalues and eigenvectors of the graph Laplacian.

    L = D - A where D is the degree matrix.

    Returns:
        eigenvalues: [k] smallest eigenvalues (λ₁=0, λ₂=algebraic connectivity)
        eigenvectors: [N, k] corresponding eigenvectors
    """
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    laplacian = np.diag(degree) - adj

    # Use sparse solver for efficiency
    L_sparse = csr_matrix(laplacian)
    k = min(n_eigenvalues, n - 1)

    eigenvalues, eigenvectors = eigsh(L_sparse, k=k, which="SM")

    # Sort by eigenvalue
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def find_hinge_residues(
    fiedler_vector: np.ndarray,
    n_hinge: int = 10,
) -> List[int]:
    """
    Find residues near the Fiedler vector zero-crossing.
    These are the hinge regions — weakest links between protein domains.
    """
    abs_fiedler = np.abs(fiedler_vector)
    return list(np.argsort(abs_fiedler)[:n_hinge])


def simulate_fragment_binding(
    adj: np.ndarray,
    pocket_residues: List[int],
    binding_strength: float = 2.0,
    cross_link_radius: int = 3,
) -> np.ndarray:
    """
    Simulate fragment binding by adding edges between pocket residues
    and their neighbors, representing the harmonic restraining terms
    introduced by a bound ligand.

    The fragment creates new, strong connections between residues that
    were previously loosely coupled — this is how a stabilizer works.
    """
    adj_new = adj.copy()
    n = adj.shape[0]

    for i in pocket_residues:
        # Strengthen connections between pocket residues
        for j in pocket_residues:
            if i != j:
                adj_new[i, j] = max(adj_new[i, j], binding_strength)
                adj_new[j, i] = max(adj_new[j, i], binding_strength)

        # Add cross-links to nearby residues (within sequence distance)
        for offset in range(-cross_link_radius, cross_link_radius + 1):
            j = i + offset
            if 0 <= j < n and j != i:
                adj_new[i, j] = max(adj_new[i, j], binding_strength * 0.5)
                adj_new[j, i] = max(adj_new[j, i], binding_strength * 0.5)

    return adj_new


def estimate_binding_affinity(
    ca_coords: np.ndarray,
    pocket_residues: List[int],
    persistence: float,
) -> float:
    """
    Proxy binding affinity (ΔG) from pocket geometry.

    Combines:
    - Pocket volume proxy (convex hull of pocket Cα coords)
    - Persistence (longer = more stable = better binding)
    - Pocket compactness (tighter = better shape complementarity)

    Returns a negative value (more negative = stronger binding).
    """
    if len(pocket_residues) < 2:
        return -1.0  # minimal binding

    pocket_coords = ca_coords[pocket_residues]

    # Volume proxy: product of principal component spreads
    centered = pocket_coords - pocket_coords.mean(axis=0)
    if centered.shape[0] >= 3:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        volume_proxy = np.prod(s[:3]) if len(s) >= 3 else np.prod(s)
    else:
        volume_proxy = np.std(pocket_coords) ** 3

    # Compactness: mean pairwise distance (lower = more compact)
    dists = cdist(pocket_coords, pocket_coords)
    compactness = 1.0 / (np.mean(dists[dists > 0]) + 1e-6)

    # Combine: ΔG proxy (more negative = stronger)
    # Scale factors chosen to produce values in [-10, -1] range (kcal/mol-like)
    dg = -(
        0.5 * np.log1p(volume_proxy) +
        2.0 * persistence +
        1.0 * compactness
    )
    return float(np.clip(dg, -15.0, -0.1))


class SpectralAnalyzer:
    """
    Stage 5: Compute algebraic connectivity and fragment stabilization scores.
    """

    def __init__(
        self,
        cutoff: float = 10.0,
        weight_decay: float = 2.0,
        binding_strength: float = 2.0,
        cross_link_radius: int = 3,
    ):
        self.cutoff = cutoff
        self.weight_decay = weight_decay
        self.binding_strength = binding_strength
        self.cross_link_radius = cross_link_radius

    def analyze(
        self,
        protein_id: str,
        ca_coords: np.ndarray,
        tda_lifted_sites: List[Dict[str, Any]],
    ) -> SpectralResult:
        """
        Run spectral analysis on a protein with TDA-identified pockets.

        Args:
            protein_id: protein identifier
            ca_coords: [N, 3] Cα coordinates
            tda_lifted_sites: lifted sites from Witness TDA (with residue_indices)
        """
        n = len(ca_coords)

        # Build adjacency and compute full spectrum (bottom + top)
        adj = build_protein_adjacency(ca_coords, self.cutoff, self.weight_decay)
        eigenvalues, eigenvectors = compute_laplacian_spectrum(adj)
        top_vals, _ = compute_top_spectrum(adj, n_top=3)
        embedding = compute_spectral_embedding(adj)

        lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        fiedler = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(n)
        hinges = find_hinge_residues(fiedler)
        lambda_n = float(top_vals[0]) if len(top_vals) > 0 else 0.0
        lambda_n_m1 = float(top_vals[1]) if len(top_vals) > 1 else 0.0
        spectral_gap_full = max(0.0, lambda_n_m1 - lambda_2)

        # Score each TDA pocket by simulated fragment binding
        fragment_scores = []
        for site in tda_lifted_sites:
            pocket_res = site.get("residue_indices", [])
            persistence = site.get("persistence", 0)
            if persistence == "inf":
                persistence = 10.0  # cap infinite persistence

            if not pocket_res or len(pocket_res) < 1:
                continue

            # Expand pocket to include sequence neighbors
            expanded = set(pocket_res)
            for r in pocket_res:
                for offset in range(-2, 3):
                    nr = r + offset
                    if 0 <= nr < n:
                        expanded.add(nr)
            expanded = sorted(expanded)

            # Simulate binding
            adj_bound = simulate_fragment_binding(
                adj, expanded, self.binding_strength, self.cross_link_radius
            )
            eig_bound, evec_bound = compute_laplacian_spectrum(adj_bound)
            lambda_2_after = float(eig_bound[1]) if len(eig_bound) > 1 else 0.0
            delta_lambda_2 = lambda_2_after - lambda_2

            # Binding affinity proxy
            dg = estimate_binding_affinity(ca_coords, expanded, persistence)

            # Joint score: 0.6 * (-ΔG) + 0.4 * Δλ₂ (from design doc)
            joint = 0.6 * (-dg) + 0.4 * delta_lambda_2

            # Fiedler displacement: how much the fragment shifts the hinge
            fiedler_after = evec_bound[:, 1] if evec_bound.shape[1] > 1 else np.zeros(n)
            fiedler_disp = float(np.mean(np.abs(fiedler_after - fiedler)))

            fragment_scores.append({
                "pocket_residues": pocket_res,
                "expanded_residues": expanded,
                "persistence": float(persistence),
                "lambda_2_before": lambda_2,
                "lambda_2_after": lambda_2_after,
                "delta_lambda_2": delta_lambda_2,
                "binding_affinity_proxy": dg,
                "joint_score": joint,
                "fiedler_displacement": fiedler_disp,
                "n_pocket_residues": len(expanded),
            })

        # Sort by joint score descending (best candidates first)
        fragment_scores.sort(key=lambda s: s["joint_score"], reverse=True)

        return SpectralResult(
            protein_id=protein_id,
            n_residues=n,
            lambda_2=lambda_2,
            fiedler_vector=fiedler,
            hinge_residues=hinges,
            lambda_n_minus1=lambda_n_m1,
            lambda_n=lambda_n,
            spectral_gap_full=spectral_gap_full,
            spectral_embedding=embedding,
            fragment_scores=fragment_scores,
        )
