"""
## CONTRACT

### Reads
- gnn_output.npz                   <- written by gnn_runner.py
  - {gdp,gtp}_cone_depth           : [N]      hyperbolic geodesic depth
  - {gdp,gtp}_ca_coords            : [N, 3]   Cα positions (for spectral adjacency)
  - {gdp,gtp}_no_midpoints         : [N, 3]   dehydron positions
  - {gdp,gtp}_aleatoric            : [N]      aleatoric uncertainty
  - {gdp,gtp}_epistemic            : [N]      epistemic uncertainty
  - curvature_c                    : scalar   learned curvature
- phase35_output                   <- written by phase35_topological_lift.py
  - lifted_sites                   : list of LiftedSite with barycenter_xyz
- phase2_output                    <- written by phase2_vulnerability_scan.py
  - doorways                       : list of Doorway objects

### Writes
- phase4_output (list of Phase4Output)
  - doorway_node, target, coupling_strength, r_eff, pathway, doorway_xyz
  - source_score, propagation_score, candidate_priority (supplementary)
  - spectral: SpectralAnalysis (lambda_2, fiedler_vector, hinge_residues,
              spectral_embedding, lambda_n_minus1, lambda_n, fragment_scores)

### GNN fields used directly
  - {gdp,gtp}_cone_depth  (resistance graph edge weights)
  - {gdp,gtp}_ca_coords   (spectral adjacency 10Å cutoff)

### What this phase adds
  - Effective resistance & allosteric conductance mapping
  - Spectral analysis (algebraic connectivity, Fiedler vector, hinge residues, spectral embedding)
  - Source-leak scoring as supplementary fields on each pathway
"""
from __future__ import annotations

import logging
import numpy as np
import numpy.linalg as nla
from typing import Dict, List, Optional

import networkx as nx
from scipy.spatial import KDTree

try:
    from .contracts import Phase2Output, Phase35Output, Phase4Output, SpectralAnalysis
except ImportError:
    from contracts import Phase2Output, Phase35Output, Phase4Output, SpectralAnalysis

try:
    from .spectral import (
        build_protein_adjacency,
        compute_laplacian_spectrum,
        find_hinge_residues,
        compute_spectral_embedding,
        compute_top_spectrum,
    )
except ImportError:
    from spectral import (
        build_protein_adjacency,
        compute_laplacian_spectrum,
        find_hinge_residues,
        compute_spectral_embedding,
        compute_top_spectrum,
    )

try:
    from .source_leak_scoring import (
        source_leak_score,
        propagation_score,
        candidate_priority,
    )
except ImportError:
    from source_leak_scoring import (
        source_leak_score,
        propagation_score,
        candidate_priority,
    )

logger_p4 = logging.getLogger("DTIE_Phase4")


def build_resistance_graph_from_gnn(
    ca_coords: np.ndarray,
    cone_depth: np.ndarray,
    cutoff: float = 8.0,
) -> nx.Graph:
    """
    Build a resistance graph from GNN cone_depth predictions.

    Edge weight = 1 / exp(cone_depth_i + cone_depth_j).
    Only connects nodes whose Cα distance is within cutoff.
    """
    n = len(ca_coords)
    G = nx.Graph()

    for i in range(n):
        G.add_node(i, xyz=ca_coords[i])

    tree = KDTree(ca_coords)
    pairs = tree.query_pairs(cutoff)

    for i, j in pairs:
        conductance = 1.0 / np.exp(cone_depth[i] + cone_depth[j])
        G.add_edge(i, j, conductance=conductance)

    return G


def _run_spectral_analysis(
    ca_coords: np.ndarray,
    tda_lifted_sites: list,
) -> SpectralAnalysis:
    """
    Run spectral analysis on the protein graph using 10Å Cα cutoff.

    Returns a SpectralAnalysis dataclass populated with lambda_2,
    fiedler_vector, hinge_residues, spectral_embedding, top spectrum,
    and fragment scores.
    """
    n = len(ca_coords)

    adj = build_protein_adjacency(ca_coords, cutoff=10.0, weight_decay=2.0)
    eigenvalues, eigenvectors = compute_laplacian_spectrum(adj)
    top_vals, _ = compute_top_spectrum(adj, n_top=3)
    embedding = compute_spectral_embedding(adj)

    lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    fiedler = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(n)
    hinges = find_hinge_residues(fiedler)
    lambda_n = float(top_vals[0]) if len(top_vals) > 0 else 0.0
    lambda_n_m1 = float(top_vals[1]) if len(top_vals) > 1 else 0.0

    # Build fragment scores from lifted sites
    fragment_scores = []
    for site in tda_lifted_sites:
        if hasattr(site, "cycle_vertices"):
            pocket_res = site.cycle_vertices
            persistence = site.birth if hasattr(site, "birth") else 0.0
        elif isinstance(site, dict):
            pocket_res = site.get("residue_indices", site.get("cycle_vertices", []))
            persistence = site.get("persistence", site.get("birth", 0.0))
        else:
            continue

        if persistence == "inf":
            persistence = 10.0

        if not pocket_res:
            continue

        fragment_scores.append({
            "pocket_residues": list(pocket_res),
            "persistence": float(persistence),
            "lambda_2": lambda_2,
        })

    return SpectralAnalysis(
        lambda_2=lambda_2,
        fiedler_vector=fiedler,
        hinge_residues=hinges,
        spectral_embedding=embedding,
        lambda_n_minus1=lambda_n_m1,
        lambda_n=lambda_n,
        fragment_scores=fragment_scores,
    )


def _compute_supplementary_scores(
    coupling_strength: float,
    r_eff: float,
) -> dict:
    """
    Compute source-leak scoring supplementary fields for a pathway.

    Uses coupling_strength and r_eff as proxies for the scoring inputs.
    """
    # Derive proxy inputs from the resistance/conductance values
    conductance_norm = min(1.0, coupling_strength / 10.0)
    depth_proxy = min(1.0, 1.0 / (r_eff + 1e-6))

    src_score = source_leak_score(
        genotype_persistence=depth_proxy,
        state_stability=conductance_norm,
        depth_persistence=depth_proxy,
        uncertainty_robustness=conductance_norm,
        leak_intensity=conductance_norm,
    )

    prop_score = propagation_score(
        conductance=conductance_norm,
        source_support=depth_proxy,
        path_coverage=conductance_norm,
        uncertainty_robustness=depth_proxy,
    )

    priority = candidate_priority(src_score, prop_score)

    return {
        "source_score": src_score,
        "propagation_score": prop_score,
        "candidate_priority": priority,
    }


def build_graph_spatial_index(
    hyperbolic_graph: nx.Graph,
    landmark_coords: Optional[Dict] = None,
):
    """Build KDTree index once; query many times in Phase 4 loop."""
    if landmark_coords is None:
        landmark_coords = {
            n: hyperbolic_graph.nodes[n].get("xyz", np.zeros(3))
            for n in hyperbolic_graph.nodes()
        }
    nodes = list(hyperbolic_graph.nodes())
    coords = np.array([landmark_coords.get(n, np.zeros(3)) for n in nodes])
    return nodes, KDTree(coords)


def find_nearest_graph_node(
    nodes: list,
    tree: KDTree,
    target_xyz: np.ndarray,
) -> int:
    """KD-tree lookup from Phase 3.5 barycenter_xyz → nearest graph node."""
    _, idx = tree.query(target_xyz)
    return nodes[idx]


def execute_phase_4_resistance_mapping(
    phase_input: "Phase4Input",
    gnn_output: Optional[dict] = None,
    condition_prefix: str = "gdp_",
) -> List["Phase4Output"]:
    """
    Phase 4: Effective Resistance & Allosteric Conductance Mapping.

    Treats the protein as a resistor network.
    Lower R_eff = stronger allosteric coupling between doorway and effector.

    When gnn_output is provided, builds the resistance graph from cone_depth
    (edge weight = 1/exp(cone_depth_i + cone_depth_j)) and runs spectral
    analysis on the 10Å Cα adjacency graph.

    Source-leak scoring functions are computed as supplementary fields on
    each pathway.
    """
    hyperbolic_graph = phase_input.hyperbolic_graph
    phase35_result = phase_input.phase35_result
    effector_sites = phase_input.effector_sites

    if gnn_output is None:
        raise ValueError("Phase 4 requires gnn_output to build the conductance graph")

    cone_depth_key = f"{condition_prefix}cone_depth"
    ca_coords_key = f"{condition_prefix}ca_coords"
    if cone_depth_key not in gnn_output or ca_coords_key not in gnn_output:
        raise KeyError(
            f"Phase 4 missing required GNN keys: {cone_depth_key} and/or {ca_coords_key}"
        )

    cone_depth = np.asarray(gnn_output[cone_depth_key])
    ca_coords = np.asarray(gnn_output[ca_coords_key])
    conductance_graph = build_resistance_graph_from_gnn(ca_coords, cone_depth)
    if conductance_graph.number_of_nodes() == 0:
        return []

    lifted_sites = (
        phase35_result.lifted_sites
        if hasattr(phase35_result, "lifted_sites")
        else phase35_result.get("lifted_sites", [])
    )
    spectral_result = _run_spectral_analysis(ca_coords, lifted_sites)
    logger_p4.info(
        f"Spectral analysis: λ₂={spectral_result.lambda_2:.4f}, "
        f"{len(spectral_result.hinge_residues)} hinge residues"
    )

    L = nx.laplacian_matrix(conductance_graph, weight="conductance").todense()
    L_pinv = nla.pinv(L)  # Moore-Penrose pseudoinverse

    # Build spatial index once from conductance graph nodes (xyz from gnn coords)
    nodes, tree = build_graph_spatial_index(conductance_graph, None)

    valid_targets = [t for t in effector_sites if t in conductance_graph.nodes]
    if not valid_targets:
        valid_targets = list(conductance_graph.nodes)[
            -min(10, conductance_graph.number_of_nodes()) :
        ]

    conductance_map = []

    for lifted in (
        phase35_result.lifted_sites
        if hasattr(phase35_result, "lifted_sites")
        else phase35_result.get("lifted_sites", [])
    ):
        barycenter = (
            lifted.barycenter_xyz
            if hasattr(lifted, "barycenter_xyz")
            else lifted.get("barycenter_xyz", np.zeros(3))
        )
        doorway_node = find_nearest_graph_node(nodes, tree, barycenter)

        for target in valid_targets:
            u, v = doorway_node, target
            if u >= L_pinv.shape[0] or v >= L_pinv.shape[0]:
                continue
            r_eff = float(L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v])
            if r_eff <= 0:
                continue

            try:
                pathway = nx.shortest_path(conductance_graph, u, v, weight="conductance")
            except nx.NetworkXNoPath:
                pathway = [u, v]

            # Compute supplementary source-leak scores
            supp = _compute_supplementary_scores(1.0 / r_eff, r_eff)

            conductance_map.append(
                Phase4Output(
                    doorway_node=u,
                    target=v,
                    coupling_strength=1.0 / r_eff,
                    r_eff=r_eff,
                    pathway=pathway,
                    doorway_xyz=(
                        barycenter.tolist()
                        if isinstance(barycenter, np.ndarray)
                        else barycenter
                    ),
                    source_score=supp["source_score"],
                    propagation_score=supp["propagation_score"],
                    candidate_priority=supp["candidate_priority"],
                    spectral=spectral_result,
                )
            )

    conductance_map.sort(key=lambda x: -x.coupling_strength)
    if conductance_map:
        logger_p4.info(
            f"Phase 4: {len(conductance_map)} allosteric pathways mapped | "
            f"top coupling = {conductance_map[0].coupling_strength:.4f}"
        )
    else:
        logger_p4.info("Phase 4: no pathways mapped.")
    return conductance_map
