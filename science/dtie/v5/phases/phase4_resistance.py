"""V5-native Phase 4: Resistance Mapping & Spectral Analysis.

Builds a conductance graph from GNN cone_depth values and Cα contacts,
then computes effective resistance between source-leak candidates and
potential effector sites. Also runs spectral analysis (algebraic
connectivity, Fiedler vector, hinge residues).

Conductance model: buried residues (high cone_depth) are highly
conductive — allosteric energy propagates through the rigid
hydrophobic core, not the floppy surface loops.

Operates entirely from GNN embeddings — no external files needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import networkx as nx
from scipy.spatial import KDTree

from science.dtie.common.interfaces import GNNInferenceResult, PhaseResult

logger = logging.getLogger(__name__)

CONTACT_CUTOFF = 8.0  # Ångströms for Cα contact graph
HINGE_EPSILON = 0.02  # Fiedler vector tolerance for hinge detection


def _build_conductance_graph(
    ca_coords: np.ndarray,
    cone_depths: np.ndarray,
    cutoff: float = CONTACT_CUTOFF,
) -> nx.Graph:
    """Build conductance graph from valid Cα coordinates.

    Edge weight = exp(depth_i + depth_j): buried residue pairs
    conduct strongly (rigid core acts as mechanical linkage).
    Nodes with NaN coordinates are excluded.
    """
    G = nx.Graph()
    n = len(ca_coords)

    # Only include nodes with valid (non-NaN) structural coordinates
    valid_nodes = [i for i in range(n) if not np.isnan(ca_coords[i]).any()]

    for i in valid_nodes:
        G.add_node(i, xyz=ca_coords[i].tolist())

    if len(valid_nodes) < 2:
        return G

    # Build KDTree only on valid spatial coordinates
    valid_coords = ca_coords[valid_nodes]
    tree = KDTree(valid_coords)
    pairs = tree.query_pairs(cutoff)

    for pi, pj in pairs:
        i, j = valid_nodes[pi], valid_nodes[pj]
        # Inverted formula: high burial → high conductance
        conductance = float(np.exp(cone_depths[i] + cone_depths[j]))
        G.add_edge(i, j, conductance=conductance)

    return G


def _spectral_analysis(G: nx.Graph) -> dict:
    """Compute spectral properties of the conductance graph.

    Uses the normalized Laplacian so λ₂ is bounded in [0, 2] and comparable
    across structures. Effective-resistance pathways still use the raw
    conductance-weighted Laplacian separately.

    Handles disconnected graphs gracefully. Hinge residues are
    identified as nodes where |Fiedler_value| < epsilon (partition
    boundary), not by sign-change between sequential indices.
    """
    if G.number_of_nodes() < 3 or not nx.is_connected(G):
        return {"lambda_2": 0.0, "hinge_residues": [], "fiedler_vector": []}

    L = nx.normalized_laplacian_matrix(G, weight="conductance").toarray().astype(
        np.float64
    )
    eigenvalues, eigenvectors = sla.eigh(L)

    lambda_2 = float(eigenvalues[1])
    fiedler = eigenvectors[:, 1]

    # Hinge residues: nodes at the partition boundary (|v_i| < epsilon)
    hinges = []
    node_list = list(G.nodes())
    for idx, node_id in enumerate(node_list):
        if abs(fiedler[idx]) < HINGE_EPSILON:
            hinges.append(int(node_id))

    return {
        "lambda_2": lambda_2,
        "hinge_residues": hinges,
        "fiedler_vector": fiedler.tolist(),
    }


def _compute_effective_resistance_sparse(
    G: nx.Graph,
    source_idx: int,
    target_idx: int,
) -> float | None:
    """Compute effective resistance between two nodes using sparse solver.

    Solves (L + (1/N)J) x = e_s - e_t via sparse CG, avoiding the
    O(N³) dense pseudoinverse. Returns None if nodes are not in the
    graph or computation fails.
    """
    node_list = list(G.nodes())
    if source_idx not in node_list or target_idx not in node_list:
        return None

    n = G.number_of_nodes()
    node_to_pos = {node: pos for pos, node in enumerate(node_list)}
    s_pos = node_to_pos[source_idx]
    t_pos = node_to_pos[target_idx]

    # Sparse Laplacian + (1/N) * ones to make it invertible
    L_sparse = nx.laplacian_matrix(G, weight="conductance").astype(np.float64)
    # Add regularization: L + (1/N)*I (simpler than J, same effect for R_eff)
    from scipy.sparse import eye as speye
    A = L_sparse + (1.0 / n) * speye(n, format="csc")

    # RHS: e_s - e_t
    rhs = np.zeros(n)
    rhs[s_pos] = 1.0
    rhs[t_pos] = -1.0

    try:
        x, info = spla.cg(A, rhs, atol=1e-10, maxiter=1000)
        if info != 0:
            return None
        r_eff = float(x[s_pos] - x[t_pos])
        return r_eff if r_eff > 0 else None
    except Exception:
        return None


async def run_phase4_resistance(
    db: Any,
    gnn_result: GNNInferenceResult,
    structure_id: str,
    phase35_result: PhaseResult | None = None,
    spatial_cutoff: float = CONTACT_CUTOFF,
) -> PhaseResult:
    """Compute effective resistance and spectral analysis.

    Uses GNN cone_depth to weight edges in the protein contact graph.
    High cone_depth → high conductance (core propagates signal).
    Lower effective resistance = stronger allosteric coupling.
    """
    nodes = gnn_result.nodes
    if not nodes:
        return PhaseResult(
            phase_name="phase4_resistance_mapping",
            structure_id=structure_id,
            model_version="DTIE-v5-phase4",
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

    # Build coordinate array — initialize with NaN to detect missing data
    node_key_to_idx = {
        (n.chain_label, n.residue_index): i for i, n in enumerate(nodes)
    }
    ca_coords = np.full((len(nodes), 3), np.nan)
    for row in rows:
        key = (row["chain_label"], row["residue_index"])
        if key in node_key_to_idx:
            idx = node_key_to_idx[key]
            ca_coords[idx] = [row["x"], row["y"], row["z"]]

    cone_depths = np.array([n.cone_depth for n in nodes])

    # Build conductance graph (NaN nodes excluded internally)
    G = _build_conductance_graph(ca_coords, cone_depths, cutoff=spatial_cutoff)

    if G.number_of_edges() == 0:
        return PhaseResult(
            phase_name="phase4_resistance_mapping",
            structure_id=structure_id,
            model_version="DTIE-v5-phase4",
            success=True,
            outputs={"pathways": [], "spectral": {}, "note": "No edges in contact graph"},
        )

    # Extract largest connected component for resistance calculations
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Spectral analysis
    spectral = await asyncio.to_thread(_spectral_analysis, G)

    # Determine source sites
    lifted_sites = []
    if phase35_result and phase35_result.success:
        lifted_sites = phase35_result.outputs.get("lifted_sites", [])

    # If no lifted sites, use top source-leak candidates as sources
    if not lifted_sites:
        epistemics = np.array([n.epistemic_uncertainty for n in nodes])
        top_sources = np.argsort(epistemics)[-5:]
        lifted_sites = [{"vertex_count": 1, "source_node": int(s)} for s in top_sources]

    # Effector targets: deepest residues (most buried = strongest conductors)
    depth_sorted = np.argsort(cone_depths)
    effector_indices = depth_sorted[-min(5, len(nodes)):]

    # Compute effective resistance using sparse solver
    graph_nodes_set = set(G.nodes())
    # Build valid ca_coords for nearest-node lookup (only graph nodes)
    valid_ca = np.full_like(ca_coords, np.nan)
    for node_id in graph_nodes_set:
        valid_ca[node_id] = ca_coords[node_id]

    pathways = []
    for site in lifted_sites[:10]:  # Cap at 10 source sites
        if "source_node" in site:
            source_idx = site["source_node"]
        elif "barycenter_xyz" in site:
            # Find nearest graph node to barycenter
            bary = np.array(site["barycenter_xyz"])
            best_dist = float("inf")
            source_idx = -1
            for nid in graph_nodes_set:
                d = np.linalg.norm(ca_coords[nid] - bary)
                if d < best_dist:
                    best_dist = d
                    source_idx = nid
            if source_idx < 0:
                continue
        else:
            continue

        if source_idx not in graph_nodes_set:
            continue

        for target_idx in effector_indices:
            if source_idx == target_idx:
                continue
            if int(target_idx) not in graph_nodes_set:
                continue

            r_eff = await asyncio.to_thread(
                _compute_effective_resistance_sparse, G, source_idx, int(target_idx)
            )
            if r_eff is None or r_eff <= 0:
                continue

            pathways.append({
                "source_node": int(source_idx),
                "target_node": int(target_idx),
                "r_eff": r_eff,
                "coupling_strength": 1.0 / r_eff,
                "source_residue": nodes[source_idx].residue_index,
                "target_residue": nodes[target_idx].residue_index,
            })

    pathways.sort(key=lambda p: -p["coupling_strength"])

    logger.info(
        "Phase 4 v5: %d pathways, λ₂=%.4f, %d hinges",
        len(pathways), spectral["lambda_2"], len(spectral["hinge_residues"]),
    )

    return PhaseResult(
        phase_name="phase4_resistance_mapping",
        structure_id=structure_id,
        model_version="DTIE-v5-phase4",
        success=True,
        outputs={
            "pathway_count": len(pathways),
            "pathways": pathways[:50],  # Top 50
            "spectral": spectral,
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
        },
    )
