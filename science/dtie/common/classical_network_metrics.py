"""Classical contact-graph network metrics from Cα coordinates (external ground truth).

Independent of DTIE ρ/τ / learned embeddings. Used as the external check for the
Jacobian flow-influence probe (Option B). Metrics mirror the classical side of
allosteric-network benchmarking: betweenness, current-flow betweenness, Fiedler
participation, and ANM fluctuation.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


DEFAULT_CONTACT_CUTOFF_A = 8.0


def build_ca_contact_graph(
    ca_coords: np.ndarray,
    *,
    cutoff_angstrom: float = DEFAULT_CONTACT_CUTOFF_A,
) -> nx.Graph:
    """Undirected Cα contact graph with distance edge weights."""
    coords = np.asarray(ca_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"ca_coords must be [N, 3], got {coords.shape}")
    n = coords.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    cutoff = float(cutoff_angstrom)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= cutoff and d > 1e-8:
                g.add_edge(i, j, weight=d, distance=d, conductance=1.0 / d)
    return g


def anm_msf(
    ca_coords: np.ndarray,
    *,
    cutoff_angstrom: float = DEFAULT_CONTACT_CUTOFF_A,
) -> np.ndarray:
    """Anisotropic Network Model mean-square fluctuation per residue.

    Kirchhoff (Laplacian) from unit springs on Cα contacts; MSF from the
    pseudoinverse of non-rigid eigenmodes (drop 6 smallest eigenvalues).
    """
    coords = np.asarray(ca_coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 8:
        return np.full(n, np.nan, dtype=np.float64)

    g = build_ca_contact_graph(coords, cutoff_angstrom=cutoff_angstrom)
    kirchhoff = np.zeros((n, n), dtype=np.float64)
    for i, j in g.edges():
        kirchhoff[i, j] = -1.0
        kirchhoff[j, i] = -1.0
    for i in range(n):
        kirchhoff[i, i] = -kirchhoff[i].sum()

    evals, evecs = np.linalg.eigh(kirchhoff)
    # Drop near-zero (rigid-body) modes; require at least one fluctuating mode.
    keep = evals > 1e-8
    if keep.sum() == 0:
        return np.full(n, np.nan, dtype=np.float64)
    # Standard ANM: exclude the six smallest modes when available.
    order = np.argsort(evals)
    drop_n = min(6, n - 1)
    keep_idx = order[drop_n:]
    keep_idx = keep_idx[evals[keep_idx] > 1e-8]
    if keep_idx.size == 0:
        return np.full(n, np.nan, dtype=np.float64)

    inv = 1.0 / evals[keep_idx]
    msf = np.sum((evecs[:, keep_idx] ** 2) * inv[np.newaxis, :], axis=1)
    return msf.astype(np.float64)


def classical_network_metrics(
    ca_coords: np.ndarray,
    *,
    cutoff_angstrom: float = DEFAULT_CONTACT_CUTOFF_A,
) -> dict[str, Any]:
    """Per-residue classical metrics on the Cα contact graph.

    Fail loudly on empty / fully disconnected graphs — no silent zeros.
    """
    coords = np.asarray(ca_coords, dtype=np.float64)
    g = build_ca_contact_graph(coords, cutoff_angstrom=cutoff_angstrom)
    n = coords.shape[0]
    if g.number_of_edges() == 0:
        raise ValueError(
            f"Cα contact graph has zero edges at cutoff={cutoff_angstrom}Å "
            f"(n={n}); refusing classical metrics"
        )

    largest = max(nx.connected_components(g), key=len)
    if len(largest) < n:
        # Restrict expensive / connectivity-requiring metrics to the LCC;
        # leave other nodes as NaN rather than fabricating zeros.
        g_lcc = g.subgraph(largest).copy()
        lcc_nodes = sorted(largest)
    else:
        g_lcc = g
        lcc_nodes = list(range(n))

    betweenness = nx.betweenness_centrality(g, weight="weight", normalized=True)
    # current_flow requires connected graph
    try:
        cfb = nx.current_flow_betweenness_centrality(
            g_lcc, weight="conductance", normalized=True
        )
    except Exception as exc:  # noqa: BLE001 — surface in report
        raise RuntimeError(
            f"current_flow_betweenness failed on LCC (n={len(lcc_nodes)}): {exc}"
        ) from exc

    fiedler_vec = np.full(n, np.nan, dtype=np.float64)
    fiedler_value = float("nan")
    try:
        fiedler_value = float(nx.algebraic_connectivity(g_lcc, weight="conductance"))
        # Per-node participation: |Fiedler eigenvector| on the LCC
        # NetworkX ≥3: fiedler_vector
        fv = nx.fiedler_vector(g_lcc, weight="conductance", normalized=True)
        for idx, node in enumerate(sorted(g_lcc.nodes())):
            fiedler_vec[int(node)] = abs(float(fv[idx]))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Fiedler computation failed: {exc}") from exc

    msf = anm_msf(coords, cutoff_angstrom=cutoff_angstrom)

    bet = np.full(n, np.nan, dtype=np.float64)
    cf = np.full(n, np.nan, dtype=np.float64)
    for node, val in betweenness.items():
        bet[int(node)] = float(val)
    for node, val in cfb.items():
        cf[int(node)] = float(val)

    return {
        "n_residues": n,
        "n_contact_edges": int(g.number_of_edges()),
        "contact_cutoff_angstrom": float(cutoff_angstrom),
        "lcc_size": int(len(lcc_nodes)),
        "betweenness": bet,
        "current_flow_betweenness": cf,
        "fiedler_value": fiedler_value,
        "fiedler_abs": fiedler_vec,
        "anm_msf": msf,
    }


__all__ = [
    "DEFAULT_CONTACT_CUTOFF_A",
    "build_ca_contact_graph",
    "anm_msf",
    "classical_network_metrics",
]
