#!/usr/bin/env python3
"""
compare_graphs.py — WT vs Mutant Graph Comparison Tool
======================================================

Takes two pipeline-produced .pt graph files (or ingestion .npz + PDB pairs)
and produces the full mechanistic analysis:

1. Edge symmetric difference + degree shifts
2. Centrality analysis (betweenness, eigenvector, closeness)
3. Per-edge wrapping check at differing edges (dehydron detection)
4. Gain-of-function contact detection (new edges in mutant)
5. Witness Complex persistence on the difference graph

Usage:
    python DTIE_GNN_ORCHESTRATION/compare_graphs.py \
        --mutant-graph results/wt_vs_g12d_active/graph_gdp.pt \
        --wt-graph results/wt_vs_g12d_active/graph_gtp.pt \
        --mutant-pdb data/kras_g12d/kras_g12d_gtp.pdb \
        --wt-pdb data/kras_g12d/kras_g12d_gtp_5p21.pdb \
        --output results/wt_vs_g12d_active/comparison_report.json

All paths can be provided as relative (recommended) or absolute paths.

If --mutant-pdb and --wt-pdb are provided, the script also computes:
  - Per-edge backbone H-bond distances and wrapping counts
  - Gain-of-function sidechain contacts (closest atom pairs)
  - Witness Complex persistence on the difference graph
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("GraphCompare")

TAU = 13.0  # dehydron threshold
WRAPPING_RADIUS = 6.5  # Angstrom


# =============================================================================
# Graph loading and conversion
# =============================================================================

def load_pyg_graph(path: Path):
    """Load a PyG Data object from .pt file."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with torch versions that do not support weights_only.
        return torch.load(str(path), map_location="cpu")


def pyg_to_nx(data) -> nx.Graph:
    """Convert PyG Data to undirected NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    return G


# =============================================================================
# 1. Edge symmetric difference + degree shifts
# =============================================================================

def compute_structural_metadata(
    mutant_pdb: Optional[Path],
    wt_pdb: Optional[Path],
    G_mut: nx.Graph,
    G_wt: nx.Graph,
) -> dict:
    """
    Compute structural comparison metadata for provenance:
    - Node counts (mutant, WT, intersection)
    - Missing residues with functional annotation
    - Ligand/cofactor state
    - Crystallographic modifications
    - Adjusted sym diff on intersection nodes only
    """
    from Bio.PDB import PDBParser

    meta = {
        "mutant_nodes": G_mut.number_of_nodes(),
        "wt_nodes": G_wt.number_of_nodes(),
    }

    # Parse PDBs for residue-level metadata
    if mutant_pdb and wt_pdb:
        parser = PDBParser(QUIET=True)
        s_mut = parser.get_structure("mut", str(mutant_pdb))
        s_wt = parser.get_structure("wt", str(wt_pdb))

        mut_residues = [r for r in s_mut.get_residues() if r.get_id()[0] == " "]
        wt_residues = [r for r in s_wt.get_residues() if r.get_id()[0] == " "]

        mut_resnums = {r.get_id()[1] for r in mut_residues}
        wt_resnums = {r.get_id()[1] for r in wt_residues}
        shared_resnums = mut_resnums & wt_resnums

        only_mut = sorted(mut_resnums - wt_resnums)
        only_wt = sorted(wt_resnums - mut_resnums)

        meta["shared_residues"] = len(shared_resnums)
        meta["only_in_mutant"] = only_mut
        meta["only_in_wt"] = only_wt
        meta["residue_range_mutant"] = (
            f"{min(mut_resnums)}-{max(mut_resnums)}" if mut_resnums else None
        )
        meta["residue_range_wt"] = (
            f"{min(wt_resnums)}-{max(wt_resnums)}" if wt_resnums else None
        )

        # Ligands / cofactors
        for label, struct in [("mutant", s_mut), ("wt", s_wt)]:
            hetatms = [r for r in struct.get_residues()
                       if r.get_id()[0] not in (" ", "W")
                       and r.get_resname().strip() != "HOH"]
            meta[f"{label}_ligands"] = [r.get_resname().strip() for r in hetatms]

        # Crystallographic modifications (common: C→S at catalytic cys, selenomethionine, etc.)
        mods = []
        for r in mut_residues:
            rname = r.get_resname().strip()
            if rname == "MSE":
                mods.append(f"SeMet at {r.get_id()[1]}")
            if rname == "SER":
                # Check if this is a known catalytic cysteine position mutated to serine
                # (common in phosphatase structures)
                pass
        meta["crystallographic_modifications"] = mods if mods else ["none detected"]

        # Adjusted sym diff: only count edges where BOTH endpoints are in shared residues
        # Build node-index-to-resnum maps
        mut_idx_to_resnum = {i: r.get_id()[1] for i, r in enumerate(mut_residues)}
        wt_idx_to_resnum = {i: r.get_id()[1] for i, r in enumerate(wt_residues)}

        edges_mut = set(G_mut.edges())
        edges_wt = set(G_wt.edges())

        # Filter to edges where both nodes map to shared residue numbers
        def filter_shared(edges, idx_to_resnum, shared):
            return {(u, v) for u, v in edges
                    if idx_to_resnum.get(u) in shared and idx_to_resnum.get(v) in shared}

        shared_edges_mut = filter_shared(edges_mut, mut_idx_to_resnum, shared_resnums)
        shared_edges_wt = filter_shared(edges_wt, wt_idx_to_resnum, shared_resnums)

        # Convert to resnum-based edges for comparison
        def to_resnum_edges(edges, idx_to_resnum):
            return {(idx_to_resnum[u], idx_to_resnum[v])
                    if idx_to_resnum[u] <= idx_to_resnum[v]
                    else (idx_to_resnum[v], idx_to_resnum[u])
                    for u, v in edges
                    if u in idx_to_resnum and v in idx_to_resnum}

        resnum_edges_mut = to_resnum_edges(shared_edges_mut, mut_idx_to_resnum)
        resnum_edges_wt = to_resnum_edges(shared_edges_wt, wt_idx_to_resnum)

        adjusted_sym_diff = resnum_edges_mut.symmetric_difference(resnum_edges_wt)
        adjusted_gained = resnum_edges_mut - resnum_edges_wt
        adjusted_lost = resnum_edges_wt - resnum_edges_mut

        meta["adjusted_sym_diff"] = len(adjusted_sym_diff)
        meta["adjusted_gained"] = len(adjusted_gained)
        meta["adjusted_lost"] = len(adjusted_lost)
        meta["edges_per_residue"] = round(len(adjusted_sym_diff) / max(len(shared_resnums), 1), 4)
        meta["raw_sym_diff_caveat"] = (
            f"Raw sym diff inflated by {len(only_mut)} mutant-only and "
            f"{len(only_wt)} WT-only residues"
            if only_mut or only_wt else "No node discrepancy"
        )
    else:
        meta["adjusted_sym_diff"] = None
        meta["raw_sym_diff_caveat"] = "PDB files not provided; cannot compute adjusted metrics"

    return meta


def compute_edge_diff(
    G_mut: nx.Graph,
    G_wt: nx.Graph,
    mut_residue_ids: Optional[np.ndarray] = None,
    wt_residue_ids: Optional[np.ndarray] = None,
) -> dict:
    """Compute edge-level differences between mutant and WT graphs.

    When residue_ids are provided (from ingestion .npz), edge lists use
    PDB residue numbers instead of raw node indices. This fixes the
    node-index ↔ PDB-residue mismatch that corrupts downstream wrapping
    and gain-of-function analysis.
    """
    edges_mut = set(G_mut.edges())
    edges_wt = set(G_wt.edges())

    gained = edges_mut - edges_wt  # new in mutant
    lost = edges_wt - edges_mut    # lost in mutant
    sym_diff = gained | lost

    # Build node_idx → PDB resnum maps
    def _idx_to_resnum(residue_ids):
        if residue_ids is None:
            return None
        mapping = {}
        for i, rid in enumerate(residue_ids):
            parts = str(rid).split(":")
            if len(parts) >= 2:
                try:
                    mapping[i] = int(parts[1])
                except ValueError:
                    mapping[i] = i + 1
            else:
                mapping[i] = i + 1
        return mapping

    mut_map = _idx_to_resnum(mut_residue_ids)
    wt_map = _idx_to_resnum(wt_residue_ids)

    def _edge_to_resnum(edges, idx_map):
        if idx_map is None:
            return sorted([(u + 1, v + 1) for u, v in edges])
        return sorted([(idx_map.get(u, u + 1), idx_map.get(v, v + 1)) for u, v in edges])

    # Degree differences (on shared node range)
    min_n = min(G_mut.number_of_nodes(), G_wt.number_of_nodes())
    deg_mut = np.array([G_mut.degree(i) for i in range(min_n)])
    deg_wt = np.array([G_wt.degree(i) for i in range(min_n)])
    deg_diff = deg_mut - deg_wt

    top_idx = np.argsort(np.abs(deg_diff))[-10:][::-1]

    # For degree changes, report PDB residue numbers
    def _resnum(idx, idx_map):
        if idx_map:
            return idx_map.get(idx, idx + 1)
        return idx + 1

    return {
        "mutant_edges": len(edges_mut),
        "wt_edges": len(edges_wt),
        "sym_diff": len(sym_diff),
        "gained_edges": len(gained),
        "lost_edges": len(lost),
        "gained_edge_list": _edge_to_resnum(gained, mut_map),
        "lost_edge_list": _edge_to_resnum(lost, wt_map),
        # Keep raw 0-indexed edges for internal use
        "_gained_0idx": sorted(gained),
        "_lost_0idx": sorted(lost),
        "top_degree_changes": [
            {"residue": _resnum(int(i), mut_map), "node_idx": int(i), "diff": int(deg_diff[i])}
            for i in top_idx
        ],
        "mean_degree_diff": float(np.mean(deg_diff)),
        "max_degree_diff": int(np.max(np.abs(deg_diff))),
    }


# =============================================================================
# 2. Centrality analysis
# =============================================================================

def compute_centrality(G_mut: nx.Graph, G_wt: nx.Graph, mutation_node: int = 11) -> dict:
    """Compute centrality metrics for both graphs, focused on mutation site."""
    bet_mut = nx.betweenness_centrality(G_mut, normalized=True)
    bet_wt = nx.betweenness_centrality(G_wt, normalized=True)
    try:
        eig_mut = nx.eigenvector_centrality(G_mut, max_iter=1000)
    except Exception:
        eig_mut = {n: 0.0 for n in G_mut.nodes}
    try:
        eig_wt = nx.eigenvector_centrality(G_wt, max_iter=1000)
    except Exception:
        eig_wt = {n: 0.0 for n in G_wt.nodes}
    clo_mut = nx.closeness_centrality(G_mut)
    clo_wt = nx.closeness_centrality(G_wt)

    # Mutation site comparison
    site = {
        "residue": mutation_node + 1,
        "betweenness_mutant": bet_mut.get(mutation_node, 0),
        "betweenness_wt": bet_wt.get(mutation_node, 0),
        "betweenness_change": bet_mut.get(mutation_node, 0) - bet_wt.get(mutation_node, 0),
        "eigenvector_mutant": eig_mut.get(mutation_node, 0),
        "eigenvector_wt": eig_wt.get(mutation_node, 0),
        "eigenvector_change": eig_mut.get(mutation_node, 0) - eig_wt.get(mutation_node, 0),
        "closeness_mutant": clo_mut.get(mutation_node, 0),
        "closeness_wt": clo_wt.get(mutation_node, 0),
        "closeness_change": clo_mut.get(mutation_node, 0) - clo_wt.get(mutation_node, 0),
        "degree_mutant": G_mut.degree(mutation_node),
        "degree_wt": G_wt.degree(mutation_node) if mutation_node < G_wt.number_of_nodes() else 0,
    }

    # Top betweenness hubs in mutant
    top_bet = sorted(bet_mut.items(), key=lambda x: x[1], reverse=True)[:5]
    hubs = [{"residue": n+1, "betweenness": v} for n, v in top_bet]

    return {"mutation_site": site, "top_hubs_mutant": hubs}


# =============================================================================
# 3. Per-edge wrapping check (dehydron detection)
# =============================================================================

def _get_residues_from_pdb(pdb_path: Path):
    """Parse PDB and return residues with their atoms."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    return residues


def _compute_wrapping_at_edge(residues, res_i_idx: int, res_j_idx: int) -> dict:
    """
    For a specific residue pair, compute:
    - Backbone N-O distance (H-bond candidate)
    - Closest atom pair distance
    - Wrapping count (apolar C within 6.5A of N-O midpoint)
    """
    if res_i_idx >= len(residues) or res_j_idx >= len(residues):
        return {"error": "residue index out of range"}

    res_i = residues[res_i_idx]
    res_j = residues[res_j_idx]

    # Backbone N-O distance
    n_atom = None
    o_atom = None
    for a in res_i.get_atoms():
        if a.get_name() == "N":
            n_atom = a
    for a in res_j.get_atoms():
        if a.get_name() == "O":
            o_atom = a

    backbone_no_dist = None
    no_midpoint = None
    if n_atom and o_atom:
        backbone_no_dist = float(np.linalg.norm(n_atom.get_vector().get_array() - o_atom.get_vector().get_array()))
        no_midpoint = (n_atom.get_vector().get_array() + o_atom.get_vector().get_array()) / 2.0

    # Also check reverse (j's N to i's O)
    n_j = None
    o_i = None
    for a in res_j.get_atoms():
        if a.get_name() == "N":
            n_j = a
    for a in res_i.get_atoms():
        if a.get_name() == "O":
            o_i = a

    reverse_no_dist = None
    if n_j and o_i:
        reverse_no_dist = float(np.linalg.norm(n_j.get_vector().get_array() - o_i.get_vector().get_array()))
        if backbone_no_dist is None or reverse_no_dist < backbone_no_dist:
            backbone_no_dist = reverse_no_dist
            no_midpoint = (n_j.get_vector().get_array() + o_i.get_vector().get_array()) / 2.0

    # Closest atom pair (any atoms)
    min_dist = float("inf")
    closest_pair = ("", "")
    for a1 in res_i.get_atoms():
        for a2 in res_j.get_atoms():
            d = float(np.linalg.norm(a1.get_vector().get_array() - a2.get_vector().get_array()))
            if d < min_dist:
                min_dist = d
                closest_pair = (a1.get_name(), a2.get_name())

    # Wrapping count at the N-O midpoint
    wrap_count = 0
    if no_midpoint is not None and backbone_no_dist is not None and backbone_no_dist < 4.0:
        POLAR_RESIDUES = {"SER", "THR", "CYS", "TYR", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS"}
        for res in residues:
            for atom in res.get_atoms():
                if atom.element != "C":
                    continue
                if atom.get_name() in ("C", "CA"):
                    continue
                if res.get_resname() in POLAR_RESIDUES:
                    continue
                d = float(np.linalg.norm(atom.get_vector().get_array() - no_midpoint))
                if d <= WRAPPING_RADIUS:
                    wrap_count += 1

    is_dehydron = backbone_no_dist is not None and backbone_no_dist < 4.0 and wrap_count < TAU

    return {
        "residue_i": res_i_idx + 1,
        "residue_j": res_j_idx + 1,
        "resname_i": res_i.get_resname(),
        "resname_j": res_j.get_resname(),
        "backbone_NO_dist": backbone_no_dist,
        "closest_atom_dist": min_dist if min_dist < float("inf") else None,
        "closest_atoms": f"{closest_pair[0]}({res_i_idx+1})—{closest_pair[1]}({res_j_idx+1})",
        "wrap_count": wrap_count if (backbone_no_dist and backbone_no_dist < 4.0) else None,
        "is_dehydron": is_dehydron,
        "is_hbond": min_dist < 3.5 if min_dist < float("inf") else False,
    }


def analyze_differing_edges(
    gained_edges_0idx: List[Tuple[int, int]],
    lost_edges_0idx: List[Tuple[int, int]],
    mutant_pdb: Path,
    wt_pdb: Path,
    mutation_node_idx: int,
    mut_residue_ids: Optional[np.ndarray] = None,
    wt_residue_ids: Optional[np.ndarray] = None,
) -> dict:
    """Analyze wrapping/contacts at edges involving the mutation site.

    All edge tuples are 0-indexed node indices (matching the graph).
    PDB residue numbers are resolved via residue_ids for display only.
    """
    mut_residues = _get_residues_from_pdb(mutant_pdb)
    wt_residues = _get_residues_from_pdb(wt_pdb)

    def _display_resnum(node_idx, residue_ids):
        if residue_ids is not None and node_idx < len(residue_ids):
            parts = str(residue_ids[node_idx]).split(":")
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return node_idx + 1

    # Focus on edges touching the mutation node
    mut_site_gained = [(u, v) for u, v in gained_edges_0idx
                       if u == mutation_node_idx or v == mutation_node_idx]
    mut_site_lost = [(u, v) for u, v in lost_edges_0idx
                     if u == mutation_node_idx or v == mutation_node_idx]

    gained_analysis = []
    for u, v in mut_site_gained:
        result = _compute_wrapping_at_edge(mut_residues, u, v)
        result["edge_type"] = "GAINED_IN_MUTANT"
        result["residue_i"] = _display_resnum(u, mut_residue_ids)
        result["residue_j"] = _display_resnum(v, mut_residue_ids)
        gained_analysis.append(result)

    lost_analysis = []
    for u, v in mut_site_lost:
        result = _compute_wrapping_at_edge(wt_residues, u, v)
        result["edge_type"] = "LOST_IN_MUTANT"
        result["residue_i"] = _display_resnum(u, wt_residue_ids)
        result["residue_j"] = _display_resnum(v, wt_residue_ids)
        lost_analysis.append(result)

    return {
        "mutation_node_idx": mutation_node_idx,
        "mutation_residue": _display_resnum(mutation_node_idx, mut_residue_ids),
        "gained_at_mutation_site": gained_analysis,
        "lost_at_mutation_site": lost_analysis,
        "total_gained_edges": len(gained_edges_0idx),
        "total_lost_edges": len(lost_edges_0idx),
    }


def scan_gain_of_function_contacts(
    mutant_pdb: Path,
    wt_pdb: Path,
    mutation_node_idx: int,
    contact_threshold: float = 3.5,
    scan_radius: int = 0,
    mut_residue_ids: Optional[np.ndarray] = None,
) -> dict:
    """
    Scan ALL residue pairs involving the mutation site for new sub-threshold
    contacts that exist in the mutant but NOT in the WT.

    This catches gain-of-function sidechain contacts (like Asp12-OD1 → Tyr32-OH)
    that a Cα-only graph misses.

    Args:
        mutant_pdb: Path to mutant structure
        wt_pdb: Path to WT structure
        mutation_node_idx: 0-based node index of the mutation in the graph
        contact_threshold: distance cutoff for "contact" (Å)
        scan_radius: if >0, also scan residues within this many positions of mutation site
        mut_residue_ids: residue IDs from ingestion for display
    """
    mut_residues = _get_residues_from_pdb(mutant_pdb)
    wt_residues = _get_residues_from_pdb(wt_pdb)

    def _display_resnum(node_idx):
        if mut_residue_ids is not None and node_idx < len(mut_residue_ids):
            parts = str(mut_residue_ids[node_idx]).split(":")
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return node_idx + 1

    # Determine which residues to scan from (mutation site + neighbors)
    scan_indices = [mutation_node_idx]
    if scan_radius > 0:
        for offset in range(-scan_radius, scan_radius + 1):
            idx = mutation_node_idx + offset
            if 0 <= idx < len(mut_residues) and idx not in scan_indices:
                scan_indices.append(idx)

    min_n = min(len(mut_residues), len(wt_residues))
    gain_of_function = []

    for src_idx in scan_indices:
        if src_idx >= min_n:
            continue
        mut_src = mut_residues[src_idx]

        for tgt_idx in range(min_n):
            if tgt_idx == src_idx:
                continue
            # Skip immediate sequence neighbors (trivial backbone contacts)
            if abs(tgt_idx - src_idx) <= 1:
                continue

            mut_tgt = mut_residues[tgt_idx]
            wt_src = wt_residues[src_idx]
            wt_tgt = wt_residues[tgt_idx]

            # Find closest atom pair in mutant
            mut_min_dist = float("inf")
            mut_closest = ("", "")
            for a1 in mut_src.get_atoms():
                for a2 in mut_tgt.get_atoms():
                    d = float(np.linalg.norm(
                        a1.get_vector().get_array() - a2.get_vector().get_array()
                    ))
                    if d < mut_min_dist:
                        mut_min_dist = d
                        mut_closest = (a1.get_name(), a2.get_name())

            # Only interested if mutant has a close contact
            if mut_min_dist >= contact_threshold:
                continue

            # Find closest atom pair in WT at same residue positions
            wt_min_dist = float("inf")
            wt_closest = ("", "")
            for a1 in wt_src.get_atoms():
                for a2 in wt_tgt.get_atoms():
                    d = float(np.linalg.norm(
                        a1.get_vector().get_array() - a2.get_vector().get_array()
                    ))
                    if d < wt_min_dist:
                        wt_min_dist = d
                        wt_closest = (a1.get_name(), a2.get_name())

            # Gain-of-function: close in mutant, distant in WT
            distance_ratio = wt_min_dist / (mut_min_dist + 1e-6)
            if distance_ratio > 2.0:  # WT is at least 2x further
                is_hbond = mut_min_dist < 3.2 and (
                    mut_closest[0][0] in "NO" or mut_closest[1][0] in "NO"
                )
                gain_of_function.append({
                    "residue_i": _display_resnum(src_idx),
                    "residue_j": _display_resnum(tgt_idx),
                    "node_idx_i": src_idx,
                    "node_idx_j": tgt_idx,
                    "resname_i": mut_src.get_resname(),
                    "resname_j": mut_tgt.get_resname(),
                    "mutant_dist": round(mut_min_dist, 3),
                    "mutant_atoms": f"{mut_closest[0]}({_display_resnum(src_idx)})—{mut_closest[1]}({_display_resnum(tgt_idx)})",
                    "wt_dist": round(wt_min_dist, 3),
                    "wt_atoms": f"{wt_closest[0]}({_display_resnum(src_idx)})—{wt_closest[1]}({_display_resnum(tgt_idx)})",
                    "distance_ratio": round(distance_ratio, 2),
                    "is_hbond": is_hbond,
                    "is_gain_of_function": True,
                    "mechanism": "H-bond" if is_hbond else "vdW contact",
                })

    # Sort by distance ratio (most dramatic gains first)
    gain_of_function.sort(key=lambda x: -x["distance_ratio"])

    return {
        "mutation_node_idx": mutation_node_idx,
        "mutation_residue": _display_resnum(mutation_node_idx),
        "contact_threshold": contact_threshold,
        "n_gain_of_function": len(gain_of_function),
        "contacts": gain_of_function,
    }


# =============================================================================
# 4. Persistence on difference graph
# =============================================================================

def compute_difference_persistence(
    gained_edges: List[Tuple[int, int]],
    ca_coords: np.ndarray,
    max_alpha: float = 20.0,
) -> dict:
    """
    Build a subgraph from gained edges, embed in 3D via Cα coords,
    and run Rips persistence to find H1 cycles in the rewiring.
    """
    try:
        import gudhi
    except ImportError:
        return {"error": "GUDHI not installed — skipping persistence analysis"}

    if len(gained_edges) < 3:
        return {"n_delta_edges": len(gained_edges), "h1_cycles": 0, "note": "too few edges for persistence"}

    # Collect unique nodes from gained edges
    nodes = sorted(set(n for e in gained_edges for n in e))
    if max(nodes) >= len(ca_coords):
        # Trim to available coords
        nodes = [n for n in nodes if n < len(ca_coords)]

    if len(nodes) < 3:
        return {"n_delta_edges": len(gained_edges), "h1_cycles": 0, "note": "too few nodes with coords"}

    # Build distance matrix for the subgraph nodes
    coords = ca_coords[nodes]
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(coords, coords)

    # Rips complex on the subgraph
    rips = gudhi.RipsComplex(distance_matrix=dist_matrix.tolist(), max_edge_length=max_alpha)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    persistence = st.persistence()

    h0_bars = [(b, d) for dim, (b, d) in persistence if dim == 0]
    h1_bars = [(b, d) for dim, (b, d) in persistence if dim == 1]

    finite_h1 = [(b, d) for b, d in h1_bars if d != float("inf")]
    terminal_h1 = [(b, d) for b, d in h1_bars if d == float("inf")]

    top_persistence = max((d - b for b, d in finite_h1), default=0.0)
    total_persistence = sum(d - b for b, d in finite_h1)
    top1_fraction = top_persistence / total_persistence if total_persistence > 0 else 0.0

    return {
        "n_delta_edges": len(gained_edges),
        "n_subgraph_nodes": len(nodes),
        "h0_bars": len(h0_bars),
        "h1_bars": len(h1_bars),
        "h1_finite": len(finite_h1),
        "h1_terminal": len(terminal_h1),
        "top_h1_persistence": top_persistence,
        "total_h1_persistence": total_persistence,
        "top1_fraction": top1_fraction,
        "distributed": top1_fraction < 0.4,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="WT vs Mutant Graph Comparison")
    parser.add_argument("--mutant-graph", required=True, help="Path to mutant .pt graph")
    parser.add_argument("--wt-graph", required=True, help="Path to WT .pt graph")
    parser.add_argument("--mutant-pdb", default=None, help="Mutant PDB for wrapping analysis")
    parser.add_argument("--wt-pdb", default=None, help="WT PDB for wrapping analysis")
    parser.add_argument("--mutation-residue", type=int, default=12, help="1-based PDB mutation residue (default: 12)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load graphs
    logger.info("Loading graphs...")
    g_mut = load_pyg_graph(Path(args.mutant_graph))
    g_wt = load_pyg_graph(Path(args.wt_graph))
    G_mut = pyg_to_nx(g_mut)
    G_wt = pyg_to_nx(g_wt)

    logger.info(f"Mutant: {G_mut.number_of_nodes()} nodes, {G_mut.number_of_edges()} edges")
    logger.info(f"WT:     {G_wt.number_of_nodes()} nodes, {G_wt.number_of_edges()} edges")

    # ── Build PDB residue number ↔ graph node index maps ──────────────
    # This is the critical fix: PDB residue numbers (e.g. 12 for G12D)
    # do NOT equal graph node indices when the PDB has gaps or doesn't
    # start at residue 1.
    run_dir = Path(args.mutant_graph).parent
    mut_ingestion_path = run_dir / "ingestion_gdp.npz"
    wt_ingestion_path = run_dir / "ingestion_gtp.npz"

    mut_residue_ids = None
    wt_residue_ids = None
    mut_resnum_to_idx = {}  # PDB resnum → node index
    wt_resnum_to_idx = {}

    if mut_ingestion_path.exists():
        ing = np.load(str(mut_ingestion_path), allow_pickle=True)
        mut_residue_ids = ing["residue_ids"]
        for i, rid in enumerate(mut_residue_ids):
            parts = str(rid).split(":")
            if len(parts) >= 2:
                try:
                    mut_resnum_to_idx[int(parts[1])] = i
                except ValueError:
                    pass
        if mut_resnum_to_idx:
            logger.info(
                f"Loaded mutant residue map: {len(mut_resnum_to_idx)} residues "
                f"(PDB range {min(mut_resnum_to_idx.keys())}-{max(mut_resnum_to_idx.keys())})"
            )
        else:
            logger.warning("Loaded mutant ingestion map but found no parseable residue numbers")

    if wt_ingestion_path.exists():
        ing = np.load(str(wt_ingestion_path), allow_pickle=True)
        wt_residue_ids = ing["residue_ids"]
        for i, rid in enumerate(wt_residue_ids):
            parts = str(rid).split(":")
            if len(parts) >= 2:
                try:
                    wt_resnum_to_idx[int(parts[1])] = i
                except ValueError:
                    pass
        if wt_resnum_to_idx:
            logger.info(
                f"Loaded WT residue map: {len(wt_resnum_to_idx)} residues "
                f"(PDB range {min(wt_resnum_to_idx.keys())}-{max(wt_resnum_to_idx.keys())})"
            )
        else:
            logger.warning("Loaded WT ingestion map but found no parseable residue numbers")

    # Resolve mutation PDB residue number → node index
    mut_pdb_resnum = args.mutation_residue
    if mut_resnum_to_idx and mut_pdb_resnum in mut_resnum_to_idx:
        mut_node_idx = mut_resnum_to_idx[mut_pdb_resnum]
        logger.info(f"Mutation residue {mut_pdb_resnum} → node index {mut_node_idx}")
    else:
        mut_node_idx = mut_pdb_resnum - 1
        logger.warning(f"No ingestion map — assuming residue {mut_pdb_resnum} → node index {mut_node_idx} (may be wrong!)")

    # 0. Structural comparison metadata (provenance layer)
    metadata = None
    if args.mutant_pdb and args.wt_pdb:
        logger.info("\n=== Structural Comparison Metadata ===")
        metadata = compute_structural_metadata(
            Path(args.mutant_pdb), Path(args.wt_pdb), G_mut, G_wt,
        )
        logger.info(f"Shared residues: {metadata.get('shared_residues', '?')}")
        logger.info(f"Only in mutant: {len(metadata.get('only_in_mutant', []))} residues")
        logger.info(f"Only in WT: {len(metadata.get('only_in_wt', []))} residues")
        logger.info(f"Mutant ligands: {metadata.get('mutant_ligands', [])}")
        logger.info(f"WT ligands: {metadata.get('wt_ligands', [])}")
        logger.info(f"Adjusted sym diff (intersection only): {metadata.get('adjusted_sym_diff', '?')}")
        logger.info(f"  (adjusted gained: {metadata.get('adjusted_gained', '?')}, lost: {metadata.get('adjusted_lost', '?')})")
        logger.info(f"  Edges/residue (adjusted): {metadata.get('edges_per_residue', '?')}")
        logger.info(f"Caveat: {metadata.get('raw_sym_diff_caveat', '')}")

    # 1. Edge differences (with proper residue ID mapping)
    logger.info("\n=== Edge Symmetric Difference ===")
    edge_diff = compute_edge_diff(G_mut, G_wt, mut_residue_ids, wt_residue_ids)
    logger.info(f"Sym diff: {edge_diff['sym_diff']} (gained: {edge_diff['gained_edges']}, lost: {edge_diff['lost_edges']})")
    logger.info(f"Mean degree diff: {edge_diff['mean_degree_diff']:.4f} | Max: {edge_diff['max_degree_diff']}")
    logger.info("Top degree changes:")
    for d in edge_diff["top_degree_changes"][:5]:
        logger.info(f"  Res {d['residue']:3d} (node {d['node_idx']}): {d['diff']:+d}")

    # Edges at mutation site (filter using PDB residue number from the edge lists)
    mut_gained_display = [(u, v) for u, v in edge_diff["gained_edge_list"]
                          if u == mut_pdb_resnum or v == mut_pdb_resnum]
    mut_lost_display = [(u, v) for u, v in edge_diff["lost_edge_list"]
                        if u == mut_pdb_resnum or v == mut_pdb_resnum]
    logger.info(f"\nGained edges at PDB residue {mut_pdb_resnum} (node {mut_node_idx}): {mut_gained_display}")
    logger.info(f"Lost edges at PDB residue {mut_pdb_resnum} (node {mut_node_idx}): {mut_lost_display}")

    # 2. Centrality (uses node indices directly)
    logger.info("\n=== Centrality Analysis ===")
    centrality = compute_centrality(G_mut, G_wt, mutation_node=mut_node_idx)
    site = centrality["mutation_site"]
    logger.info(f"Residue {mut_pdb_resnum} (node {mut_node_idx}):")
    logger.info(f"  Betweenness: {site['betweenness_mutant']:.5f} (mut) vs {site['betweenness_wt']:.5f} (wt) → {site['betweenness_change']:+.5f}")
    logger.info(f"  Eigenvector: {site['eigenvector_mutant']:.5f} (mut) vs {site['eigenvector_wt']:.5f} (wt) → {site['eigenvector_change']:+.5f}")
    logger.info(f"  Degree:      {site['degree_mutant']} (mut) vs {site['degree_wt']} (wt)")
    logger.info("Top betweenness hubs in mutant:")
    for h in centrality["top_hubs_mutant"]:
        logger.info(f"  Res {h['residue']:3d}: {h['betweenness']:.5f}")

    # 3. Wrapping analysis (if PDBs provided) — uses 0-indexed edges
    wrapping = None
    gain_of_function = None
    if args.mutant_pdb and args.wt_pdb:
        logger.info("\n=== Wrapping / Dehydron Analysis ===")
        wrapping = analyze_differing_edges(
            gained_edges_0idx=edge_diff["_gained_0idx"],
            lost_edges_0idx=edge_diff["_lost_0idx"],
            mutant_pdb=Path(args.mutant_pdb),
            wt_pdb=Path(args.wt_pdb),
            mutation_node_idx=mut_node_idx,
            mut_residue_ids=mut_residue_ids,
            wt_residue_ids=wt_residue_ids,
        )
        logger.info(f"Gained contacts at residue {mut_pdb_resnum} (node {mut_node_idx}):")
        for g in wrapping["gained_at_mutation_site"]:
            dehydron_flag = " ← DEHYDRON" if g["is_dehydron"] else ""
            hbond_flag = " ← H-BOND" if g["is_hbond"] else ""
            dist_str = f"backbone N-O: {g['backbone_NO_dist']:.2f}Å" if g.get('backbone_NO_dist') else "no backbone H-bond"
            logger.info(
                f"  ({g['residue_i']}, {g['residue_j']}) "
                f"{g['resname_i']}-{g['resname_j']} | "
                f"closest: {g['closest_atom_dist']:.2f}Å ({g['closest_atoms']}) | "
                f"{dist_str}{dehydron_flag}{hbond_flag}"
            )

        # Gain-of-function scan
        logger.info("\n=== Gain-of-Function Contact Scan ===")
        gain_of_function = scan_gain_of_function_contacts(
            mutant_pdb=Path(args.mutant_pdb),
            wt_pdb=Path(args.wt_pdb),
            mutation_node_idx=mut_node_idx,
            contact_threshold=3.5,
            scan_radius=2,
            mut_residue_ids=mut_residue_ids,
        )
        logger.info(f"Found {gain_of_function['n_gain_of_function']} gain-of-function contacts:")
        for c in gain_of_function["contacts"]:
            flag = " ★ H-BOND" if c["is_hbond"] else ""
            logger.info(
                f"  ({c['residue_i']}, {c['residue_j']}) "
                f"{c['resname_i']}-{c['resname_j']} | "
                f"mutant: {c['mutant_dist']:.2f}Å ({c['mutant_atoms']}) | "
                f"WT: {c['wt_dist']:.2f}Å | "
                f"ratio: {c['distance_ratio']:.1f}x{flag}"
            )

    # 4. Persistence on difference graph — uses 0-indexed edges directly
    persistence = None
    if mut_ingestion_path.exists():
        logger.info("\n=== Persistence on Difference Graph ===")
        ing = np.load(str(mut_ingestion_path), allow_pickle=True)
        ca_coords = ing["ca_coords"]
        persistence = compute_difference_persistence(edge_diff["_gained_0idx"], ca_coords)
        if "error" in persistence:
            logger.warning(f"Persistence unavailable: {persistence['error']}")
        else:
            logger.info(
                f"Subgraph: {persistence.get('n_subgraph_nodes', 0)} nodes "
                f"from {persistence.get('n_delta_edges', 0)} gained edges"
            )
            logger.info(
                f"H1 cycles: {persistence.get('h1_bars', 0)} "
                f"(finite: {persistence.get('h1_finite', 0)}, terminal: {persistence.get('h1_terminal', 0)})"
            )
            logger.info(f"Top H1 persistence: {persistence.get('top_h1_persistence', 0):.2f}Å")
            logger.info(f"Top-1 fraction: {persistence.get('top1_fraction', 0):.1%}")
            logger.info(f"Distributed rewiring: {'YES' if persistence.get('distributed') else 'NO'}")

    # ── Save results ──────────────────────────────────────────────────
    # Strip internal 0-indexed edge lists from the report
    report_edge_diff = {k: v for k, v in edge_diff.items() if not k.startswith("_")}

    report = {
        "residue_mapping": {
            "mutation_pdb_resnum": mut_pdb_resnum,
            "mutation_node_idx": mut_node_idx,
            "mutant_residue_count": len(mut_residue_ids) if mut_residue_ids is not None else G_mut.number_of_nodes(),
            "wt_residue_count": len(wt_residue_ids) if wt_residue_ids is not None else G_wt.number_of_nodes(),
        },
        "structural_metadata": metadata,
        "edge_diff": report_edge_diff,
        "centrality": centrality,
        "wrapping": wrapping,
        "gain_of_function": gain_of_function,
        "persistence": persistence,
    }

    output_path = Path(args.output) if args.output else run_dir / "comparison_report.json"

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=_serialize)
    logger.info(f"\n✅ Report saved: {output_path}")


if __name__ == "__main__":
    main()
