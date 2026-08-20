"""Euclidean shortcut edges for chem-MVP reach ablation (train-side, v66).

New relation ID (chem stack, containment OFF):

- ``euclidean_shortcut`` → relation 7

Frozen rules v2 (docs/specs/v66_chem_MVP/ablation_euclidean_reach.md):

1. Graph distance on baseline role+chem undirected multigraph > 6 hops
2. Same chain: |auth seq| ≥ 10; different chains: seq rule waived
3. Top-K=2 nearest Euclidean neighbors among survivors with Cα–Cα ≤ 20 Å

Mutually exclusive with ``containment_edge_mp`` (both claim relation 7).
Does **not** write Normalizer / ``fact_graph_edge``. No SSE / hierarchy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from science.dtie.common.dehydron_barcode_features import EDGE_BARCODE_DIM
from science.dtie.v66.chem_edge_graph import (
    EDGE_ATTR_CHEM_DIM,
    NUM_ROLE_RELATIONS_WITH_CHEM,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM, parse_auth_seq_ids

ROLE_AUX_DIM = 1 + EDGE_BARCODE_DIM

ROLE_EUCLIDEAN_SHORTCUT = 7
NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT = 8
ROLE_ONEHOT_DIM_EUC = NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT
SPOKE_RHO_COL_EUC = GEO_DIM + ROLE_ONEHOT_DIM_EUC
COUPLING_STRENGTH_COL_EUC = SPOKE_RHO_COL_EUC
DEHYDRON_BARCODE_COL_EUC = SPOKE_RHO_COL_EUC + 1
EDGE_ATTR_EUC_DIM = GEO_DIM + ROLE_ONEHOT_DIM_EUC + ROLE_AUX_DIM

# Top-K governor (ablation v2)
EUC_SPATIAL_CUTOFF_A = 20.0
EUC_TOP_K = 2
EUC_MIN_GRAPH_HOPS = 6
EUC_MIN_SEQ_SEP = 10


def pad_chem_edge_attr_for_euc_shortcut(
    edge_attr: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Insert one zero one-hot column after chem's 7 slots (before aux)."""
    is_torch = isinstance(edge_attr, torch.Tensor)
    ea = edge_attr if is_torch else np.asarray(edge_attr)
    if ea.shape[-1] == EDGE_ATTR_EUC_DIM:
        return ea
    if ea.shape[-1] != EDGE_ATTR_CHEM_DIM:
        raise ValueError(
            f"expected chem edge_attr width {EDGE_ATTR_CHEM_DIM} or "
            f"{EDGE_ATTR_EUC_DIM}, got {ea.shape[-1]}"
        )
    head = ea[..., : GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM]
    aux = ea[..., GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM :]
    if is_torch:
        zeros = torch.zeros(*ea.shape[:-1], 1, dtype=ea.dtype, device=ea.device)
        return torch.cat([head, zeros, aux], dim=-1)
    zeros = np.zeros((*ea.shape[:-1], 1), dtype=ea.dtype)
    return np.concatenate([head, zeros, aux], axis=-1)


def _parse_chains(residue_ids: Sequence[str] | None, n: int) -> np.ndarray:
    """Chain labels from ``A:32:``-style ids; fallback ``A``."""
    out = np.array(["A"] * n, dtype=object)
    if residue_ids is None or len(residue_ids) != n:
        return out
    for i, rid in enumerate(residue_ids):
        parts = str(rid).strip().split(":")
        if parts:
            out[i] = parts[0] or "A"
    return out


def undirected_hop_distances(
    edge_index: torch.Tensor | np.ndarray, num_nodes: int
) -> np.ndarray:
    """All-pairs unweighted hop distances on the undirected baseline graph."""
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().cpu().numpy()
    else:
        ei = np.asarray(edge_index)
    if ei.size == 0:
        dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
        np.fill_diagonal(dist, 0.0)
        return dist
    src = ei[0].astype(np.int64)
    dst = ei[1].astype(np.int64)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=np.float64)
    adj = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = (adj > 0).astype(np.float64)
    return shortest_path(csgraph=adj, directed=False, unweighted=True)


def compute_euclidean_shortcut_pairs(
    coords: np.ndarray,
    edge_index_baseline: torch.Tensor | np.ndarray,
    *,
    residue_ids: Sequence[str] | None,
    spatial_cutoff: float = EUC_SPATIAL_CUTOFF_A,
    min_hops: int = EUC_MIN_GRAPH_HOPS,
    min_seq_sep: int = EUC_MIN_SEQ_SEP,
    top_k: int = EUC_TOP_K,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return undirected pairs ``(i[], j[])`` with i < j under Top-K governor.

    For each node, among partners with hop > min_hops, seq/chain rules, and
    Euclidean distance ≤ spatial_cutoff, keep the ``top_k`` nearest.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N,3], got {coords.shape}")
    k = max(int(top_k), 0)

    seq = parse_auth_seq_ids(residue_ids, n)
    chains = _parse_chains(residue_ids, n)
    hop = undirected_hop_distances(edge_index_baseline, n)
    dmat = cdist(coords, coords, metric="euclidean")

    pair_set: set[tuple[int, int]] = set()
    n_cand = 0
    n_seq_fail = 0
    n_hop_fail = 0
    n_dist_fail = 0

    for i in range(n):
        cand: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            d = float(dmat[i, j])
            if not (1e-6 < d <= float(spatial_cutoff)):
                n_dist_fail += 1
                continue
            same_chain = chains[i] == chains[j]
            if same_chain:
                if abs(int(seq[i]) - int(seq[j])) < int(min_seq_sep):
                    n_seq_fail += 1
                    continue
            # Cross-chain pairs are ALWAYS eligible for Top-K (seq exclusion waived).
            # Required for IGPS HisF↔HisH void; do NOT require same_chain.
            h = float(hop[i, j])
            if not (h > float(min_hops)):
                n_hop_fail += 1
                continue
            n_cand += 1
            cand.append((d, j))
        cand.sort(key=lambda t: t[0])
        for _, j in cand[:k]:
            a, b = (i, j) if i < j else (j, i)
            pair_set.add((a, b))

    keep_i = np.asarray([p[0] for p in sorted(pair_set)], dtype=np.int64)
    keep_j = np.asarray([p[1] for p in sorted(pair_set)], dtype=np.int64)

    finite = hop[np.isfinite(hop)]
    diameter = float(finite.max()) if finite.size else 0.0
    stats: dict[str, float | int] = {
        "n_nodes": n,
        "top_k": k,
        "spatial_cutoff_a": float(spatial_cutoff),
        "n_candidate_directed": n_cand,
        "n_seq_fail": n_seq_fail,
        "n_hop_fail": n_hop_fail,
        "n_dist_fail": n_dist_fail,
        "n_shortcut_undirected": int(len(pair_set)),
        "max_undirected_bound": int(n * k),
        "graph_diameter": diameter,
    }
    return keep_i, keep_j, stats


def attach_euclidean_shortcut_graph(
    data: Data,
    ca_coords: torch.Tensor | np.ndarray,
    *,
    residue_ids: Sequence[str] | None,
    force: bool = False,
) -> Data:
    """Pad chem ``edge_attr`` and append bidirectional ``euclidean_shortcut`` rows."""
    if not isinstance(data, Data):
        return data
    if getattr(data, "euclidean_shortcut_graph", False) and not force:
        return data
    if not getattr(data, "chem_edge_graph", False) and not force:
        raise ValueError(
            "attach_euclidean_shortcut_graph requires chem_edge_graph first"
        )

    if isinstance(ca_coords, torch.Tensor):
        coords_np = ca_coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(ca_coords, dtype=np.float64)

    n = int(data.x.size(0))
    if coords_np.shape[0] != n:
        raise ValueError(f"ca_coords length {coords_np.shape[0]} != data.x N={n}")

    device = data.edge_index.device
    dtype = data.edge_attr.dtype if data.edge_attr is not None else torch.float32

    baseline_index = data.edge_index.detach().clone()
    padded = pad_chem_edge_attr_for_euc_shortcut(data.edge_attr)

    src_u, dst_u, stats = compute_euclidean_shortcut_pairs(
        coords_np,
        baseline_index,
        residue_ids=residue_ids,
    )

    new_src: list[int] = []
    new_dst: list[int] = []
    new_rows: list[torch.Tensor] = []
    for i, j in zip(src_u.tolist(), dst_u.tolist(), strict=True):
        diff = coords_np[j] - coords_np[i]
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            continue
        for s, t, vec in ((i, j, diff), (j, i, -diff)):
            row = torch.zeros(EDGE_ATTR_EUC_DIM, dtype=dtype, device=device)
            row[0] = float(vec[0])
            row[1] = float(vec[1])
            row[2] = float(vec[2])
            row[3] = dist
            row[GEO_DIM + ROLE_EUCLIDEAN_SHORTCUT] = 1.0
            new_src.append(s)
            new_dst.append(t)
            new_rows.append(row)

    if new_rows:
        euc_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        euc_attr = torch.stack(new_rows, dim=0)
        data.edge_index = torch.cat([data.edge_index, euc_index], dim=1)
        data.edge_attr = torch.cat([padded, euc_attr], dim=0)
    else:
        data.edge_attr = padded

    data.euclidean_shortcut_graph = True  # type: ignore[attr-defined]
    data.euclidean_shortcut_stats = stats  # type: ignore[attr-defined]
    data.euclidean_shortcut_count_undirected = int(  # type: ignore[attr-defined]
        stats["n_shortcut_undirected"]
    )
    return data


__all__ = [
    "ROLE_EUCLIDEAN_SHORTCUT",
    "NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT",
    "EDGE_ATTR_EUC_DIM",
    "SPOKE_RHO_COL_EUC",
    "COUPLING_STRENGTH_COL_EUC",
    "DEHYDRON_BARCODE_COL_EUC",
    "EUC_SPATIAL_CUTOFF_A",
    "EUC_TOP_K",
    "EUC_MIN_GRAPH_HOPS",
    "EUC_MIN_SEQ_SEP",
    "pad_chem_edge_attr_for_euc_shortcut",
    "undirected_hop_distances",
    "compute_euclidean_shortcut_pairs",
    "attach_euclidean_shortcut_graph",
]
