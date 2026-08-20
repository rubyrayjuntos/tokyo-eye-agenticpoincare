"""KRAS G12 residue-12 graft helpers (matched-state WT ← mut transplant).

GNN inputs are ρ/τ/ss (topology), not AA one-hots. Grafting copies mut's
node features + Cα at resseq 12 onto the WT scaffold and rebuilds contacts.
Asp vs Gly enters via deposited local physics/geometry at that site.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from science.dtie.common.kras_topo_matrix import parse_resseq, residue_index_map, spearman_rho
from science.dtie.v5.gnn.model import precompute_clustering

EDGE_CUTOFF = 10.0
GRAFT_RESSEQ = 12
DEFAULT_SCRAMBLE_RESSEQ = 80  # distal control donor on mut (single-site)
SWITCH_LOCK_PARTNERS = (32, 60, 61)
SWITCH_I_RANGE = range(25, 41)  # inclusive 25–40
SWITCH_II_RANGE = range(57, 76)  # inclusive 57–75


def ca_neighbors_of_resseq(
    prot: dict[str, Any],
    resseq: int,
    *,
    cutoff_a: float = EDGE_CUTOFF,
) -> list[int]:
    """Resseqs with Cα distance ≤ cutoff to ``resseq`` (excludes self)."""
    idx_map = residue_index_map(list(prot.get("residue_ids") or []))
    if resseq not in idx_map:
        raise KeyError(f"missing resseq {resseq}")
    ca = prot["ca_coords"]
    ca = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
    i0 = idx_map[resseq]
    neighbors: list[int] = []
    for rs, i in idx_map.items():
        if rs == resseq:
            continue
        if i >= ca.shape[0]:
            continue
        d = float(np.linalg.norm(ca[i] - ca[i0]))
        if 1e-8 < d <= float(cutoff_a):
            neighbors.append(int(rs))
    return sorted(neighbors)


def neighborhood_n12(
    mut: dict[str, Any],
    wt: dict[str, Any],
    *,
    seed_resseq: int = GRAFT_RESSEQ,
    cutoff_a: float = EDGE_CUTOFF,
    include_switch_lock: bool = True,
) -> list[int]:
    """Build N₁₂ = seed ∪ mut Cα-neighbors ∪ optional Switch lock partners ∩ WT."""
    wt_map = residue_index_map(list(wt.get("residue_ids") or []))
    mut_map = residue_index_map(list(mut.get("residue_ids") or []))
    n_set = {seed_resseq}
    n_set.update(ca_neighbors_of_resseq(mut, seed_resseq, cutoff_a=cutoff_a))
    if include_switch_lock:
        n_set.update(SWITCH_LOCK_PARTNERS)
    # Present on both WT and mut
    out = sorted(r for r in n_set if r in wt_map and r in mut_map)
    if seed_resseq not in out:
        raise ValueError(f"seed {seed_resseq} missing from WT∩mut intersection")
    return out


def distal_scramble_donors(
    mut: dict[str, Any],
    *,
    n_needed: int,
    exclude: set[int] | None = None,
) -> list[int]:
    """Sorted mut resseqs outside seed/Switch bands, length ``n_needed``."""
    mut_map = residue_index_map(list(mut.get("residue_ids") or []))
    banned = set(exclude or ())
    banned.add(GRAFT_RESSEQ)
    banned.update(SWITCH_I_RANGE)
    banned.update(SWITCH_II_RANGE)
    banned.update(SWITCH_LOCK_PARTNERS)
    candidates = sorted(r for r in mut_map if r not in banned)
    if len(candidates) < n_needed:
        raise ValueError(
            f"need {n_needed} distal scramble donors, only {len(candidates)} available"
        )
    return candidates[:n_needed]


def graft_resseq_map(
    wt: dict[str, Any],
    mut: dict[str, Any],
    *,
    target_to_donor: dict[int, int],
) -> dict[str, Any]:
    """Copy mut features+Cα for each target←donor pair onto WT; rebuild contacts."""
    wt_map = residue_index_map(list(wt.get("residue_ids") or []))
    mut_map = residue_index_map(list(mut.get("residue_ids") or []))
    out = copy.deepcopy(wt)
    data = out["data"]
    x = data.x.detach().cpu().clone()
    x_mut = mut["data"].x.detach().cpu()
    sasa = None
    sasa_mut = None
    if hasattr(data, "sasa") and data.sasa is not None:
        sasa = data.sasa.detach().cpu().clone()
        if hasattr(mut["data"], "sasa") and mut["data"].sasa is not None:
            sasa_mut = mut["data"].sasa.detach().cpu()

    ca_wt = out["ca_coords"]
    ca_wt = ca_wt.detach().cpu().numpy() if torch.is_tensor(ca_wt) else np.asarray(ca_wt)
    ca_mut = mut["ca_coords"]
    ca_mut = ca_mut.detach().cpu().numpy() if torch.is_tensor(ca_mut) else np.asarray(ca_mut)
    ca_np = np.asarray(ca_wt, dtype=np.float64).copy()

    pairs: list[dict[str, int]] = []
    for target_rs, donor_rs in sorted(target_to_donor.items()):
        if target_rs not in wt_map:
            raise KeyError(f"WT missing target {target_rs}")
        if donor_rs not in mut_map:
            raise KeyError(f"mut missing donor {donor_rs}")
        i_wt = wt_map[target_rs]
        i_mut = mut_map[donor_rs]
        if x_mut.shape[-1] != x.shape[-1]:
            d = min(x.shape[-1], x_mut.shape[-1])
            x[i_wt, :d] = x_mut[i_mut, :d]
        else:
            x[i_wt] = x_mut[i_mut]
        if sasa is not None and sasa_mut is not None and i_mut < sasa_mut.shape[0]:
            sasa[i_wt] = sasa_mut[i_mut]
        ca_np[i_wt] = np.asarray(ca_mut[i_mut], dtype=np.float64)
        pairs.append(
            {
                "target_resseq": int(target_rs),
                "donor_resseq": int(donor_rs),
                "wt_index": int(i_wt),
                "mut_index": int(i_mut),
            }
        )

    data.x = x
    if sasa is not None:
        data.sasa = sasa
    out = rebuild_ca_graph(out, ca_np)
    out["graft_meta"] = {"pairs": pairs, "n_sites": len(pairs)}
    return out


def graft_neighborhood_matched(
    wt: dict[str, Any],
    mut: dict[str, Any],
    n12: list[int],
) -> dict[str, Any]:
    """Matched neighborhood graft: each r∈N₁₂ gets mut's residue r."""
    mapping = {r: r for r in n12}
    out = graft_resseq_map(wt, mut, target_to_donor=mapping)
    out["graft_meta"]["kind"] = "neighborhood_matched"
    out["graft_meta"]["n12"] = list(n12)
    return out


def graft_neighborhood_scramble(
    wt: dict[str, Any],
    mut: dict[str, Any],
    n12: list[int],
) -> dict[str, Any]:
    """Scramble: sorted distal donors mapped onto sorted N₁₂."""
    donors = distal_scramble_donors(mut, n_needed=len(n12))
    targets = sorted(n12)
    mapping = {t: d for t, d in zip(targets, donors)}
    out = graft_resseq_map(wt, mut, target_to_donor=mapping)
    out["graft_meta"]["kind"] = "neighborhood_scramble"
    out["graft_meta"]["n12"] = list(n12)
    out["graft_meta"]["scramble_donors"] = list(donors)
    return out


def neighborhood_verdict(
    *,
    rho_wt_mut: float,
    rho_single_mut: float,
    rho_neighborhood_mut: float,
    rho_scramble_mut: float,
) -> dict[str, Any]:
    """Pass: neighborhood Δρ vs WT exceeds single-site Δρ, and beats scramble."""
    d_single = float(rho_single_mut - rho_wt_mut)
    d_neigh = float(rho_neighborhood_mut - rho_wt_mut)
    improved_more = bool(
        np.isfinite(rho_neighborhood_mut)
        and np.isfinite(rho_single_mut)
        and np.isfinite(rho_wt_mut)
        and d_neigh > d_single
    )
    beats_scramble = bool(
        np.isfinite(rho_neighborhood_mut)
        and np.isfinite(rho_scramble_mut)
        and rho_neighborhood_mut > rho_scramble_mut
    )
    ok = improved_more and beats_scramble
    return {
        "pass": ok,
        "gates": {
            "neighborhood_delta_exceeds_single_site": improved_more,
            "neighborhood_beats_scramble": beats_scramble,
        },
        "rho_wt_mut": float(rho_wt_mut),
        "rho_single_mut": float(rho_single_mut),
        "rho_neighborhood_mut": float(rho_neighborhood_mut),
        "rho_scramble_mut": float(rho_scramble_mut),
        "delta_rho_single": d_single,
        "delta_rho_neighborhood": d_neigh,
        "rule": (
            "Δρ_neigh = ρ(N,mut)−ρ(WT,mut) > Δρ_single = ρ(site12,mut)−ρ(WT,mut) "
            "AND ρ(N,mut) > ρ(scramble,mut)"
        ),
        "outcome": "Pass" if ok else "Fail",
    }

def rebuild_ca_graph(prot: dict[str, Any], ca_np: np.ndarray) -> dict[str, Any]:
    """Deep-copy prot and rebuild Cα contact edges from coordinates."""
    out = copy.deepcopy(prot)
    ca_np = np.asarray(ca_np, dtype=np.float64)
    n = ca_np.shape[0]
    data0 = out["data"]
    x = data0.x.detach().cpu().clone()
    if x.shape[0] != n:
        raise ValueError(f"ca/x length mismatch: {n} vs {x.shape[0]}")

    dists = cdist(ca_np, ca_np)
    src, dst = np.where((dists < EDGE_CUTOFF) & (dists > 0.1))
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    rel_pos = ca_np[dst] - ca_np[src]
    edge_dist = dists[src, dst]
    edge_attr = np.column_stack([rel_pos, edge_dist]).astype(np.float32)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    if hasattr(data0, "sasa") and data0.sasa is not None:
        data.sasa = data0.sasa.detach().cpu().clone()
    data = precompute_clustering(data)
    out["data"] = data
    out["ca_coords"] = torch.tensor(ca_np, dtype=torch.float32)
    return out


def graft_resseq_from_donor(
    wt: dict[str, Any],
    mut: dict[str, Any],
    *,
    target_resseq: int = GRAFT_RESSEQ,
    donor_resseq: int | None = None,
) -> dict[str, Any]:
    """Copy mut node features + Cα at ``donor_resseq`` onto WT at ``target_resseq``.

    Default: matched G12 graft (donor_resseq=target_resseq=12).
    Scramble: donor_resseq=80 (or other distal), target_resseq=12.
    """
    if donor_resseq is None:
        donor_resseq = target_resseq

    wt_ids = list(wt.get("residue_ids") or [])
    mut_ids = list(mut.get("residue_ids") or [])
    wt_map = residue_index_map(wt_ids)
    mut_map = residue_index_map(mut_ids)
    if target_resseq not in wt_map:
        raise KeyError(f"WT missing target resseq {target_resseq}")
    if donor_resseq not in mut_map:
        raise KeyError(f"mut missing donor resseq {donor_resseq}")

    i_wt = wt_map[target_resseq]
    i_mut = mut_map[donor_resseq]

    out = copy.deepcopy(wt)
    data = out["data"]
    x = data.x.detach().cpu().clone()
    x_mut = mut["data"].x.detach().cpu()
    if x_mut.shape[-1] != x.shape[-1]:
        # Align feature width to WT (topology truncate if needed).
        d = min(x.shape[-1], x_mut.shape[-1])
        x[i_wt, :d] = x_mut[i_mut, :d]
    else:
        x[i_wt] = x_mut[i_mut]
    data.x = x
    if hasattr(data, "sasa") and data.sasa is not None and hasattr(mut["data"], "sasa"):
        sasa = data.sasa.detach().cpu().clone()
        sasa_mut = mut["data"].sasa.detach().cpu()
        if sasa_mut is not None and i_mut < sasa_mut.shape[0]:
            sasa[i_wt] = sasa_mut[i_mut]
            data.sasa = sasa

    ca_wt = out["ca_coords"]
    ca_wt = ca_wt.detach().cpu().numpy() if torch.is_tensor(ca_wt) else np.asarray(ca_wt)
    ca_mut = mut["ca_coords"]
    ca_mut = ca_mut.detach().cpu().numpy() if torch.is_tensor(ca_mut) else np.asarray(ca_mut)
    ca_np = np.asarray(ca_wt, dtype=np.float64).copy()
    ca_np[i_wt] = np.asarray(ca_mut[i_mut], dtype=np.float64)
    out = rebuild_ca_graph(out, ca_np)
    out["graft_meta"] = {
        "target_resseq": int(target_resseq),
        "donor_resseq": int(donor_resseq),
        "wt_index": int(i_wt),
        "mut_index": int(i_mut),
    }
    return out


def align_out_effects(
    values_by_label: dict[str, np.ndarray],
    maps_by_label: dict[str, dict[int, int]],
    labels: list[str],
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Intersect resseqs and return aligned out_effect vectors."""
    shared = set(maps_by_label[labels[0]].keys())
    for lab in labels[1:]:
        shared &= set(maps_by_label[lab].keys())
    shared_sorted = sorted(shared)
    if not shared_sorted:
        raise ValueError("empty residue intersection")
    aligned: dict[str, np.ndarray] = {}
    for lab in labels:
        vals = np.asarray(values_by_label[lab], dtype=np.float64).reshape(-1)
        idx_map = maps_by_label[lab]
        aligned[lab] = np.asarray([vals[idx_map[r]] for r in shared_sorted], dtype=np.float64)
    return shared_sorted, aligned


def graft_direction_verdict(
    *,
    rho_wt_mut: float,
    rho_graft_mut: float,
    rho_scramble_mut: float,
) -> dict[str, Any]:
    """Pass: graft closer to mut than WT is, and beat scramble."""
    improved = bool(
        np.isfinite(rho_graft_mut)
        and np.isfinite(rho_wt_mut)
        and rho_graft_mut > rho_wt_mut
    )
    beats_scramble = bool(
        np.isfinite(rho_graft_mut)
        and np.isfinite(rho_scramble_mut)
        and rho_graft_mut > rho_scramble_mut
    )
    ok = improved and beats_scramble
    return {
        "pass": ok,
        "gates": {
            "graft_closer_to_mut_than_wt": improved,
            "graft_beats_scramble": beats_scramble,
        },
        "rho_wt_mut": float(rho_wt_mut),
        "rho_graft_mut": float(rho_graft_mut),
        "rho_scramble_mut": float(rho_scramble_mut),
        "delta_rho_graft_vs_wt": float(rho_graft_mut - rho_wt_mut)
        if np.isfinite(rho_graft_mut) and np.isfinite(rho_wt_mut)
        else float("nan"),
        "rule": "ρ(graft,mut) > ρ(WT,mut) AND ρ(graft,mut) > ρ(scramble,mut)",
    }


def delta_field_spearman(
    oe_a: np.ndarray,
    oe_b: np.ndarray,
    oe_ref_a: np.ndarray,
    oe_ref_b: np.ndarray,
) -> float:
    """Spearman of (oe_b - oe_a) vs (oe_ref_b - oe_ref_a)."""
    return spearman_rho(oe_b - oe_a, oe_ref_b - oe_ref_a)


def splice_x_hyp(
    x_hyp_host: torch.Tensor,
    host_map: dict[int, int],
    x_hyp_donor: torch.Tensor,
    donor_map: dict[int, int],
    *,
    target_to_donor: dict[int, int],
    curvature: float | torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, list[dict[str, int]]]:
    """Copy donor ball coords onto host at target←donor pairs; return depth.

    Does **not** re-run Euclidean MP. Depth is ``dist0`` of the spliced ball
    (post-lift intervention only).
    """
    from geoopt.manifolds.stereographic import math as pmath

    if x_hyp_host.ndim != 2 or x_hyp_donor.ndim != 2:
        raise ValueError("x_hyp must be [N, D]")
    out = x_hyp_host.detach().float().clone()
    donor = x_hyp_donor.detach().float()
    if out.shape[-1] != donor.shape[-1]:
        raise ValueError(
            f"x_hyp dim mismatch host={out.shape[-1]} donor={donor.shape[-1]}"
        )
    c = float(curvature.detach().cpu()) if torch.is_tensor(curvature) else float(curvature)
    k = torch.tensor(-c, dtype=out.dtype, device=out.device)
    pairs: list[dict[str, int]] = []
    for target_rs, donor_rs in sorted(target_to_donor.items()):
        if target_rs not in host_map:
            raise KeyError(f"host missing target {target_rs}")
        if donor_rs not in donor_map:
            raise KeyError(f"donor missing {donor_rs}")
        i_h = host_map[target_rs]
        i_d = donor_map[donor_rs]
        out[i_h] = donor[i_d]
        pairs.append(
            {
                "target_resseq": int(target_rs),
                "donor_resseq": int(donor_rs),
                "host_index": int(i_h),
                "donor_index": int(i_d),
            }
        )
    out = pmath.project(out, k=k)
    depth = pmath.dist0(out, k=k, keepdim=False).detach().cpu().numpy().astype(np.float64)
    return out, depth, pairs


def hyp_latent_neighborhood_matched(
    x_hyp_wt: torch.Tensor,
    map_wt: dict[int, int],
    x_hyp_mut: torch.Tensor,
    map_mut: dict[int, int],
    n12: list[int],
    *,
    curvature: float | torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
    """Matched hyp graft: each r∈N₁₂ gets mut's ``x_hyp`` at r."""
    mapping = {r: r for r in n12}
    x_out, depth, pairs = splice_x_hyp(
        x_hyp_wt,
        map_wt,
        x_hyp_mut,
        map_mut,
        target_to_donor=mapping,
        curvature=curvature,
    )
    meta = {"kind": "hyp_neighborhood_matched", "n12": list(n12), "pairs": pairs}
    return x_out, depth, meta


def hyp_latent_neighborhood_scramble(
    x_hyp_wt: torch.Tensor,
    map_wt: dict[int, int],
    x_hyp_mut: torch.Tensor,
    map_mut: dict[int, int],
    n12: list[int],
    *,
    curvature: float | torch.Tensor,
    mut_prot_for_donors: dict[str, Any],
) -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
    """Hyp scramble: distal mut ``x_hyp`` donors onto sorted N₁₂."""
    donors = distal_scramble_donors(mut_prot_for_donors, n_needed=len(n12))
    targets = sorted(n12)
    mapping = {t: d for t, d in zip(targets, donors)}
    x_out, depth, pairs = splice_x_hyp(
        x_hyp_wt,
        map_wt,
        x_hyp_mut,
        map_mut,
        target_to_donor=mapping,
        curvature=curvature,
    )
    meta = {
        "kind": "hyp_neighborhood_scramble",
        "n12": list(n12),
        "scramble_donors": list(donors),
        "pairs": pairs,
    }
    return x_out, depth, meta


def hyp_latent_verdict(
    *,
    rho_wt_mut: float,
    rho_hyp_graft_mut: float,
    rho_euc_neigh_mut: float,
    rho_hyp_scramble_mut: float,
) -> dict[str, Any]:
    """Pass: hyp graft Δρ > euc-neigh Δρ (both on hyp depth), and beats hyp scramble."""
    d_hyp = float(rho_hyp_graft_mut - rho_wt_mut)
    d_euc = float(rho_euc_neigh_mut - rho_wt_mut)
    beats_euc = bool(
        np.isfinite(rho_hyp_graft_mut)
        and np.isfinite(rho_euc_neigh_mut)
        and np.isfinite(rho_wt_mut)
        and d_hyp > d_euc
    )
    beats_scramble = bool(
        np.isfinite(rho_hyp_graft_mut)
        and np.isfinite(rho_hyp_scramble_mut)
        and rho_hyp_graft_mut > rho_hyp_scramble_mut
    )
    ok = beats_euc and beats_scramble
    return {
        "pass": ok,
        "gates": {
            "hyp_delta_exceeds_euc_neighborhood": beats_euc,
            "hyp_graft_beats_hyp_scramble": beats_scramble,
        },
        "rho_wt_mut": float(rho_wt_mut),
        "rho_hyp_graft_mut": float(rho_hyp_graft_mut),
        "rho_euc_neigh_mut": float(rho_euc_neigh_mut),
        "rho_hyp_scramble_mut": float(rho_hyp_scramble_mut),
        "delta_rho_hyp": d_hyp,
        "delta_rho_euc_neigh": d_euc,
        "rule": (
            "Δρ_hyp = ρ(depth_hyp_graft,mut)−ρ(WT,mut) > "
            "Δρ_euc = ρ(depth_euc_neigh,mut)−ρ(WT,mut) "
            "AND ρ(depth_hyp_graft,mut) > ρ(depth_hyp_scramble,mut)"
        ),
        "outcome": "Pass" if ok else "Fail",
    }


__all__ = [
    "DEFAULT_SCRAMBLE_RESSEQ",
    "EDGE_CUTOFF",
    "GRAFT_RESSEQ",
    "SWITCH_I_RANGE",
    "SWITCH_II_RANGE",
    "SWITCH_LOCK_PARTNERS",
    "align_out_effects",
    "ca_neighbors_of_resseq",
    "delta_field_spearman",
    "distal_scramble_donors",
    "graft_direction_verdict",
    "graft_neighborhood_matched",
    "graft_neighborhood_scramble",
    "graft_resseq_from_donor",
    "graft_resseq_map",
    "hyp_latent_neighborhood_matched",
    "hyp_latent_neighborhood_scramble",
    "hyp_latent_verdict",
    "neighborhood_n12",
    "neighborhood_verdict",
    "parse_resseq",
    "rebuild_ca_graph",
    "residue_index_map",
    "spearman_rho",
    "splice_x_hyp",
]
