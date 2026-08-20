#!/usr/bin/env python3
"""Gradient-influence / edge-importance probe: euc_shortcut vs dehydron/spoke.

Hypothesis under test (chem-mvp euc-reach Fail autopsy):
  Euclidean shortcuts cause **Signal Dilution via Shortcut Competition** —
  MP routes through spatial wormholes, starving dehydron/spoke biology paths.

Methods (honest; see SIGNAL_SIPHON_PROBE.md):

1. **Primary — relation message-ablation**
   Remove all directed edges of a relation class; measure signed Δ on
   mean ``cone_depth`` over frozen pathway set S (and |Δ| share across classes).
   Hard one-hots in ``EquivariantConvMultiRel`` make one-hot IG near-useless;
   ablation is the causal edge-class instrument.

2. **Secondary — geo-channel gradient × activation**
   ``attr_e = Σ_{c∈{0..3}} |∂L/∂edge_attr[e,c] · edge_attr[e,c]|`` with
   ``L = Σ_{i∈S} cone_depth_i``, aggregated by relation one-hot mask.
   Captures continuous (Δx,Δy,Δz,dist) sensitivity only.

Target metric: model ``cone_depth`` = ``dist0(x_hyp)`` after learned lift from
Euclidean MP trunk (not structural SSOT; not disc-alone; not ``cone_depth_routed``).

v66 only. No training. Writes under
``checkpoints/v66/diagnostics/euclidean_reach/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.diagnostics.euclidean_reach_grade_first_ckpt import (
    BASELINE_CKPT,
    DEFAULT_OUT,
    REPO,
    SUITE,
    load_frozen_S,
)
from experiments.training.v66.manifold_ssot import load_suite_prot
from experiments.training.v66.train_loop import (
    attach_v6_features,
    prepare_training_batch,
    residue_node_count,
)
from science.dtie.v66.chem_edge_graph import (
    NUM_ROLE_RELATIONS_WITH_CHEM,
    ROLE_COVALE,
    ROLE_DISULF,
    attach_chem_edge_graph,
)
from science.dtie.v66.euclidean_shortcut_graph import (
    NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT,
    ROLE_EUCLIDEAN_SHORTCUT,
    attach_euclidean_shortcut_graph,
)
from science.dtie.v66.role_edge_graph import (
    ROLE_COUPLING,
    ROLE_DEHYDRON,
    ROLE_PACKING,
    ROLE_RIBBON,
    ROLE_SPOKE,
    attach_role_edge_graph,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM
from science.training.gnn_lineage import load_model_from_checkpoint

SITE_DIR = DEFAULT_OUT / "site_lists"
TREATMENT_CKPT = REPO / "checkpoints/v66/runs/chem_mvp_euc_reach_v1/v66_best.pt"

REL_NAME = {
    ROLE_PACKING: "packing",
    ROLE_DEHYDRON: "dehydron",
    ROLE_SPOKE: "spoke",
    ROLE_RIBBON: "ribbon",
    ROLE_COUPLING: "coupling",
    ROLE_DISULF: "disulf",
    ROLE_COVALE: "covale",
    ROLE_EUCLIDEAN_SHORTCUT: "euclidean_shortcut",
}

# Report buckets requested by the scientific ask.
BUCKETS: dict[str, tuple[int, ...]] = {
    "euc_shortcut": (ROLE_EUCLIDEAN_SHORTCUT,),
    "dehydron_spoke": (ROLE_DEHYDRON, ROLE_SPOKE),
    "chem": (ROLE_DISULF, ROLE_COVALE),
    "other_role": (ROLE_PACKING, ROLE_RIBBON, ROLE_COUPLING),
}


def _parse_chain_auth(rid: Any) -> tuple[str, int]:
    parts = str(rid).split(":")
    return parts[0], int(parts[1])


def prepare_part0_parity_batch(
    model: torch.nn.Module, prot: dict[str, Any], device: str
) -> Any:
    """Rebuild with Part 0 feature/loader recipe so euc shortcuts can appear.

    Grade ``load_suite_prot`` + ``resolve_residue_records`` yields denser role
    graphs (1GPW diameter ~7 → 0 shortcuts). Part 0 used its own multichain CA
    loader (different ρ/τ) without resolved records (diameter ~63 → 684 sc).

    This returns a batch built from the **Part 0 prot**, not a surgical edit of
    the grade graph. Counterfactual only — ρ/τ differ from grade SSOT.
    """
    from experiments.diagnostics.euclidean_reach_part0 import (
        DEFAULT_PDB_DIRS,
        _ensure_pdb,
        _load_multichain_ca,
        _load_single_chain,
    )

    pdb_id = str(prot.get("pdb_id") or "").upper()
    chains_raw = str(prot.get("chain") or "A")
    chains = tuple(c for c in chains_raw.split("+") if c)
    path = _ensure_pdb(pdb_id, list(DEFAULT_PDB_DIRS))
    if len(chains) > 1:
        prot_p0 = _load_multichain_ca(path, pdb_id, chains=chains)
    else:
        prot_p0 = _load_single_chain(path, chains[0] if chains else "A")
        prot_p0["pdb_id"] = pdb_id

    # Ensure Data wrapper for attach_v6_features (node_emb expects 3 feats).
    if "data" not in prot_p0:
        x = prot_p0["x"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.size(-1) > 3:
            x = x[:, :3].contiguous()
        prot_p0["data"] = Data(
            x=x.clone(),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 4),
        )
    else:
        data0 = prot_p0["data"]
        if data0.x is not None and data0.x.size(-1) > 3:
            if getattr(data0, "sasa", None) is None:
                data0.sasa = data0.x[:, 3].detach().clone()
            data0.x = data0.x[:, :3].contiguous()
        prot_p0["data"] = data0

    data = attach_v6_features(prot_p0["data"].clone().to(device))
    if data.x.size(-1) > 3:
        data.x = data.x[:, :3].contiguous()
    ca = prot_p0.get("ca_coords")
    if ca is None:
        raise ValueError("part0 parity requires ca_coords")
    ca_t = ca.to(device) if hasattr(ca, "to") else ca
    residue_ids = prot_p0.get("residue_ids")
    data = attach_role_edge_graph(
        data,
        ca_t,
        residue_ids=residue_ids,
        residue_records=None,
        dehydron_edge_lookup=prot_p0.get("dehydron_edge_lookup"),
        enable_coupling_edges=bool(getattr(model, "role_coupling_edges", False)),
        dehydron_exclusivity=bool(getattr(model, "dehydron_exclusivity", True)),
        spoke_edge_scale=float(getattr(model, "spoke_edge_scale", 1.0)),
        ribbon_edge_scale=float(getattr(model, "ribbon_edge_scale", 1.0)),
        ha_edges=bool(getattr(model, "ha_edge_mp", False)),
    )
    if bool(getattr(model, "chem_edge_mp", False)):
        if residue_ids is None:
            raise ValueError("chem_edge_mp requires residue_ids")
        data = attach_chem_edge_graph(
            data,
            ca_t,
            prot_p0.get("covalent_bonds") or [],
            residue_ids=residue_ids,
            structure_id=str(pdb_id.lower()),
            chain_label=str(prot_p0.get("chain") or "A"),
        )
        if bool(getattr(model, "euclidean_shortcut_mp", False)):
            data = attach_euclidean_shortcut_graph(
                data, ca_t, residue_ids=residue_ids
            )
    data.pdb_id = pdb_id  # type: ignore[attr-defined]
    # Stash residue ids for S remapping by caller
    data._part0_residue_ids = list(residue_ids or [])  # type: ignore[attr-defined]

    # Match grade batch side-channels (clustering / degree after final edges).
    from science.dtie.v5.gnn.model import precompute_clustering
    from torch_geometric.utils import degree as pyg_degree

    data = precompute_clustering(data)
    data.degree = pyg_degree(  # type: ignore[attr-defined]
        data.edge_index[0],
        num_nodes=data.x.size(0),
        dtype=data.x.dtype,
    )
    if not hasattr(data, "rho") or data.rho is None:
        data.rho = data.x[:, 0].clone()  # type: ignore[attr-defined]
    return data


def _num_relations(model: torch.nn.Module, edge_attr: torch.Tensor) -> int:
    convs = getattr(model, "convs", None)
    if convs is not None and len(convs) > 0 and hasattr(convs[0], "num_relations"):
        return int(convs[0].num_relations)
    # Fallback from attr width: GEO + onehot + aux
    w = int(edge_attr.size(-1))
    if w >= GEO_DIM + NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT + 1:
        return NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT
    return NUM_ROLE_RELATIONS_WITH_CHEM


def _relation_masks(
    edge_attr: torch.Tensor, *, n_rel: int
) -> dict[int, torch.Tensor]:
    onehots = edge_attr[:, GEO_DIM : GEO_DIM + n_rel]
    masks = {r: onehots[:, r] > 0.5 for r in range(n_rel)}
    assigned = torch.stack(list(masks.values()), dim=-1).any(dim=-1)
    if not bool(assigned.all()) and 0 in masks:
        masks[0] = masks[0] | (~assigned)
    return masks


def _edge_counts(masks: dict[int, torch.Tensor]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r, m in masks.items():
        out[REL_NAME.get(r, f"rel_{r}")] = int(m.sum().item())
    for bucket, ids in BUCKETS.items():
        out[f"bucket_{bucket}"] = int(
            sum(int(masks[r].sum().item()) for r in ids if r in masks)
        )
    out["total_directed"] = int(sum(int(m.sum().item()) for m in masks.values()))
    return out


def _clone_drop_relations(
    data: Data, drop_ids: tuple[int, ...], *, n_rel: int
) -> tuple[Data, int]:
    """Return a clone with directed edges of given relation IDs removed."""
    masks = _relation_masks(data.edge_attr, n_rel=n_rel)
    drop = torch.zeros(data.edge_attr.size(0), dtype=torch.bool, device=data.edge_attr.device)
    for r in drop_ids:
        if r in masks:
            drop = drop | masks[r]
    keep = ~drop
    d = data.clone()
    d.edge_index = data.edge_index[:, keep].contiguous()
    d.edge_attr = data.edge_attr[keep].contiguous()
    return d, int(drop.sum().item())


def _cone_on_S(
    model: torch.nn.Module, data: Data, S_indices: list[int]
) -> dict[str, float]:
    with torch.no_grad():
        out = model(data)
    depth = out["cone_depth"].detach().float().reshape(-1)
    n = depth.numel()
    s_idx = [i for i in S_indices if 0 <= i < n]
    if not s_idx:
        raise RuntimeError("empty S indices after clamp")
    s_t = torch.tensor(s_idx, device=depth.device, dtype=torch.long)
    mask = torch.zeros(n, dtype=torch.bool, device=depth.device)
    mask[s_t] = True
    mean_s = float(depth[s_t].mean().item())
    mean_c = float(depth[~mask].mean().item()) if bool((~mask).any()) else float("nan")
    return {
        "mean_cone_depth_S": mean_s,
        "mean_cone_depth_complement": mean_c,
        "gap_S_minus_complement": mean_s - mean_c,
        "sum_cone_depth_S": float(depth[s_t].sum().item()),
        "mean_cone_depth_all": float(depth.mean().item()),
    }


def ablation_attribution(
    model: torch.nn.Module,
    data: Data,
    S_indices: list[int],
    *,
    n_rel: int,
) -> dict[str, Any]:
    """Primary method: leave-one-relation-class-out message ablation."""
    base = _cone_on_S(model, data, S_indices)
    per_rel: dict[str, Any] = {}
    abs_deltas: dict[str, float] = {}

    for r in range(n_rel):
        name = REL_NAME.get(r, f"rel_{r}")
        ablated, n_dropped = _clone_drop_relations(data, (r,), n_rel=n_rel)
        if n_dropped == 0:
            per_rel[name] = {
                "relation_id": r,
                "n_edges_dropped": 0,
                "skipped": True,
                "delta_mean_cone_S": 0.0,
                "abs_delta_mean_cone_S": 0.0,
            }
            abs_deltas[name] = 0.0
            continue
        metrics = _cone_on_S(model, ablated, S_indices)
        d_mean = base["mean_cone_depth_S"] - metrics["mean_cone_depth_S"]
        d_gap = base["gap_S_minus_complement"] - metrics["gap_S_minus_complement"]
        per_rel[name] = {
            "relation_id": r,
            "n_edges_dropped": n_dropped,
            "skipped": False,
            "ablated": metrics,
            "delta_mean_cone_S": d_mean,
            "abs_delta_mean_cone_S": abs(d_mean),
            "delta_gap_S_minus_complement": d_gap,
            "abs_delta_gap": abs(d_gap),
            # Positive delta_mean_cone_S ⇒ those edges raised mean depth on S.
            "interpretation_hint": (
                "edges_raise_mean_cone_S"
                if d_mean > 0
                else "edges_lower_mean_cone_S"
                if d_mean < 0
                else "null"
            ),
        }
        abs_deltas[name] = abs(d_mean)

    # Bucket ablations (drop whole bucket at once) for the headline fractions.
    bucket_rows: dict[str, Any] = {}
    bucket_abs: dict[str, float] = {}
    for bucket, ids in BUCKETS.items():
        present = tuple(r for r in ids if r < n_rel)
        if not present:
            bucket_rows[bucket] = {
                "relation_ids": list(ids),
                "n_edges_dropped": 0,
                "skipped": True,
                "abs_delta_mean_cone_S": 0.0,
                "fraction_of_bucket_abs_mass": 0.0,
            }
            bucket_abs[bucket] = 0.0
            continue
        ablated, n_dropped = _clone_drop_relations(data, present, n_rel=n_rel)
        if n_dropped == 0:
            bucket_rows[bucket] = {
                "relation_ids": list(present),
                "n_edges_dropped": 0,
                "skipped": True,
                "delta_mean_cone_S": 0.0,
                "abs_delta_mean_cone_S": 0.0,
            }
            bucket_abs[bucket] = 0.0
            continue
        metrics = _cone_on_S(model, ablated, S_indices)
        d_mean = base["mean_cone_depth_S"] - metrics["mean_cone_depth_S"]
        d_gap = base["gap_S_minus_complement"] - metrics["gap_S_minus_complement"]
        bucket_rows[bucket] = {
            "relation_ids": list(present),
            "n_edges_dropped": n_dropped,
            "skipped": False,
            "ablated": metrics,
            "delta_mean_cone_S": d_mean,
            "abs_delta_mean_cone_S": abs(d_mean),
            "delta_gap_S_minus_complement": d_gap,
            "abs_delta_gap": abs(d_gap),
            "interpretation_hint": (
                "edges_raise_mean_cone_S"
                if d_mean > 0
                else "edges_lower_mean_cone_S"
                if d_mean < 0
                else "null"
            ),
        }
        bucket_abs[bucket] = abs(d_mean)

    total_b = sum(bucket_abs.values()) or 1e-12
    for bucket, row in bucket_rows.items():
        row["fraction_of_bucket_abs_mass"] = float(bucket_abs[bucket] / total_b)

    total_r = sum(abs_deltas.values()) or 1e-12
    per_rel_frac = {k: float(v / total_r) for k, v in abs_deltas.items()}

    return {
        "method": "relation_message_ablation",
        "target": "mean(cone_depth[S])",
        "baseline_metrics": base,
        "per_relation": per_rel,
        "per_relation_fraction_abs_delta_mean_cone_S": per_rel_frac,
        "buckets": bucket_rows,
        "headline_fractions_abs_delta_mean_cone_S": {
            k: float(bucket_abs[k] / total_b) for k in BUCKETS
        },
        "notes": [
            "Hard relation one-hots gate MP paths; ablation removes those messages.",
            "Fraction = |Δ mean cone_depth on S| / sum_buckets |Δ|.",
            "Does not equal Integrated Gradients on one-hots (those are non-diff).",
        ],
    }


def geo_grad_act_attribution(
    model: torch.nn.Module,
    data: Data,
    S_indices: list[int],
    *,
    n_rel: int,
) -> dict[str, Any]:
    """Secondary: |grad×act| on GEO_DIM channels, summed by relation mask."""
    model.zero_grad(set_to_none=True)
    d = data.clone()
    ea = d.edge_attr.detach().clone().requires_grad_(True)
    d.edge_attr = ea

    out = model(d)
    depth = out["cone_depth"].reshape(-1)
    s_idx = [i for i in S_indices if 0 <= i < depth.numel()]
    L = depth[torch.tensor(s_idx, device=depth.device, dtype=torch.long)].sum()
    L.backward()

    if ea.grad is None:
        raise RuntimeError("edge_attr.grad is None — cone_depth not connected to edges?")

    g = ea.grad
    # Continuous geometry only (rel_pos + dist). Role one-hots are hard-thresholded.
    per_edge = (g[:, :GEO_DIM] * ea[:, :GEO_DIM]).abs().sum(dim=-1).detach()
    masks = _relation_masks(ea.detach(), n_rel=n_rel)

    per_rel_mass: dict[str, float] = {}
    for r in range(n_rel):
        name = REL_NAME.get(r, f"rel_{r}")
        m = masks[r]
        per_rel_mass[name] = float(per_edge[m].sum().item()) if bool(m.any()) else 0.0

    bucket_mass: dict[str, float] = {}
    for bucket, ids in BUCKETS.items():
        mass = 0.0
        for r in ids:
            if r < n_rel:
                mass += per_rel_mass.get(REL_NAME.get(r, f"rel_{r}"), 0.0)
        bucket_mass[bucket] = mass

    total_b = sum(bucket_mass.values()) or 1e-12
    total_r = sum(per_rel_mass.values()) or 1e-12

    return {
        "method": "geo_channel_grad_times_activation",
        "target": "sum(cone_depth[S])",
        "grad_channels": "edge_attr[:, :4]  # Δxyz + dist",
        "onehot_channels": "excluded (hard >0.5 masks — non-diff routing)",
        "L": float(L.detach().item()),
        "per_relation_mass": per_rel_mass,
        "per_relation_fraction": {k: float(v / total_r) for k, v in per_rel_mass.items()},
        "headline_fractions": {k: float(bucket_mass[k] / total_b) for k in BUCKETS},
        "bucket_mass": bucket_mass,
        "total_geo_grad_act_mass": float(per_edge.sum().item()),
    }


def _probe_on_data(
    *,
    name: str,
    checkpoint: Path,
    model: torch.nn.Module,
    data: Any,
    prot: dict[str, Any],
    S_indices: list[int],
    graph_kind: str,
) -> dict[str, Any]:
    n = residue_node_count(data, prot)
    n_rel = _num_relations(model, data.edge_attr)
    masks = _relation_masks(data.edge_attr, n_rel=n_rel)
    counts = _edge_counts(masks)
    stats = dict(getattr(data, "euclidean_shortcut_stats", {}) or {})

    abl = ablation_attribution(model, data, S_indices, n_rel=n_rel)
    try:
        # Fresh clone for grad (ablation mutated nothing on ``data`` but clear grads).
        data_g = data.clone()
        grad = geo_grad_act_attribution(model, data_g, S_indices, n_rel=n_rel)
        grad_ok = True
        grad_err = None
    except Exception as exc:  # noqa: BLE001 — probe must still emit ablation
        grad = None
        grad_ok = False
        grad_err = f"{type(exc).__name__}: {exc}"

    euc_share_abl = abl["headline_fractions_abs_delta_mean_cone_S"].get(
        "euc_shortcut", 0.0
    )
    ds_share_abl = abl["headline_fractions_abs_delta_mean_cone_S"].get(
        "dehydron_spoke", 0.0
    )
    return {
        "arm": name,
        "checkpoint": str(checkpoint),
        "graph_kind": graph_kind,
        "euclidean_shortcut_mp": bool(getattr(model, "euclidean_shortcut_mp", False)),
        "chem_edge_mp": bool(getattr(model, "chem_edge_mp", False)),
        "role_edge_mp": bool(getattr(model, "role_edge_mp", False)),
        "N": n,
        "n_relations": n_rel,
        "S_indices_count": len(S_indices),
        "edge_counts": counts,
        "construction": {
            "euclidean_shortcut_count_undirected": int(
                getattr(data, "euclidean_shortcut_count_undirected", 0) or 0
            ),
            "euclidean_shortcut_stats": stats,
            "graph_diameter": stats.get("graph_diameter"),
            "n_directed_edges": int(data.edge_index.size(1)),
        },
        "primary_ablation": abl,
        "secondary_geo_grad_act": grad,
        "secondary_geo_grad_act_ok": grad_ok,
        "secondary_geo_grad_act_error": grad_err,
        "headline": {
            "ablation_fraction_euc_shortcut": euc_share_abl,
            "ablation_fraction_dehydron_spoke": ds_share_abl,
            "ablation_fraction_chem": abl["headline_fractions_abs_delta_mean_cone_S"].get(
                "chem", 0.0
            ),
            "ablation_fraction_other_role": abl[
                "headline_fractions_abs_delta_mean_cone_S"
            ].get("other_role", 0.0),
            "grad_fraction_euc_shortcut": (
                None
                if grad is None
                else grad["headline_fractions"].get("euc_shortcut", 0.0)
            ),
            "grad_fraction_dehydron_spoke": (
                None
                if grad is None
                else grad["headline_fractions"].get("dehydron_spoke", 0.0)
            ),
        },
    }


def probe_arm(
    *,
    name: str,
    checkpoint: Path,
    prot: dict[str, Any],
    S_keys: list[tuple[str, int]],
    device: str,
    graph_loader: str = "grade_ssot",
    run_part0_parity_if_empty: bool = True,
) -> dict[str, Any]:
    """Probe one checkpoint.

    ``graph_loader``:
    - ``grade_ssot`` — train/grade ``prepare_training_batch`` (real ρ/τ).
    - ``part0_legacy`` — **primary autopsy** under Part0 constant-ρ multichain
      construction (align grade path to Part0 for Manifold Reconcile autopsy).
    """
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()

    if graph_loader == "part0_legacy":
        data = prepare_part0_parity_batch(model, prot, device)
        rids = list(getattr(data, "_part0_residue_ids", []) or [])
        if not rids:
            rids = list(prot["residue_ids"])[: residue_node_count(data, prot)]
        prot_use = {
            **prot,
            "residue_ids": rids,
            "n_residues": len(rids),
            "data": data,
            "loader": "part0_legacy",
        }
        graph_kind = "part0_legacy_primary_autopsy"
        print(
            f"  [{name}] primary graph = part0_legacy "
            "(Manifold Reconcile autopsy; not grade SSOT)",
            flush=True,
        )
    elif graph_loader == "grade_ssot":
        data = prepare_training_batch(model, prot, device)
        prot_use = prot
        graph_kind = "prepare_training_batch_grade_ssot"
    else:
        raise ValueError(
            f"unknown graph_loader {graph_loader!r}; use grade_ssot|part0_legacy"
        )

    n = residue_node_count(data, prot_use)
    residue_ids = list(prot_use["residue_ids"])[:n]
    idx_map = {_parse_chain_auth(r): i for i, r in enumerate(residue_ids)}
    missing = [f"{c}:{a}" for c, a in S_keys if (c, a) not in idx_map]
    if missing:
        raise RuntimeError(f"{name}: S missing from graph: {missing}")
    S_indices = [idx_map[k] for k in S_keys]

    primary = _probe_on_data(
        name=name,
        checkpoint=checkpoint,
        model=model,
        data=data,
        prot=prot_use,
        S_indices=S_indices,
        graph_kind=graph_kind,
    )
    primary["S_keys"] = [f"{c}:{a}" for c, a in S_keys]
    primary["graph_loader"] = graph_loader

    sc = int(primary["construction"]["euclidean_shortcut_count_undirected"])
    # Only when primary is grade SSOT and empty: keep dual-path sensitivity note.
    if (
        graph_loader == "grade_ssot"
        and run_part0_parity_if_empty
        and bool(getattr(model, "euclidean_shortcut_mp", False))
        and sc == 0
    ):
        print(f"  [{name}] grade graph has 0 euc shortcuts — Part0-parity counterfactual", flush=True)
        data_p0 = prepare_part0_parity_batch(model, prot, device)
        rids_p0 = list(getattr(data_p0, "_part0_residue_ids", []) or [])
        if not rids_p0:
            rids_p0 = list(prot["residue_ids"])[: residue_node_count(data_p0, prot)]
        idx_p0 = {_parse_chain_auth(r): i for i, r in enumerate(rids_p0)}
        missing_p0 = [f"{c}:{a}" for c, a in S_keys if (c, a) not in idx_p0]
        if missing_p0:
            primary["counterfactual_part0_parity_error"] = f"S missing: {missing_p0}"
        else:
            S_p0 = [idx_p0[k] for k in S_keys]
            prot_p0 = {
                **prot,
                "residue_ids": rids_p0,
                "n_residues": len(rids_p0),
                "data": data_p0,
            }
            primary["counterfactual_part0_parity_graph"] = _probe_on_data(
                name=name,
                checkpoint=checkpoint,
                model=model,
                data=data_p0,
                prot=prot_p0,
                S_indices=S_p0,
                graph_kind="part0_loader_parity_counterfactual",
            )
            primary["counterfactual_part0_parity_graph"]["S_keys"] = primary["S_keys"]
            primary["counterfactual_note"] = (
                "Grade prepare_training_batch yielded 0 euc shortcuts (dense role "
                "graph; diameter~7). Counterfactual rebuilds with Part 0 multichain "
                "loader ρ/τ (diameter~63, sc~684). Node features differ from grade "
                "SSOT — sensitivity only, not suite evidence."
            )
    return primary


def _read_fractions(arm: dict[str, Any]) -> dict[str, float]:
    h = arm["headline"]
    return {
        "euc_shortcut": float(h["ablation_fraction_euc_shortcut"]),
        "dehydron_spoke": float(h["ablation_fraction_dehydron_spoke"]),
        "chem": float(h["ablation_fraction_chem"]),
        "other_role": float(h["ablation_fraction_other_role"]),
    }


def _verdict_from_shares(
    *,
    t_euc: float,
    t_ds: float,
    b_ds: float,
    n_euc_edges: int,
    euc_flag: bool,
) -> tuple[str, str]:
    euc_dominates_ds = t_euc > t_ds and t_euc >= 0.35
    ds_shrunk = (b_ds - t_ds) >= 0.10

    if euc_flag and n_euc_edges == 0:
        return (
            "empty_lever_blocker",
            "Treatment flag on but zero euc_shortcut edges on this autopsy graph "
            "— siphon untestable here; lever was empty at forward time.",
        )
    if not euc_flag:
        return "n/a", "No euc shortcuts on this arm."
    if euc_dominates_ds and ds_shrunk:
        return (
            "supports_siphon",
            "Euc-shortcut |Δ| mass exceeds dehydron/spoke and dehydron/spoke "
            "share fell vs baseline — consistent with shortcut competition for "
            "cone_depth[S]. Does NOT alone explain suite Fail (baseline cone "
            "P@K already ~0).",
        )
    if euc_dominates_ds and not ds_shrunk:
        return (
            "partial_consistent",
            "Shortcuts carry large |Δ| mass on cone_depth[S], but dehydron/spoke "
            "share did not clearly shrink vs baseline — competition possible, "
            "biology-path siphon not shown.",
        )
    if t_euc < 0.15 and t_ds >= t_euc:
        return (
            "contradicts_siphon",
            "Euc-shortcut ablation share is small vs dehydron/spoke — does not "
            "support 'shortcuts steal cone_depth signal' on this PDB.",
        )
    return (
        "inconclusive",
        "Attribution shares are mixed; cannot cleanly support or falsify "
        "shortcut siphon. Suite Fail was Δ≈0 enrichment (both arms ~0 cone "
        "P@K on best ckpt), not proof a 6-hop path vanished.",
    )


def interpret_structure(
    *,
    pdb_id: str,
    baseline: dict[str, Any],
    treatment: dict[str, Any],
) -> dict[str, Any]:
    """Scientifically careful read — Fail was Δ≈0 enrichment, not vanished pathways."""
    t_frac = _read_fractions(treatment)
    b_frac = _read_fractions(baseline)
    n_euc = int(treatment.get("edge_counts", {}).get("euclidean_shortcut", 0))
    verdict, claim = _verdict_from_shares(
        t_euc=t_frac["euc_shortcut"],
        t_ds=t_frac["dehydron_spoke"],
        b_ds=b_frac["dehydron_spoke"],
        n_euc_edges=n_euc,
        euc_flag=bool(treatment.get("euclidean_shortcut_mp")),
    )

    out: dict[str, Any] = {
        "pdb_id": pdb_id,
        "verdict": verdict,
        "claim": claim,
        "graph_kind": treatment.get("graph_kind"),
        "graph_loader": treatment.get("graph_loader"),
        "treatment_construction": treatment.get("construction"),
        "treatment_ablation_euc_vs_dehydron_spoke": t_frac,
        "baseline_ablation_dehydron_spoke": b_frac["dehydron_spoke"],
        "caveats": [
            "cone_depth P@K Fail = enrichment Δ≈0; baseline also ~0 on these PDBs.",
            "Ablation |Δ| mass ≠ pathway vanished; signed deltas reported separately.",
            "Geometry lock: MP is Euclidean construction; cone_depth is post-lift.",
            (
                "Headline fractions use Part0 legacy primary autopsy graph "
                "(constant ρ≡12 multichain) — Manifold Reconcile; not grade SSOT."
                if treatment.get("graph_loader") == "part0_legacy"
                else "Headline fractions use grade SSOT graph (prepare_training_batch)."
            ),
            "No governor redesign; existing chem_mvp_euc_reach_v1 weights only.",
        ],
    }

    cf = treatment.get("counterfactual_part0_parity_graph")
    if cf is not None:
        cf_frac = _read_fractions(cf)
        cf_n = int(cf.get("edge_counts", {}).get("euclidean_shortcut", 0))
        cf_v, cf_c = _verdict_from_shares(
            t_euc=cf_frac["euc_shortcut"],
            t_ds=cf_frac["dehydron_spoke"],
            b_ds=b_frac["dehydron_spoke"],
            n_euc_edges=cf_n,
            euc_flag=True,
        )
        out["counterfactual_part0_parity"] = {
            "verdict": cf_v,
            "claim": cf_c,
            "fractions": cf_frac,
            "construction": cf.get("construction"),
            "note": treatment.get("counterfactual_note"),
        }
    return out


def run_structure(
    *,
    pdb_id: str,
    tier: str,
    chains: tuple[str, ...],
    baseline_ckpt: Path,
    treatment_ckpt: Path,
    device: str,
    out_dir: Path,
    graph_loader: str = "grade_ssot",
    artifact_suffix: str = "",
) -> dict[str, Any]:
    site_path = SITE_DIR / f"{pdb_id.lower()}_pathway_residues.json"
    S_keys = load_frozen_S(site_path)
    # Seed prot always from grade concat for chain/id metadata; Part0 rebuild
    # inside probe_arm when graph_loader=part0_legacy.
    prot = load_suite_prot(pdb_id, chains)

    print(f"=== {pdb_id} baseline (loader={graph_loader}) ===", flush=True)
    baseline = probe_arm(
        name="chem_mvp_baseline",
        checkpoint=baseline_ckpt,
        prot=prot,
        S_keys=S_keys,
        device=device,
        graph_loader=graph_loader,
        run_part0_parity_if_empty=(graph_loader == "grade_ssot"),
    )
    print(f"=== {pdb_id} treatment (loader={graph_loader}) ===", flush=True)
    treatment = probe_arm(
        name="chem_mvp_euc_reach_v1",
        checkpoint=treatment_ckpt,
        prot=prot,
        S_keys=S_keys,
        device=device,
        graph_loader=graph_loader,
        run_part0_parity_if_empty=(graph_loader == "grade_ssot"),
    )
    reading = interpret_structure(
        pdb_id=pdb_id, baseline=baseline, treatment=treatment
    )

    blob = {
        "schema_version": 2,
        "probe": "shortcut_signal_siphon",
        "graph_loader": graph_loader,
        "autopsy_mode": graph_loader == "part0_legacy",
        "ssot": "docs/specs/v66_chem_MVP/ablation_euclidean_reach.md",
        "method_note": str(out_dir / "SIGNAL_SIPHON_PROBE.md"),
        "structure": pdb_id,
        "tier": tier,
        "chains": list(chains),
        "site_list": str(site_path),
        "device": device,
        "target_metric": "cone_depth (pre-routing dist0; model out['cone_depth'])",
        "primary_method": "relation_message_ablation",
        "secondary_method": "geo_channel_grad_times_activation",
        "baseline": baseline,
        "treatment": treatment,
        "interpretation": reading,
        "honesty": {
            "suite_fail_was_delta_enrichment_near_zero": True,
            "baseline_cone_P_at_K_also_near_zero_on_best_ckpt": True,
            "do_not_equate_siphon_with_suite_fail": True,
            "no_soft_partial_to_pass": True,
            "no_governor_redesign": True,
            "existing_run_autopsy_only": True,
            "part0_legacy_is_constant_rho": graph_loader == "part0_legacy",
        },
    }
    stem = f"shortcut_signal_siphon_{pdb_id.lower()}"
    if artifact_suffix:
        stem = f"{stem}_{artifact_suffix}"
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"Wrote {out_path}", flush=True)
    print(
        f"  verdict={reading['verdict']}  "
        f"treat euc={reading['treatment_ablation_euc_vs_dehydron_spoke']['euc_shortcut']:.3f} "
        f"dh+spoke={reading['treatment_ablation_euc_vs_dehydron_spoke']['dehydron_spoke']:.3f}",
        flush=True,
    )
    return blob


def write_method_note(out_dir: Path) -> Path:
    path = out_dir / "SIGNAL_SIPHON_PROBE.md"
    path.write_text(
        "# Signal Siphon Probe — Euclidean Shortcut vs Dehydron/Spoke\n"
        "\n"
        "**Date:** 2026-07-19  \n"
        "**Parent SSOT:** `docs/specs/v66_chem_MVP/ablation_euclidean_reach.md`  \n"
        "**Script:** `experiments/diagnostics/euclidean_reach_shortcut_signal_siphon.py`\n"
        "\n"
        "## Hypothesis\n"
        "\n"
        "Euclidean shortcuts (relation 7) cause **Signal Dilution via Shortcut Competition**:\n"
        "MoE/MP routes through spatial wormholes and bypasses dehydron/spoke biology paths —\n"
        "proposed autopsy for healthy train stats + Fail `cone_depth` P@K.\n"
        "\n"
        "## Target\n"
        "\n"
        '`out["cone_depth"]` = hyperbolic `dist0(x_hyp)` after **learned** lift from the\n'
        "Euclidean MP trunk (`radial_head` / `angular_head`). Not structural SSOT, not\n"
        "disc radius alone, not `cone_depth_routed`.\n"
        "\n"
        "Pathway set S: frozen site lists under `site_lists/` (same as suite grade).\n"
        "Structures: **1GPW (A+B)**, **1F88**.\n"
        "\n"
        "Checkpoints: treatment `chem_mvp_euc_reach_v1/v66_best.pt` vs baseline\n"
        "`chem_mvp_stage_a12_cold_v1/v66_best.pt`.\n"
        "\n"
        "## Methods (exact)\n"
        "\n"
        "### Primary — relation message-ablation\n"
        "\n"
        "1. Build graph via v66 `prepare_training_batch` (role + chem; + euc if flag).\n"
        "2. Record m0 = mean cone_depth on S.\n"
        "3. For each relation class / bucket "
        "`{euc_shortcut, dehydron_spoke, chem, other_role}`: "
        "drop all directed edges whose role one-hot > 0.5, re-forward, record m.\n"
        "4. Importance = |m0 - m|. Headline fraction = importance / sum over buckets.\n"
        "\n"
        "**Why not Integrated Gradients on role one-hots?**  \n"
        "`EquivariantConvMultiRel` selects relations with a hard `> 0.5` mask, then runs\n"
        "separate radial MLPs. One-hot identity is effectively discrete — IG/grad on those\n"
        "columns does not measure message routing. Ablation matches the discrete gate.\n"
        "\n"
        "### Secondary — geo-channel gradient × activation\n"
        "\n"
        "L = sum_{i in S} cone_depth_i, differentiate w.r.t. continuous "
        "`edge_attr[:, :4]` (Δxyz + dist), attribute "
        "sum_c |∂L/∂a_c · a_c| per edge, sum by relation mask.\n"
        "\n"
        "### Construction caveat (1GPW)\n"
        "\n"
        "Grade SSOT `prepare_training_batch` + `load_suite_prot` yields denser role\n"
        "graphs (1GPW diameter ~7 → **0** euc shortcuts). Part 0 used a different\n"
        "multichain CA loader (different ρ/τ) and got diameter ~63 → 684 shortcuts.\n"
        "When grade sc=0, the probe runs a **Part0-loader counterfactual**; ρ/τ and\n"
        "edges differ from grade — labeled sensitivity only; does not reopen Pass.\n"
        "\n"
        "## Honesty frame\n"
        "\n"
        "Suite Fail was **Δ≈0 enrichment** on frozen S (best-ckpt cone P@K = 0 on both\n"
        "arms for 1GPW/1F88). Baseline was already ~0 — Fail is not evidence that a\n"
        "6-hop biological path vanished, only that shortcuts did not create enrichment.\n"
        "Siphon (if present) is a *mechanism candidate*, not an automatic Fail explanation.\n"
        "\n"
        "## Artifacts\n"
        "\n"
        "- `shortcut_signal_siphon_1gpw.json`\n"
        "- `shortcut_signal_siphon_1f88.json`\n"
        "- `shortcut_signal_siphon_summary.json`\n"
        "- This note\n"
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--baseline", type=Path, default=BASELINE_CKPT)
    ap.add_argument("--treatment", type=Path, default=TREATMENT_CKPT)
    ap.add_argument(
        "--pdbs",
        nargs="+",
        default=["1GPW", "1F88"],
        help="Subset of suite PDBs (default: 1GPW 1F88)",
    )
    ap.add_argument(
        "--loader",
        choices=("grade_ssot", "part0_legacy"),
        default="grade_ssot",
        help=(
            "Primary graph construction. part0_legacy = Manifold Reconcile "
            "autopsy (align probe to Part0 constant-ρ loader; no governor redesign)."
        ),
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_method_note(out_dir)

    suite_by_id = {s["pdb_id"]: s for s in SUITE}
    suffix = "part0_autopsy" if args.loader == "part0_legacy" else ""
    results = []
    for pdb_id in args.pdbs:
        spec = suite_by_id[pdb_id.upper()]
        results.append(
            run_structure(
                pdb_id=spec["pdb_id"],
                tier=spec["tier"],
                chains=spec["chains"],
                baseline_ckpt=args.baseline,
                treatment_ckpt=args.treatment,
                device=args.device,
                out_dir=out_dir,
                graph_loader=args.loader,
                artifact_suffix=suffix,
            )
        )

    summary_name = (
        "shortcut_signal_siphon_summary_part0_autopsy.json"
        if args.loader == "part0_legacy"
        else "shortcut_signal_siphon_summary.json"
    )
    summary = {
        "schema_version": 2,
        "probe": "shortcut_signal_siphon_summary",
        "graph_loader": args.loader,
        "autopsy_mode": args.loader == "part0_legacy",
        "treatment_ckpt": str(args.treatment),
        "baseline_ckpt": str(args.baseline),
        "structures": {
            r["structure"]: {
                "verdict": r["interpretation"]["verdict"],
                "claim": r["interpretation"]["claim"],
                "fractions": r["interpretation"][
                    "treatment_ablation_euc_vs_dehydron_spoke"
                ],
                "construction": r["interpretation"].get("treatment_construction"),
                "counterfactual_part0_parity": r["interpretation"].get(
                    "counterfactual_part0_parity"
                ),
                "artifact": str(
                    out_dir
                    / (
                        f"shortcut_signal_siphon_{r['structure'].lower()}"
                        f"{'_' + suffix if suffix else ''}.json"
                    )
                ),
            }
            for r in results
        },
    }
    summary_path = out_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
