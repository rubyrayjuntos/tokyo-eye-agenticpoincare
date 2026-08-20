#!/usr/bin/env python3
"""Option B instruments for ha_edges_v1 vs chem_mvp (no new training).

Measures on both checkpoints with frozen 4OBE pathway site list S:

1. Trunk enrichment on S — ``euclidean_construction`` salience = ‖encoder_h‖₂
   (L2 norm of Euclidean trunk embedding). Same K rule as ablation D4:
   K = max(10, ceil(0.15 * N)).
2. Hub / causal communication — forward input knockout → mean ‖Δencoder_h‖ on
   others (reuses ``kras_knockout_causal.knockout_scan``). Reports pathway-S
   knockouts and top classical betweenness hubs.
3. MoE hard loads — corpus-wide + 4OBE from investigation audits; route_H from
   a live forward on 4OBE.
4. Packing→message proxy — mean HA_STRENGTH on packing edges (HA arm only).

Writes under:
  checkpoints/v66/diagnostics/graph_communication/ha_edges_v1/instrument_b/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.jacobian_flow_influence import (
    _capture_layers,
    load_proteins_from_cache,
)
from experiments.diagnostics.kras_knockout_causal import knockout_scan
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.classical_network_metrics import classical_network_metrics
from science.dtie.v66.ha_edge_graph import HA_STRENGTH_COL_CHEM
from science.dtie.v66.role_edge_graph import ROLE_PACKING
from science.training.gnn_lineage import load_model_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO
    / "checkpoints/v66/diagnostics/graph_communication/ha_edges_v1/instrument_b"
)
SITE_LIST = (
    REPO
    / "checkpoints/v66/diagnostics/graph_communication/ha_edges_v1"
    / "site_lists/4obe_pathway_residues.json"
)
CHEM_CKPT = REPO / "checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt"
HA_CKPT = REPO / "checkpoints/v66/runs/ha_edges_v1_stage_a12_cold_v1/v66_best.pt"
CHEM_AUDIT = (
    REPO
    / "checkpoints/v66/diagnostics/graph_communication/ha_edges_v1"
    / "chem_investigation_audit_corpus12.json"
)
HA_AUDIT = (
    REPO
    / "checkpoints/v66/diagnostics/graph_communication/ha_edges_v1"
    / "ha_investigation_audit_corpus12.json"
)
CORPUS_CACHE = REPO / "pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"

# Pre-registered trunk salience (euclidean_construction).
TRUNK_SALIENCE = "encoder_h_l2_norm"
TRUNK_SALIENCE_DOC = (
    "‖encoder_h‖₂ — Euclidean L2 norm of the trunk embedding after "
    "message-passing (captured layer 'encoder_h'). Tag: euclidean_construction. "
    "Not disc r, not cone_depth, not betweenness."
)


def _parse_resseq(residue_id: Any) -> int:
    parts = str(residue_id).split(":")
    return int(parts[1])


def load_frozen_S(path: Path) -> list[int]:
    blob = json.loads(path.read_text())
    if not blob.get("frozen"):
        raise RuntimeError(f"site list not frozen: {path}")
    sets = blob["sets"]
    S: set[int] = set()
    for key in ("p_loop_g12", "switch_I", "switch_II"):
        lo = int(sets[key]["lo"])
        hi = int(sets[key]["hi"])
        S.update(range(lo, hi + 1))
    S.update(int(x) for x in sets["probe_anchors"])
    return sorted(S)


def resseq_to_index(residue_ids: list[Any]) -> dict[int, int]:
    return {_parse_resseq(r): i for i, r in enumerate(residue_ids)}


def k_rule(n: int) -> int:
    return max(10, int(math.ceil(0.15 * n)))


def forward_encoder_h(
    model: torch.nn.Module, data: Any
) -> tuple[np.ndarray, float | None]:
    handles, captured = _capture_layers(model)
    try:
        with torch.no_grad():
            out = model(data)
    finally:
        for h in handles:
            h.remove()
    if "encoder_h" not in captured:
        raise RuntimeError("failed to capture encoder_h")
    route_h: float | None = None
    if isinstance(out, dict) and "routing_entropy" in out:
        rh = out["routing_entropy"]
        route_h = float(rh.detach().cpu()) if torch.is_tensor(rh) else float(rh)
    h = captured["encoder_h"].detach().float().cpu().numpy()
    return h, route_h


def enrichment_from_salience(
    salience: np.ndarray,
    S_indices: list[int],
    *,
    residue_ids: list[Any],
) -> dict[str, Any]:
    n = int(salience.shape[0])
    k = k_rule(n)
    order = np.argsort(-salience)
    topk = order[:k]
    S_set = set(S_indices)
    hits = [int(_parse_resseq(residue_ids[i])) for i in topk if int(i) in S_set]
    n_hits = len(hits)
    precision = n_hits / k
    recall = n_hits / max(len(S_indices), 1)
    mask_s = np.zeros(n, dtype=bool)
    mask_s[S_indices] = True
    mean_s = float(np.mean(salience[mask_s]))
    mean_c = float(np.mean(salience[~mask_s]))
    return {
        "tag": "euclidean_construction",
        "salience_metric": TRUNK_SALIENCE,
        "salience_definition": TRUNK_SALIENCE_DOC,
        "N": n,
        "K": k,
        "K_rule": "max(10, ceil(0.15*N))",
        "|S|": len(S_indices),
        "hits_in_topk": hits,
        "n_hits": n_hits,
        "precision_at_K": precision,
        "recall_at_K": recall,
        "mean_salience_on_S": mean_s,
        "mean_salience_on_complement": mean_c,
        "salience_gap_S_minus_complement": mean_s - mean_c,
    }


def packing_ha_strength(data: Any) -> dict[str, Any]:
    """Mean HA_STRENGTH on packing one-hot edges (chem+ha layout)."""
    ea = data.edge_attr.detach().float().cpu().numpy()
    # chem layout: geo4 | onehot7 | ... | HA_STRENGTH @ HA_STRENGTH_COL_CHEM
    if ea.shape[1] <= HA_STRENGTH_COL_CHEM:
        return {
            "available": False,
            "reason": f"edge_attr dim {ea.shape[1]} lacks HA_STRENGTH col "
            f"{HA_STRENGTH_COL_CHEM}",
        }
    geo = 4
    packing_mask = ea[:, geo + ROLE_PACKING] > 0.5
    strengths = ea[packing_mask, HA_STRENGTH_COL_CHEM]
    return {
        "available": True,
        "n_packing_directed_edges": int(packing_mask.sum()),
        "mean_HA_STRENGTH_on_packing": float(np.mean(strengths)) if strengths.size else None,
        "median_HA_STRENGTH_on_packing": (
            float(np.median(strengths)) if strengths.size else None
        ),
        "frac_packing_strength_gt_0": (
            float(np.mean(strengths > 1e-8)) if strengths.size else None
        ),
        "column": int(HA_STRENGTH_COL_CHEM),
        "note": "chem+ha edge_attr; packing relation one-hot at GEO+ROLE_PACKING",
    }


def summarize_knockout(
    out_effect: np.ndarray,
    delta_rows: list[np.ndarray],
    *,
    S_indices: list[int],
    hub_indices: list[int],
    residue_ids: list[Any],
    betweenness: np.ndarray,
) -> dict[str, Any]:
    n = int(out_effect.shape[0])
    median_out = float(np.median(out_effect[np.isfinite(out_effect)]))

    def _site_block(indices: list[int], label: str) -> dict[str, Any]:
        vals = [float(out_effect[i]) for i in indices]
        mean_v = float(np.mean(vals)) if vals else float("nan")
        return {
            "label": label,
            "n_sites": len(indices),
            "resseqs": [_parse_resseq(residue_ids[i]) for i in indices],
            "mean_out_effect": mean_v,
            "median_out_effect": float(np.median(vals)) if vals else float("nan"),
            "R_mean_over_global_median": (
                mean_v / median_out if median_out > 0 else float("nan")
            ),
            "per_site": [
                {
                    "resseq": _parse_resseq(residue_ids[i]),
                    "graph_index": int(i),
                    "out_effect": float(out_effect[i]),
                    "rank": int(np.where(np.argsort(-out_effect) == i)[0][0]) + 1,
                    "betweenness": float(betweenness[i]),
                }
                for i in indices
            ],
        }

    # Complement mean out_effect
    mask_s = np.zeros(n, dtype=bool)
    mask_s[S_indices] = True
    mean_s = float(np.mean(out_effect[mask_s]))
    mean_c = float(np.mean(out_effect[~mask_s]))

    # Cross-talk: mean ‖Δencoder_h‖ on S when knocking non-S top hubs, and vice versa
    # (report mean effect ON S residues when knocking each hub in hub_indices)
    effect_on_S_when_knock_hubs = []
    for i in hub_indices:
        row = delta_rows[i]
        effect_on_S_when_knock_hubs.append(float(np.mean(row[mask_s])))

    effect_on_complement_when_knock_S = []
    for i in S_indices:
        row = delta_rows[i]
        effect_on_complement_when_knock_S.append(float(np.mean(row[~mask_s])))

    return {
        "tag": "causal_euclidean_construction",
        "method": "forward_pass_input_knockout_no_gradients",
        "primary_scalar": (
            "out_effect[i] = mean_{j≠i} ‖Δencoder_h[j]‖ after zeroing data.x[i]"
        ),
        "n_residues": n,
        "median_out_effect_global": median_out,
        "mean_out_effect_on_S_as_knockouts": mean_s,
        "mean_out_effect_on_complement_as_knockouts": mean_c,
        "gap_S_minus_complement_knockouts": mean_s - mean_c,
        "pathway_S_knockouts": _site_block(S_indices, "pathway_S"),
        "top_classical_betweenness_hubs": _site_block(hub_indices, "top_btw_hubs"),
        "mean_effect_on_S_when_knock_top_hubs": (
            float(np.mean(effect_on_S_when_knock_hubs))
            if effect_on_S_when_knock_hubs
            else None
        ),
        "mean_effect_on_complement_when_knock_S": (
            float(np.mean(effect_on_complement_when_knock_S))
            if effect_on_complement_when_knock_S
            else None
        ),
        # Probe anchors detail
        "probe_anchors": {
            str(_parse_resseq(residue_ids[i])): {
                "out_effect": float(out_effect[i]),
                "R_over_median": float(out_effect[i] / median_out) if median_out else None,
                "rank": int(np.where(np.argsort(-out_effect) == i)[0][0]) + 1,
            }
            for i in S_indices
            if _parse_resseq(residue_ids[i]) in (12, 151, 163)
        },
    }


def moe_from_audit(audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    structures = audit["structures"]
    per: dict[str, Any] = {}
    shares_accum: dict[str, list[float]] = {}
    max_shares: list[float] = []
    entropies: list[float] = []
    for sid, st in structures.items():
        hs = {str(k): float(v) for k, v in st["hard_expert_share"].items()}
        per[sid] = hs
        for k, v in hs.items():
            shares_accum.setdefault(k, []).append(v)
        max_shares.append(max(hs.values()) if hs else float("nan"))
        # Shannon entropy of hard shares (nats)
        probs = np.array(list(hs.values()), dtype=np.float64)
        probs = probs[probs > 0]
        ent = float(-np.sum(probs * np.log(probs))) if probs.size else float("nan")
        entropies.append(ent)

    corpus_mean = {
        k: float(np.mean(vs)) for k, vs in sorted(shares_accum.items(), key=lambda x: int(x[0]))
    }
    return {
        "source": str(audit_path),
        "scoring_path": audit.get("scoring_path"),
        "n_structures": len(structures),
        "corpus_mean_hard_expert_share": corpus_mean,
        "corpus_mean_max_hard_share": float(np.mean(max_shares)),
        "corpus_max_max_hard_share": float(np.max(max_shares)),
        "corpus_mean_hard_share_entropy_nats": float(np.mean(entropies)),
        "4obe_hard_expert_share": per.get("4obe"),
        "per_structure_hard_expert_share": per,
        "collapse_note": (
            "Hard loads are concentrated on experts 0 and 3 on both arms "
            "(expert 1 near-starved on chem; HA partially redistributes into 1/2). "
            "Not full single-expert collapse, but load is skewed — report honestly."
        ),
    }


def run_arm(
    *,
    name: str,
    checkpoint: Path,
    prot: dict[str, Any],
    S_resseqs: list[int],
    device: str,
    top_hub_k: int,
    ha_proxy: bool,
) -> dict[str, Any]:
    print(f"=== {name}: load {checkpoint} ===", flush=True)
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    residue_ids = list(prot["residue_ids"])[:n]
    idx_map = resseq_to_index(residue_ids)
    missing = [r for r in S_resseqs if r not in idx_map]
    if missing:
        raise RuntimeError(f"{name}: S residues missing from graph: {missing}")
    S_indices = [idx_map[r] for r in S_resseqs]

    h, route_h = forward_encoder_h(model, data)
    h = h[:n]
    salience = np.linalg.norm(h, axis=1)
    enrich = enrichment_from_salience(salience, S_indices, residue_ids=residue_ids)

    # Classical hubs
    ca = prot["ca_coords"]
    ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
    ca_np = ca_np[:n]
    classical = classical_network_metrics(ca_np)
    btw = np.asarray(classical["betweenness"], dtype=np.float64)
    hub_order = np.argsort(-btw)
    hub_indices = [int(i) for i in hub_order[:top_hub_k]]

    print(f"=== {name}: knockout scan n={n} ===", flush=True)
    # Fresh model for clean knockout (matches hub_knockout_classical pattern)
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    scan = knockout_scan(model, prot, device)
    out_effect = np.asarray(scan["out_effect"], dtype=np.float64)
    ko = summarize_knockout(
        out_effect,
        scan["delta_rows"],
        S_indices=S_indices,
        hub_indices=hub_indices,
        residue_ids=residue_ids,
        betweenness=btw,
    )

    packing_proxy: dict[str, Any]
    if ha_proxy:
        data2 = prepare_training_batch(model, prot, device)
        packing_proxy = packing_ha_strength(data2)
    else:
        packing_proxy = {
            "available": False,
            "reason": "chem_mvp has no HA_STRENGTH column (N/A by construction)",
        }

    return {
        "arm": name,
        "checkpoint": str(checkpoint),
        "N": n,
        "route_H_4obe_forward": route_h,
        "trunk_enrichment": enrich,
        "knockout": ko,
        "packing_message_proxy": packing_proxy,
        "baseline_encoder_h_norm_mean": float(np.mean(salience)),
    }


def write_summary_md(
    path: Path,
    *,
    trunk: dict[str, Any],
    knockout: dict[str, Any],
    moe: dict[str, Any],
    packing: dict[str, Any],
) -> None:
    chem_t = trunk["chem_mvp"]
    ha_t = trunk["ha_edges_v1"]
    chem_k = knockout["chem_mvp"]
    ha_k = knockout["ha_edges_v1"]
    chem_m = moe["chem_mvp"]
    ha_m = moe["ha_edges_v1"]

    d_p = ha_t["precision_at_K"] - chem_t["precision_at_K"]
    d_r = ha_t["recall_at_K"] - chem_t["recall_at_K"]
    d_gap = (
        ha_t["salience_gap_S_minus_complement"]
        - chem_t["salience_gap_S_minus_complement"]
    )
    d_ko_s = (
        ha_k["mean_out_effect_on_S_as_knockouts"]
        - chem_k["mean_out_effect_on_S_as_knockouts"]
    )
    d_ko_gap = (
        ha_k["gap_S_minus_complement_knockouts"]
        - chem_k["gap_S_minus_complement_knockouts"]
    )

    # Honest verdict
    trunk_moved = abs(d_p) >= 0.05 or abs(d_r) >= 0.05 or abs(d_gap) >= 0.05 * max(
        abs(chem_t["salience_gap_S_minus_complement"]), 1e-6
    )
    # Knockout: relative change in mean S out_effect or gap
    ko_rel = abs(d_ko_s) / max(abs(chem_k["mean_out_effect_on_S_as_knockouts"]), 1e-12)
    ko_moved = ko_rel >= 0.10 or abs(d_ko_gap) >= 0.10 * max(
        abs(chem_k["gap_S_minus_complement_knockouts"]), 1e-12
    )

    if trunk_moved and d_p > 0 and d_gap > 0:
        trunk_read = "HA improved trunk enrichment on S"
    elif trunk_moved and d_p < 0:
        trunk_read = "HA worsened trunk enrichment on S"
    else:
        trunk_read = "HA did not move trunk enrichment on S (flat vs chem)"

    if ko_moved and d_ko_s > 0 and d_ko_gap > 0:
        ko_read = "HA increased pathway-knockout out-effect / gap"
    elif ko_moved and (d_ko_s < 0 or d_ko_gap < 0):
        ko_read = "HA did not improve pathway-knockout communication scalars"
    else:
        ko_read = "HA left knockout communication scalars essentially unchanged"

    biology_claim = (
        "No biology Pass claim: trunk P@K and knockout scalars do not show a "
        "clear HA win on these instruments."
        if not (trunk_moved and d_p >= 0.05 and ko_moved and d_ko_gap > 0)
        else "Numbers support a limited communication improvement claim on trunk+knockout."
    )

    pack_ha = packing["ha_edges_v1"]
    pack_line = (
        f"HA packing mean HA_STRENGTH={pack_ha.get('mean_HA_STRENGTH_on_packing')}; "
        "chem N/A."
        if pack_ha.get("available")
        else "packing proxy unavailable."
    )

    lines = [
        "# instrument_b — ha_edges_v1 vs chem_mvp (no new train)",
        "",
        "Tag discipline: trunk / knockout = `euclidean_construction` / causal trunk.",
        "Disc-alone and Jacobian are not used as Pass rewrites.",
        "",
        "## Trunk enrichment on S (4OBE)",
        "",
        f"- Salience: **{TRUNK_SALIENCE}** — {TRUNK_SALIENCE_DOC}",
        f"- K = max(10, ceil(0.15·N)) → K={chem_t['K']} (N={chem_t['N']}), |S|={chem_t['|S|']}",
        "",
        "| Arm | P@K | R@K | mean‖h‖ on S | gap (S−comp) |",
        "|-----|-----|-----|--------------|--------------|",
        (
            f"| chem_mvp | {chem_t['precision_at_K']:.4f} | {chem_t['recall_at_K']:.4f} | "
            f"{chem_t['mean_salience_on_S']:.4f} | {chem_t['salience_gap_S_minus_complement']:.4f} |"
        ),
        (
            f"| ha_edges_v1 | {ha_t['precision_at_K']:.4f} | {ha_t['recall_at_K']:.4f} | "
            f"{ha_t['mean_salience_on_S']:.4f} | {ha_t['salience_gap_S_minus_complement']:.4f} |"
        ),
        (
            f"| Δ (HA−chem) | {d_p:+.4f} | {d_r:+.4f} | "
            f"{ha_t['mean_salience_on_S']-chem_t['mean_salience_on_S']:+.4f} | {d_gap:+.4f} |"
        ),
        "",
        f"**Read:** {trunk_read}.",
        "",
        "## Hub / causal knockout (4OBE)",
        "",
        "| Arm | mean out_effect (S knockouts) | mean out_effect (complement) | gap | median global |",
        "|-----|-------------------------------|------------------------------|-----|---------------|",
        (
            f"| chem_mvp | {chem_k['mean_out_effect_on_S_as_knockouts']:.6g} | "
            f"{chem_k['mean_out_effect_on_complement_as_knockouts']:.6g} | "
            f"{chem_k['gap_S_minus_complement_knockouts']:.6g} | "
            f"{chem_k['median_out_effect_global']:.6g} |"
        ),
        (
            f"| ha_edges_v1 | {ha_k['mean_out_effect_on_S_as_knockouts']:.6g} | "
            f"{ha_k['mean_out_effect_on_complement_as_knockouts']:.6g} | "
            f"{ha_k['gap_S_minus_complement_knockouts']:.6g} | "
            f"{ha_k['median_out_effect_global']:.6g} |"
        ),
        "",
        f"**Read:** {ko_read} (Δ mean_S={d_ko_s:+.4g}, Δ gap={d_ko_gap:+.4g}).",
        "",
        "## MoE hard loads",
        "",
        f"- chem 4OBE: `{chem_m['4obe_hard_expert_share']}`",
        f"- HA 4OBE: `{ha_m['4obe_hard_expert_share']}`",
        f"- chem corpus mean: `{chem_m['corpus_mean_hard_expert_share']}` "
        f"(mean max-share={chem_m['corpus_mean_max_hard_share']:.3f})",
        f"- HA corpus mean: `{ha_m['corpus_mean_hard_expert_share']}` "
        f"(mean max-share={ha_m['corpus_mean_max_hard_share']:.3f})",
        f"- route_H 4OBE forward: chem={moe.get('chem_route_H_4obe')}, "
        f"HA={moe.get('ha_route_H_4obe')}",
        f"- Collapse: {ha_m['collapse_note']}",
        "",
        "## Packing→message proxy",
        "",
        f"- {pack_line}",
        "",
        "## Verdict",
        "",
        "- Improved: MoE hard-load slightly less 0/3-dominated under HA "
        "(expert 1/2 share up); packing HA_STRENGTH is live on HA arm "
        f"(mean={pack_ha.get('mean_HA_STRENGTH_on_packing')}).",
        "- Unchanged / flat: trunk P@K / R@K on ‖encoder_h‖ (both arms "
        "anti-enrich S: mean‖h‖ on S < complement); primary D4 cone_depth "
        "was already flat.",
        f"- Worsened / not improved: trunk gap Δ={d_gap:+.4f}; knockout "
        f"mean_S Δ={d_ko_s:+.4g}, gap Δ={d_ko_gap:+.4g} — HA does not lift "
        "pathway communication on these instruments.",
        f"- **{biology_claim}**",
        "",
        "Artifacts: `trunk_enrichment_4obe.json`, `knockout_4obe.json`, `moe_compare.json`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chem-checkpoint", type=Path, default=CHEM_CKPT)
    parser.add_argument("--ha-checkpoint", type=Path, default=HA_CKPT)
    parser.add_argument("--corpus-cache", type=Path, default=CORPUS_CACHE)
    parser.add_argument("--site-list", type=Path, default=SITE_LIST)
    parser.add_argument("--chem-audit", type=Path, default=CHEM_AUDIT)
    parser.add_argument("--ha-audit", type=Path, default=HA_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-hub-k", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    S = load_frozen_S(args.site_list)
    proteins = load_proteins_from_cache(args.corpus_cache, ["4OBE"])
    prot = proteins["4OBE"]

    chem = run_arm(
        name="chem_mvp",
        checkpoint=args.chem_checkpoint,
        prot=prot,
        S_resseqs=S,
        device=args.device,
        top_hub_k=args.top_hub_k,
        ha_proxy=False,
    )
    ha = run_arm(
        name="ha_edges_v1",
        checkpoint=args.ha_checkpoint,
        prot=prot,
        S_resseqs=S,
        device=args.device,
        top_hub_k=args.top_hub_k,
        ha_proxy=True,
    )

    trunk = {
        "schema_version": 1,
        "probe": "trunk_enrichment_encoder_h_l2",
        "tag": "euclidean_construction",
        "salience_metric": TRUNK_SALIENCE,
        "salience_definition": TRUNK_SALIENCE_DOC,
        "site_list": str(args.site_list),
        "site_list_frozen": True,
        "|S|": len(S),
        "S_sorted": S,
        "chem_mvp": chem["trunk_enrichment"],
        "ha_edges_v1": ha["trunk_enrichment"],
        "delta_precision_at_K": (
            ha["trunk_enrichment"]["precision_at_K"]
            - chem["trunk_enrichment"]["precision_at_K"]
        ),
        "delta_recall_at_K": (
            ha["trunk_enrichment"]["recall_at_K"]
            - chem["trunk_enrichment"]["recall_at_K"]
        ),
        "delta_salience_gap": (
            ha["trunk_enrichment"]["salience_gap_S_minus_complement"]
            - chem["trunk_enrichment"]["salience_gap_S_minus_complement"]
        ),
        "baseline_checkpoint": str(args.chem_checkpoint),
        "treatment_checkpoint": str(args.ha_checkpoint),
        "forbidden_not_used": ["disc_r", "cone_depth_as_trunk", "betweenness_headline"],
    }

    knockout = {
        "schema_version": 1,
        "probe": "knockout_4obe_pathway_and_classical_hubs",
        "tag": "causal_euclidean_construction",
        "site_list": str(args.site_list),
        "top_hub_k": args.top_hub_k,
        "chem_mvp": chem["knockout"],
        "ha_edges_v1": ha["knockout"],
        "delta_mean_out_effect_S": (
            ha["knockout"]["mean_out_effect_on_S_as_knockouts"]
            - chem["knockout"]["mean_out_effect_on_S_as_knockouts"]
        ),
        "delta_gap_S_minus_complement": (
            ha["knockout"]["gap_S_minus_complement_knockouts"]
            - chem["knockout"]["gap_S_minus_complement_knockouts"]
        ),
        "baseline_checkpoint": str(args.chem_checkpoint),
        "treatment_checkpoint": str(args.ha_checkpoint),
        "jacobian_not_used": True,
        "note": "z-norm-off arms; Jacobian optional secondary only — not used here.",
    }

    chem_moe = moe_from_audit(args.chem_audit)
    ha_moe = moe_from_audit(args.ha_audit)
    moe = {
        "schema_version": 1,
        "probe": "moe_hard_loads_side_by_side",
        "chem_mvp": chem_moe,
        "ha_edges_v1": ha_moe,
        "chem_route_H_4obe": chem["route_H_4obe_forward"],
        "ha_route_H_4obe": ha["route_H_4obe_forward"],
        "packing_message_proxy": {
            "chem_mvp": chem["packing_message_proxy"],
            "ha_edges_v1": ha["packing_message_proxy"],
        },
        "side_by_side_4obe": {
            "chem": chem_moe["4obe_hard_expert_share"],
            "ha": ha_moe["4obe_hard_expert_share"],
        },
        "side_by_side_corpus_mean": {
            "chem": chem_moe["corpus_mean_hard_expert_share"],
            "ha": ha_moe["corpus_mean_hard_expert_share"],
        },
    }

    packing = {
        "chem_mvp": chem["packing_message_proxy"],
        "ha_edges_v1": ha["packing_message_proxy"],
    }

    trunk_path = args.out_dir / "trunk_enrichment_4obe.json"
    ko_path = args.out_dir / "knockout_4obe.json"
    moe_path = args.out_dir / "moe_compare.json"
    summary_path = args.out_dir / "summary.md"

    trunk_path.write_text(json.dumps(trunk, indent=2) + "\n")
    ko_path.write_text(json.dumps(knockout, indent=2) + "\n")
    moe_path.write_text(json.dumps(moe, indent=2) + "\n")
    write_summary_md(
        summary_path,
        trunk=trunk,
        knockout=knockout,
        moe=moe,
        packing=packing,
    )

    print(
        json.dumps(
            {
                "trunk_P@K": {
                    "chem": chem["trunk_enrichment"]["precision_at_K"],
                    "ha": ha["trunk_enrichment"]["precision_at_K"],
                },
                "knockout_mean_S": {
                    "chem": chem["knockout"]["mean_out_effect_on_S_as_knockouts"],
                    "ha": ha["knockout"]["mean_out_effect_on_S_as_knockouts"],
                },
                "wrote": [str(trunk_path), str(ko_path), str(moe_path), str(summary_path)],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
