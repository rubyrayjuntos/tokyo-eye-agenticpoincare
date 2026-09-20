"""Crosstab Poincaré radius-stripe membership vs discrete residue features.

Hypothesis (Ray): identical two-stripe ‖z‖ bands across all folds look mechanical,
not biology. Find which categorical feature predicts stripe membership.
Also check whether bimodality already exists at z_lift (pre attn/MoE).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.training.v8.equ_geoopt_restore import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE,
    RV_TAU_END,
    assert_frontend_bank,
    load_boot_split,
)
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import (
    _load_train_batch,
    _resolve_entry,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.r0_r5_graph import (
    R0_COVALENT,
    R1_HBOND,
    R2_DEHYDRON,
    R3_HYDROPHOBIC_PI,
    R4_SALT_BRIDGE,
    R5_LOCAL_NEIGHBORHOOD,
    set_dehydron_wrap_max,
)

BEST = Path("checkpoints/tokyoeye/runs/eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt")
OUT = Path("data/local_objects/frontend_ablation/stripe_membership_crosstab.json")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")

EDGE_NAMES = {
    R0_COVALENT: "R0_covalent",
    R1_HBOND: "R1_hbond",
    R2_DEHYDRON: "R2_dehydron",
    R3_HYDROPHOBIC_PI: "R3_hydrophobic_pi",
    R4_SALT_BRIDGE: "R4_salt",
    R5_LOCAL_NEIGHBORHOOD: "R5_local",
}


def kmeans2_1d(r: np.ndarray) -> np.ndarray:
    """2-means on 1D radii; label 1 = higher-mean cluster."""
    r = np.asarray(r, dtype=np.float64)
    c0, c1 = float(np.percentile(r, 25)), float(np.percentile(r, 75))
    for _ in range(40):
        d0 = np.abs(r - c0)
        d1 = np.abs(r - c1)
        lab = (d1 < d0).astype(np.int64)
        if lab.sum() == 0 or lab.sum() == len(lab):
            # fallback median split
            med = float(np.median(r))
            return (r >= med).astype(np.int64)
        n0 = float((~lab.astype(bool)).sum())
        n1 = float(lab.sum())
        c0 = float(r[lab == 0].mean()) if n0 else c0
        c1 = float(r[lab == 1].mean()) if n1 else c1
    # ensure label 1 is the higher-radius stripe
    if c0 > c1:
        lab = 1 - lab
    return lab


def strongest_incoming_edge(edge_index: np.ndarray, edge_type: np.ndarray, n: int) -> np.ndarray:
    """Per-node: edge type of strongest priority among edges touching the node.
    Priority: R2 > R1 > R4 > R3 > R0 > R5 (biology-ish), else first seen.
    """
    priority = {
        R2_DEHYDRON: 60,
        R1_HBOND: 50,
        R4_SALT_BRIDGE: 40,
        R3_HYDROPHOBIC_PI: 30,
        R0_COVALENT: 20,
        R5_LOCAL_NEIGHBORHOOD: 10,
    }
    best_p = np.full(n, -1, dtype=np.int64)
    best_t = np.full(n, -1, dtype=np.int64)
    if edge_type.size == 0:
        return best_t
    for e in range(edge_type.shape[0]):
        t = int(edge_type[e])
        p = priority.get(t, 0)
        a = int(edge_index[0, e])
        b = int(edge_index[1, e])
        for node in (a, b):
            if 0 <= node < n and p > best_p[node]:
                best_p[node] = p
                best_t[node] = t
    return best_t


def purity(stripe: np.ndarray, cat: np.ndarray) -> float:
    """Fraction correctly assigned if each stripe picks its majority category."""
    ok = 0
    n = len(stripe)
    for s in (0, 1):
        m = stripe == s
        if not m.any():
            continue
        vals, counts = np.unique(cat[m], return_counts=True)
        ok += int(counts.max())
    return ok / max(n, 1)


def crosstab(stripe: np.ndarray, cat: np.ndarray) -> dict:
    table = defaultdict(lambda: {0: 0, 1: 0})
    for s, c in zip(stripe.tolist(), cat.tolist()):
        table[str(c)][int(s)] += 1
    # accuracy if cat perfectly predicts stripe via majority mapping
    # also: can we predict stripe from cat with 1-1 mapping?
    # max accuracy over functions cat -> stripe
    mapping_correct = 0
    cats = np.unique(cat)
    for c in cats:
        m = cat == c
        if not m.any():
            continue
        # assign this category to the stripe that holds more of it
        s0 = int((stripe[m] == 0).sum())
        s1 = int((stripe[m] == 1).sum())
        mapping_correct += max(s0, s1)
    acc = mapping_correct / max(len(stripe), 1)
    try:
        ari = float(adjusted_rand_score(stripe, cat)) if len(np.unique(cat)) > 1 else float("nan")
        nmi = float(normalized_mutual_info_score(stripe, cat)) if len(np.unique(cat)) > 1 else float("nan")
    except Exception:
        ari, nmi = float("nan"), float("nan")
    return {
        "counts": {k: v for k, v in table.items()},
        "predict_stripe_from_cat_acc": float(acc),
        "purity_stripe_majority_cat": float(purity(stripe, cat)),
        "ARI": ari,
        "NMI": nmi,
    }


def main():
    set_dehydron_wrap_max(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    system, _ = build_equiformer_pool_system(
        cfg,
        equiformer_ckpt=DEFAULT_FRONTEND_CKPT,
        device=device,
        freeze_backbone=True,
        max_neighbors=50,
    )
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    blob = torch.load(BEST, map_location=device, weights_only=False)
    system.load_state_dict(blob["model"], strict=False)
    system.eval()
    print("loaded", BEST, "epoch", blob.get("epoch"))

    split = load_boot_split(DEFAULT_MANIFEST)
    entries = [_resolve_entry(e, PDB_DIR) for e in (split["train"] + split["probe"])]

    per_pdb = []
    # pooled cats
    pooled = {
        "idx_parity": [],
        "idx_quartile": [],
        "moe_expert": [],
        "strongest_edge": [],
        "dehydron": [],
        "sdrp": [],
        "stripe_z": [],
        "stripe_lift": [],
        "r_z": [],
        "r_lift": [],
        "pdb": [],
    }

    with torch.no_grad():
        for entry in entries:
            pdb_id = str(entry["pdb_id"]).upper()
            chain = str(entry["chain"])
            batch = _load_train_batch(
                entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE
            )
            out = system(
                batch["x"],
                batch["edge_index"],
                batch["edge_type"],
                tau_ceiling=RV_TAU_END,
            )
            z = out["z_hyp"].detach().float().cpu()
            z_lift = out["z_lift"].detach().float().cpu()
            r = torch.linalg.vector_norm(z, dim=-1).numpy()
            r_lift = torch.linalg.vector_norm(z_lift, dim=-1).numpy()
            stripe = kmeans2_1d(r)
            stripe_lift = kmeans2_1d(r_lift)

            n = int(batch["num_nodes"])
            idx = np.arange(n)
            parity = idx % 2
            # quartile by sequence index
            q = np.clip((idx * 4) // max(n, 1), 0, 3)

            routing = out["moe_aux"]["routing"].detach().float().cpu().numpy()
            expert = routing.argmax(axis=-1).astype(np.int64)

            ei = batch["edge_index"].detach().cpu().numpy()
            et = batch["edge_type"].detach().cpu().numpy()
            strong = strongest_incoming_edge(ei, et, n)
            dehyd = batch["dehydron_labels"].detach().cpu().numpy().astype(np.int64)
            sdrp = batch["sdrp_target"].detach().cpu().numpy().astype(np.int64)

            cats = {
                "idx_parity": parity,
                "idx_quartile": q,
                "moe_expert": expert,
                "strongest_edge": strong,
                "dehydron": dehyd,
                "sdrp": sdrp,
            }
            local = {
                "pdb_id": pdb_id,
                "chain": chain,
                "n": n,
                "r_mean_stripe0": float(r[stripe == 0].mean()) if (stripe == 0).any() else None,
                "r_mean_stripe1": float(r[stripe == 1].mean()) if (stripe == 1).any() else None,
                "n_stripe0": int((stripe == 0).sum()),
                "n_stripe1": int((stripe == 1).sum()),
                "lift_vs_z_stripe_ARI": float(adjusted_rand_score(stripe, stripe_lift)),
                "r_z_hist": np.histogram(r, bins=20, range=(0, 0.6))[0].tolist(),
                "features": {},
            }
            for name, cat in cats.items():
                ct = crosstab(stripe, cat)
                # humanize edge types
                if name == "strongest_edge":
                    ct["counts_named"] = {
                        EDGE_NAMES.get(int(k), f"t{k}"): v for k, v in ct["counts"].items()
                    }
                local["features"][name] = ct
                pooled[name].append(cat)
            pooled["stripe_z"].append(stripe)
            pooled["stripe_lift"].append(stripe_lift)
            pooled["r_z"].append(r)
            pooled["r_lift"].append(r_lift)
            pooled["pdb"].append(f"{pdb_id}:{chain}")

            # print top predictors locally
            ranked = sorted(
                local["features"].items(),
                key=lambda kv: kv[1]["predict_stripe_from_cat_acc"],
                reverse=True,
            )
            top = ", ".join(f"{k}={v['predict_stripe_from_cat_acc']:.3f}" for k, v in ranked[:4])
            print(
                f"{pdb_id}:{chain} n={n} stripes={local['n_stripe0']}/{local['n_stripe1']} "
                f"r≈{local['r_mean_stripe0']:.3f}/{local['r_mean_stripe1']:.3f} "
                f"liftARI={local['lift_vs_z_stripe_ARI']:.3f} | {top}"
            )
            per_pdb.append(local)

    # pooled
    stripe_all = np.concatenate(pooled["stripe_z"])
    pooled_report = {}
    for name in ["idx_parity", "idx_quartile", "moe_expert", "strongest_edge", "dehydron", "sdrp"]:
        cat_all = np.concatenate(pooled[name])
        pooled_report[name] = crosstab(stripe_all, cat_all)
        if name == "strongest_edge":
            pooled_report[name]["counts_named"] = {
                EDGE_NAMES.get(int(k), f"t{k}"): v
                for k, v in pooled_report[name]["counts"].items()
            }

    # Does lift already have the same stripes?
    stripe_lift_all = np.concatenate(pooled["stripe_lift"])
    lift_agree = float((stripe_all == stripe_lift_all).mean())
    # expert load overall
    expert_all = np.concatenate(pooled["moe_expert"])
    expert_frac = {f"E{i}": float((expert_all == i).mean()) for i in range(int(expert_all.max()) + 1)}

    ranked_pool = sorted(
        pooled_report.items(),
        key=lambda kv: kv[1]["predict_stripe_from_cat_acc"],
        reverse=True,
    )

    report = {
        "gate_id": "tokyo_eye_equ_geoopt_restore",
        "ckpt": str(BEST),
        "epoch": blob.get("epoch"),
        "stripe_def": "2-means on ambient ||z_hyp||_2 (not PCA)",
        "note": (
            "CA-residue graph only — no backbone/sidechain atom flag exists in this pipeline. "
            "Candidates: MoE expert, strongest R0-R5 edge, idx parity/quartile, dehydron, SDRP."
        ),
        "pooled": {
            "n_residues": int(len(stripe_all)),
            "n_structures": len(per_pdb),
            "stripe_frac_high": float(stripe_all.mean()),
            "lift_stripe_agreement": lift_agree,
            "lift_vs_z_ARI": float(adjusted_rand_score(stripe_all, stripe_lift_all)),
            "moe_expert_frac": expert_frac,
            "feature_predict_acc_ranked": [
                {"feature": k, **{kk: vv for kk, vv in v.items() if kk != "counts"}}
                for k, v in ranked_pool
            ],
            "features": pooled_report,
        },
        "per_pdb": per_pdb,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\n=== POOLED (predict stripe from cat accuracy) ===")
    for k, v in ranked_pool:
        print(
            f"  {k:16s} acc={v['predict_stripe_from_cat_acc']:.4f} "
            f"ARI={v['ARI']:.4f} NMI={v['NMI']:.4f}"
        )
    print(f"lift↔z stripe agreement={lift_agree:.4f} ARI={report['pooled']['lift_vs_z_ARI']:.4f}")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
