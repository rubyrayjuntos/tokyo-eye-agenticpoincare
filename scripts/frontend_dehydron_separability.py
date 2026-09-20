"""Stage 0 — no-train: do frozen Equiformer (s,v) separate dehydron vs wrapped H-bond residues?"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.training.v8.equ_theme_restore import DEFAULT_MANIFEST, load_boot_split
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import _resolve_entry
from science.tokyo_eye.v8.equiformer_pool_frontend import EquiformerPoolFrontend, MPTRJ_GRADIENT_KWARGS
from science.tokyo_eye.v8.loader import ensure_pdb_cached, load_structure_batch
from science.tokyo_eye.v8.r0_r5_graph import R1_HBOND, R2_DEHYDRON, set_dehydron_wrap_max

CKPT = Path("checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt")
OUT = Path("data/local_objects/frontend_ablation")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via Mann–Whitney (no sklearn). labels in {0,1}."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(tie)
    # vectorized
    # sort all
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    sum_ranks_pos = float(ranks[labels > 0.5].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def node_classes(edge_index: torch.Tensor, edge_type: torch.Tensor, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_mask, neg_mask) bool arrays length n."""
    ei = edge_index.detach().cpu().numpy()
    et = edge_type.detach().cpu().numpy()
    r2 = np.zeros(n, dtype=bool)
    r1 = np.zeros(n, dtype=bool)
    if et.size:
        m2 = et == R2_DEHYDRON
        m1 = et == R1_HBOND
        if m2.any():
            r2[ei[0, m2]] = True
            r2[ei[1, m2]] = True
        if m1.any():
            r1[ei[0, m1]] = True
            r1[ei[1, m1]] = True
    pos = r2  # dehydron-incident
    neg = r1 & ~r2  # wrapped H-bond only
    return pos, neg


def fisher_and_auc(X: np.ndarray, y: np.ndarray) -> dict:
    """X [N,D], y {0,1}. Score = (x-μ_all)·(μ1-μ0)."""
    y = y.astype(np.float64)
    X = X.astype(np.float64)
    pos = X[y > 0.5]
    neg = X[y < 0.5]
    out = {"n_pos": int(pos.shape[0]), "n_neg": int(neg.shape[0]), "dim": int(X.shape[1])}
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        out.update(auc=float("nan"), centroid_cos=float("nan"), mean_norm_pos=float("nan"), mean_norm_neg=float("nan"))
        return out
    mu1 = pos.mean(axis=0)
    mu0 = neg.mean(axis=0)
    diff = mu1 - mu0
    dn = np.linalg.norm(diff) + 1e-12
    # cosine between centroids from origin
    n1 = np.linalg.norm(mu1) + 1e-12
    n0 = np.linalg.norm(mu0) + 1e-12
    centroid_cos = float(np.dot(mu1, mu0) / (n1 * n0))
    scores = (X - X.mean(axis=0)) @ (diff / dn)
    out.update(
        auc=float(roc_auc(scores, y)),
        centroid_cos=centroid_cos,
        mean_norm_pos=float(np.linalg.norm(mu1)),
        mean_norm_neg=float(np.linalg.norm(mu0)),
        mean_diff_norm=float(dn),
    )
    return out


@torch.no_grad()
def frontend_features(frontend: EquiformerPoolFrontend, x: torch.Tensor) -> np.ndarray:
    s, v = frontend(x)
    return torch.cat([s, v], dim=-1).detach().float().cpu().numpy()


def main() -> None:
    set_dehydron_wrap_max(1)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_boot_split(DEFAULT_MANIFEST)
    entries = [_resolve_entry(e, PDB_DIR) for e in (split["train"] + split["probe"])]

    # --- MPtrj frozen ---
    mptrj = EquiformerPoolFrontend(equiformer_ckpt=CKPT, freeze=True, max_neighbors=50).to(device)
    mptrj.eval()
    # --- random same arch ---
    rand = EquiformerPoolFrontend.__new__(EquiformerPoolFrontend)
    torch.nn.Module.__init__(rand)
    rand.scalar_dim = 128
    rand.vector_dim = 3
    rand.equiformer_ckpt = CKPT
    from science.tokyo_eye.v8.equiformer_pool_frontend import _ensure_vendor_path
    _ensure_vendor_path()
    from equiformer_v3_model.equiformer_v3 import EquiformerV3_OC
    kwargs = dict(MPTRJ_GRADIENT_KWARGS)
    kwargs["max_neighbors"] = 50
    bb = EquiformerV3_OC(**kwargs).to(device)
    # fresh random weights (do not load ckpt)
    rand._backbone = bb
    rand._s_proj = torch.nn.Identity()
    rand._v_proj = torch.nn.Identity()
    rand.frontend_mode = "equiformer_v3_pool_random"
    rand.live_backbone = False
    for p in rand._backbone.parameters():
        p.requires_grad_(False)
    rand.eval()

    pools = {
        "mptrj_frozen": {"Xs": [], "ys": [], "per_pdb": []},
        "random_frozen": {"Xs": [], "ys": [], "per_pdb": []},
        "ca_coords": {"Xs": [], "ys": [], "per_pdb": []},
    }

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry["chain"])
        batch = load_structure_batch(
            pdb_id, chain, pdb_dir=PDB_DIR, device=device, graph_cache_dir=GRAPH_CACHE
        )
        n = int(batch["num_nodes"])
        pos, neg = node_classes(batch["edge_index"], batch["edge_type"], n)
        # keep only labeled residues
        keep = pos | neg
        if keep.sum() < 4 or pos.sum() < 2 or neg.sum() < 2:
            print(f"skip {pdb_id}:{chain} pos={pos.sum()} neg={neg.sum()}")
            continue
        y = np.zeros(n, dtype=np.float64)
        y[pos] = 1.0
        y = y[keep]

        feats_m = frontend_features(mptrj, batch["x"])[keep]
        feats_r = frontend_features(rand, batch["x"])[keep]
        feats_c = batch["x"].detach().float().cpu().numpy()[keep]

        for name, feats in (
            ("mptrj_frozen", feats_m),
            ("random_frozen", feats_r),
            ("ca_coords", feats_c),
        ):
            pools[name]["Xs"].append(feats)
            pools[name]["ys"].append(y)
            local = fisher_and_auc(feats, y)
            local.update(pdb_id=pdb_id, chain=chain, n_keep=int(keep.sum()))
            pools[name]["per_pdb"].append(local)
            print(
                f"{name} {pdb_id}:{chain} n_pos={local['n_pos']} n_neg={local['n_neg']} "
                f"auc={local['auc']:.4f} cos={local['centroid_cos']:.4f}"
            )

    report = {
        "gate_id": "tokyo_eye_equ_frontend_ablation",
        "stage": 0,
        "wrap_max": 1,
        "pos": "incident_R2_dehydron",
        "neg": "incident_R1_hbond_and_not_R2",
        "score": "projection_onto_mean_diff",
        "frontend_ckpt": str(CKPT),
        "arms": {},
    }
    for name, pool in pools.items():
        if not pool["Xs"]:
            continue
        X = np.concatenate(pool["Xs"], axis=0)
        y = np.concatenate(pool["ys"], axis=0)
        agg = fisher_and_auc(X, y)
        aucs = [r["auc"] for r in pool["per_pdb"] if r["auc"] == r["auc"]]
        report["arms"][name] = {
            "aggregate": agg,
            "mean_per_pdb_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "per_pdb": pool["per_pdb"],
        }
        print(
            f"AGG {name}: n_pos={agg['n_pos']} n_neg={agg['n_neg']} "
            f"auc={agg['auc']:.4f} mean_pdb_auc={report['arms'][name]['mean_per_pdb_auc']:.4f}"
        )

    out_path = OUT / "stage0_separability.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
