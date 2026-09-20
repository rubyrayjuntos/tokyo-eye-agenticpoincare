"""Stage 0 — LOOCV+L2 dehydron vs wrapped-H-bond separability (all frontend arms).

Prior Stage 0 AUCs were in-sample mean-diff (fit+score on same residues).
GearNet's ~3072-d vs Equiformer ~131-d can inflate that. This script reports:
  - in_sample_mean_diff_auc (legacy continuity)
  - loocv_l2: leave-one-PDB-out L2 logistic (fixed C=1.0), train-fold StandardScaler
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.training.v8.equ_theme_restore import DEFAULT_MANIFEST, load_boot_split
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import _resolve_entry
from science.tokyo_eye.v8.equiformer_pool_frontend import EquiformerPoolFrontend, MPTRJ_GRADIENT_KWARGS
from science.tokyo_eye.v8.loader import load_structure_batch
from science.tokyo_eye.v8.r0_r5_graph import R1_HBOND, R2_DEHYDRON, set_dehydron_wrap_max

CKPT = Path("checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt")
GEARNET_CKPT = Path("checkpoints/tokyoeye/pretrained/gearnet/mc_gearnet_edge.pth")
OUT = Path("data/local_objects/frontend_ablation")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")
L2_C = 1.0
RANDOM_SEED = 0


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
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


def node_classes(edge_index: torch.Tensor, edge_type: torch.Tensor, n: int):
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
    return r2, r1 & ~r2


def fisher_and_auc(X: np.ndarray, y: np.ndarray) -> dict:
    y = y.astype(np.float64)
    X = X.astype(np.float64)
    pos = X[y > 0.5]
    neg = X[y < 0.5]
    out = {"n_pos": int(pos.shape[0]), "n_neg": int(neg.shape[0]), "dim": int(X.shape[1])}
    if pos.shape[0] < 2 or neg.shape[0] < 2:
        out.update(auc=float("nan"))
        return out
    mu1 = pos.mean(axis=0)
    mu0 = neg.mean(axis=0)
    diff = mu1 - mu0
    dn = np.linalg.norm(diff) + 1e-12
    scores = (X - X.mean(axis=0)) @ (diff / dn)
    out["auc"] = float(roc_auc(scores, y))
    return out


def loocv_l2(per_struct: list[dict], C: float = L2_C) -> dict:
    """Leave-one-structure-out L2 logistic. per_struct items: X,y,pdb_id,chain."""
    n = len(per_struct)
    fold_aucs = []
    all_scores = []
    all_labels = []
    all_pdb = []
    for i in range(n):
        te = per_struct[i]
        tr = [per_struct[j] for j in range(n) if j != i]
        Xtr = np.concatenate([t["X"] for t in tr], axis=0)
        ytr = np.concatenate([t["y"] for t in tr], axis=0)
        Xte = te["X"]
        yte = te["y"]
        if (ytr > 0.5).sum() < 2 or (ytr < 0.5).sum() < 2:
            fold_aucs.append(float("nan"))
            continue
        if (yte > 0.5).sum() < 1 or (yte < 0.5).sum() < 1:
            fold_aucs.append(float("nan"))
            continue
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr_s, ytr)
        scores = clf.decision_function(Xte_s)
        auc = float(roc_auc(scores, yte))
        fold_aucs.append(auc)
        all_scores.append(scores)
        all_labels.append(yte)
        all_pdb.append(f"{te['pdb_id']}:{te['chain']}")
    valid = [a for a in fold_aucs if a == a]
    pooled = float("nan")
    if all_scores:
        pooled = float(roc_auc(np.concatenate(all_scores), np.concatenate(all_labels)))
    return {
        "C": C,
        "scaler": "StandardScaler_fit_on_train_fold",
        "mean_per_pdb_auc": float(np.mean(valid)) if valid else float("nan"),
        "std_per_pdb_auc": float(np.std(valid)) if valid else float("nan"),
        "n_valid_folds": len(valid),
        "pooled_heldout_auc": pooled,
        "per_fold_auc": [
            {"pdb": f"{per_struct[i]['pdb_id']}:{per_struct[i]['chain']}", "auc": fold_aucs[i]}
            for i in range(n)
        ],
    }


@torch.no_grad()
def frontend_features(frontend, x: torch.Tensor) -> np.ndarray:
    s, v = frontend(x)
    return torch.cat([s, v], dim=-1).detach().float().cpu().numpy()


def align_by_ca(ref_xyz: np.ndarray, src_xyz: np.ndarray, src_feat: np.ndarray, tol: float = 1.0):
    """Map each ref CA to nearest src CA; return features aligned to ref, and match mask."""
    from scipy.spatial import cKDTree

    tree = cKDTree(src_xyz)
    dist, idx = tree.query(ref_xyz, k=1)
    ok = dist <= tol
    out = np.zeros((ref_xyz.shape[0], src_feat.shape[1]), dtype=np.float64)
    out[ok] = src_feat[idx[ok]]
    return out, ok, dist


def gearnet_features_for_pdb(pdb_path: Path, chain: str, device: torch.device, model, gc):
    from torchdrug import data

    protein = data.Protein.from_pdb(
        str(pdb_path), atom_feature=None, bond_feature=None, residue_feature="default"
    )
    # Filter to chain if chain_id present (torchdrug encodes chains as ints)
    # Prefer filtering atoms by matching chain letter via residue numbers after AlphaCarbon
    batch = data.Protein.pack([protein])
    g = gc(batch)
    with torch.no_grad():
        out = model(g.to(device) if device.type == "cuda" else g, g.residue_feature.float().to(device))
        feat = out["node_feature"].detach().float().cpu().numpy()
    pos = g.node_position.detach().float().cpu().numpy()
    return feat, pos


def build_random_frontend(device):
    from science.tokyo_eye.v8.equiformer_pool_frontend import _ensure_vendor_path

    _ensure_vendor_path()
    from equiformer_v3_model.equiformer_v3 import EquiformerV3_OC

    rand = EquiformerPoolFrontend.__new__(EquiformerPoolFrontend)
    torch.nn.Module.__init__(rand)
    rand.scalar_dim = 128
    rand.vector_dim = 3
    rand.equiformer_ckpt = CKPT
    kwargs = dict(MPTRJ_GRADIENT_KWARGS)
    kwargs["max_neighbors"] = 50
    bb = EquiformerV3_OC(**kwargs).to(device)
    rand._backbone = bb
    rand._s_proj = torch.nn.Identity()
    rand._v_proj = torch.nn.Identity()
    rand.frontend_mode = "equiformer_v3_pool_random"
    rand.live_backbone = False
    for p in rand._backbone.parameters():
        p.requires_grad_(False)
    rand.eval()
    return rand


def main():
    set_dehydron_wrap_max(1)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_boot_split(DEFAULT_MANIFEST)
    entries = [_resolve_entry(e, PDB_DIR) for e in (split["train"] + split["probe"])]

    mptrj = EquiformerPoolFrontend(equiformer_ckpt=CKPT, freeze=True, max_neighbors=50).to(device)
    mptrj.eval()
    rand = build_random_frontend(device)

    # GearNet
    gearnet_ok = False
    gearnet_err = None
    gearnet_model = None
    gc = None
    try:
        from torchdrug import models, layers

        gearnet_model = models.GearNet(
            input_dim=21,
            hidden_dims=[512] * 6,
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            concat_hidden=True,
            short_cut=True,
            batch_norm=True,
        )
        sd = torch.load(GEARNET_CKPT, map_location="cpu")
        gearnet_model.load_state_dict(sd)
        gearnet_model = gearnet_model.to(device)
        gearnet_model.eval()
        gc = layers.GraphConstruction(
            node_layers=[layers.geometry.AlphaCarbonNode()],
            edge_layers=[
                layers.geometry.SequentialEdge(max_distance=2),
                layers.geometry.SpatialEdge(radius=10.0, min_distance=5),
                layers.geometry.KNNEdge(k=10, min_distance=5),
            ],
            edge_feature="gearnet",
        )
        gearnet_ok = True
        print("GearNet ready")
    except Exception as e:
        gearnet_err = f"{type(e).__name__}: {e}"
        print("GearNet BLOCKED:", gearnet_err)

    pools = {
        "mptrj_frozen": [],
        "random_frozen": [],
        "ca_coords": [],
        "gearnet_mc_edge": [],
    }

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry["chain"])
        batch = load_structure_batch(
            pdb_id, chain, pdb_dir=PDB_DIR, device=device, graph_cache_dir=GRAPH_CACHE
        )
        n = int(batch["num_nodes"])
        pos, neg = node_classes(batch["edge_index"], batch["edge_type"], n)
        keep = pos | neg
        if keep.sum() < 4 or pos.sum() < 2 or neg.sum() < 2:
            print(f"skip {pdb_id}:{chain} pos={pos.sum()} neg={neg.sum()}")
            continue
        y = np.zeros(n, dtype=np.float64)
        y[pos] = 1.0
        y_k = y[keep]
        xyz = batch["x"].detach().float().cpu().numpy()

        feats = {
            "mptrj_frozen": frontend_features(mptrj, batch["x"])[keep],
            "random_frozen": frontend_features(rand, batch["x"])[keep],
            "ca_coords": xyz[keep],
        }
        if gearnet_ok:
            pdb_path = PDB_DIR / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                # try lowercase / ensure_pdb style
                alts = list(PDB_DIR.glob(f"{pdb_id}*.pdb")) + list(PDB_DIR.glob(f"{pdb_id.lower()}*.pdb"))
                pdb_path = alts[0] if alts else pdb_path
            try:
                gfeat, gpos = gearnet_features_for_pdb(pdb_path, chain, device, gearnet_model, gc)
                aligned, ok, dist = align_by_ca(xyz, gpos, gfeat, tol=1.5)
                # keep only labeled residues that matched
                keep2 = keep & ok
                if keep2.sum() < 4 or (y[keep2] > 0.5).sum() < 2 or (y[keep2] < 0.5).sum() < 2:
                    print(f"gearnet align skip {pdb_id}:{chain} matched={ok.sum()}/{n} keep2={keep2.sum()}")
                else:
                    # For GearNet use matched subset; for fairness note n may differ slightly
                    pools["gearnet_mc_edge"].append(
                        {
                            "X": aligned[keep2],
                            "y": y[keep2],
                            "pdb_id": pdb_id,
                            "chain": chain,
                            "n_match": int(ok.sum()),
                            "mean_match_dist": float(dist[ok].mean()) if ok.any() else float("nan"),
                        }
                    )
                    print(
                        f"gearnet {pdb_id}:{chain} n={keep2.sum()} dim=3072 "
                        f"match={ok.sum()}/{n} mean_d={dist[ok].mean():.3f}"
                    )
            except Exception as e:
                print(f"gearnet fail {pdb_id}:{chain}: {type(e).__name__}: {e}")

        for name, X in feats.items():
            pools[name].append({"X": X, "y": y_k, "pdb_id": pdb_id, "chain": chain})
            ins = fisher_and_auc(X, y_k)
            print(f"{name} {pdb_id}:{chain} dim={ins['dim']} in_sample_auc={ins['auc']:.4f}")

    report = {
        "gate_id": "tokyo_eye_equ_frontend_ablation",
        "stage": 0,
        "method_note": (
            "Prior stage0_separability.json AUCs were IN-SAMPLE mean-diff "
            "(fit μ1-μ0 on same residues then score). This report adds leave-one-PDB-out "
            f"L2 logistic (C={L2_C}) with StandardScaler fit on train folds only."
        ),
        "wrap_max": 1,
        "pos": "incident_R2_dehydron",
        "neg": "incident_R1_hbond_and_not_R2",
        "l2_C": L2_C,
        "gearnet_ckpt": str(GEARNET_CKPT),
        "gearnet_ok": gearnet_ok,
        "gearnet_error": gearnet_err,
        "arms": {},
    }

    for name, items in pools.items():
        if not items:
            report["arms"][name] = {"status": "empty"}
            continue
        X = np.concatenate([it["X"] for it in items], axis=0)
        y = np.concatenate([it["y"] for it in items], axis=0)
        ins = fisher_and_auc(X, y)
        loo = loocv_l2(items, C=L2_C)
        report["arms"][name] = {
            "dim": int(X.shape[1]),
            "n_structures": len(items),
            "n_pos": int(ins["n_pos"]),
            "n_neg": int(ins["n_neg"]),
            "in_sample_mean_diff_auc": float(ins["auc"]),
            "loocv_l2": loo,
        }
        print(
            f"AGG {name}: dim={X.shape[1]} in_sample={ins['auc']:.4f} "
            f"loocv_mean={loo['mean_per_pdb_auc']:.4f} loocv_pooled={loo['pooled_heldout_auc']:.4f}"
        )

    out_path = OUT / "stage0_separability_loocv.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
