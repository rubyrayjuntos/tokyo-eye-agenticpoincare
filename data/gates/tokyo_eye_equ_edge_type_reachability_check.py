"""Code-level check (no harness training): does edge_type reach mechanism_score?

Lite path  : init-state sensitivity + closed-form ridge decodability on frozen init features.
Pool path  : bitwise invariance test under edge_type perturbation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score as ap

sys.path.insert(0, "/workspace")
spec = importlib.util.spec_from_file_location(
    "gf", "/workspace/scripts/wrap1_dehydron_loso_g_fit.py"
)
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)
from science.tokyo_eye.v8.loader import R2_DEHYDRON, R5_LOCAL_NEIGHBORHOOD  # noqa: E402

DEV = torch.device("cpu")
HOLD = "1MBN:A"
TAU = 0.995
OUT = Path("/workspace/data/gates/tokyo_eye_equ_edge_type_reachability_check.json")


def rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def perturb(et: torch.Tensor, kind: str, seed: int = 0) -> torch.Tensor:
    if kind == "shuffle":
        g = torch.Generator().manual_seed(seed)
        return et[torch.randperm(et.numel(), generator=g)]
    if kind == "r2_to_r5":
        out = et.clone()
        out[out == R2_DEHYDRON] = R5_LOCAL_NEIGHBORHOOD
        return out
    if kind == "all_r5":
        return torch.full_like(et, R5_LOCAL_NEIGHBORHOOD)
    raise ValueError(kind)


def mech(system, batch, et):
    with torch.no_grad():
        out = system(
            batch["x"], batch["edge_index"], et,
            tau_ceiling=TAU, chem=batch.get("gate_chem"),
        )
    return out["mechanism_score"].detach().double().numpy()


def sensitivity(system, batch, state: str = "init") -> dict:
    y = batch["dehydron_labels"].numpy() > 0
    base = mech(system, batch, batch["edge_type"])
    sd = float(base.std())
    res = {"model_state": state, "score_std": sd, "AUPRC_true_type": float(ap(y, base))}
    for kind, seeds in (("shuffle", (0, 1, 2)), ("r2_to_r5", (0,)), ("all_r5", (0,))):
        rows = []
        for s in seeds:
            p = mech(system, batch, perturb(batch["edge_type"], kind, s))
            d = np.abs(p - base)
            rows.append({
                "max_abs_diff": float(d.max()),
                "mean_abs_diff_over_std": float(d.mean() / sd) if sd > 0 else float("nan"),
                "spearman_vs_true": spearman(base, p),
                "mean_abs_diff_pos_over_std": float(d[y].mean() / sd) if sd > 0 else float("nan"),
                "mean_abs_diff_neg_over_std": float(d[~y].mean() / sd) if sd > 0 else float("nan"),
            })
        res[kind] = rows if len(rows) > 1 else rows[0]
    return res


def frontend_feats(system, batch, et) -> np.ndarray:
    with torch.no_grad():
        s, _ = system.frontend(batch["x"], edge_index=batch["edge_index"], edge_type=et)
    return s.detach().double().numpy()


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ y)
    return w, mu, sd


def ridge_pred(X, w, mu, sd):
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Z @ w


def decodability(system, train_batches, hold_batch, lam=10.0) -> dict:
    res = {}
    conds = [("true_type", None), ("shuffled_type_s0", ("shuffle", 0)),
             ("shuffled_type_s1", ("shuffle", 1)), ("all_R5_type", ("all_r5", 0))]
    for name, pk in conds:
        def et_of(b):
            return b["edge_type"] if pk is None else perturb(b["edge_type"], pk[0], pk[1])
        Xtr = np.vstack([frontend_feats(system, b, et_of(b)) for b in train_batches])
        ytr = np.concatenate([b["dehydron_labels"].numpy() for b in train_batches]).astype(np.float64)
        w, mu, sd = ridge_fit(Xtr, ytr, lam)
        tr_auprc = float(ap(ytr > 0, ridge_pred(Xtr, w, mu, sd)))
        Xh = frontend_feats(system, hold_batch, et_of(hold_batch))
        yh = hold_batch["dehydron_labels"].numpy() > 0
        ho_auprc = float(ap(yh, ridge_pred(Xh, w, mu, sd)))
        res[name] = {"ridge_train_pooled_AUPRC": tr_auprc, "ridge_heldout_AUPRC": ho_auprc,
                     "train_prevalence": float(ytr.mean())}
    return res


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = gf.load_weight_map(gf.REPO_ROOT / gf.DEFAULT_WEIGHT_MAP)
    tags = gf._all_structure_tags()
    train_tags = [t for t in tags if t != HOLD]
    batches = {t: gf._load_batch(t, DEV) for t in tags}
    result: dict = {"hold": HOLD, "n_train_structures": len(train_tags)}

    # ---- lite (same construction/seed as G_fit fold 1) ----
    lite, _ = gf.build_system(cfg, equiformer_ckpt=None, device=DEV, freeze_backbone=False)
    lite.set_moe_mode("ablated")
    lite.eval()
    result["lite"] = {
        "sensitivity_init_state_on_hold": sensitivity(lite, batches[HOLD]),
        "ridge_decodability_frozen_init_features": decodability(
            lite, [batches[t] for t in train_tags], batches[HOLD]
        ),
    }

    # ---- pool (cold_init architecture; same forward contract) ----
    try:
        from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system

        pool, info = build_equiformer_pool_system(
            cfg, equiformer_ckpt=None, device=DEV, freeze_backbone=False, cold_init=True
        )
        pool.set_moe_mode("ablated")
        pool.eval()
        result["pool"] = {"constructed": True, "load_info_keys": sorted(info.keys())[:12],
                          "sensitivity_cold_init_on_hold": sensitivity(pool, batches[HOLD])}
    except Exception as exc:  # noqa: BLE001
        result["pool"] = {"constructed": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback_tail": traceback.format_exc().splitlines()[-6:]}
    print(json.dumps(result, indent=1))
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path("/tmp/edge_type_reachability_result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
