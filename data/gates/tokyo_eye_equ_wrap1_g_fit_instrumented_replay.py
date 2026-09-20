"""Instrumented replay of ONE G_fit fold. Reuses the sealed harness loop unchanged.

Adds (without editing scripts/wrap1_dehydron_loso_g_fit.py):
  * MLflow params (steps, per-group lr, seed, frontend, coeffs, script sha)
  * per-bucket grad norm + parameter drift from init, logged every 10 steps
  * post-training edge_type sensitivity of the trained model (hold + one train fold)
Does NOT write E200/E400 stamps or touch the disposition.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gf = _load("gf", "/workspace/scripts/wrap1_dehydron_loso_g_fit.py")
ret = _load(
    "ret", str(Path(__file__).with_name("tokyo_eye_equ_edge_type_reachability_check.py"))
)

S: dict = {"init": None, "trace": [], "groups": None, "ps_trace": [], "npz": {}}
PS_STEPS = {0, 50, 100, 150, 200, 250, 300, 350}
_orig_epoch = gf.run_epoch_mean_structures


def bucket(name: str) -> str:
    return ".".join(name.split(".")[:2])


def snapshot(system) -> dict:
    acc: dict = {}
    for n, p in system.named_parameters():
        b = bucket(n)
        d = acc.setdefault(
            b, {"g2": 0.0, "d2": 0.0, "i2": 0.0, "dmax": 0.0, "n": 0, "none": 0, "zero": 0, "nz": 0}
        )
        p0 = S["init"][n].to(p.device)
        delta = (p.detach() - p0)
        d["d2"] += float(delta.pow(2).sum())
        d["i2"] += float(p0.pow(2).sum())
        d["dmax"] = max(d["dmax"], float(delta.abs().max())) if delta.numel() else d["dmax"]
        d["n"] += p.numel()
        if p.grad is None:
            d["none"] += 1
        else:
            g2p = float(p.grad.detach().pow(2).sum())
            d["g2"] += g2p
            d["zero" if g2p == 0.0 else "nz"] += 1
    out = {}
    for b, d in acc.items():
        out[b] = {
            "grad_l2": d["g2"] ** 0.5,
            "drift_l2": d["d2"] ** 0.5,
            "drift_rel": (d["d2"] ** 0.5) / max(d["i2"] ** 0.5, 1e-12),
            "drift_max_abs": d["dmax"],
            "n_params": d["n"],
            "n_tensors_grad_none": d["none"],
            "n_tensors_grad_exact_zero": d["zero"],
            "n_tensors_grad_nonzero": d["nz"],
        }
    for grp in ("frontend", "spine"):
        keys = [b for b in out if b.startswith(grp + ".")]
        out[f"ALL.{grp}"] = {
            "grad_l2": sum(out[k]["grad_l2"] ** 2 for k in keys) ** 0.5,
            "drift_l2": sum(out[k]["drift_l2"] ** 2 for k in keys) ** 0.5,
            "drift_rel": float("nan"),
            "drift_max_abs": max((out[k]["drift_max_abs"] for k in keys), default=0.0),
            "n_params": sum(out[k]["n_params"] for k in keys),
        }
    return out


def _mlflow_active():
    try:
        import mlflow

        return mlflow if mlflow.active_run() is not None else None
    except Exception:  # noqa: BLE001
        return None


from sklearn.metrics import roc_auc_score  # noqa: E402


def _pct(a: np.ndarray) -> np.ndarray:
    return (np.argsort(np.argsort(a)).astype(np.float64) + 0.5) / len(a)


def _one(system, b, tau):
    with torch.no_grad():
        out = system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=tau, chem=b.get("gate_chem"))
    lg = out["mechanism_score"].detach().float().cpu()
    return lg, torch.sigmoid(lg), b["dehydron_labels"].detach().float().cpu()


def _metrics(lg, s, y):
    prev = float(y.mean())
    a = float(gf.binary_auprc(s, y))
    yy = y.numpy() > 0
    return {"n": int(y.numel()), "prevalence": prev, "auprc": a,
            "lift": a / prev if prev > 0 else float("nan"),
            "roc_auc": float(roc_auc_score(yy, lg.numpy())) if 0 < yy.sum() < len(yy) else float("nan"),
            "logit_mean": float(lg.mean()), "logit_std": float(lg.std())}


def per_structure_eval(system, batches, hold_batch, tau, keep_arrays=False):
    was = system.training
    system.eval()
    rows, S_, Y_, LG_, RN_, Z_ = {}, [], [], [], [], []
    for b in batches:
        lg, s, y = _one(system, b, tau)
        tag = f"{b['pdb_id']}:{b['chain']}"
        rows[tag] = _metrics(lg, s, y)
        S_.append(s); Y_.append(y); LG_.append(lg)
        a = lg.numpy().astype(np.float64)
        RN_.append(torch.from_numpy(_pct(a))); Z_.append(torch.from_numpy((a - a.mean()) / (a.std() + 1e-9)))
        if keep_arrays:
            k = tag.replace(":", "_")
            S["npz"][f"logit__{k}"] = lg.numpy(); S["npz"][f"label__{k}"] = y.numpy()
    Y = torch.cat(Y_)
    aus = [r["auprc"] for r in rows.values()]
    prevs = [r["prevalence"] for r in rows.values()]
    res = {
        "per_structure": rows,
        "pooled_auprc_sealed_method": float(gf.binary_auprc(torch.cat(S_), Y)),
        "pooled_auprc_rank_normalised_per_structure": float(gf.binary_auprc(torch.cat(RN_), Y)),
        "pooled_auprc_zscored_logit_per_structure": float(gf.binary_auprc(torch.cat(Z_), Y)),
        "macro_mean_auprc": float(np.nanmean(aus)),
        "macro_mean_lift": float(np.nanmean([a / p for a, p in zip(aus, prevs)])),
        "min_structure_auprc": float(np.nanmin(aus)),
        "n_structures": len(rows),
    }
    lg, s, y = _one(system, hold_batch, tau)
    res["heldout"] = {f"{hold_batch['pdb_id']}:{hold_batch['chain']}": _metrics(lg, s, y)}
    if keep_arrays:
        k = f"{hold_batch['pdb_id']}_{hold_batch['chain']}"
        S["npz"][f"logit__{k}__HELDOUT"] = lg.numpy(); S["npz"][f"label__{k}__HELDOUT"] = y.numpy()
    if was:
        system.train()
    return res


def instrumented_epoch(system, optimizer, batches, **kw):
    if S["init"] is None:
        S["init"] = {n: p.detach().clone().cpu() for n, p in system.named_parameters()}
        S["system"] = system
        S["groups"] = [
            {"name": g.get("name"), "lr": g["lr"], "n_tensors": len(g["params"])}
            for g in optimizer.param_groups
        ]
        mlf = _mlflow_active()
        if mlf is not None:
            params = {
                "steps_total_optimizer": S["epochs"],
                "steps_per_epoch": 1,
                "batch_reduction": "mean over 11 train structures, 1 Adam step/epoch",
                "optimizer": type(optimizer).__name__,
                "seed": S["seed"],
                "hold": S["hold"],
                "frontend": "cold_se3_lite",
                "moe_mode": "ablated",
                "sdrp_coeff": gf.SDRP_COEFF,
                "dehydron_coeff": kw.get("dehydron_coeff"),
                "margin_coeff": kw.get("margin_coeff"),
                "max_grad_norm": kw.get("max_grad_norm", 1.0),
                "g_fit_threshold": gf.G_FIT_THRESHOLD,
                "instrumented_replay": "true",
                "harness_script_sha256": S["script_sha"],
            }
            for g in S["groups"]:
                params[f"lr_{g['name']}"] = g["lr"]
            mlf.log_params({k: str(v) for k, v in params.items()})
            mlf.set_tag("instrumented_replay", "true")
            mlf.set_tag("mlflow.runName", f"INSTR_replay_E{S['epochs']}_hold_{S['hold'].replace(':', '')}")
    m = _orig_epoch(system, optimizer, batches, **kw)
    ep = int(kw["epoch"])
    if ep < 3 or ep % 10 == 0:
        snap = snapshot(system)
        S["trace"].append({"epoch": ep, "buckets": snap})
        mlf = _mlflow_active()
        if mlf is not None:
            flat = {}
            for b, d in snap.items():
                for k in ("grad_l2", "drift_l2", "drift_rel", "drift_max_abs", "n_tensors_grad_none"):
                    v = d.get(k, float("nan"))
                    if v == v:
                        flat[f"{k}/{b}"] = float(v)
            if isinstance(m.get("grad_norm"), float):
                flat["grad_norm_preclip_total"] = m["grad_norm"]
            mlf.log_metrics(flat, step=ep)
    if ep in PS_STEPS or ep == S["epochs"] - 1:
        last = ep == S["epochs"] - 1
        ps = per_structure_eval(system, batches, S["hold_batch"], S["tau_eval"], keep_arrays=last)
        S["ps_trace"].append({"epoch": ep, **ps})
        mlf = _mlflow_active()
        if mlf is not None:
            mlf.log_metrics({"ps/pooled_auprc": ps["pooled_auprc_sealed_method"],
                             "ps/macro_mean_auprc": ps["macro_mean_auprc"],
                             "ps/min_structure_auprc": ps["min_structure_auprc"],
                             "ps/heldout_auprc": next(iter(ps["heldout"].values()))["auprc"]}, step=ep)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", default="1MBN:A")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("STOP: cuda requested but unavailable")
    S.update(epochs=a.epochs, seed=a.seed, hold=a.hold,
             script_sha=hashlib.sha256(Path(gf.__file__).read_bytes()).hexdigest())
    gf.run_epoch_mean_structures = instrumented_epoch
    dev = torch.device(a.device)
    S["hold_batch"] = gf._load_batch(a.hold, dev)
    S["tau_eval"] = float(gf.load_weight_map(gf.REPO_ROOT / gf.DEFAULT_WEIGHT_MAP)["tau_end"])
    row = gf.run_one_fold(a.hold, device=dev, epochs=a.epochs, seed=a.seed, no_mlflow=a.no_mlflow)

    system = S["system"].cpu().eval()
    hold_b = gf._load_batch(a.hold, torch.device("cpu"))
    tr_tag = next(t for t in gf._all_structure_tags() if t != a.hold)
    tr_b = gf._load_batch(tr_tag, torch.device("cpu"))
    post = {"hold": ret.sensitivity(system, hold_b, state="trained"), f"train:{tr_tag}": ret.sensitivity(system, tr_b, state="trained")}
    final = snapshot(system)
    res = {
        "kind": "instrumented_replay_of_sealed_G_fit_fold",
        "not_a_new_card": "same config as sealed E400 fold; diagnostic only; no stamps/disposition touched",
        "hold": a.hold, "epochs": a.epochs, "seed": a.seed, "device": a.device,
        "harness_script_sha256": S["script_sha"],
        "instrument_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "param_groups": S["groups"],
        "final_train_pooled_auprc": row["final"]["train_pooled_auprc"],
        "finite": row["finite"],
        "curve_every_10": row["curve_every_10"],
        "trace": S["trace"],
        "per_structure_trace": S["ps_trace"],
        "final_buckets": final,
        "post_training_edge_type_sensitivity": post,
    }
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    if S["npz"]:
        np.savez_compressed(str(Path(a.out).with_suffix("")) + ".scores.npz", **S["npz"])
    print(f"[instr] wrote {a.out}")


if __name__ == "__main__":
    main()
