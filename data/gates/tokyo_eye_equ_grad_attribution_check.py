"""Per-loss gradient attribution with positive controls (no optimizer step, no training).

For every loss term in run_epoch_mean_structures, backprop it ALONE and report per-bucket:
  NONE  = params not in the autograd graph of that loss (grad is None)
  ZERO  = in graph, gradient exactly 0
  value = grad L2 norm
Separates 'excluded by config' from 'structurally disconnected', and cross-checks the logger.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")
spec = importlib.util.spec_from_file_location("gf", "/workspace/scripts/wrap1_dehydron_loso_g_fit.py")
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)
from science.tokyo_eye.v8.heads import mechanism_margin_loss_v2, sdrp_cross_entropy  # noqa: E402

DEV = torch.device("cpu")
HOLD = "1MBN:A"
TAG_BATCHES = ["1LYZ:A", "1BG1:A"]  # two train structures, mean-reduced like the loop


def bucket(n: str) -> str:
    return ".".join(n.split(".")[:2])


def build(cfg, mode: str):
    torch.manual_seed(0)
    np.random.seed(0)
    system, _ = gf.build_system(cfg, equiformer_ckpt=None, device=DEV, freeze_backbone=False)
    system.set_moe_mode(mode)
    system.train()
    system.set_moe_temperature(float(cfg["gumbel_tau_start"]))
    system.set_moe_explore_epsilon(float(cfg.get("eps_start", 0.20)))
    return system


def comps(system, batches, cfg):
    """Return dict name -> scalar loss (mean over batches), mirroring the loop's terms."""
    acc: dict[str, list] = {k: [] for k in
        ("BCE_dehydron", "margin", "SDRP_CE", "balance_composed", "evidence_SYNTHETIC")}
    tau = float(cfg["tau_start"])  # epoch 0 ceiling
    for b in batches:
        out = system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=tau, chem=b.get("gate_chem"))
        aux = out["moe_aux"]
        z = torch.zeros((), dtype=out["mechanism_score"].dtype)

        def g(k):
            v = aux.get(k)
            return z if v is None else v

        acc["BCE_dehydron"].append(torch.nn.functional.binary_cross_entropy_with_logits(
            out["mechanism_score"], b["dehydron_labels"]))
        acc["margin"].append(mechanism_margin_loss_v2(out["mechanism_score"], b["mechanism_pos"], b["mechanism_neg"]))
        acc["SDRP_CE"].append(sdrp_cross_entropy(out["sdrp_logits"], b["sdrp_target"]))
        acc["balance_composed"].append(
            aux["cv_loss"] * float(cfg["cv_coeff"]) + g("quota_loss") * float(cfg["moe_quota_coeff"])
            + g("majority_hinge_loss") + g("switch_lb_loss")
            + g("soft_quota_loss") * float(cfg["moe_quota_coeff"]) + g("eval_proxy_lb_loss")
            + g("eval_proxy_quota_loss") * float(cfg["eval_proxy_quota_coeff"]))
        acc["evidence_SYNTHETIC"].append(out["evidence"].float().pow(2).mean())  # NOT a real loss term
    n = len(batches)
    res = {k: sum(v) / n for k, v in acc.items()}
    res["G_FIT_TOTAL(sdrp=0)"] = res["BCE_dehydron"] + 0.0 * res["SDRP_CE"] + res["margin"] + res["balance_composed"]
    res["TOTAL_with_sdrp=1"] = res["BCE_dehydron"] + 1.0 * res["SDRP_CE"] + res["margin"] + res["balance_composed"]
    return res


def grad_table(system, loss):
    system.zero_grad(set_to_none=True)
    if not loss.requires_grad:
        return None
    loss.backward(retain_graph=True)
    per: dict[str, dict] = {}
    for n, p in system.named_parameters():
        d = per.setdefault(bucket(n), {"g2": 0.0, "none": 0, "zero": 0, "nz": 0, "n": 0})
        d["n"] += 1
        if p.grad is None:
            d["none"] += 1
        else:
            s = float(p.grad.pow(2).sum())
            d["g2"] += s
            d["zero" if s == 0.0 else "nz"] += 1
    glob = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in system.parameters() if p.grad is not None) or torch.tensor(0.0)))
    return per, glob


def cell(d) -> str:
    if d["nz"] == 0 and d["zero"] == 0:
        return "NONE"
    if d["nz"] == 0:
        return "ZERO"
    return f"{d['g2'] ** 0.5:.3g}"


def main() -> None:
    cfg = gf.load_weight_map(gf.REPO_ROOT / gf.DEFAULT_WEIGHT_MAP)
    batches = [gf._load_batch(t, DEV) for t in TAG_BATCHES]
    out = {}
    for mode in ("ablated", "live"):
        system = build(cfg, mode)
        cs = comps(system, batches, cfg)
        table, glob, bucket_names = {}, {}, None
        for name, loss in cs.items():
            r = grad_table(system, loss)
            if r is None:
                table[name] = "CONSTANT (no graph)"
                continue
            per, gnorm = r
            table[name] = per
            glob[name] = gnorm
            bucket_names = sorted(per)
        out[mode] = {"table": table, "global_grad_norm": glob}
        print(f"\n=== moe_mode={mode} : grad L2 by bucket (NONE = not in graph, ZERO = in graph but 0) ===")
        cols = [c for c in cs if isinstance(table[c], dict)]
        short = {"BCE_dehydron": "BCE", "margin": "margin", "SDRP_CE": "SDRP_CE", "balance_composed": "balance",
                 "evidence_SYNTHETIC": "evid*", "G_FIT_TOTAL(sdrp=0)": "GFIT_TOT", "TOTAL_with_sdrp=1": "TOT_sdrp1"}
        print(f"{'bucket':24s}" + "".join(f"{short[c]:>11s}" for c in cols))
        for b in bucket_names:
            print(f"{b:24s}" + "".join(f"{cell(table[c][b]):>11s}" for c in cols))
        for c in cs:
            if not isinstance(table[c], dict):
                print(f"  [{short[c]}] {table[c]}")
        # logger completeness: sum of per-bucket norms == torch global norm for the G_fit total
        per = table["G_FIT_TOTAL(sdrp=0)"]
        ssum = sum(d["g2"] for d in per.values()) ** 0.5
        print(f"  logger check (G_FIT_TOTAL): sqrt(sum bucket g2)={ssum:.6g} vs torch global grad norm={glob['G_FIT_TOTAL(sdrp=0)']:.6g}")
        out[mode]["logger_check"] = {"bucket_sum": ssum, "torch_global": glob["G_FIT_TOTAL(sdrp=0)"]}
    Path("/tmp/grad_attribution_result.json").write_text(json.dumps(out, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
