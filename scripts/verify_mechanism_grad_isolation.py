"""Single-step loss-attribution check for the decoupled spine (Fold 0, 11 train structures).

Three isolated backward passes from the same initial weights:
  bce_only   dehydron_coeff=1.0, sdrp_coeff=0.0
  sdrp_only  dehydron_coeff=0.0, sdrp_coeff=0.1
  combined   dehydron_coeff=1.0, sdrp_coeff=0.1   (production multi-task state)

For each: per-bucket parameter grad L2 and the gradient w.r.t. the two halves of the
MechanismScoreHead input, ``z_hyp`` (through the hyperbolic spine) and ``h_euc``
(Euclidean skip). Under bce_only, non-zero grad on ``projector`` / ``attn_layers`` can
only have arrived through ``z_hyp`` -- that is the proof BCE supervises the spine.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
for p in (str(REPO_ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("D", SCRIPTS / "verify_grad_flow_decoupled.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)  # type: ignore[union-attr]
G, M2 = D.G, D.M2
from science.tokyo_eye.v8.grad_reachability import bucket_grad_stats  # noqa: E402
from experiments.training.v8.run_v8_experiment import DEFAULT_WEIGHT_MAP, load_weight_map  # noqa: E402

PASSES = {
    "bce_only": (0.0, 1.0),
    "sdrp_only": (0.1, 0.0),
    "combined": (0.1, 1.0),
}


def run_pass(system, batches, tau, sdrp_coeff, dehydron_coeff) -> dict:
    system.train()
    system.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    n = len(batches)
    sq_z = sq_h = 0.0
    cnt_z = cnt_h = 0
    for b in batches:
        out = system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=tau,
                     chem=b.get("gate_chem"))
        z, h = out["z_hyp"], out["h_euc"]
        z.retain_grad()
        h.retain_grad()
        loss = 0.0
        if sdrp_coeff:
            loss = loss + sdrp_coeff * F.cross_entropy(out["sdrp_logits"], b["sdrp_target"])
        if dehydron_coeff:
            loss = loss + dehydron_coeff * F.binary_cross_entropy_with_logits(
                out["mechanism_score"], b["dehydron_labels"])
        (loss / n).backward()
        if z.grad is not None:
            sq_z += float(z.grad.pow(2).sum()); cnt_z += z.grad.numel()
        if h.grad is not None:
            sq_h += float(h.grad.pow(2).sum()); cnt_h += h.grad.numel()
    table = bucket_grad_stats(system.spine)
    buckets = {k: float(v["grad_l2"]) for k, v in table.items()}
    z_l2, h_l2 = sq_z ** 0.5, sq_h ** 0.5
    z_rms = (sq_z / max(cnt_z, 1)) ** 0.5
    h_rms = (sq_h / max(cnt_h, 1)) ** 0.5
    system.zero_grad(set_to_none=True)
    return {
        "bucket_grad_l2": buckets,
        "grad_input_z_hyp_l2": z_l2, "grad_input_h_euc_l2": h_l2,
        "grad_input_z_hyp_rms": z_rms, "grad_input_h_euc_rms": h_rms,
        "z_over_h_l2": z_l2 / h_l2 if h_l2 > 0 else float("inf"),
        "z_over_h_rms": z_rms / h_rms if h_rms > 0 else float("inf"),
    }


def main() -> int:
    device = torch.device("cuda")
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    tags = G._all_structure_tags()
    hold, train = tags[0], tags[1:]
    torch.manual_seed(0)
    system, _ = D._build_system(device, cfg)
    batches = [G._load_batch(t, device, decouple_r2_input=True) for t in train]
    tau = float(cfg["tau_start"])
    res = {name: run_pass(system, batches, tau, *coefs) for name, coefs in PASSES.items()}
    bce = res["bce_only"]["bucket_grad_l2"]
    checks = {
        "bce_reaches_projector": bce.get("projector", 0.0) > 0.0,
        "bce_reaches_attn_layers": bce.get("attn_layers", 0.0) > 0.0,
        "bce_reaches_mechanism_head": bce.get("mechanism_head", 0.0) > 0.0,
        "bce_grad_wrt_z_hyp_nonzero": res["bce_only"]["grad_input_z_hyp_l2"] > 0.0,
        "sdrp_only_mechanism_head_zero": res["sdrp_only"]["bucket_grad_l2"].get("mechanism_head", 0.0) == 0.0,
    }
    out = {"hold": hold, "n_train": len(train), "tau": tau, "passes": res, "checks": checks,
           "all_pass": all(checks.values())}
    path = REPO_ROOT / "data" / "gates" / "verify_mechanism_grad_isolation_result.json"
    path.write_text(json.dumps(out, indent=2, default=float))

    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(D.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"verify_mechanism_grad_isolation_fold0_{hold.replace(':', '')}"):
        mlflow.set_tags({"diagnostic": "true", "do_not_promote": "true", "mode": "loss-attribution",
                         "all_checks": "PASS" if out["all_pass"] else "FAIL"})
        for name, r in res.items():
            m = {f"{name}/grad_l2_{k}": v for k, v in r["bucket_grad_l2"].items()}
            m.update({f"{name}/{k}": v for k, v in r.items() if k != "bucket_grad_l2"})
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v and abs(v) != float("inf")})
        mlflow.log_artifact(str(path))
    print(json.dumps({k: out[k] for k in ("checks", "all_pass")}, indent=2))
    for name, r in res.items():
        b = r["bucket_grad_l2"]
        print(name, {k: round(b.get(k, 0.0), 5) for k in ("mechanism_head", "projector", "attn_layers", "moe", "sdrp_head", "euc_skip")},
              "|grad z_hyp|", round(r["grad_input_z_hyp_l2"], 5), "|grad h_euc|", round(r["grad_input_h_euc_l2"], 5),
              "z/h l2", round(r["z_over_h_l2"], 3), "z/h rms", round(r["z_over_h_rms"], 3))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
