"""100-step gradient-flow / train-mode diagnostic for the decoupled-R2 spine.

Modes
-----
``--mode verify-grad-flow``   100 steps on Fold 0, SDRP CE + dehydron BCE (coeff 1.0),
                              logs per-bucket grad L2 to MLflow and evaluates the gates.
``--mode verify-train-mode``  backbone Dropout/GraphDropPath modules all report
                              ``.training == True`` after ``system.train()``.
``--mode all``                both (default).

Not a sealed card: diagnostic only, logs to ``diag/gradflow-decoupled-r2``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
for p in (str(REPO_ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import wrap1_zhyp_g_fit as G  # noqa: E402
from experiments.training.v8.run_v8_experiment import (  # noqa: E402
    DEFAULT_WEIGHT_MAP,
    build_equiformer_pool_system,
    load_weight_map,
)
from science.tokyo_eye.v8.engine import (  # noqa: E402
    CurriculumRadiusController,
    EpsilonGreedySchedule,
    GumbelTemperatureSchedule,
)
from science.tokyo_eye.v8.r0_r5_graph import R2_DEHYDRON  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "wrap1_zhyp_m2_pool_lr", SCRIPTS / "wrap1_zhyp_m2_pool_lr.py"
)
M2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M2)  # type: ignore[union-attr]

MLFLOW_EXPERIMENT = "diag/gradflow-decoupled-r2"
MAX_GRAD_NORM = 1.0
DEHYDRON_COEFF = 1.0
# name -> (bucket key, minimum grad L2)
GRAD_GATES = {
    "grad_l2_mechanism_head": ("mechanism_head", 1e-3),
    "grad_l2_sdrp_head": ("sdrp_head", 1e-3),
    "grad_l2_attn_layers": ("attn_layers", 1e-3),
    "grad_l2_moe": ("moe", 1e-4),
}
BCE_STEP0_MIN = 0.30


def _build_system(device: torch.device, cfg: dict[str, Any]):
    system, load_info = build_equiformer_pool_system(
        cfg,
        equiformer_ckpt=None,
        device=device,
        freeze_backbone=False,
        max_neighbors=M2.MAX_NEIGHBORS,
        cold_init=True,
        spine_kwargs={"decoupled_arch": True},
    )
    system.set_moe_mode("ablated")
    system.frontend._backbone.gradient_checkpointing_block_list = [1] * int(
        system.frontend._backbone.num_layers
    )
    M2.apply_backbone_train_mode(system)
    return system, load_info


def _value_path_grad_l2(system: torch.nn.Module) -> float:
    total = 0.0
    for name, p in system.spine.named_parameters():
        if ("R_v_rel" in name or name.endswith("v_scale")) and p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total**0.5


def _input_leak_checks(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Structural (not statistical) leakage checks on what the model is fed."""
    n_r2_edges = 0
    tau_col_max = 0.0
    for b in batches:
        n_r2_edges += int((b["edge_type"] == R2_DEHYDRON).sum().item())
        tau_col_max = max(tau_col_max, float(b["gate_chem"][:, 1].abs().max().item()))
    return {
        "input_r2_edges": n_r2_edges,
        "gate_chem_tau_abs_max": tau_col_max,
        "pass": n_r2_edges == 0 and tau_col_max == 0.0,
    }


@torch.no_grad()
def _step0_dehydron_bce(system, batch, tau: float) -> dict[str, float]:
    system.eval()
    out = system(
        batch["x"], batch["edge_index"], batch["edge_type"],
        tau_ceiling=tau, chem=batch.get("gate_chem"),
    )
    y = batch["dehydron_labels"]
    bce = float(F.binary_cross_entropy_with_logits(out["mechanism_score"], y))
    p = float(y.mean().clamp(1e-6, 1 - 1e-6))
    prior_bce = float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
    system.train()
    return {"bce_step0": bce, "label_prior": p, "bce_constant_prior": prior_bce}


def verify_train_mode(system: torch.nn.Module) -> dict[str, Any]:
    system.train()
    live, total = M2.backbone_reg_modules_live(system)
    return {"live": live, "total": total, "pass": total == 14 and live == total}


def verify_grad_flow(
    device: torch.device, steps: int, seed: int, no_mlflow: bool
) -> dict[str, Any]:
    torch.manual_seed(seed)
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    all_tags = G._all_structure_tags()
    hold = all_tags[0]  # Fold 0
    train_tags = [t for t in all_tags if t != hold]

    system, _ = _build_system(device, cfg)
    train_batches = [G._load_batch(t, device, decouple_r2_input=True) for t in train_tags]
    hold_batch = G._load_batch(hold, device, decouple_r2_input=True)

    leak = _input_leak_checks(train_batches + [hold_batch])
    bce0 = _step0_dehydron_bce(system, hold_batch, tau=float(cfg["tau_start"]))
    train_mode = verify_train_mode(system)

    groups = [
        {"params": [p for p in system.frontend.parameters() if p.requires_grad],
         "lr": 1e-4, "name": "backbone"},
        {"params": [p for p in system.spine.parameters() if p.requires_grad],
         "lr": float(cfg["lr_hyperbolic"]), "name": "hyperbolic"},
    ]
    optimizer = torch.optim.Adam(groups)
    radius = CurriculumRadiusController(float(cfg["tau_start"]), float(cfg["tau_end"]), steps)
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]), float(cfg["gumbel_tau_end"]), steps,
        schedule=str(cfg.get("gumbel_schedule", "exponential")),
        alpha=float(cfg["gumbel_exp_alpha"]) if cfg.get("gumbel_exp_alpha") is not None else None,
        half_epochs=int(cfg.get("gumbel_exp_half_epochs", 12)),
    )
    eps_sched = EpsilonGreedySchedule(
        float(cfg.get("eps_start", 0.20)), float(cfg.get("eps_end", 0.0)),
        half_epochs=int(cfg.get("eps_half_epochs", int(cfg.get("gumbel_exp_half_epochs", 12)))),
    )

    mlflow = None
    if not no_mlflow:
        import mlflow as _mlflow

        _mlflow.set_tracking_uri("http://127.0.0.1:5000")
        _mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow = _mlflow
        mlflow.start_run(run_name=f"verify_grad_flow_decoupled_r2_fold0_{hold.replace(':', '')}")
        mlflow.set_tags({
            "diagnostic": "true", "do_not_promote": "true",
            "arch": "decoupled_arch", "hold": hold, "mode": "verify-grad-flow",
        })
        mlflow.log_params({
            "steps": steps, "seed": seed, "dehydron_coeff": DEHYDRON_COEFF,
            "sdrp_coeff": M2.SDRP_COEFF, "lr_frontend": 1e-4,
            "lr_hyperbolic": float(cfg["lr_hyperbolic"]), "max_neighbors": M2.MAX_NEIGHBORS,
            **{f"summary_{k}": v for k, v in system.spine.model_summary().items()},
        })
        mlflow.log_metrics({
            "bce_step0": bce0["bce_step0"], "label_prior": bce0["label_prior"],
            "bce_constant_prior": bce0["bce_constant_prior"],
            "input_r2_edges": float(leak["input_r2_edges"]),
            "backbone_reg_modules_live": float(train_mode["live"]),
            "backbone_reg_modules_total": float(train_mode["total"]),
        }, step=0)

    per_step: list[dict[str, float]] = []
    try:
        for step in range(steps):
            m = G.run_step_sdrp_only(
                system, optimizer, train_batches, epoch=step, radius=radius, gumbel=gumbel,
                explore_epsilon=float(eps_sched.epsilon(step)), max_grad_norm=MAX_GRAD_NORM,
                sdrp_coeff=M2.SDRP_COEFF, dehydron_coeff=DEHYDRON_COEFF,
            )
            if m.get("nan_abort", 0.0) >= 1.0 or not math.isfinite(float(m["loss_total"])):
                raise RuntimeError(f"non-finite at step {step}: {m}")
            row = {"step": float(step), "loss_total": float(m["loss_total"]),
                   "preclip_norm": float(m["preclip_norm"])}
            for name, (bucket, _) in GRAD_GATES.items():
                row[name] = float(m["bucket_grad_l2"].get(bucket, 0.0))
            # grads were cleared by optimizer.step()->zero on next call; read value-path now
            row["grad_l2_attn_value_path"] = _value_path_grad_l2(system)
            per_step.append(row)
            if mlflow is not None and (step % 10 == 0 or step == steps - 1):
                mlflow.log_metrics({k: v for k, v in row.items() if k != "step"}, step=step)
            if step % 10 == 0 or step == steps - 1:
                print(f"[verify-grad-flow] step={step} loss={row['loss_total']:.4f} "
                      + " ".join(f"{k.replace('grad_l2_', '')}={row[k]:.2e}" for k in
                                 [*GRAD_GATES, "grad_l2_attn_value_path"]))

        gates: dict[str, Any] = {}
        for name, (_, thr) in GRAD_GATES.items():
            vals = [r[name] for r in per_step]
            gates[name] = {"min": min(vals), "final": vals[-1], "threshold": thr,
                           "pass": min(vals) > thr}
        vp = [r["grad_l2_attn_value_path"] for r in per_step]
        gates["grad_l2_attn_value_path"] = {"min": min(vp), "final": vp[-1],
                                            "threshold": 0.0, "pass": min(vp) > 0.0}
        gates["bce_step0"] = {**bce0, "threshold": BCE_STEP0_MIN,
                              "pass": bce0["bce_step0"] > BCE_STEP0_MIN}
        gates["no_r2_in_model_input"] = leak
        gates["train_mode_14_of_14"] = train_mode
        result = {"hold": hold, "steps": steps, "gates": gates,
                  "all_pass": all(g["pass"] for g in gates.values()),
                  "model_summary": system.spine.model_summary()}
        if mlflow is not None:
            for name, g in gates.items():
                mlflow.set_tag(f"gate_{name}", "PASS" if g["pass"] else "FAIL")
            mlflow.set_tag("all_gates", "PASS" if result["all_pass"] else "FAIL")
            out = REPO_ROOT / "data" / "gates" / "verify_grad_flow_decoupled_result.json"
            out.write_text(json.dumps(result, indent=2, default=float))
            mlflow.log_artifact(str(out))
        return result
    finally:
        if mlflow is not None:
            mlflow.end_run()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["verify-grad-flow", "verify-train-mode", "all"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device)
    rc = 0
    if a.mode in ("verify-train-mode",):
        cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
        system, _ = _build_system(device, cfg)
        r = verify_train_mode(system)
        print(json.dumps(r))
        rc = 0 if r["pass"] else 1
    if a.mode in ("verify-grad-flow", "all"):
        res = verify_grad_flow(device, a.steps, a.seed, a.no_mlflow)
        print(json.dumps(res, indent=2, default=float))
        rc = 0 if res["all_pass"] else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
