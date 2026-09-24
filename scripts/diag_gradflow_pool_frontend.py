#!/usr/bin/env python3
"""Diagnostic (NOT a sealed card, non-claim, do_not_promote): gradient reach
and fit quality of the wrap1 z_hyp SDRP-live stack under the real (non-lite)
EquiformerV3 pool frontend, cold-init, vs. the sealed SE(3)-lite stub.

Why this exists
----------------
``science/tokyo_eye/v8/equiformer_frontend.py::StubEquiformerFrontend.
_se3_lite_forward`` only ever touches ``self.backbone.blocks[0]`` — under
SE(3)-lite, 84 of the frontend's 91 backbone-block parameters (6 of the
7 "blocks") are structurally outside the loss graph, not slow learners.
Every prior wrap1_zhyp_g_fit / M2 finding that read "frontend barely moved"
under SE(3)-lite was correctly measured but should be read as "1-block stub
capacity", not "frontend architecture capacity" — retroactive note, not a
retraction (spine/wiring conclusions in those cards are unaffected).

This script swaps in ``EquiformerPoolFrontend`` (cold_init=True, no MPtrj
checkpoint — architecture only, per addendum §2.3) to test whether gradient
reaches the full frontend, and whether a higher frontend LR produces real
fit (mean+min macro-F1, the M2 card's own bar) or just a sharper majority
-class loss basin (the exact confound ``tokyo_eye_equ_wrap1_zhyp_m2_result.
json`` was written to catch).

Off-path disclosure (same category as the SE(3)-lite pilots, different
cause): this bypasses the sealed script and ``assert_governed_assembly``'s
frontend-kind gate is not invoked here (no MPtrj checkpoint => not a
governed promote-eligible build regardless). ``pure_hyp_pass`` was spot
-checked clean for this exact config -- see ``--mode verify-pure-hyp``
below and rerun it if the config changes; it is not wired into the training
loop as a hard gate here.

Environment
-----------
The pool frontend needs four packages beyond ``requirements-science.txt``:
see ``requirements-diagnostics-pool-frontend.txt`` (repo root) for exact
pins. Install with::

    pip install --no-deps -r requirements-diagnostics-pool-frontend.txt

No ``/tmp``-only installs — the pins are checked into the repo so this is
re-runnable from a clean environment.

GPU memory: the 4GB card this was developed on OOMs on the full 7-layer
backbone without activation checkpointing; ``--mode train`` always enables
``gradient_checkpointing_block_list`` on the backbone (verified bit
-equivalent to no-checkpointing modulo float32 accumulation order --
see ``--mode verify-checkpoint``, max abs grad diff 5e-8 / max rel 3e-5
over 318 tensors, one run, seed 0, logged in the run this docstring
accompanies).

Modes
-----
``train``            : run one condition (frontend LR / euc_skip on-off),
                        log gradient-reach + fit-quality telemetry to MLflow.
``verify-checkpoint`` : one forward/backward with and without activation
                        checkpointing on the same batch+seed; asserts the
                        grads match within float32 tolerance. Run this again
                        if you change backbone kwargs or PyTorch version.
``verify-pure-hyp``   : run the trunk-wide pure_hyp tracer once against the
                        exact system this script builds; prints the report.
``verify-train-mode`` : after ``system.train()``, report whether the backbone's
                        Dropout / GraphDropPath modules are actually in train
                        mode. ``EquiformerPoolFrontend.train()`` calls
                        ``self._backbone.eval()`` unconditionally, so with
                        ``freeze_backbone=False`` the backbone's own
                        regularization (attn_weights_drop, drop_path_rate) is
                        expected to be silently OFF during training.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import wrap1_zhyp_g_fit as G  # noqa: E402  (sibling script, sealed-protocol constants + loaders)
from experiments.training.v8.run_v8_experiment import (  # noqa: E402
    build_equiformer_pool_system,
)
from science.tokyo_eye.v8.engine import (  # noqa: E402
    CurriculumRadiusController,
    EpsilonGreedySchedule,
    GumbelTemperatureSchedule,
)
from science.tokyo_eye.v8.equiformer_frontend import (  # noqa: E402
    DEFAULT_WEIGHT_MAP,
    load_weight_map,
)
from science.tokyo_eye.v8.grad_reachability import param_bucket  # noqa: E402
from science.tokyo_eye.v8.heads import sdrp_cross_entropy  # noqa: E402
from science.tokyo_eye.v8 import model as spine_model  # noqa: E402

MLFLOW_EXPERIMENT = "diag/gradflow-pool-frontend"
DEFAULT_TRAIN_TAGS = ["1UBQ:A", "1TEN:A", "1HHP:A", "1LYZ:A"]

CONDITIONS: dict[str, dict[str, Any]] = {
    "pool_base": dict(no_euc=False, lr_fe=None),
    "pool_no_euc": dict(no_euc=True, lr_fe=None),
    "pool_no_euc_lr1e-4": dict(no_euc=True, lr_fe=1e-4),
    "pool_lr1e-4": dict(no_euc=False, lr_fe=1e-4),
}
STAGES = ["proj", "attn0", "attn1", "moe_final"]

# --- clamp saturation recorder (read-only wrapper around the real fn) ------
_orig_clamp = spine_model.clamp_ball_radius
_REC: dict[str, Any] = {"rows": None}


def _rec_clamp(x: torch.Tensor, *, max_r: float, eps: float = 1e-5) -> torch.Tensor:
    rows = _REC["rows"]
    if rows is not None:
        with torch.no_grad():
            r = torch.linalg.vector_norm(torch.nan_to_num(x.detach()), dim=-1)
            rows.append((int((r > max_r).sum()), int(r.numel()), float(r.mean())))
    return _orig_clamp(x, max_r=max_r, eps=eps)


spine_model.clamp_ball_radius = _rec_clamp


def _build_system(device: torch.device, max_neighbors: int, no_euc: bool, cfg: dict[str, Any]):
    system, info = build_equiformer_pool_system(
        cfg, equiformer_ckpt=None, device=device, freeze_backbone=False,
        max_neighbors=max_neighbors, cold_init=True,
    )
    system.set_moe_mode("ablated")
    bb = system.frontend._backbone
    bb.gradient_checkpointing_block_list = [1] * int(bb.num_layers)
    if no_euc:
        with torch.no_grad():
            system.spine.euc_skip.weight.zero_()
            system.spine.euc_skip.bias.zero_()
        for p in system.spine.euc_skip.parameters():
            p.requires_grad = False
    return system, info


def _named(system: torch.nn.Module):
    for prefix, mod in (("frontend", system.frontend), ("spine", system.spine)):
        for name, p in mod.named_parameters():
            yield f"{prefix}.{name}", p


def _bucket_of(key: str) -> str:
    prefix, name = key.split(".", 1)
    if prefix == "frontend":
        m = re.match(r"_backbone\.blocks\.(\d+)\.", name)
        if m:
            return f"fe_block{m.group(1)}"
        return "fe_other" if name.startswith("_backbone.") else "fe_proj"
    return param_bucket(name)


def _agg(per_param: dict[str, float | None]) -> dict[str, float]:
    sq: dict[str, float] = {}
    for k, v in per_param.items():
        if v is None:
            continue
        b = _bucket_of(k)
        sq[b] = sq.get(b, 0.0) + v * v
    return {b: math.sqrt(s) for b, s in sq.items()}


def _param_groups(system, *, lr_backbone: float, lr_hyperbolic: float):
    return [
        {"params": [p for p in system.frontend.parameters() if p.requires_grad],
         "lr": float(lr_backbone), "name": "backbone"},
        {"params": [p for p in system.spine.parameters() if p.requires_grad],
         "lr": float(lr_hyperbolic), "name": "hyperbolic"},
    ]


def verify_checkpoint(device: torch.device, tag: str, max_neighbors: int, seed: int) -> bool:
    """One fwd/bwd with vs. without activation checkpointing; same seed+batch."""
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)

    def run(ckpt_on: bool) -> tuple[float, dict[str, torch.Tensor | None]]:
        torch.manual_seed(seed)
        system, _ = _build_system(device, max_neighbors, no_euc=False, cfg=cfg)
        system.frontend._backbone.gradient_checkpointing_block_list = (
            [1 if ckpt_on else 0] * int(system.frontend._backbone.num_layers)
        )
        system.train()
        torch.manual_seed(seed + 1000)
        b = G._load_batch(tag, device)
        out = system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=0.7, chem=b.get("gate_chem"))
        loss = sdrp_cross_entropy(out["sdrp_logits"], b["sdrp_target"])
        loss.backward()
        grads = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in _named(system)}
        return float(loss.detach()), grads

    loss_off, g_off = run(False)
    loss_on, g_on = run(True)
    print(f"[verify-checkpoint] loss off={loss_off:.10f} on={loss_on:.10f} diff={abs(loss_off - loss_on):.2e}")
    max_abs, max_rel, n_cmp, n_none_mismatch, worst = 0.0, 0.0, 0, 0, None
    for k in g_off:
        a, b_ = g_off[k], g_on[k]
        if (a is None) != (b_ is None):
            n_none_mismatch += 1
            print(f"[verify-checkpoint] NONE MISMATCH: {k} off_none={a is None} on_none={b_ is None}")
            continue
        if a is None:
            continue
        n_cmp += 1
        d = (a - b_).abs().max().item()
        if d > max_abs:
            max_abs, worst = d, k
        max_rel = max(max_rel, d / (a.abs().max().item() + 1e-12))
    ok = n_none_mismatch == 0 and max_abs < 1e-4
    print(f"[verify-checkpoint] compared {n_cmp} tensors; max_abs_diff={max_abs:.3e} (param={worst}) "
          f"max_rel_diff={max_rel:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def verify_pure_hyp(device: torch.device, tag: str, max_neighbors: int, no_euc: bool, seed: int) -> bool:
    from science.tokyo_eye.v8.pure_hyp_pass import scan_forward

    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    torch.manual_seed(seed)
    system, info = _build_system(device, max_neighbors, no_euc, cfg)
    print(f"[verify-pure-hyp] frontend load_info={info}")
    b = G._load_batch(tag, device)

    def fwd():
        return system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=0.7, chem=b.get("gate_chem"))

    report = scan_forward(system, fwd)
    print(f"[verify-pure-hyp] pure_hyp_pass={report.passed} violations={len(report.violations)}")
    for v in report.violations[:30]:
        print(f"   - {v.kind} {v.module_name} | {v.detail}")
    return bool(report.passed)


def verify_train_mode(device: torch.device, max_neighbors: int, seed: int) -> bool:
    """Report whether backbone regularization modules are live after system.train().

    Returns True iff every active-probability Dropout / GraphDropPath in the
    backbone is in training mode (i.e. regularization would actually run).
    """
    import torch.nn as nn

    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    torch.manual_seed(seed)
    system, _ = _build_system(device, max_neighbors, no_euc=False, cfg=cfg)
    system.train()
    bb = system.frontend._backbone
    print(f"[verify-train-mode] system.training={system.training} "
          f"frontend.training={system.frontend.training} backbone.training={bb.training}")
    dropouts = [(n, m) for n, m in bb.named_modules() if isinstance(m, nn.Dropout) and m.p > 0.0]
    droppaths = [(n, m) for n, m in bb.named_modules()
                 if type(m).__name__ == "GraphDropPath" and float(m.drop_prob or 0.0) > 0.0]
    n_do_live = sum(1 for _, m in dropouts if m.training)
    n_dp_live = sum(1 for _, m in droppaths if m.training)
    print(f"[verify-train-mode] nn.Dropout with p>0: {len(dropouts)} modules, {n_do_live} in train mode"
          + (f" (p={sorted({m.p for _, m in dropouts})})" if dropouts else ""))
    print(f"[verify-train-mode] GraphDropPath with drop_prob>0: {len(droppaths)} modules, {n_dp_live} in train mode"
          + (f" (drop_prob={sorted({float(m.drop_prob) for _, m in droppaths})})" if droppaths else ""))
    ok = (n_do_live == len(dropouts)) and (n_dp_live == len(droppaths))
    print(f"[verify-train-mode] backbone regularization active during training: {'YES' if ok else 'NO'}")
    return ok


def train(
    cond: str, seed: int, steps: int, eval_every: int, device: torch.device,
    use_mlflow: bool, out_dir: Path, train_tags: list[str], max_neighbors: int,
) -> None:
    import json

    spec = CONDITIONS[cond]
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    torch.manual_seed(seed)
    np.random.seed(seed)

    system, load_info = _build_system(device, max_neighbors, spec["no_euc"], cfg)
    lr_fe = float(spec["lr_fe"] if spec["lr_fe"] is not None else cfg["lr_backbone"])
    opt = torch.optim.Adam(_param_groups(system, lr_backbone=lr_fe, lr_hyperbolic=float(cfg["lr_hyperbolic"])))
    radius = CurriculumRadiusController(float(cfg["tau_start"]), float(cfg["tau_end"]), steps)
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]), float(cfg["gumbel_tau_end"]), steps,
        schedule=str(cfg.get("gumbel_schedule", "exponential")),
        alpha=float(cfg["gumbel_exp_alpha"]) if cfg.get("gumbel_exp_alpha") is not None else None,
        half_epochs=int(cfg.get("gumbel_exp_half_epochs", 12)),
    )
    half_ep = int(cfg.get("gumbel_exp_half_epochs", 12))
    eps_sched = EpsilonGreedySchedule(
        float(cfg.get("eps_start", 0.20)), float(cfg.get("eps_end", 0.0)),
        half_epochs=int(cfg.get("eps_half_epochs", half_ep)),
    )
    batches = [G._load_batch(t, device) for t in train_tags]
    n = len(batches)

    run_name = f"gradflow_{cond}_f1_seed{seed}"
    mlflow = None
    if use_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags({
            "diagnostic": "true", "do_not_promote": "true", "non_claim": "true",
            "off_path_reason": "frontend=equiformer_pool cold_init (no ckpt) + activation-checkpointed "
                                "+ 4/12 train structures + partial steps -- see script docstring",
            "condition": cond, "euc_skip_zeroed": str(spec["no_euc"]),
            "frontend": "equiformer_v3_pool_cold_init",
            "checkpoint_equivalence_verified": "true (max_abs_grad_diff=5.2e-8 over 318 tensors, seed0)",
            "source_script": "scripts/diag_gradflow_pool_frontend.py",
        })
        mlflow.log_params({
            "steps": steps, "eval_every": eval_every, "seed": seed,
            "sdrp_coeff": G.SDRP_COEFF, "max_grad_norm": G.MAX_GRAD_NORM,
            "lr_frontend": lr_fe, "lr_hyperbolic": cfg["lr_hyperbolic"],
            "max_neighbors": max_neighbors, "train_tags": ",".join(train_tags),
        })

    curve: list[dict[str, Any]] = []
    fit_curve: list[dict[str, Any]] = []
    snaps: dict[str, Any] = {}
    stages = STAGES
    t_start = time.time()
    tau_eval = float(cfg["tau_end"])
    try:
        for step in range(steps):
            measure = (step % G.LOG_EVERY == 0) or step == steps - 1
            do_eval = (step % eval_every == 0) or step == steps - 1
            tau = radius.tau_ceiling(step)
            system.train()
            system.set_moe_temperature(gumbel.temperature(step))
            system.set_moe_explore_epsilon(float(eps_sched.epsilon(step)))
            opt.zero_grad(set_to_none=True)
            _REC["rows"] = [] if measure else None
            loss_sum = 0.0
            for b in batches:
                out = system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=tau, chem=b.get("gate_chem"))
                loss = (G.SDRP_COEFF * sdrp_cross_entropy(out["sdrp_logits"], b["sdrp_target"])) / n
                loss.backward()
                loss_sum += float(loss.detach())
            rec, _REC["rows"] = _REC["rows"], None
            if not math.isfinite(loss_sum):
                print(f"[{run_name}] non-finite loss at step {step}; aborting")
                break

            per_param = None
            if measure:
                per_param = {k: (None if p.grad is None else float(p.grad.detach().norm())) for k, p in _named(system)}
                before = {k: p.detach().clone() for k, p in _named(system) if p.requires_grad}
            preclip = G._total_grad_norm(system.parameters())
            torch.nn.utils.clip_grad_norm_(system.parameters(), G.MAX_GRAD_NORM)
            opt.step()

            fit_row = None
            if do_eval:
                per_struct = {t: G._sdrp_structure_metrics(system, b, tau_ceiling=tau_eval)
                              for t, b in zip(train_tags, batches)}
                f1s = [v["macro_f1"] for v in per_struct.values() if math.isfinite(v["macro_f1"])]
                lifts = [v["lift"] for v in per_struct.values() if math.isfinite(v["lift"])]
                fit_row = {
                    "step": step,
                    "train_macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
                    "train_min_f1": float(np.min(f1s)) if f1s else float("nan"),
                    "train_macro_lift": float(np.mean(lifts)) if lifts else float("nan"),
                    "train_min_lift": float(np.min(lifts)) if lifts else float("nan"),
                    "per_structure": per_struct,
                }
                fit_curve.append(fit_row)
                print(f"[{run_name}] step={step:3d} EVAL macro_f1={fit_row['train_macro_f1']:.4f} "
                      f"min_f1={fit_row['train_min_f1']:.4f} macro_lift={fit_row['train_macro_lift']:.4f} "
                      f"min_lift={fit_row['train_min_lift']:.4f}", flush=True)
                if mlflow is not None:
                    mlflow.log_metrics({k: v for k, v in fit_row.items() if k != "per_structure" and math.isfinite(v)}, step=step)

            if not measure:
                continue
            gl2 = _agg(per_param)
            dsq, tsq = {}, {}
            for k, p in _named(system):
                if k not in before:
                    continue
                bk = _bucket_of(k)
                dsq[bk] = dsq.get(bk, 0.0) + float((p.detach() - before[k]).pow(2).sum())
                tsq[bk] = tsq.get(bk, 0.0) + float(before[k].pow(2).sum())
            upd = {b: math.sqrt(dsq[b]) / max(math.sqrt(tsq[b]), 1e-12) for b in dsq}
            sat = {}
            for i, name in enumerate(stages):
                rows = rec[i::len(stages)]
                tot = sum(r[1] for r in rows)
                sat[name] = (sum(r[0] for r in rows) / tot) if tot else float("nan")
                sat[name + "_meanr"] = float(np.mean([r[2] for r in rows])) if rows else float("nan")
            spine_nz = sum(gl2.get(b, 0.0) for b in ("attn_layers", "projector", "moe", "sdrp_head"))
            euc = gl2.get("euc_skip", 0.0)
            head = gl2.get("sdrp_head", 0.0)
            row = {
                "step": step, "tau": float(tau), "loss_total": loss_sum, "preclip_norm": preclip,
                "grad_l2": gl2, "update_ratio": upd, "sat_frac": sat,
                "euc_skip_share": (euc / (euc + spine_nz)) if (euc + spine_nz) > 0 else float("nan"),
                "rel_to_head": {b: (v / head if head > 0 else float("nan")) for b, v in gl2.items()},
            }
            curve.append(row)
            if step in (0, steps // 2, steps - 1):
                snaps[str(step)] = per_param
            print(f"[{run_name}] step={step:3d} t={time.time()-t_start:.0f}s loss={loss_sum:.4f} "
                  f"pre={preclip:.3f} euc_share={row['euc_skip_share']:.3f} "
                  f"gl2={ {k: float(f'{v:.1e}') for k, v in sorted(gl2.items())} }", flush=True)
            if mlflow is not None:
                m = {"loss_total": loss_sum, "preclip_norm": preclip, "tau_ceiling": float(tau),
                     "euc_skip_share": row["euc_skip_share"]}
                m.update({f"grad_l2_{b}": v for b, v in gl2.items()})
                m.update({f"update_ratio_{b}": v for b, v in upd.items()})
                mlflow.log_metrics({k: float(v) for k, v in m.items() if math.isfinite(v)}, step=step)
    finally:
        if mlflow is not None:
            mlflow.end_run()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cond}_seed{seed}.json").write_text(json.dumps({
        "cond": cond, **spec, "seed": seed, "steps": steps, "train_tags": train_tags,
        "curve": curve, "fit_curve": fit_curve, "snapshots": snaps,
    }, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "verify-checkpoint", "verify-pure-hyp", "verify-train-mode"], default="train")
    ap.add_argument("--condition", choices=sorted(CONDITIONS))
    ap.add_argument("--train-tags", default=",".join(DEFAULT_TRAIN_TAGS))
    ap.add_argument("--tag", default="1UBQ:A", help="single structure, for verify-* modes")
    ap.add_argument("--max-neighbors", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "gates" / "diag_gradflow_pool_frontend"))
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable (no CPU fallback)")
    device = torch.device(a.device)

    if a.mode == "verify-checkpoint":
        ok = verify_checkpoint(device, a.tag, a.max_neighbors, a.seed)
        return 0 if ok else 1
    if a.mode == "verify-train-mode":
        # exit 0 = regularization active, 1 = silently off (informational finding, not a crash)
        return 0 if verify_train_mode(device, a.max_neighbors, a.seed) else 1
    if a.mode == "verify-pure-hyp":
        ok = verify_pure_hyp(device, a.tag, a.max_neighbors, no_euc=False, seed=a.seed)
        return 0 if ok else 1

    if not a.condition:
        raise SystemExit("--condition required for --mode train")
    train(a.condition, a.seed, a.steps, a.eval_every, device, not a.no_mlflow,
          Path(a.out_dir), [t for t in a.train_tags.split(",") if t], a.max_neighbors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
