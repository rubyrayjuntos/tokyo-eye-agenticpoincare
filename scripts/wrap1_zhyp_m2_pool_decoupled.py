#!/usr/bin/env python3
"""wrap=1 z_hyp M2 macro-F1 -- decoupled-R2 architecture card (DRAFT, unsealed).

Derived from ``scripts/wrap1_zhyp_m2_pool_lr.py`` (arm ``lr_1e-4_bbtrain``). Same
frontend (EquiformerPoolFrontend cold_init, backbone train mode live), lr_frontend
1e-4, 400 steps, 12-fold LOSO, seed 0, MoE ablated, M2 bars. Differs by a BUNDLE of
changes (see the prereg ``changes_vs_baseline``), so a result is not attributable to
any single one of them:

  * model input carries no R2 edges / wrap values / R2 gate features
    (``decouple_r2_input``); labels still use wrap <= DEHYDRON_WRAP_MAX (=1);
  * salt bridges outrank coincident H-bonds in the model-facing typing only;
  * MechanismScoreHead reads [z_hyp, h_euc]; dehydron BCE coeff 1.0 in the loss;
  * relation typing on the attention value path (beta_R removed);
  * evidential head structurally excluded; curvature frozen.

Launch requires the prereg ``status == READY_TO_RUN`` (operator-set), like the
parent card. ``--smoke`` bypasses that for a 1-fold, 3-step wiring check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import git_provenance  # noqa: E402
import wrap1_zhyp_g_fit as G  # noqa: E402  (sibling script -- reuse, not fork)
from experiments.training.v8.run_v8_experiment import (  # noqa: E402
    build_equiformer_pool_system,
)
from science.tokyo_eye.v8.assembly_gate import (  # noqa: E402
    AssemblyGateError,
    assert_governed_assembly,
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
from science.tokyo_eye.v8.freeze_reconciliation import (  # noqa: E402
    CANONICAL_MLFLOW_EXPERIMENT,
)
from science.tokyo_eye.v8.grad_reachability import (  # noqa: E402
    SPINE_NZ_BUCKETS,
    bucket_grad_stats,
)
from science.tokyo_eye.v8.heads import sdrp_cross_entropy  # noqa: E402

FOLDS_FROZEN = REPO_ROOT / "data" / "gates" / "wrap1_dehydron_loso" / "folds_frozen.json"
PREREG_DECOUPLED = REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_zhyp_m2_pool_decoupled_prereg.json"
RESULT_DIR = REPO_ROOT / "data" / "gates" / "wrap1_zhyp_m2_pool_decoupled"

# Card lock -- amendment_rule in the prereg: any change here needs a new signed card.
STEPS = 400
SEED = 0
SDRP_COEFF = 0.1
DEHYDRON_COEFF = 1.0  # BCE on MechanismScoreHead([z_hyp, h_euc]) is in the loss graph
SPINE_KWARGS = {"decoupled_arch": True}
DECOUPLE_R2_INPUT = True
MAX_GRAD_NORM = 1.0
MAX_NEIGHBORS = 16  # pool-frontend memory bound on a 4GB card
LOG_EVERY = 10
DENSE_LOG_STEPS = 150  # log EVERY step through here (sparse spikes fell between LOG_EVERY points)
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
S2_OF_N = 12
G_FIT_TRAIN_MACRO_MIN = 0.40
G_FIT_TRAIN_MIN_STRUCT = 0.40
S1_MEAN_MIN = 0.40
S1_CI_LB_MIN: float | None = None  # M2 family: no CI gate, matches sealed M2 card
S2_MIN_ABOVE = 10
S2_THRESHOLD = 0.40

# One row per arm. Arms of the same card differ in exactly one field.
ARMS: dict[str, dict[str, Any]] = {
    "decoupled": {"lr_frontend": 1e-4, "backbone_train_mode": True, "prereg": PREREG_DECOUPLED},
}
LR_ARMS: dict[str, float] = {k: float(v["lr_frontend"]) for k, v in ARMS.items()}


def _lb(tag: str, device: torch.device) -> dict[str, Any]:
    """Batch loader for this card: model input carries no R2 information."""
    return G._load_batch(tag, device, decouple_r2_input=DECOUPLE_R2_INPUT)


def apply_backbone_train_mode(system: torch.nn.Module) -> None:
    """Make ``system.train()`` reach the pool backbone's own regularization.

    ``EquiformerPoolFrontend.train()`` ends with ``self._backbone.eval()``
    unconditionally (science/tokyo_eye/v8/equiformer_pool_frontend.py), so even
    with ``freeze_backbone=False`` the backbone's Dropout / GraphDropPath modules
    are silently OFF during training. This binds an instance-level ``train`` that
    skips that override and leaves the shared module untouched. ``system.eval()``
    still puts everything in eval mode, so evaluation is unaffected.
    """

    def _train_full(self, mode: bool = True):
        return torch.nn.Module.train(self, mode)

    system.frontend.train = types.MethodType(_train_full, system.frontend)


def backbone_reg_modules(system: torch.nn.Module) -> list[torch.nn.Module]:
    """Backbone modules that are stochastic when in train mode (p > 0)."""
    mods: list[torch.nn.Module] = []
    for m in system.frontend._backbone.modules():
        if isinstance(m, torch.nn.Dropout) and m.p > 0.0:
            mods.append(m)
        elif type(m).__name__ == "GraphDropPath" and float(m.drop_prob or 0.0) > 0.0:
            mods.append(m)
    return mods


def backbone_reg_modules_live(system: torch.nn.Module) -> tuple[int, int]:
    """(# in train mode, # total) over ``backbone_reg_modules``."""
    mods = backbone_reg_modules(system)
    return sum(1 for m in mods if m.training), len(mods)


def assert_backbone_regularization_live(system: torch.nn.Module) -> tuple[int, int]:
    """Raise unless every backbone regularization module is in train mode."""
    live, total = backbone_reg_modules_live(system)
    if not (total > 0 and live == total):
        raise RuntimeError(
            f"backbone regularization not active: {live}/{total} Dropout/GraphDropPath modules in "
            "train mode; arm requires all. Aborting rather than running a mislabeled card."
        )
    return live, total


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _card_paths(arm: str) -> G.CardPaths:
    """Reuse the sealed script's scoring dataclass/logic -- new paths per arm."""
    return G.CardPaths(
        card=f"m2_pool_{arm}",
        metric_family="M2_macro_f1",
        prereg=ARMS[arm]["prereg"],
        result=REPO_ROOT / "data" / "gates" / f"tokyo_eye_equ_wrap1_zhyp_m2_pool_{arm}_result.json",
        out_dir=RESULT_DIR / arm,
        gate_id=f"tokyo_eye_equ_wrap1_zhyp_m2_pool_{arm}_result",
        g_fit_train_macro_min=G_FIT_TRAIN_MACRO_MIN,
        g_fit_train_min_struct=G_FIT_TRAIN_MIN_STRUCT,
        s1_mean_min=S1_MEAN_MIN,
        s1_ci_lb_min=S1_CI_LB_MIN,
        s2_min_above=S2_MIN_ABOVE,
        s2_threshold=S2_THRESHOLD,
    )


def run_one_fold(
    hold: str, arm: str, *, device: torch.device, steps: int, seed: int,
    no_mlflow: bool, card: G.CardPaths,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    lr_frontend = LR_ARMS[arm]

    all_tags = G._all_structure_tags()
    train_tags = [t for t in all_tags if t != hold]
    assert len(train_tags) == 11, train_tags

    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    system, load_info = build_equiformer_pool_system(
        cfg, equiformer_ckpt=None, device=device, freeze_backbone=False,
        max_neighbors=MAX_NEIGHBORS, cold_init=True,
        spine_kwargs=SPINE_KWARGS,
    )
    system.set_moe_mode("ablated")
    system.frontend._backbone.gradient_checkpointing_block_list = (
        [1] * int(system.frontend._backbone.num_layers)
    )
    if ARMS[arm]["backbone_train_mode"]:
        apply_backbone_train_mode(system)
    load_info["frontend_pilot"] = "equiformer_v3_pool_cold_init"
    load_info["arm"] = f"zhyp_m2_pool_{arm}"

    # Real governed-frontend gate -- NOT off-path (frontend is pool, not lite/stub).
    try:
        probe = train_tags[0]

        def _fwd():
            b = _lb(probe, device)
            return system(
                b["x"], b["edge_index"], b["edge_type"],
                tau_ceiling=0.70, chem=b.get("gate_chem"),
            )

        gate_result = assert_governed_assembly(
            frontend_kind="equiformer_pool",
            load_info=load_info,
            system=system,
            forward_fn=_fwd,
            allow_off_path_frontend=False,
            require_pure_hyp=True,
            check_deps=True,
            claim_bearing_biology=False,
        )
        load_info["assembly_gate"] = gate_result.detail
        load_info["assembly_gate_passed"] = True
    except AssemblyGateError as exc:
        raise SystemExit(f"[m2_pool_lr] assembly_gate FAIL hold={hold} arm={arm}: {exc}") from exc

    groups = [
        {"params": [p for p in system.frontend.parameters() if p.requires_grad],
         "lr": float(lr_frontend), "name": "backbone"},
        {"params": [p for p in system.spine.parameters() if p.requires_grad],
         "lr": float(cfg["lr_hyperbolic"]), "name": "hyperbolic"},
    ]
    lr_hyperbolic = float(cfg["lr_hyperbolic"])
    optimizer = torch.optim.Adam(groups)
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

    train_batches = [_lb(t, device) for t in train_tags]
    hold_batch = _lb(hold, device)
    c_init = G._curvature_c(system)

    step0 = G._step0_spine_ok(system, train_batches, tau=float(cfg["tau_start"]))
    if not step0["ok"]:
        return {
            "arm": arm, "seed": seed, "heldout": [hold], "train": train_tags, "steps": steps,
            "finite": True, "G_grad_spine": False, "step0_grad": step0, "abort": "ABORT_WIRING",
            "c_init": c_init, "c_final": c_init, "lr_frontend": lr_frontend, "lr_hyperbolic": lr_hyperbolic,
        }

    finite = True
    clip_active_count = 0
    clip_logged = 0
    steps_run = 0
    clip_all_steps = 0
    curve: list[dict[str, Any]] = []
    abort_wiring_c = False

    mlflow = None
    if not no_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(CANONICAL_MLFLOW_EXPERIMENT)
        except Exception as exc:  # noqa: BLE001
            print(f"[m2_pool_lr] MLflow unavailable ({exc}); continuing without")
            mlflow = None

    run_ctx = (
        mlflow.start_run(run_name=f"wrap1_zhyp_m2_pool_{arm}_hold_{hold.replace(':', '')}")
        if mlflow is not None else None
    )
    try:
        if mlflow is not None:
            mlflow.set_tags({
                "diagnostic": "false", "do_not_promote": "true",
                "card": f"m2_pool_{arm}", "metric_family": "M2_macro_f1",
                "moe_mode": "ablated", "sdrp_coeff": str(SDRP_COEFF),
                "dehydron_coeff": str(DEHYDRON_COEFF), "margin_coeff": "0.0",
                "frontend": "equiformer_v3_pool_cold_init", "lr_arm": arm,
                "hold": hold, "gate_id": card.gate_id.replace("_result", "_prereg"),
                "G_grad_spine_step0": "PASS" if step0["ok"] else "FAIL",
                "frontend_governed": "true (allow_off_path_frontend=False)",
                "backbone_train_mode": str(ARMS[arm]["backbone_train_mode"]),
            })
            mlflow.log_params({
                "lr_frontend": lr_frontend, "lr_hyperbolic": lr_hyperbolic,
                "sdrp_coeff": SDRP_COEFF, "max_grad_norm": MAX_GRAD_NORM,
                "steps": steps, "seed": seed, "max_neighbors": MAX_NEIGHBORS,
                "backbone_train_mode": str(ARMS[arm]["backbone_train_mode"]),
                "card": card.card, "metric_family": card.metric_family,
                **git_provenance.provenance_params(REPO_ROOT),
            })
            git_provenance.log_dirty_diff(mlflow, REPO_ROOT)
            step0_metrics = {"G_grad_spine_ok": 1.0 if step0["ok"] else 0.0, "curvature_c": float(c_init)}
            for b, gval in (step0.get("grad_l2") or {}).items():
                step0_metrics[f"grad_l2_{b}"] = float(gval)
            if step0.get("euc_skip_share") is not None:
                step0_metrics["euc_skip_share"] = float(step0["euc_skip_share"])
            mlflow.log_metrics(step0_metrics, step=0)
            print(f"[m2_pool_lr] hold={hold} arm={arm} step0 G_grad_spine="
                  f"{'PASS' if step0['ok'] else 'FAIL'} grad_l2={step0.get('grad_l2')}")

        for step in range(steps):
            eps_t = float(eps_sched.epsilon(step))
            try:
                metrics = G.run_step_sdrp_only(
                    system, optimizer, train_batches, epoch=step, radius=radius,
                    gumbel=gumbel, explore_epsilon=eps_t,
                    max_grad_norm=MAX_GRAD_NORM, sdrp_coeff=SDRP_COEFF,
                    dehydron_coeff=DEHYDRON_COEFF,
                )
            except Exception as exc:  # noqa: BLE001
                finite = False
                print(f"[m2_pool_lr] NONFINITE/ERROR hold={hold} arm={arm} step={step}: {exc}")
                break
            if metrics.get("nan_abort", 0.0) >= 1.0 or not math.isfinite(float(metrics.get("loss_total", float("nan")))):
                finite = False
                break
            steps_run += 1
            if metrics.get("clip_active", 0.0) >= 1.0:
                clip_all_steps += 1
            sparse_point = step % LOG_EVERY == 0 or step == steps - 1
            if step < DENSE_LOG_STEPS or sparse_point:
                if sparse_point:  # keeps clip_active_fraction comparable to the baseline card
                    clip_logged += 1
                    if metrics.get("clip_active", 0.0) >= 1.0:
                        clip_active_count += 1
                last_bucket = dict(metrics.get("bucket_grad_l2") or {})
                euc_share = float(metrics.get("euc_skip_share", float("nan")))
                c_now = G._curvature_c(system)
                if ARMS[arm]["backbone_train_mode"]:
                    reg_live, reg_total = assert_backbone_regularization_live(system)
                else:
                    reg_live, reg_total = backbone_reg_modules_live(system)
                curve.append({
                    "step": step, "loss_total": metrics["loss_total"],
                    "preclip_norm": metrics["preclip_norm"], "postclip_norm": metrics["postclip_norm"],
                    "clip_active": metrics["clip_active"], "c": c_now,
                    "loss_sdrp_ce": metrics.get("loss_sdrp_ce"),
                    "loss_dehydron_bce": metrics.get("loss_dehydron_bce"),
                    "tau_ceiling": metrics.get("tau_ceiling"),
                    "gumbel_temperature": metrics.get("gumbel_temperature"),
                    "explore_epsilon": metrics.get("explore_epsilon"),
                    "bce_per_structure": metrics.get("bce_per_structure"),
                    "bucket_grad_l2": last_bucket, "euc_skip_share": euc_share,
                })
                print(f"[m2_pool_lr] hold={hold} arm={arm} step={step} "
                      f"loss={metrics['loss_total']:.4f} preclip={metrics['preclip_norm']:.3f} "
                      f"clip={int(metrics['clip_active'])} euc_share={euc_share:.3f}", flush=True)
                if mlflow is not None:
                    payload = {
                        "loss_total": float(metrics["loss_total"]), "preclip_norm": float(metrics["preclip_norm"]),
                        "postclip_norm": float(metrics["postclip_norm"]), "clip_active": float(metrics["clip_active"]),
                        "curvature_c": float(c_now), "G_grad_spine_ok": 1.0, "euc_skip_share": euc_share,
                    }
                    payload["backbone_reg_modules_live"] = float(reg_live)
                    for k in ("loss_sdrp_ce", "loss_dehydron_bce", "tau_ceiling",
                              "gumbel_temperature", "explore_epsilon"):
                        if metrics.get(k) is not None:
                            payload[k] = float(metrics[k])
                    bps = metrics.get("bce_per_structure") or []
                    if bps:
                        payload["bce_per_structure_max"] = float(max(bps))
                        payload["bce_per_structure_min"] = float(min(bps))
                    for b, gval in last_bucket.items():
                        payload[f"grad_l2_{b}"] = float(gval)
                    mlflow.log_metrics(payload, step=step)

        c_final = G._curvature_c(system)
        if abs(c_final - c_init) > 1e-5:
            abort_wiring_c = True

        tau_eval = float(cfg["tau_end"])
        train_per: dict[str, Any] = {
            t: G._sdrp_structure_metrics(system, b, tau_ceiling=tau_eval)
            for t, b in zip(train_tags, train_batches)
        }
        held = G._sdrp_structure_metrics(system, hold_batch, tau_ceiling=tau_eval)
        shuffle = G._edge_type_shuffle_sensitivity(system, hold_batch, tau_ceiling=tau_eval, seed=seed)

        train_f1s = [float(v["macro_f1"]) for v in train_per.values() if math.isfinite(v["macro_f1"])]
        macro_f1_mean = float(np.mean(train_f1s)) if train_f1s else float("nan")
        min_f1 = float(np.min(train_f1s)) if train_f1s else float("nan")
        train_lifts = [float(v["lift"]) for v in train_per.values() if math.isfinite(v["lift"])]
        macro_lift = float(np.mean(train_lifts)) if train_lifts else float("nan")
        min_lift = float(np.min(train_lifts)) if train_lifts else float("nan")

        g_fit_train = bool(
            finite and step0["ok"] and not abort_wiring_c
            and math.isfinite(macro_f1_mean) and math.isfinite(min_f1)
            and macro_f1_mean >= card.g_fit_train_macro_min
            and min_f1 >= card.g_fit_train_min_struct
        )
        clip_frac = float(clip_active_count) / float(clip_logged) if clip_logged else float("nan")
        clip_frac_all = float(clip_all_steps) / float(steps_run) if steps_run else float("nan")

        if mlflow is not None:
            final = {
                "train_macro_f1": float(macro_f1_mean), "train_min_f1": float(min_f1),
                "train_macro_lift": float(macro_lift), "train_min_lift": float(min_lift),
                "heldout_macro_f1": float(held["macro_f1"]), "heldout_lift": float(held["lift"]),
                "heldout_sdrp_top1": float(held["sdrp_top1_acc"]), "heldout_majority": float(held["majority_rate"]),
                "g_fit_train_pass_fold": 1.0 if g_fit_train else 0.0,
                "G_grad_spine_ok": 1.0 if step0["ok"] else 0.0, "G_finite": 1.0 if finite else 0.0,
                "c_drift": float(c_final - c_init),
                "clip_active_fraction": float(clip_frac) if math.isfinite(clip_frac) else float("nan"),
                "clip_active_fraction_all_steps": float(clip_frac_all) if math.isfinite(clip_frac_all) else float("nan"),
                "edge_type_shuffle_abs_delta": float(shuffle.get("sdrp_logits_abs_delta_mean", float("nan"))),
            }
            mlflow.log_metrics(final, step=steps)
            mlflow.set_tags({
                "g_fit_train_pass_fold": "PASS" if g_fit_train else "FAIL",
                "G_finite": "PASS" if finite else "FAIL",
            })
            print(f"[m2_pool_lr] hold={hold} arm={arm} FINAL train_f1={macro_f1_mean:.4f}/{min_f1:.4f} "
                  f"held_f1={held['macro_f1']:.4f} g_fit_train={'PASS' if g_fit_train else 'FAIL'}")
    finally:
        if run_ctx is not None:
            mlflow.end_run()

    return {
        "arm": arm, "card": card.card, "metric_family": card.metric_family, "seed": seed,
        "heldout": [hold], "train": train_tags, "steps": steps, "finite": finite,
        "G_grad_spine": bool(step0["ok"]), "step0_grad": step0,
        "abort": "ABORT_WIRING" if abort_wiring_c or not step0["ok"] else None,
        "moe_mode": "ablated", "sdrp_coeff": SDRP_COEFF, "dehydron_coeff": DEHYDRON_COEFF, "margin_coeff": 0.0,
        "tau_eval": tau_eval, "frontend": "equiformer_v3_pool_cold_init",
        "c_init": c_init, "c_final": c_final, "c_drift": float(c_final - c_init),
        "lr_frontend": lr_frontend, "lr_hyperbolic": lr_hyperbolic,
        "backbone_train_mode": ARMS[arm]["backbone_train_mode"],
        "clip_active_fraction": clip_frac, "clip_active_fraction_all_steps": clip_frac_all,
        "train_per_structure": train_per,
        "train_macro_lift": macro_lift, "train_min_lift": min_lift,
        "train_macro_f1": macro_f1_mean, "train_min_f1": min_f1,
        "train_score_macro": macro_f1_mean, "train_score_min": min_f1,
        "g_fit_train_pass_fold": g_fit_train, "heldout_metrics": {hold: held},
        "edge_type_shuffle": shuffle, "curve_every_10": curve,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr-frontend-arm", choices=sorted(LR_ARMS), default="decoupled")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--folds", default="", help="comma-separated hold tags to run; default = all 12")
    p.add_argument("--smoke", action="store_true", help="1 fold, 3 steps, no stamps -- bypasses READY_TO_RUN")
    p.add_argument("--allow-unsealed", action="store_true")
    p.add_argument("--allow-dirty", action="store_true",
                   help="run even if tracked code differs from HEAD (diff is logged to MLflow)")
    args = p.parse_args()
    git_provenance.require_clean_code(REPO_ROOT, args.allow_dirty)
    arm = args.lr_frontend_arm
    card = _card_paths(arm)

    if not FOLDS_FROZEN.is_file():
        raise SystemExit(f"missing {FOLDS_FROZEN}")
    if not card.prereg.is_file():
        raise SystemExit(f"missing {card.prereg}")
    prereg = json.loads(card.prereg.read_text())
    status = str(prereg.get("status", ""))
    if not args.smoke and not args.allow_unsealed and status != "READY_TO_RUN":
        raise SystemExit(
            f"[m2_pool_lr] STOP: prereg status={status!r}; need READY_TO_RUN (or --smoke / --allow-unsealed)."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[m2_pool_lr] STOP: --device cuda requested but CUDA unavailable. No CPU fallback.")

    all_tags = G._all_structure_tags()
    assert len(all_tags) == 12, all_tags
    if args.smoke:
        folds = all_tags[:1]
    elif args.folds:
        folds = [t for t in args.folds.split(",") if t]
    else:
        folds = all_tags
    steps = 3 if args.smoke else int(args.steps)
    if not args.smoke and int(args.steps) != STEPS:
        raise SystemExit(f"[m2_pool_lr] STOP: card locks steps={STEPS}; got {args.steps}.")

    card.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[m2_pool_lr] arm={arm} lr_frontend={LR_ARMS[arm]} steps={steps} seed={args.seed} folds={len(folds)}")

    fold_results: list[dict[str, Any]] = []
    for hold in folds:
        hold_key = hold.replace(":", "")
        out = card.out_dir / f"seed{args.seed}_hold_{hold_key}.json"
        if not args.smoke and out.is_file():
            print(f"[m2_pool_lr] skip hold={hold} (already have {out}, resuming)")
            fold_results.append(json.loads(out.read_text()))
            continue
        row = run_one_fold(hold, arm, device=device, steps=steps, seed=int(args.seed),
                            no_mlflow=bool(args.no_mlflow or args.smoke), card=card)
        fold_results.append(row)
        if not args.smoke:
            out.write_text(json.dumps(row, indent=2) + "\n")
            print(f"[m2_pool_lr] wrote {out}")
        if row.get("abort") == "ABORT_WIRING":
            print(f"[m2_pool_lr] ABORT_WIRING on hold={hold}; stopping remaining folds")
            break

    if args.smoke:
        print("[m2_pool_lr] smoke done -- no stamps")
        print(json.dumps({r["heldout"][0]: r.get("abort") or r.get("train_macro_f1") for r in fold_results}, indent=2))
        return 0

    if len(fold_results) < len(folds):
        print("[m2_pool_lr] incomplete fold set; not scoring a partial card")
        return 0

    scored = G._score_card(fold_results, card)
    verdict = scored["verdict"]
    stamp: dict[str, Any] = {
        "schema_version": 1, "gate_id": card.gate_id, "status": verdict,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "display_lineage": "Tokyo Eye EQU", "do_not_promote": True,
        "card": str(card.prereg.relative_to(REPO_ROOT)), "metric_family": card.metric_family,
        "lr_frontend_arm": arm, "lr_frontend": LR_ARMS[arm],
        "backbone_train_mode": ARMS[arm]["backbone_train_mode"],
        "seed": int(args.seed), "steps": STEPS, "folds": [r["heldout"][0] for r in fold_results],
        "frontend": "equiformer_v3_pool_cold_init", "moe_mode": "ablated", "sdrp_coeff": SDRP_COEFF,
        "dehydron_coeff": DEHYDRON_COEFF, "margin_coeff": 0.0, "scoring": scored,
        "per_fold_summary": {
            r["heldout"][0]: {
                "finite": r["finite"], "G_grad_spine": r.get("G_grad_spine"),
                "train_macro_f1": r.get("train_macro_f1"), "train_min_f1": r.get("train_min_f1"),
                "g_fit_train_pass_fold": r.get("g_fit_train_pass_fold"),
                "heldout_macro_f1": r.get("heldout_metrics", {}).get(r["heldout"][0], {}).get("macro_f1"),
                "c_drift": r.get("c_drift"), "clip_active_fraction": r.get("clip_active_fraction"),
            }
            for r in fold_results
        },
        "script_sha256": _sha256(Path(__file__)),
        "prereg_sha256": _sha256(card.prereg),
        "folds_frozen_sha256": _sha256(FOLDS_FROZEN),
        "git_commit": prereg.get("pins", {}).get("git_commit"),
        "not_claims": prereg.get("not_claims", []),
        "signed": "auto from wrap1_zhyp_m2_pool_lr.py",
    }
    card.result.write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"[m2_pool_lr] RESULT={verdict} arm={arm} wrote {card.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
