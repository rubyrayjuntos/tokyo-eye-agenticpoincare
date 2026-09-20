#!/usr/bin/env python3
"""wrap=1 dehydron LOSO — G_fit only (typed train AUPRC on 3 pinned folds).

Task packet: data/gates/tokyo_eye_equ_wrap1_dehydron_loso_G_fit_task.json

Resolves PENDING_G_FIT from Arm B. Does NOT open a T science question.
Typed/leaked inputs only — a pass clears Arm B ambiguity, not biology validation.

Protocol (card-locked):
  * Cold SE(3)-lite, moe_mode=ablated, sdrp_coeff=0, E=200, seed 0
  * Folds: hold out each of 1MBN:A, 1LYZ:A, 1BG1:A; train on the other 11 (typed)
  * G_fit PASS iff last-epoch train-fold pooled AUPRC >= 0.80 on EVERY fold
  * Stamp: data/gates/tokyo_eye_equ_wrap1_dehydron_loso_G_fit_result.json
  * Update disposition to FAIL_NO_SIGNAL or INCONCLUSIVE_UNDERFIT

No edits to moe.py / model.py / engine.py / loader / loss / weight map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.training.v8.run_v8_experiment import (  # noqa: E402
    build_system,
    run_epoch_mean_structures,
)
from science.tokyo_eye.v8.engine import (  # noqa: E402
    CurriculumRadiusController,
    EpsilonGreedySchedule,
    GumbelTemperatureSchedule,
    PoincareDiagnosticsEngine,
)
from science.tokyo_eye.v8.equiformer_frontend import (  # noqa: E402
    DEFAULT_WEIGHT_MAP,
    build_param_groups,
    load_weight_map,
)
from science.tokyo_eye.v8.freeze_reconciliation import (  # noqa: E402
    CANONICAL_MLFLOW_EXPERIMENT,
)
from science.tokyo_eye.v8.metrics import binary_auprc  # noqa: E402
from science.tokyo_eye.v8.r0_r5_graph import get_dehydron_wrap_max  # noqa: E402

MANIFEST = REPO_ROOT / "manifests" / "v8_stage_a_small_v1.json"
PDB_DIR = REPO_ROOT / "pdb_cache"
GRAPH_CACHE = PDB_DIR / "v8_graph_cache"
FOLDS_FROZEN = REPO_ROOT / "data" / "gates" / "wrap1_dehydron_loso" / "folds_frozen.json"
OUT_DIR = REPO_ROOT / "data" / "gates" / "wrap1_dehydron_loso"
RESULT_PATH = (
    REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_dehydron_loso_G_fit_result.json"
)
DISPOSITION_PATH = (
    REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_dehydron_loso_B_disposition.json"
)
TASK_PATH = (
    REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_dehydron_loso_G_fit_task.json"
)
APPROVED = (
    REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_dehydron_loso_prereg.APPROVED.json"
)

G_FIT_FOLDS = ["1MBN:A", "1LYZ:A", "1BG1:A"]
G_FIT_THRESHOLD = 0.80
EPOCHS = 200
SEED = 0
SDRP_COEFF = 0.0
E400_PREREG = (
    REPO_ROOT
    / "data"
    / "gates"
    / "tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_prereg.json"
)
# Pre-registered plateau / r-trajectory thresholds (E400 prereg).
R_DRIFT_TOL = 0.10
PLATEAU_FRACTION = 0.50
PLATEAU_ABS_IF_G3_NONPOS = 0.005


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _parse_tag(tag: str) -> tuple[str, str]:
    pdb, chain = tag.split(":")
    return pdb, chain


def _load_batch(tag: str, device: torch.device) -> dict[str, Any]:
    from science.tokyo_eye.v8.loader import load_structure_batch

    pdb, chain = _parse_tag(tag)
    batch = load_structure_batch(
        pdb,
        chain,
        pdb_dir=PDB_DIR,
        device=device,
        use_graph_cache=True,
        graph_cache_dir=GRAPH_CACHE,
    )
    batch["pdb_id"] = pdb
    batch["chain"] = chain
    return batch


def _pooled_train_auprc(
    system: torch.nn.Module,
    batches: list[dict[str, Any]],
    *,
    tau_ceiling: float,
) -> tuple[float, float]:
    """Node-pooled dehydron AUPRC + prevalence over train batches (eval mode)."""
    was = system.training
    system.eval()
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            out = system(
                batch["x"],
                batch["edge_index"],
                batch["edge_type"],
                tau_ceiling=tau_ceiling,
                chem=batch.get("gate_chem"),
            )
            scores.append(torch.sigmoid(out["mechanism_score"]).detach().float().cpu())
            labels.append(batch["dehydron_labels"].detach().float().cpu())
    if was:
        system.train()
    s = torch.cat(scores)
    y = torch.cat(labels)
    prev = float(y.mean().item()) if y.numel() else 0.0
    auprc = float(binary_auprc(s, y))
    return auprc, prev


def _all_structure_tags() -> list[str]:
    folds = json.loads(FOLDS_FROZEN.read_text())["folds"]
    return [f[0] for f in folds]


def run_one_fold(
    hold: str,
    *,
    device: torch.device,
    epochs: int,
    seed: int,
    no_mlflow: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_tags = _all_structure_tags()
    train_tags = [t for t in all_tags if t != hold]
    assert len(train_tags) == 11, train_tags

    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    system, load_info = build_system(
        cfg, equiformer_ckpt=None, device=device, freeze_backbone=False
    )
    system.set_moe_mode("ablated")
    load_info["frontend_pilot"] = "cold_se3_lite"
    load_info["allow_off_path_frontend"] = True
    load_info["arm"] = "T_typed_G_fit_only"

    groups = build_param_groups(
        system.frontend,
        system.spine,
        lr_backbone=float(cfg["lr_backbone"]),
        lr_hyperbolic=float(cfg["lr_hyperbolic"]),
        freeze_backbone=False,
    )
    optimizer = torch.optim.Adam(groups)
    radius = CurriculumRadiusController(
        float(cfg["tau_start"]), float(cfg["tau_end"]), epochs
    )
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        epochs,
        schedule=str(cfg.get("gumbel_schedule", "exponential")),
        alpha=float(cfg["gumbel_exp_alpha"])
        if cfg.get("gumbel_exp_alpha") is not None
        else None,
        half_epochs=int(cfg.get("gumbel_exp_half_epochs", 12)),
    )
    half_ep = int(cfg.get("gumbel_exp_half_epochs", 12))
    eps_sched = EpsilonGreedySchedule(
        float(cfg.get("eps_start", 0.20)),
        float(cfg.get("eps_end", 0.0)),
        half_epochs=int(cfg.get("eps_half_epochs", half_ep)),
    )
    diagnostics = PoincareDiagnosticsEngine()
    cv_coeff = float(cfg.get("cv_coeff", 10.0))
    moe_quota_coeff = float(cfg.get("moe_quota_coeff", 5.0))
    eval_proxy_quota_coeff = float(cfg.get("eval_proxy_quota_coeff", 140.0))

    train_batches = [_load_batch(t, device) for t in train_tags]
    # typed: do not blind edge_type / gate_chem

    finite = True
    last_train_auprc = float("nan")
    last_train_prev = float("nan")
    curve: list[dict[str, Any]] = []

    mlflow = None
    if not no_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(CANONICAL_MLFLOW_EXPERIMENT)
        except Exception as exc:  # noqa: BLE001
            print(f"[g_fit] MLflow unavailable ({exc}); continuing without")
            mlflow = None

    run_ctx = (
        mlflow.start_run(run_name=f"wrap1_loso_G_fit_hold_{hold.replace(':', '')}")
        if mlflow is not None
        else None
    )
    try:
        if mlflow is not None:
            mlflow.set_tags(
                {
                    "diagnostic": "true",
                    "do_not_promote": "true",
                    "arm": "T_typed_G_fit_only",
                    "moe_mode": "ablated",
                    "moe_claim": "none",
                    "sdrp_coeff": "0.0",
                    "frontend": "cold_se3_lite",
                    "hold": hold,
                    "gate_id": "tokyo_eye_equ_wrap1_dehydron_loso_G_fit",
                }
            )
        for epoch in range(epochs):
            eps_t = float(eps_sched.epsilon(epoch))
            eps_at_floor = eps_t <= float(eps_sched.eps_end) + 1e-12
            try:
                metrics = run_epoch_mean_structures(
                    system,
                    optimizer,
                    train_batches,
                    epoch=epoch,
                    radius=radius,
                    gumbel=gumbel,
                    diagnostics=diagnostics,
                    cv_coeff=cv_coeff,
                    moe_quota_coeff=moe_quota_coeff,
                    eval_proxy_quota_coeff=eval_proxy_quota_coeff,
                    explore_epsilon=eps_t,
                    eps_at_floor=eps_at_floor,
                    sdrp_coeff=SDRP_COEFF,
                    dehydron_coeff=1.0,
                    margin_coeff=1.0,
                )
            except Exception as exc:  # noqa: BLE001
                finite = False
                print(f"[g_fit] NONFINITE/ERROR hold={hold} epoch={epoch}: {exc}")
                break
            for v in metrics.values():
                if isinstance(v, float) and not math.isfinite(v):
                    finite = False
            tau_eval = float(cfg["tau_end"])
            if epoch % 10 == 0 or epoch == epochs - 1:
                auprc, prev = _pooled_train_auprc(
                    system, train_batches, tau_ceiling=tau_eval
                )
                last_train_auprc, last_train_prev = auprc, prev
                row = {
                    "epoch": epoch,
                    "train_pooled_auprc": auprc,
                    "train_prevalence": prev,
                    "loss_total": metrics.get("loss_total"),
                }
                curve.append(row)
                print(
                    f"[g_fit] hold={hold} epoch={epoch} "
                    f"train_auprc={auprc:.4f} prev={prev:.3f}"
                )
                if mlflow is not None:
                    mlflow.log_metrics(
                        {
                            "train_pooled_auprc": auprc,
                            "train_prevalence": prev,
                            "loss_total": float(metrics.get("loss_total", 0.0)),
                        },
                        step=epoch,
                    )
    finally:
        if run_ctx is not None:
            mlflow.end_run()

    return {
        "arm": "T",
        "g_fit_only": True,
        "seed": seed,
        "heldout": [hold],
        "train": train_tags,
        "epochs": epochs,
        "finite": finite,
        "moe_mode": "ablated",
        "sdrp_coeff": SDRP_COEFF,
        "tau_eval": float(cfg["tau_end"]),
        "frontend": "cold SE(3)-lite",
        "final": {
            "train_pooled_auprc": last_train_auprc,
            "train_prevalence": last_train_prev,
        },
        "curve_every_10": curve,
        "g_fit_pass_fold": bool(
            finite and last_train_auprc >= G_FIT_THRESHOLD
        ),
        "g_fit_threshold": G_FIT_THRESHOLD,
    }


def _a_at(curve: list[dict[str, Any]], target: int) -> float:
    """Nearest logged epoch ≤ target (curve_every_10)."""
    eligible = [r for r in curve if int(r["epoch"]) <= target]
    if not eligible:
        return float(curve[0]["train_pooled_auprc"])
    row = max(eligible, key=lambda r: int(r["epoch"]))
    return float(row["train_pooled_auprc"])


def _classify_e400(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply E400 prereg read order: r-trajectory → 0.80 → plateau."""
    per_fold: dict[str, Any] = {}
    any_climbing = False
    failing: list[str] = []
    for r in fold_results:
        hold = r["heldout"][0]
        curve = r["curve_every_10"]
        a0 = _a_at(curve, 0)
        a100 = _a_at(curve, 100)
        a200 = _a_at(curve, 200)
        a300 = _a_at(curve, 300)
        a400 = _a_at(curve, 400)
        g1, g2, g3, g4 = a100 - a0, a200 - a100, a300 - a200, a400 - a300
        r12 = (g2 / g1) if g1 > 1e-12 else float("nan")
        r23 = (g3 / g2) if g2 > 1e-12 else float("nan")
        r34 = (g4 / g3) if g3 > 1e-12 else float("nan")
        climbing = False
        if g1 > 1e-12 and g2 > 1e-12 and (r23 > r12 + R_DRIFT_TOL):
            climbing = True
        if g3 > 1e-12 and g2 > 1e-12 and (r34 > r23 + R_DRIFT_TOL):
            climbing = True
        if climbing:
            any_climbing = True
        if g3 > 0:
            plateau = g4 <= PLATEAU_FRACTION * g3
        else:
            plateau = g4 <= PLATEAU_ABS_IF_G3_NONPOS
        clears_fold = a400 >= G_FIT_THRESHOLD
        if not clears_fold:
            failing.append(hold)
        per_fold[hold] = {
            "A": {"0": a0, "100": a100, "200": a200, "300": a300, "400": a400},
            "gains": {"g1": g1, "g2": g2, "g3": g3, "g4": g4},
            "ratios": {"r12": r12, "r23": r23, "r34": r34},
            "r_climbing": climbing,
            "plateau": plateau,
            "clears_0_80": clears_fold,
        }

    r_traj = "R_CLIMBING" if any_climbing else "R_STABLE"
    clears = all(v["clears_0_80"] for v in per_fold.values())
    plateau_confirmed = all(
        per_fold[h]["plateau"] for h in failing
    ) if failing else True

    if clears:
        outcome = "CLEARS_0_80"
    elif r_traj == "R_CLIMBING" or not plateau_confirmed:
        outcome = "DECAY_UNSETTLED"
    else:
        outcome = "PLATEAU_BELOW"

    return {
        "r_trajectory": r_traj,
        "plateau_confirmed": plateau_confirmed,
        "outcome": outcome,
        "per_fold_blocks": per_fold,
    }


def _update_disposition(
    verdict: str,
    result_path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    if not DISPOSITION_PATH.is_file():
        return
    d = json.loads(DISPOSITION_PATH.read_text())
    d["status"] = verdict
    d["g_fit_resolved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    d["g_fit_result"] = str(result_path.relative_to(REPO_ROOT))
    d["active_branch"] = verdict
    if extra:
        d.update(extra)
    d["next_command_note"] = (
        f"G_fit resolved → {verdict}. See after_g_fit_branch / E400 outcomes. "
        "Do not promote. Do not over-read typed G_fit as biology validation."
    )
    payload = json.dumps(d, indent=2) + "\n"
    try:
        DISPOSITION_PATH.write_text(payload)
    except PermissionError:
        # appuser (999) cannot overwrite host-owned disposition; write sibling.
        alt = DISPOSITION_PATH.with_suffix(".UPDATED.json")
        alt.write_text(payload)
        print(
            f"[g_fit] WARN disposition PermissionError on {DISPOSITION_PATH}; "
            f"wrote sibling {alt} — host should merge or chmod o+w the original."
        )


def main() -> int:
    p = argparse.ArgumentParser(description="wrap1 LOSO G_fit only")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="1 fold, 3 epochs, no stamps (wiring check only)",
    )
    args = p.parse_args()

    if get_dehydron_wrap_max() != 1:
        raise SystemExit(f"wrap_max must be 1, got {get_dehydron_wrap_max()}")
    if not FOLDS_FROZEN.is_file():
        raise SystemExit(f"missing {FOLDS_FROZEN}")
    if not APPROVED.is_file():
        raise SystemExit(f"missing {APPROVED}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "[g_fit] STOP: --device cuda requested but torch.cuda.is_available() is False. "
            "Do not fall back to cpu for sealed cards. Fix GPU passthrough "
            "(tokyoeye_mlflow deploy.resources.reservations.devices) and retry."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = G_FIT_FOLDS if not args.smoke else G_FIT_FOLDS[:1]
    epochs = 3 if args.smoke else int(args.epochs)

    print(
        f"[g_fit] device={device} epochs={epochs} seed={args.seed} "
        f"folds={folds} threshold={G_FIT_THRESHOLD}"
    )

    fold_results: list[dict[str, Any]] = []
    for hold in folds:
        row = run_one_fold(
            hold,
            device=device,
            epochs=epochs,
            seed=int(args.seed),
            no_mlflow=bool(args.no_mlflow or args.smoke),
        )
        fold_results.append(row)
        hold_key = hold.replace(":", "")
        if int(args.epochs) == 400:
            out = OUT_DIR / f"T_seed{args.seed}_hold_{hold_key}_G_fit_E400.json"
        else:
            out = OUT_DIR / f"T_seed{args.seed}_hold_{hold_key}_G_fit.json"
        if not args.smoke:
            out.write_text(json.dumps(row, indent=2) + "\n")
            print(f"[g_fit] wrote {out}")

    if args.smoke:
        print("[g_fit] smoke done — no stamps")
        return 0

    finite_all = all(r["finite"] for r in fold_results)
    per_fold = {
        r["heldout"][0]: {
            "train_pooled_auprc": r["final"]["train_pooled_auprc"],
            "g_fit_pass_fold": r["g_fit_pass_fold"],
        }
        for r in fold_results
    }
    g_fit_pass = finite_all and all(r["g_fit_pass_fold"] for r in fold_results)

    e400_class: dict[str, Any] | None = None
    if not finite_all:
        verdict = "ABORT_NONFINITE"
    elif int(args.epochs) == 400:
        e400_class = _classify_e400(fold_results)
        # Map named harness outcomes; CLEARS_0_80 → FAIL_NO_SIGNAL for B branch.
        outcome = e400_class["outcome"]
        if outcome == "CLEARS_0_80":
            verdict = "FAIL_NO_SIGNAL"
        else:
            verdict = outcome  # PLATEAU_BELOW | DECAY_UNSETTLED
        e400_class["card_verdict_raw"] = outcome
    elif g_fit_pass:
        verdict = "FAIL_NO_SIGNAL"
    else:
        verdict = "INCONCLUSIVE_UNDERFIT"

    if int(args.epochs) == 400:
        result_path = (
            REPO_ROOT
            / "data"
            / "gates"
            / "tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_result.json"
        )
        gate_id = "tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_result"
        task_rel = (
            str(E400_PREREG.relative_to(REPO_ROOT)) if E400_PREREG.is_file() else None
        )
    else:
        result_path = RESULT_PATH
        gate_id = "tokyo_eye_equ_wrap1_dehydron_loso_G_fit_result"
        task_rel = (
            str(TASK_PATH.relative_to(REPO_ROOT)) if TASK_PATH.is_file() else None
        )

    stamp: dict[str, Any] = {
        "schema_version": 1,
        "gate_id": gate_id,
        "status": verdict,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "display_lineage": "Tokyo Eye EQU",
        "do_not_promote": True,
        "task": task_rel,
        "parent_b_result": "data/gates/tokyo_eye_equ_wrap1_dehydron_loso_B_result.json",
        "disposition": str(DISPOSITION_PATH.relative_to(REPO_ROOT)),
        "arm": "T_typed_G_fit_only",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "folds": folds,
        "g_fit_threshold": G_FIT_THRESHOLD,
        "g_fit_pass": g_fit_pass,
        "G_finite": finite_all,
        "per_fold": per_fold,
        "frontend": "cold SE(3)-lite",
        "moe_mode": "ablated",
        "sdrp_coeff": SDRP_COEFF,
        "g_fit_meaning": {
            "what_it_is": "Typed/leaked train-fold AUPRC — can the model fit when given the answer",
            "what_it_is_not": "Not architecture validation, not biology discovery, not pool seal",
        },
        "card_verdict": verdict,
        "script_sha256": _sha256(Path(__file__)),
        "folds_frozen_sha256": _sha256(FOLDS_FROZEN),
        "approved_sha256": _sha256(APPROVED),
        "signed": "auto from wrap1_dehydron_loso_g_fit.py",
    }
    if e400_class is not None:
        stamp["e400_classification"] = e400_class
        stamp["read_order_applied"] = [
            "r_trajectory",
            "auprc_vs_0.80",
            "plateau_test",
        ]
        stamp["parent_g_fit_result"] = (
            "data/gates/tokyo_eye_equ_wrap1_dehydron_loso_G_fit_result.json"
        )

    result_path.write_text(json.dumps(stamp, indent=2) + "\n")
    extra = None
    if e400_class is not None:
        extra = {
            "e400_result": str(result_path.relative_to(REPO_ROOT)),
            "e400_outcome": e400_class["outcome"],
            "e400_r_trajectory": e400_class["r_trajectory"],
            "epoch_extend_prereg": str(E400_PREREG.relative_to(REPO_ROOT)),
        }
    _update_disposition(verdict, result_path, extra=extra)
    print(f"[g_fit] RESULT={verdict} wrote {result_path}")
    if e400_class is not None:
        print(
            f"[g_fit] E400 outcome={e400_class['outcome']} "
            f"r_traj={e400_class['r_trajectory']} "
            f"plateau_confirmed={e400_class['plateau_confirmed']}"
        )
    print(f"[g_fit] per_fold={json.dumps(per_fold)}")
    return 0 if finite_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
