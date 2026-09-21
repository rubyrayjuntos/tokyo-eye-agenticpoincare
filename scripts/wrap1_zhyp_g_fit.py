#!/usr/bin/env python3
"""wrap=1 z_hyp G_fit — SDRP-live sole loss on cold SE(3)-lite (12-fold LOSO).

Cards:
  * lift (legacy): data/gates/tokyo_eye_equ_wrap1_zhyp_g_fit_prereg.json
  * m2:           data/gates/tokyo_eye_equ_wrap1_zhyp_m2_prereg.json

Harness/capacity probe with intentional SDRP leak. Not biology seal.
Not defect-B close. Not learned-curvature training (_log_c detached).

Protocol (card-locked):
  * Cold SE(3)-lite + allow_off_path; moe ablated; pool FORBIDDEN
  * sdrp_coeff=0.1, dehydron_coeff=0, margin_coeff=0 — zero-coeff terms OMITTED
    from the loss graph (not multiplied by 0.0)
  * 400 optimizer steps; last-step only
  * Bars: G_finite, G_grad_spine, G_fit_train, S1, S2 (see active prereg)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.training.v8.run_v8_experiment import (  # noqa: E402
    build_system,
)
from science.tokyo_eye.v8.assembly_gate import (  # noqa: E402
    AssemblyGateError,
    assert_governed_assembly,
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
from science.tokyo_eye.v8.grad_reachability import (  # noqa: E402
    SPINE_NZ_BUCKETS,
    bucket_grad_stats,
    bucket_status,
)
from science.tokyo_eye.v8.heads import sdrp_cross_entropy  # noqa: E402
from science.tokyo_eye.v8.r0_r5_graph import get_dehydron_wrap_max  # noqa: E402

MANIFEST = REPO_ROOT / "manifests" / "v8_stage_a_small_v1.json"
PDB_DIR = REPO_ROOT / "pdb_cache"
GRAPH_CACHE = PDB_DIR / "v8_graph_cache"
FOLDS_FROZEN = REPO_ROOT / "data" / "gates" / "wrap1_dehydron_loso" / "folds_frozen.json"

STEPS = 400
SEED = 0
SDRP_COEFF = 0.1
MAX_GRAD_NORM = 1.0
LOG_EVERY = 10
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
S2_OF_N = 12


@dataclass(frozen=True)
class CardPaths:
    card: str
    metric_family: str
    prereg: Path
    result: Path
    out_dir: Path
    gate_id: str
    # Train G_fit bars (metric depends on family)
    g_fit_train_macro_min: float
    g_fit_train_min_struct: float
    # Held-out S1/S2
    s1_mean_min: float
    s1_ci_lb_min: float | None  # None = no CI gate (M2)
    s2_min_above: int
    s2_threshold: float  # lift>1.0 or macro_f1>=0.40


CARDS: dict[str, CardPaths] = {
    "lift": CardPaths(
        card="lift",
        metric_family="lift",
        prereg=REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_zhyp_g_fit_prereg.json",
        result=REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_zhyp_g_fit_result.json",
        out_dir=REPO_ROOT / "data" / "gates" / "wrap1_zhyp_g_fit",
        gate_id="tokyo_eye_equ_wrap1_zhyp_g_fit_result",
        g_fit_train_macro_min=1.30,
        g_fit_train_min_struct=1.10,
        s1_mean_min=1.30,
        s1_ci_lb_min=1.10,
        s2_min_above=10,
        s2_threshold=1.0,  # count lift > 1.0
    ),
    "m2": CardPaths(
        card="m2",
        metric_family="M2_macro_f1",
        prereg=REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_zhyp_m2_prereg.json",
        result=REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_wrap1_zhyp_m2_result.json",
        out_dir=REPO_ROOT / "data" / "gates" / "wrap1_zhyp_m2",
        gate_id="tokyo_eye_equ_wrap1_zhyp_m2_result",
        g_fit_train_macro_min=0.40,
        g_fit_train_min_struct=0.40,
        s1_mean_min=0.40,
        s1_ci_lb_min=None,
        s2_min_above=10,
        s2_threshold=0.40,  # count macro_f1 >= 0.40
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _all_structure_tags() -> list[str]:
    folds = json.loads(FOLDS_FROZEN.read_text())["folds"]
    return [f[0] for f in folds]


def _curvature_c(system: torch.nn.Module) -> float:
    spine = system.spine
    return float(F.softplus(spine._log_c).detach() + 1e-4)


def _sdrp_structure_metrics(
    system: torch.nn.Module,
    batch: dict[str, Any],
    *,
    tau_ceiling: float,
) -> dict[str, float]:
    """Per-structure SDRP top-1 / majority / lift / macro-F1 (present classes)."""
    was = system.training
    system.eval()
    with torch.no_grad():
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=tau_ceiling,
            chem=batch.get("gate_chem"),
        )
        logits = out["sdrp_logits"]
        pred = logits.argmax(dim=-1)
        y = batch["sdrp_target"].long()
        n = int(y.numel())
        acc = float((pred == y).float().mean().item()) if n else float("nan")
        # majority class rate on this structure
        counts = torch.bincount(y, minlength=int(logits.shape[-1]))
        maj = float(counts.max().item() / n) if n else float("nan")
        lift = float(acc / maj) if maj and maj > 0 else float("nan")
        # macro-F1 over classes present in y
        present = (counts > 0).nonzero(as_tuple=False).view(-1).tolist()
        f1s: list[float] = []
        for c in present:
            tp = int(((pred == c) & (y == c)).sum().item())
            fp = int(((pred == c) & (y != c)).sum().item())
            fn = int(((pred != c) & (y == c)).sum().item())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(
                0.0 if (prec + rec) == 0 else 2.0 * prec * rec / (prec + rec)
            )
        macro_f1 = float(sum(f1s) / len(f1s)) if f1s else float("nan")
    if was:
        system.train()
    return {
        "n": float(n),
        "sdrp_top1_acc": acc,
        "majority_rate": maj,
        "lift": lift,
        "macro_f1": macro_f1,
    }


def _edge_type_shuffle_sensitivity(
    system: torch.nn.Module,
    batch: dict[str, Any],
    *,
    tau_ceiling: float,
    seed: int,
) -> dict[str, float]:
    """Diagnostic: shuffle edge_type; Δ on mechanism_score and sdrp_logits."""
    was = system.training
    system.eval()
    g = torch.Generator()
    g.manual_seed(int(seed))
    with torch.no_grad():
        out0 = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=tau_ceiling,
            chem=batch.get("gate_chem"),
        )
        perm = torch.randperm(int(batch["edge_type"].numel()), generator=g)
        et_shuf = batch["edge_type"][perm.to(batch["edge_type"].device)]
        out1 = system(
            batch["x"],
            batch["edge_index"],
            et_shuf,
            tau_ceiling=tau_ceiling,
            chem=batch.get("gate_chem"),
        )
        d_mech = float(
            (out0["mechanism_score"] - out1["mechanism_score"]).abs().mean().item()
        )
        d_sdrp = float(
            (out0["sdrp_logits"] - out1["sdrp_logits"]).abs().mean().item()
        )
    if was:
        system.train()
    return {
        "mech_score_abs_delta_mean": d_mech,
        "sdrp_logits_abs_delta_mean": d_sdrp,
    }


def _total_grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return float(total**0.5)


def run_step_sdrp_only(
    system: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[dict[str, Any]],
    *,
    epoch: int,
    radius: CurriculumRadiusController,
    gumbel: GumbelTemperatureSchedule,
    explore_epsilon: float,
    max_grad_norm: float = MAX_GRAD_NORM,
    sdrp_coeff: float = SDRP_COEFF,
) -> dict[str, float]:
    """One mean-over-structures step; SDRP CE only (omit zero-coeff terms)."""
    if not batches:
        raise ValueError("batches must be non-empty")
    n = len(batches)
    system.train()
    tau_ceil = radius.tau_ceiling(epoch)
    gumbel_tau = gumbel.temperature(epoch)
    system.set_moe_temperature(gumbel_tau)
    system.set_moe_explore_epsilon(float(explore_epsilon))
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    for batch in batches:
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=tau_ceil,
            chem=batch.get("gate_chem"),
        )
        loss_sdrp = sdrp_cross_entropy(out["sdrp_logits"], batch["sdrp_target"])
        # Sole task loss — do NOT multiply disabled terms by 0.0.
        loss = (float(sdrp_coeff) * loss_sdrp) / float(n)
        if not torch.isfinite(loss):
            return {
                "loss_total": float("nan"),
                "tau_ceiling": float(tau_ceil),
                "gumbel_temperature": float(gumbel_tau),
                "nan_abort": 1.0,
                "preclip_norm": float("nan"),
                "postclip_norm": float("nan"),
                "clip_active": 0.0,
            }
        loss.backward()
        loss_sum += float(loss.detach())

    preclip = _total_grad_norm(system.parameters())
    if not math.isfinite(preclip):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss_total": float(loss_sum),
            "tau_ceiling": float(tau_ceil),
            "gumbel_temperature": float(gumbel_tau),
            "nan_abort": 1.0,
            "preclip_norm": float("nan"),
            "postclip_norm": float("nan"),
            "clip_active": 0.0,
        }
    torch.nn.utils.clip_grad_norm_(system.parameters(), max_grad_norm)
    postclip = min(preclip, float(max_grad_norm))
    clip_active = 1.0 if preclip > float(max_grad_norm) else 0.0
    optimizer.step()
    return {
        "loss_total": float(loss_sum),
        "tau_ceiling": float(tau_ceil),
        "gumbel_temperature": float(gumbel_tau),
        "nan_abort": 0.0,
        "preclip_norm": float(preclip),
        "postclip_norm": float(postclip),
        "clip_active": float(clip_active),
        "max_grad_norm": float(max_grad_norm),
    }


def _step0_spine_ok(system: torch.nn.Module, batch: dict[str, Any], *, tau: float) -> dict[str, Any]:
    """G_grad_spine at step-0: one SDRP backward, check spine buckets NZ."""
    system.train()
    optimizer_dummy = None  # noqa: F841 — explicit no opt; grads only
    system.zero_grad(set_to_none=True)
    out = system(
        batch["x"],
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau,
        chem=batch.get("gate_chem"),
    )
    loss = float(SDRP_COEFF) * sdrp_cross_entropy(out["sdrp_logits"], batch["sdrp_target"])
    loss.backward()
    table = bucket_grad_stats(system.spine)
    statuses = {b: bucket_status(table.get(b, {"nz": 0, "zero": 0, "none": 1})) for b in SPINE_NZ_BUCKETS}
    log_c = bucket_status(table.get("_log_c", {"nz": 0, "zero": 0, "none": 1}))
    ok = all(statuses[b] == "NZ" for b in SPINE_NZ_BUCKETS)
    system.zero_grad(set_to_none=True)
    return {
        "ok": ok,
        "statuses": statuses,
        "_log_c_status": log_c,
        "grad_l2": {b: float(table.get(b, {}).get("grad_l2", 0.0)) for b in SPINE_NZ_BUCKETS},
    }


def _bootstrap_ci_lb(lifts: list[float], *, draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    arr = np.asarray(lifts, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float("nan")
    means = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        means[i] = float(arr[idx].mean())
    return float(np.percentile(means, 2.5))


def run_one_fold(
    hold: str,
    *,
    device: torch.device,
    steps: int,
    seed: int,
    no_mlflow: bool,
    card: CardPaths,
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
    load_info["frontend_cli"] = "stub"
    load_info["arm"] = "zhyp_g_fit_sdrp_live"

    # Assembly gate: off-path lite pilot (non-claim).
    try:
        probe = train_tags[0]

        def _fwd():
            b = _load_batch(probe, device)
            return system(
                b["x"], b["edge_index"], b["edge_type"],
                tau_ceiling=0.70, chem=b.get("gate_chem"),
            )

        gate_result = assert_governed_assembly(
            frontend_kind="stub",
            load_info=load_info,
            system=system,
            forward_fn=_fwd,
            allow_off_path_frontend=True,
            require_pure_hyp=True,
            check_deps=True,
            claim_bearing_biology=False,
        )
        load_info["assembly_gate"] = gate_result.detail
        load_info["assembly_gate_passed"] = True
    except AssemblyGateError as exc:
        raise SystemExit(f"[zhyp_g_fit] assembly_gate FAIL: {exc}") from exc

    groups = build_param_groups(
        system.frontend,
        system.spine,
        lr_backbone=float(cfg["lr_backbone"]),
        lr_hyperbolic=float(cfg["lr_hyperbolic"]),
        freeze_backbone=False,
    )
    lr_frontend = float(cfg["lr_backbone"])
    lr_hyperbolic = float(cfg["lr_hyperbolic"])
    optimizer = torch.optim.Adam(groups)
    radius = CurriculumRadiusController(
        float(cfg["tau_start"]), float(cfg["tau_end"]), steps
    )
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        steps,
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
    _ = PoincareDiagnosticsEngine()  # parity with other runners; unused under SDRP-only

    train_batches = [_load_batch(t, device) for t in train_tags]
    hold_batch = _load_batch(hold, device)
    c_init = _curvature_c(system)

    # Step-0 spine reachability (G_grad_spine)
    step0 = _step0_spine_ok(
        system, train_batches[0], tau=float(cfg["tau_start"])
    )
    if not step0["ok"]:
        return {
            "arm": "zhyp_g_fit",
            "seed": seed,
            "heldout": [hold],
            "train": train_tags,
            "steps": steps,
            "finite": True,
            "G_grad_spine": False,
            "step0_grad": step0,
            "abort": "ABORT_WIRING",
            "c_init": c_init,
            "c_final": c_init,
            "lr_frontend": lr_frontend,
            "lr_hyperbolic": lr_hyperbolic,
        }

    finite = True
    clip_active_count = 0
    clip_logged = 0
    curve: list[dict[str, Any]] = []
    last_bucket: dict[str, Any] = {}
    abort_wiring_c = False

    mlflow = None
    if not no_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(CANONICAL_MLFLOW_EXPERIMENT)
        except Exception as exc:  # noqa: BLE001
            print(f"[zhyp_g_fit] MLflow unavailable ({exc}); continuing without")
            mlflow = None

    run_ctx = (
        mlflow.start_run(
            run_name=f"wrap1_zhyp_{card.card}_hold_{hold.replace(':', '')}"
        )
        if mlflow is not None
        else None
    )
    try:
        if mlflow is not None:
            mlflow.set_tags(
                {
                    "diagnostic": "true",
                    "do_not_promote": "true",
                    "arm": "zhyp_g_fit_sdrp_live",
                    "card": card.card,
                    "metric_family": card.metric_family,
                    "moe_mode": "ablated",
                    "sdrp_coeff": str(SDRP_COEFF),
                    "dehydron_coeff": "0.0",
                    "margin_coeff": "0.0",
                    "frontend": "cold_se3_lite",
                    "hold": hold,
                    "gate_id": card.gate_id.replace("_result", "_prereg"),
                    "G_grad_spine_step0": "PASS" if step0["ok"] else "FAIL",
                }
            )
            mlflow.log_params(
                {
                    "lr_frontend": lr_frontend,
                    "lr_hyperbolic": lr_hyperbolic,
                    "sdrp_coeff": SDRP_COEFF,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "steps": steps,
                    "seed": seed,
                    "card": card.card,
                    "metric_family": card.metric_family,
                    "g_fit_train_macro_min": card.g_fit_train_macro_min,
                    "g_fit_train_min_struct": card.g_fit_train_min_struct,
                }
            )
            # Gate telemetry — what we are actually testing
            step0_metrics = {
                "G_grad_spine_ok": 1.0 if step0["ok"] else 0.0,
                "curvature_c": float(c_init),
            }
            for b, g in (step0.get("grad_l2") or {}).items():
                step0_metrics[f"grad_l2_{b}"] = float(g)
            mlflow.log_metrics(step0_metrics, step=0)
            print(
                f"[zhyp_g_fit] hold={hold} step0 G_grad_spine="
                f"{'PASS' if step0['ok'] else 'FAIL'} "
                f"grad_l2={step0.get('grad_l2')}"
            )

        for step in range(steps):
            eps_t = float(eps_sched.epsilon(step))
            try:
                metrics = run_step_sdrp_only(
                    system,
                    optimizer,
                    train_batches,
                    epoch=step,
                    radius=radius,
                    gumbel=gumbel,
                    explore_epsilon=eps_t,
                    max_grad_norm=MAX_GRAD_NORM,
                    sdrp_coeff=SDRP_COEFF,
                )
            except Exception as exc:  # noqa: BLE001
                finite = False
                print(f"[zhyp_g_fit] NONFINITE/ERROR hold={hold} step={step}: {exc}")
                break
            if metrics.get("nan_abort", 0.0) >= 1.0 or not math.isfinite(
                float(metrics.get("loss_total", float("nan")))
            ):
                finite = False
                break
            if step % LOG_EVERY == 0 or step == steps - 1:
                clip_logged += 1
                if metrics.get("clip_active", 0.0) >= 1.0:
                    clip_active_count += 1
                # bucket grads: re-run one SDRP backward for telemetry (no step)
                system.zero_grad(set_to_none=True)
                b0 = train_batches[0]
                out = system(
                    b0["x"], b0["edge_index"], b0["edge_type"],
                    tau_ceiling=float(metrics["tau_ceiling"]),
                    chem=b0.get("gate_chem"),
                )
                (
                    float(SDRP_COEFF)
                    * sdrp_cross_entropy(out["sdrp_logits"], b0["sdrp_target"])
                ).backward()
                last_bucket = {
                    k: float(v.get("grad_l2", 0.0))
                    for k, v in bucket_grad_stats(system.spine).items()
                }
                system.zero_grad(set_to_none=True)

                c_now = _curvature_c(system)
                row = {
                    "step": step,
                    "loss_total": metrics["loss_total"],
                    "preclip_norm": metrics["preclip_norm"],
                    "postclip_norm": metrics["postclip_norm"],
                    "clip_active": metrics["clip_active"],
                    "c": c_now,
                    "bucket_grad_l2": last_bucket,
                }
                curve.append(row)
                spine_nz = {
                    b: last_bucket.get(b, 0.0) for b in SPINE_NZ_BUCKETS
                }
                print(
                    f"[zhyp_g_fit] hold={hold} step={step} "
                    f"loss={metrics['loss_total']:.4f} "
                    f"preclip={metrics['preclip_norm']:.3f} "
                    f"clip={int(metrics['clip_active'])} "
                    f"spine_grad={{{', '.join(f'{k}={v:.2e}' for k, v in spine_nz.items())}}}"
                )
                if mlflow is not None:
                    payload = {
                        "loss_total": float(metrics["loss_total"]),
                        "preclip_norm": float(metrics["preclip_norm"]),
                        "postclip_norm": float(metrics["postclip_norm"]),
                        "clip_active": float(metrics["clip_active"]),
                        "curvature_c": float(c_now),
                        "G_grad_spine_ok": 1.0,  # still live if we got here
                    }
                    for b, g in last_bucket.items():
                        payload[f"grad_l2_{b}"] = float(g)
                    for b in SPINE_NZ_BUCKETS:
                        payload[f"spine_nz_{b}"] = (
                            1.0 if float(last_bucket.get(b, 0.0)) > 0.0 else 0.0
                        )
                    mlflow.log_metrics(payload, step=step)

        c_final = _curvature_c(system)
        if abs(c_final - c_init) > 1e-5:
            abort_wiring_c = True

        tau_eval = float(cfg["tau_end"])
        train_per: dict[str, Any] = {}
        for t, b in zip(train_tags, train_batches):
            train_per[t] = _sdrp_structure_metrics(system, b, tau_ceiling=tau_eval)
        held = _sdrp_structure_metrics(system, hold_batch, tau_ceiling=tau_eval)
        shuffle = _edge_type_shuffle_sensitivity(
            system, hold_batch, tau_ceiling=tau_eval, seed=seed
        )

        train_lifts = [
            float(v["lift"]) for v in train_per.values() if math.isfinite(v["lift"])
        ]
        macro_lift = float(np.mean(train_lifts)) if train_lifts else float("nan")
        min_lift = float(np.min(train_lifts)) if train_lifts else float("nan")
        train_f1s = [
            float(v["macro_f1"])
            for v in train_per.values()
            if math.isfinite(v["macro_f1"])
        ]
        macro_f1_mean = float(np.mean(train_f1s)) if train_f1s else float("nan")
        min_f1 = float(np.min(train_f1s)) if train_f1s else float("nan")

        if card.metric_family == "M2_macro_f1":
            train_macro = macro_f1_mean
            train_min = min_f1
        else:
            train_macro = macro_lift
            train_min = min_lift

        g_fit_train = bool(
            finite
            and step0["ok"]
            and not abort_wiring_c
            and math.isfinite(train_macro)
            and math.isfinite(train_min)
            and train_macro >= card.g_fit_train_macro_min
            and train_min >= card.g_fit_train_min_struct
        )
        clip_frac = (
            float(clip_active_count) / float(clip_logged) if clip_logged else float("nan")
        )

        if mlflow is not None:
            final = {
                "train_macro_lift": float(macro_lift),
                "train_min_lift": float(min_lift),
                "train_macro_f1": float(macro_f1_mean),
                "train_min_f1": float(min_f1),
                "train_score_macro": float(train_macro),
                "train_score_min": float(train_min),
                "heldout_lift": float(held["lift"]),
                "heldout_macro_f1": float(held["macro_f1"]),
                "heldout_sdrp_top1": float(held["sdrp_top1_acc"]),
                "heldout_majority": float(held["majority_rate"]),
                "g_fit_train_pass_fold": 1.0 if g_fit_train else 0.0,
                "G_grad_spine_ok": 1.0 if step0["ok"] else 0.0,
                "G_finite": 1.0 if finite else 0.0,
                "c_drift": float(c_final - c_init),
                "clip_active_fraction": float(clip_frac)
                if math.isfinite(clip_frac)
                else float("nan"),
                "edge_type_shuffle_abs_delta": float(
                    shuffle.get("sdrp_logits_abs_delta_mean", float("nan"))
                ),
            }
            mlflow.log_metrics(final, step=steps)
            mlflow.set_tags(
                {
                    "g_fit_train_pass_fold": "PASS" if g_fit_train else "FAIL",
                    "G_finite": "PASS" if finite else "FAIL",
                }
            )
            print(
                f"[zhyp_g_fit] hold={hold} FINAL "
                f"train_score={train_macro:.4f}/{train_min:.4f} "
                f"held_f1={held['macro_f1']:.4f} held_lift={held['lift']:.4f} "
                f"g_fit_train={'PASS' if g_fit_train else 'FAIL'}"
            )
    finally:
        if run_ctx is not None:
            mlflow.end_run()

    return {
        "arm": "zhyp_g_fit",
        "card": card.card,
        "metric_family": card.metric_family,
        "seed": seed,
        "heldout": [hold],
        "train": train_tags,
        "steps": steps,
        "finite": finite,
        "G_grad_spine": bool(step0["ok"]),
        "step0_grad": step0,
        "abort": "ABORT_WIRING" if abort_wiring_c or not step0["ok"] else None,
        "moe_mode": "ablated",
        "sdrp_coeff": SDRP_COEFF,
        "dehydron_coeff": 0.0,
        "margin_coeff": 0.0,
        "tau_eval": tau_eval,
        "frontend": "cold SE(3)-lite",
        "c_init": c_init,
        "c_final": c_final,
        "c_drift": float(c_final - c_init),
        "lr_frontend": lr_frontend,
        "lr_hyperbolic": lr_hyperbolic,
        "clip_active_fraction": clip_frac,
        "train_per_structure": train_per,
        "train_macro_lift": macro_lift,
        "train_min_lift": min_lift,
        "train_macro_f1": macro_f1_mean,
        "train_min_f1": min_f1,
        "train_score_macro": train_macro,
        "train_score_min": train_min,
        "g_fit_train_pass_fold": g_fit_train,
        "heldout_metrics": {hold: held},
        "edge_type_shuffle": shuffle,
        "curve_every_10": curve,
    }


def _score_card(fold_results: list[dict[str, Any]], card: CardPaths) -> dict[str, Any]:
    finite_all = all(r["finite"] for r in fold_results)
    spine_all = all(r.get("G_grad_spine") for r in fold_results)
    c_ok = all(abs(float(r.get("c_drift", 0.0))) <= 1e-5 for r in fold_results)

    if not finite_all:
        return {"verdict": "ABORT_NONFINITE", "G_finite": False}

    if not spine_all or not c_ok:
        return {
            "verdict": "ABORT_WIRING",
            "G_finite": True,
            "G_grad_spine": spine_all,
            "c_ok": c_ok,
        }

    # G_fit_train: per-fold mean+min over 11 train structures; all folds must pass.
    g_fit_all = all(r.get("g_fit_train_pass_fold") for r in fold_results)
    if not g_fit_all:
        return {
            "verdict": "INCONCLUSIVE_UNDERFIT",
            "G_finite": True,
            "G_grad_spine": True,
            "G_fit_train": False,
            "metric_family": card.metric_family,
        }

    held_scores: list[float] = []
    for r in fold_results:
        hold = r["heldout"][0]
        m = r["heldout_metrics"][hold]
        if card.metric_family == "M2_macro_f1":
            held_scores.append(float(m["macro_f1"]))
        else:
            held_scores.append(float(m["lift"]))

    mean_held = float(np.mean(held_scores))
    ci_lb = _bootstrap_ci_lb(held_scores, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    if card.metric_family == "M2_macro_f1":
        n_above = int(sum(1 for x in held_scores if x >= card.s2_threshold))
        s1 = bool(mean_held >= card.s1_mean_min)
        s1_payload = {
            "pass": s1,
            "mean_macro_f1": mean_held,
            "bar_mean": card.s1_mean_min,
            "ci95_lb_diagnostic": ci_lb,
        }
        s2_payload = {
            "pass": bool(n_above >= card.s2_min_above),
            "n_macro_f1_ge_bar": n_above,
            "of_n": S2_OF_N,
            "bar_n": card.s2_min_above,
            "threshold": card.s2_threshold,
        }
    else:
        n_above = int(sum(1 for x in held_scores if x > card.s2_threshold))
        assert card.s1_ci_lb_min is not None
        s1 = bool(mean_held >= card.s1_mean_min and ci_lb > card.s1_ci_lb_min)
        s1_payload = {
            "pass": s1,
            "mean_lift": mean_held,
            "ci95_lb": ci_lb,
            "bar_mean": card.s1_mean_min,
            "bar_ci_lb": card.s1_ci_lb_min,
        }
        s2_payload = {
            "pass": bool(n_above >= card.s2_min_above),
            "n_lift_gt_1": n_above,
            "of_n": S2_OF_N,
            "bar": card.s2_min_above,
        }

    s2 = bool(s2_payload["pass"])
    if s1 and s2:
        verdict = "PASS_SIGNAL"
    else:
        verdict = "FAIL_NO_SIGNAL"

    return {
        "verdict": verdict,
        "G_finite": True,
        "G_grad_spine": True,
        "G_fit_train": True,
        "metric_family": card.metric_family,
        "S1": s1_payload,
        "S2": s2_payload,
        "held_scores": held_scores,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="wrap1 z_hyp G_fit (SDRP-live)")
    p.add_argument(
        "--card",
        choices=sorted(CARDS.keys()),
        default="lift",
        help="Active prereg/scoring family (default: lift for completed card)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="1 fold, 3 steps, no stamps (wiring check; bypasses READY_TO_RUN)",
    )
    p.add_argument(
        "--allow-unsealed",
        action="store_true",
        help="Allow full run before READY_TO_RUN (operator override only)",
    )
    args = p.parse_args()
    card = CARDS[args.card]

    if get_dehydron_wrap_max() != 1:
        raise SystemExit(f"wrap_max must be 1, got {get_dehydron_wrap_max()}")
    if not FOLDS_FROZEN.is_file():
        raise SystemExit(f"missing {FOLDS_FROZEN}")
    if not card.prereg.is_file():
        raise SystemExit(f"missing {card.prereg}")

    prereg = json.loads(card.prereg.read_text())
    status = str(prereg.get("status", ""))
    if not args.smoke and not args.allow_unsealed and status != "READY_TO_RUN":
        raise SystemExit(
            f"[zhyp_g_fit] STOP: prereg status={status!r}; need READY_TO_RUN "
            "(or --smoke / --allow-unsealed)."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "[zhyp_g_fit] STOP: --device cuda requested but CUDA unavailable. "
            "No CPU fallback for sealed cards."
        )

    all_tags = _all_structure_tags()
    assert len(all_tags) == 12, all_tags
    folds = all_tags if not args.smoke else all_tags[:1]
    steps = 3 if args.smoke else int(args.steps)
    if not args.smoke and int(args.steps) != STEPS:
        raise SystemExit(
            f"[zhyp_g_fit] STOP: sealed card locks steps={STEPS}; got {args.steps}. "
            "No extensions without a new signed card."
        )

    card.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[zhyp_g_fit] card={card.card} metric={card.metric_family} "
        f"device={device} steps={steps} seed={args.seed} "
        f"folds={len(folds)} sdrp_coeff={SDRP_COEFF}"
    )

    fold_results: list[dict[str, Any]] = []
    for hold in folds:
        row = run_one_fold(
            hold,
            device=device,
            steps=steps,
            seed=int(args.seed),
            no_mlflow=bool(args.no_mlflow or args.smoke),
            card=card,
        )
        fold_results.append(row)
        hold_key = hold.replace(":", "")
        out = card.out_dir / f"zhyp_seed{args.seed}_hold_{hold_key}.json"
        if not args.smoke:
            out.write_text(json.dumps(row, indent=2) + "\n")
            print(f"[zhyp_g_fit] wrote {out}")
        if row.get("abort") == "ABORT_WIRING":
            print(f"[zhyp_g_fit] ABORT_WIRING on hold={hold}; stopping remaining folds")
            break

    if args.smoke:
        print("[zhyp_g_fit] smoke done — no stamps")
        key = (
            "train_macro_f1"
            if card.metric_family == "M2_macro_f1"
            else "train_macro_lift"
        )
        print(
            json.dumps(
                {r["heldout"][0]: r.get("abort") or r.get(key) for r in fold_results},
                indent=2,
            )
        )
        return 0

    scored = _score_card(fold_results, card)
    verdict = scored["verdict"]

    stamp: dict[str, Any] = {
        "schema_version": 1,
        "gate_id": card.gate_id,
        "status": verdict,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "display_lineage": "Tokyo Eye EQU",
        "do_not_promote": True,
        "card": str(card.prereg.relative_to(REPO_ROOT)),
        "metric_family": card.metric_family,
        "seed": int(args.seed),
        "steps": STEPS,
        "folds": [r["heldout"][0] for r in fold_results],
        "frontend": "cold SE(3)-lite",
        "moe_mode": "ablated",
        "sdrp_coeff": SDRP_COEFF,
        "dehydron_coeff": 0.0,
        "margin_coeff": 0.0,
        "scoring": scored,
        "per_fold_summary": {
            r["heldout"][0]: {
                "finite": r["finite"],
                "G_grad_spine": r.get("G_grad_spine"),
                "train_macro_lift": r.get("train_macro_lift"),
                "train_min_lift": r.get("train_min_lift"),
                "train_macro_f1": r.get("train_macro_f1"),
                "train_min_f1": r.get("train_min_f1"),
                "train_score_macro": r.get("train_score_macro"),
                "train_score_min": r.get("train_score_min"),
                "g_fit_train_pass_fold": r.get("g_fit_train_pass_fold"),
                "heldout_lift": r.get("heldout_metrics", {})
                .get(r["heldout"][0], {})
                .get("lift"),
                "heldout_macro_f1": r.get("heldout_metrics", {})
                .get(r["heldout"][0], {})
                .get("macro_f1"),
                "c_drift": r.get("c_drift"),
                "clip_active_fraction": r.get("clip_active_fraction"),
            }
            for r in fold_results
        },
        "script_sha256": _sha256(Path(__file__)),
        "prereg_sha256": _sha256(card.prereg),
        "folds_frozen_sha256": _sha256(FOLDS_FROZEN),
        "git_commit": prereg.get("pins", {}).get("git_commit"),
        "not_claims": prereg.get("not_claims", []),
        "signed": "auto from wrap1_zhyp_g_fit.py",
    }
    card.result.write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"[zhyp_g_fit] RESULT={verdict} wrote {card.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
