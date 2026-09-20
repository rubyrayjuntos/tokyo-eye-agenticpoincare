#!/usr/bin/env python3
"""MoE / ε card runner — A0 note + R (mean-over-12) + A (ablated) arms.

Card: ``data/gates/tokyo_eye_equ_moe_eps_card_prereg.json`` (operator-approved).
Pins locked; no coefficient / schedule retune.

  R: each epoch averages grads over all 12 Stage-A-12 structures into ONE
     optimizer step; pooled node-weighted eval MoE utilization. Seeds 0/1/2.
     PASS only if every post-floor row (25) passes locked moe_eval_util on all seeds.
  A: ``moe_mode=ablated`` finite train + exact load [1,0,0,0]; no performance bar.

Also mirrored at ``/tmp/moe_eps_card_run.py`` for the card's next_command path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow

from experiments.training.v8.run_v8_experiment import (
    SDRP_LOSS_COEFF,
    build_system,
    run_epoch,
    run_epoch_mean_structures,
)
from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
    EpsilonGreedySchedule,
    GumbelTemperatureSchedule,
    PoincareDiagnosticsEngine,
)
from science.tokyo_eye.v8.equiformer_frontend import (
    DEFAULT_WEIGHT_MAP,
    build_param_groups,
    load_weight_map,
)
from science.tokyo_eye.v8.freeze_reconciliation import (
    CANONICAL_MLFLOW_EXPERIMENT,
    freeze_reconciliation_mlflow_params,
)
from science.tokyo_eye.v8.loader import TokyoEyeCuratedDataset
from science.tokyo_eye.v8.r0_r5_graph import get_dehydron_wrap_max

MANIFEST_PATH = REPO_ROOT / "manifests" / "v8_stage_a_small_v1.json"
PDB_DIR = REPO_ROOT / "pdb_cache"
GRAPH_CACHE_DIR = PDB_DIR / "v8_graph_cache"
CARD_PATH = REPO_ROOT / "data" / "gates" / "tokyo_eye_equ_moe_eps_card_prereg.json"
# Prefer data/gates (new files ok). Fall back: /tmp. Avoid gates_agent —
# prior root-owned writes there caused Assistant PermissionError and skipped A.
STAMP_CANDIDATES = [
    REPO_ROOT / "data" / "gates",
    Path("/tmp"),
]
POST_FLOOR_HOLD_EPOCHS = 24
TOTAL_EPOCHS = 60


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_card() -> dict:
    return json.loads(CARD_PATH.read_text())


def _write_json(name: str, obj: dict) -> Path:
    """Write stamp to first writable candidate; raise if all fail."""
    payload = json.dumps(obj, indent=2) + "\n"
    errors: list[str] = []
    for parent in STAMP_CANDIDATES:
        path = parent / name
        try:
            parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
            return path
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    raise PermissionError(
        "could not write stamp; tried: " + "; ".join(errors)
    )


def run_a0() -> Path:
    """Sidecar attribution: NEVER_LIVE (no compute)."""
    card = _load_card()
    stamp = {
        "schema_version": 1,
        "gate_id": "tokyo_eye_equ_moe_eps_a0_attribution",
        "status": "RECORDED",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "parent_stamp": card["evidence"]["stamp"],
        "parent_sha256": card["evidence"]["stamp_sha256"],
        "card": str(CARD_PATH.relative_to(REPO_ROOT)),
        "relabel": {
            "from": "COLLAPSED_POST_FLOOR",
            "to": "NEVER_LIVE",
            "rationale": card["evidence"]["verified_findings"]["F1_never_live_in_eval"],
        },
        "do_not_promote": True,
        "diagnostic": True,
    }
    return _write_json("tokyo_eye_equ_moe_eps_a0_attribution.json", stamp)


def _build_live_system(cfg: dict, device: torch.device, moe_mode: str = "live"):
    system, load_info = build_system(
        cfg, equiformer_ckpt=None, device=device, freeze_backbone=False
    )
    system.set_moe_mode(moe_mode)
    groups = build_param_groups(
        system.frontend,
        system.spine,
        lr_backbone=float(cfg["lr_backbone"]),
        lr_hyperbolic=float(cfg["lr_hyperbolic"]),
        freeze_backbone=False,
    )
    optimizer = torch.optim.Adam(groups)
    return system, load_info, optimizer


def _schedules(cfg: dict, total_epochs: int):
    radius = CurriculumRadiusController(
        float(cfg["tau_start"]), float(cfg["tau_end"]), total_epochs
    )
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        total_epochs,
        schedule=str(cfg.get("gumbel_schedule", "exponential")),
        alpha=cfg.get("gumbel_exp_alpha"),
        half_epochs=int(cfg.get("gumbel_exp_half_epochs", 12)),
    )
    eps_sched = EpsilonGreedySchedule(
        float(cfg.get("eps_start", 0.20)),
        float(cfg.get("eps_end", 0.0)),
        half_epochs=int(cfg.get("eps_half_epochs", 12)),
    )
    return radius, gumbel, eps_sched, PoincareDiagnosticsEngine()


def run_r_seed(seed: int, device: torch.device) -> dict:
    card = _load_card()
    pins = card["pins_at_draft"]
    torch.manual_seed(seed)
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    # Pin coefficients from card (must match weight map; no drift).
    assert float(cfg["cv_coeff"]) == float(pins["cv_coeff"])
    assert float(cfg["moe_quota_coeff"]) == float(pins["moe_quota_coeff"])
    assert float(cfg["eval_proxy_quota_coeff"]) == float(pins["eval_proxy_quota_coeff"])

    system, load_info, optimizer = _build_live_system(cfg, device, moe_mode="live")
    dataset = TokyoEyeCuratedDataset(
        manifest_path=MANIFEST_PATH,
        pdb_dir=PDB_DIR,
        use_graph_cache=True,
        graph_cache_dir=GRAPH_CACHE_DIR,
    )
    n = len(dataset)
    assert n == 12, f"expected 12 structures, got {n}"
    radius, gumbel, eps_sched, diagnostics = _schedules(cfg, TOTAL_EPOCHS)
    telemetry = dict(cfg.get("telemetry") or {})

    epoch_rows: list[dict] = []
    floor_epoch: int | None = None
    post_floor_rows: list[dict] = []

    run_name = f"tokyo_eye_equ_moe_eps_card_R_seed{seed}"
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(CANONICAL_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        rec = freeze_reconciliation_mlflow_params()
        mlflow.set_tags(
            {
                "gate_id": "tokyo_eye_equ_moe_eps_card_R",
                "arm": "R",
                "moe_mode": "live",
                "moe_claim": "live",
                "diagnostic": "true",
                "do_not_promote": "true",
                "seed": str(seed),
                "grad_aggregation": "mean_over_12",
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "cv_coeff": pins["cv_coeff"],
                "moe_quota_coeff": pins["moe_quota_coeff"],
                "eval_proxy_quota_coeff": pins["eval_proxy_quota_coeff"],
                "eps_start": pins["eps_start"],
                "eps_end": pins["eps_end"],
                "eps_half_epochs": pins["eps_half_epochs"],
                "dehydron_wrap_max": get_dehydron_wrap_max(),
                "addendum_id": rec["addendum_id"],
                "backbone_mode": load_info.get("backbone_mode"),
            }
        )

        for epoch in range(TOTAL_EPOCHS):
            batches = [dataset.get_on_device(i, device) for i in range(n)]
            eps_t = eps_sched.epsilon(epoch)
            eps_at_floor = bool(eps_t <= eps_sched.eps_end + 1e-12)
            metrics = run_epoch_mean_structures(
                system,
                optimizer,
                batches,
                epoch=epoch,
                radius=radius,
                gumbel=gumbel,
                diagnostics=diagnostics,
                cv_coeff=float(pins["cv_coeff"]),
                moe_quota_coeff=float(pins["moe_quota_coeff"]),
                eval_proxy_quota_coeff=float(pins["eval_proxy_quota_coeff"]),
                explore_epsilon=eps_t,
                eps_at_floor=eps_at_floor,
                telemetry=telemetry,
                sdrp_coeff=float(SDRP_LOSS_COEFF),
            )
            if metrics.get("nan_abort", 0.0) >= 1.0:
                return {
                    "seed": seed,
                    "verdict": "ABORT_NONFINITE",
                    "floor_epoch": floor_epoch,
                    "epoch_log": epoch_rows,
                    "mlflow_run_id": mlflow.active_run().info.run_id,
                }

            if eps_t <= eps_sched.eps_end and floor_epoch is None:
                floor_epoch = epoch

            row = {
                "epoch": epoch,
                "eps_t": eps_t,
                "loss_total": metrics.get("loss_total"),
                "eval_moe_load": [metrics.get(f"eval_moe_load_e{i}") for i in range(4)],
                "eval_moe_n_alive": metrics.get("eval_moe_n_alive"),
                "eval_moe_load_min": metrics.get("eval_moe_load_min"),
                "eval_moe_entropy_norm": metrics.get("eval_moe_entropy_norm"),
                "eval_moe_liveness_pass": bool(
                    metrics.get("eval_moe_liveness_pass", 0.0) >= 1.0
                ),
            }
            epoch_rows.append(row)
            if floor_epoch is not None and epoch >= floor_epoch:
                post_floor_rows.append(row)

            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                    mlflow.log_metric(k, float(v), step=epoch)

            print(
                f"[R seed={seed}] epoch={epoch} eps={eps_t:.4f} "
                f"loss={row['loss_total']:.4f} "
                f"n_alive={row['eval_moe_n_alive']} "
                f"min_load={row['eval_moe_load_min']:.3f} "
                f"Hnorm={row['eval_moe_entropy_norm']:.3f} "
                f"pass={row['eval_moe_liveness_pass']}"
            )

            if floor_epoch is not None and epoch >= floor_epoch + POST_FLOOR_HOLD_EPOCHS:
                break

        # Verdict table from card.
        pre = None
        if floor_epoch is not None and floor_epoch > 0:
            pre = epoch_rows[floor_epoch - 1]
        held = (
            floor_epoch is not None
            and len(post_floor_rows) >= POST_FLOOR_HOLD_EPOCHS + 1
            and all(r["eval_moe_liveness_pass"] for r in post_floor_rows)
        )
        if held:
            verdict = "HELD"
        elif pre is not None and not pre["eval_moe_liveness_pass"]:
            verdict = "NEVER_LIVE"
        elif pre is not None and pre["eval_moe_liveness_pass"]:
            verdict = "COLLAPSED_POST_FLOOR"
        else:
            verdict = "NEVER_LIVE"

        result = {
            "seed": seed,
            "verdict": verdict,
            "floor_epoch": floor_epoch,
            "post_floor_hold_epochs_observed": len(post_floor_rows),
            "final_eval_load": epoch_rows[-1]["eval_moe_load"] if epoch_rows else None,
            "epoch_log": epoch_rows,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        mlflow.set_tag("verdict", verdict)
        mlflow.log_dict(result, f"moe_eps_card_R_seed{seed}.json")
        return result


def run_a_seed(seed: int, device: torch.device, epochs: int = 38) -> dict:
    """Ablated arm: finite train, load [1,0,0,0], zero MoE terms."""
    torch.manual_seed(seed)
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    system, load_info, optimizer = _build_live_system(cfg, device, moe_mode="ablated")
    dataset = TokyoEyeCuratedDataset(
        manifest_path=MANIFEST_PATH,
        pdb_dir=PDB_DIR,
        use_graph_cache=True,
        graph_cache_dir=GRAPH_CACHE_DIR,
    )
    radius, gumbel, eps_sched, diagnostics = _schedules(cfg, epochs)
    telemetry = dict(cfg.get("telemetry") or {})

    rows = []
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(CANONICAL_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"tokyo_eye_equ_moe_eps_card_A_seed{seed}"):
        mlflow.set_tags(
            {
                "arm": "A",
                "moe_mode": "ablated",
                "moe_claim": "none",
                "diagnostic": "true",
                "do_not_promote": "true",
                "decay_floor_hold": "N/A_ABLATED",
            }
        )
        for epoch in range(epochs):
            batch = dataset.get_on_device(epoch % len(dataset), device)
            eps_t = eps_sched.epsilon(epoch)
            metrics = run_epoch(
                system,
                optimizer,
                batch,
                epoch=epoch,
                radius=radius,
                gumbel=gumbel,
                diagnostics=diagnostics,
                cv_coeff=float(cfg["cv_coeff"]),
                moe_quota_coeff=float(cfg["moe_quota_coeff"]),
                eval_proxy_quota_coeff=float(cfg["eval_proxy_quota_coeff"]),
                explore_epsilon=0.0,  # ablated ignores eps; keep zero
                eps_at_floor=bool(eps_t <= eps_sched.eps_end + 1e-12),
                telemetry=telemetry,
                sdrp_coeff=float(SDRP_LOSS_COEFF),
            )
            if metrics.get("nan_abort", 0.0) >= 1.0:
                return {"seed": seed, "pass": False, "reason": "ABORT_NONFINITE"}
            load = [metrics.get(f"eval_moe_load_e{i}") for i in range(4)]
            rows.append({"epoch": epoch, "eval_moe_load": load, "loss": metrics["loss_total"]})
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                    mlflow.log_metric(k, float(v), step=epoch)
        ok_load = all(
            r["eval_moe_load"] is not None
            and abs(r["eval_moe_load"][0] - 1.0) < 1e-6
            and all(abs(x) < 1e-6 for x in r["eval_moe_load"][1:])
            for r in rows
        )
        result = {
            "seed": seed,
            "pass": bool(ok_load),
            "decay_floor_hold": "N/A_ABLATED",
            "backbone_mode": load_info.get("backbone_mode"),
            "n_epochs": len(rows),
            "final_eval_load": rows[-1]["eval_moe_load"] if rows else None,
        }
        mlflow.log_dict(result, f"moe_eps_card_A_seed{seed}.json")
        return result


def main() -> int:
    p = argparse.ArgumentParser(description="MoE/ε card runner")
    p.add_argument("--arm", choices=["A0", "R", "A", "all"], default="A0")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--a-epochs", type=int, default=38)
    args = p.parse_args()
    device = torch.device(args.device)

    # Keep /tmp mirror for the card's next_command.
    tmp = Path("/tmp/moe_eps_card_run.py")
    try:
        shutil.copy2(Path(__file__), tmp)
    except OSError:
        pass

    summary: dict = {
        "card": str(CARD_PATH),
        "operator_decisions_applied": {
            "P1_hold_window": "every post-floor row (25)",
            "seeds": "all must pass",
            "grad_aggregation": "mean_over_12",
        },
        "code_sha256": {
            "moe.py": _sha256(REPO_ROOT / "science/tokyo_eye/v8/moe.py"),
            "model.py": _sha256(REPO_ROOT / "science/tokyo_eye/v8/model.py"),
            "run_v8_experiment.py": _sha256(
                REPO_ROOT / "experiments/training/v8/run_v8_experiment.py"
            ),
            "moe_eps_card_run.py": _sha256(Path(__file__)),
        },
        "arms": {},
    }

    if args.arm in ("A0", "all"):
        path = run_a0()
        summary["arms"]["A0"] = {"status": "RECORDED", "path": str(path)}
        print(f"[A0] wrote {path}")

    if args.arm in ("R", "all"):
        r_results = [run_r_seed(s, device) for s in args.seeds]
        all_held = all(r["verdict"] == "HELD" for r in r_results)
        summary["arms"]["R"] = {
            "pass": all_held,
            "seeds": r_results,
            "fallback_to_A": not all_held,
        }
        out = _write_json(
            "tokyo_eye_equ_moe_eps_card_R_result.json", summary["arms"]["R"]
        )
        print(f"[R] pass={all_held} → {out}")

    if args.arm in ("A", "all") or (
        args.arm == "R" and summary.get("arms", {}).get("R", {}).get("fallback_to_A")
    ):
        # Auto-fallback when R fails, or explicit A.
        a_results = [run_a_seed(s, device, epochs=args.a_epochs) for s in args.seeds[:1]]
        summary["arms"]["A"] = {
            "pass": all(r.get("pass") for r in a_results),
            "seeds": a_results,
            "decay_floor_hold": "N/A_ABLATED",
            "moe_mode": "ablated",
            "moe_claim": "none",
        }
        out = _write_json(
            "tokyo_eye_equ_moe_eps_card_A_result.json", summary["arms"]["A"]
        )
        print(f"[A] pass={summary['arms']['A']['pass']} → {out}")

    out_sum = _write_json("tokyo_eye_equ_moe_eps_card_run_summary.json", summary)
    print(f"summary → {out_sum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
