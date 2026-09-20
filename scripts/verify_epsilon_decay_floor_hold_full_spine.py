"""§2.6 check #3, full-spine extension — same protocol, real trunk + Stage-A-12.

Extends ``scripts/verify_epsilon_decay_floor_hold.py`` (that script's isolated
``TopologyAwareHardMoE``-only synthetic diagnostic is unchanged and still the
canonical stamp at ``data/gates/tokyo_eye_equ_epsilon_decay_floor_hold.json``).
Per CLAUDE.md: "Needed: same post-floor eval_moe_utilization protocol on real
``TokyoEyesHyperbolicV8`` + Stage-A-12 batches via the training harness."

This script does exactly that and nothing more:
  * builds the real spine via ``experiments/training/v8/run_v8_experiment
    .build_system`` (stub-but-live SE(3)-lite frontend, cold random-init,
    addendum §2.3 default — no MPtrj checkpoint required/loaded)
  * trains/evals with that harness's own ``run_epoch`` (not a
    reimplementation of the loss/eval-routing call sites)
  * cycles round-robin over the real 12-structure Stage-A-12 manifest
    (``manifests/v8_stage_a_small_v1.json``, the ``enabled: true`` subset of
    the v8 Stage A small corpus — 1MBN, 1LYZ, 1BG1, 1F88, 2Z6H, 1HHP, 1TEN,
    1UBQ, 1TIM, 4OBE, 1IVO, 2SHP), one structure per epoch, matching the
    harness's own "one graph per step" convention (Mode B round-robin)
  * uses the same ``EpsilonGreedySchedule``/eps_start=0.20 -> eps_end=0.0 /
    half_epochs=12 config as the weight-map SSOT
    (``science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json``)
  * applies the same decisive criterion, verbatim from addendum §2.6: ANY
    post-floor eval-mode collapse (``eval_moe_liveness_pass`` false, i.e. the
    ``moe_eval_utilization`` SSOT gate) fails the hold, even if pre-floor
    epochs looked healthy.

Logs an MLflow run under the canonical full-stack experiment
(``tokyoeye/equiformer-v3-moe/geometric/full-stack``, per
``freeze_reconciliation.CANONICAL_MLFLOW_EXPERIMENT``) tagged
``diagnostic=true`` / ``do_not_promote=true`` — this is not a train/promote
candidate, it is the binding evidence-standard reproduction of the router
diagnostic per addendum §2.6/§2.7.

Writes a **sibling** stamp,
``data/gates/tokyo_eye_equ_epsilon_decay_floor_hold_full_spine.json`` — does
NOT overwrite ``tokyo_eye_equ_epsilon_decay_floor_hold.json`` (that stamp's
scope is explicitly the isolated router; CLAUDE.md allows a sibling stamp
with explicit scope in lieu of overwriting).

Placement note: written under /tmp/ because this account has no write access
to /workspace/scripts/ (verified: `touch` there fails with EPERM; world-write
only extends to /workspace/data/gates/). Host agent: fold this into
scripts/verify_epsilon_decay_floor_hold.py as the ``--full-spine`` code path
described in that file's module docstring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import torch

import mlflow

from experiments.training.v8.run_v8_experiment import (
    SDRP_LOSS_COEFF,
    build_system,
    run_epoch,
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

REPO_ROOT = Path("/workspace")
MANIFEST_PATH = REPO_ROOT / "manifests" / "v8_stage_a_small_v1.json"
PDB_DIR = REPO_ROOT / "pdb_cache"
GRAPH_CACHE_DIR = PDB_DIR / "v8_graph_cache"

TOTAL_EPOCHS = 60
POST_FLOOR_HOLD_EPOCHS = 24  # same window as the isolated-router diagnostic

MLFLOW_EXPERIMENT = CANONICAL_MLFLOW_EXPERIMENT
MLFLOW_RUN_NAME = "tokyo_eye_equ_epsilon_decay_floor_hold_full_spine"
GATE_ID = "tokyo_eye_equ_epsilon_decay_floor_hold_full_spine"
STAMP_PATH = REPO_ROOT / "data" / "gates" / f"{GATE_ID}.json"


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    # addendum §2.3: cold random-init default — no MPtrj ckpt, freeze_backbone=False
    # (live_se3_lite), matching build_system(...) with equiformer_ckpt=None.
    system, load_info = build_system(
        cfg, equiformer_ckpt=None, device=device, freeze_backbone=False
    )

    dataset = TokyoEyeCuratedDataset(
        manifest_path=MANIFEST_PATH,
        pdb_dir=PDB_DIR,
        use_graph_cache=True,
        graph_cache_dir=GRAPH_CACHE_DIR,
    )
    assert dataset.mode == "manifest"
    n_structures = len(dataset)
    structure_tags = [
        f"{e['pdb_id']}:{e['chain']}" for e in dataset.entries
    ]

    groups = build_param_groups(
        system.frontend,
        system.spine,
        lr_backbone=float(cfg["lr_backbone"]),
        lr_hyperbolic=float(cfg["lr_hyperbolic"]),
        freeze_backbone=False,
    )
    optimizer = torch.optim.Adam(groups)

    radius = CurriculumRadiusController(
        float(cfg["tau_start"]), float(cfg["tau_end"]), TOTAL_EPOCHS
    )
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        TOTAL_EPOCHS,
        schedule=str(cfg.get("gumbel_schedule", "exponential")),
        alpha=cfg.get("gumbel_exp_alpha"),
        half_epochs=int(cfg.get("gumbel_exp_half_epochs", 12)),
    )
    eps_sched = EpsilonGreedySchedule(
        float(cfg.get("eps_start", 0.20)),
        float(cfg.get("eps_end", 0.0)),
        half_epochs=int(cfg.get("eps_half_epochs", 12)),
    )
    diagnostics = PoincareDiagnosticsEngine()
    telemetry = dict(cfg.get("telemetry") or {})

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    epoch_rows: list[dict] = []
    floor_epoch: int | None = None
    post_floor_rows: list[dict] = []

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        rec = freeze_reconciliation_mlflow_params()
        mlflow.set_tags(
            {
                "gate_id": GATE_ID,
                "addendum_id": rec["addendum_id"],
                "addendum_clause": "freeze_reconciliation_addendum_2.6_check_3",
                "diagnostic": "true",
                "do_not_promote": "true",
                "scope": "full_spine_stage_a_12",
            }
        )
        mlflow.log_params(
            {
                "hidden_dim": cfg["hidden_dim"],
                "scalar_dim": cfg["scalar_dim"],
                "vector_dim": cfg["vector_dim"],
                "num_experts": 4,
                "eps_start": eps_sched.eps_start,
                "eps_end": eps_sched.eps_end,
                "eps_half_epochs": eps_sched.half_epochs,
                "total_epochs": TOTAL_EPOCHS,
                "post_floor_hold_epochs": POST_FLOOR_HOLD_EPOCHS,
                "lr_backbone": cfg["lr_backbone"],
                "lr_hyperbolic": cfg["lr_hyperbolic"],
                "cv_coeff": cfg["cv_coeff"],
                "moe_quota_coeff": cfg["moe_quota_coeff"],
                "eval_proxy_quota_coeff": cfg["eval_proxy_quota_coeff"],
                "backbone_mode": load_info.get("backbone_mode"),
                "equiformer_mode": load_info.get("mode"),
                "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
                "n_structures": n_structures,
                "structures": ",".join(structure_tags),
                "dehydron_wrap_max": get_dehydron_wrap_max(),
            }
        )

        for epoch in range(TOTAL_EPOCHS):
            batch = dataset.get_on_device(epoch % n_structures, device)
            eps_t = eps_sched.epsilon(epoch)
            eps_at_floor = bool(eps_t <= eps_sched.eps_end + 1e-12)

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
                explore_epsilon=eps_t,
                eps_at_floor=eps_at_floor,
                telemetry=telemetry,
                sdrp_coeff=float(SDRP_LOSS_COEFF),
            )

            if metrics.get("nan_abort", 0.0) >= 1.0:
                print(f"[full_spine] ABORT non-finite loss/grad at epoch={epoch}")
                break

            if eps_t <= eps_sched.eps_end and floor_epoch is None:
                floor_epoch = epoch

            row = {
                "epoch": epoch,
                "pdb_id": batch.get("pdb_id"),
                "chain": batch.get("chain"),
                "num_nodes": metrics.get("num_nodes"),
                "eps_t": eps_t,
                "loss_total": metrics.get("loss_total"),
                "val_dehydron_auprc": metrics.get("val_dehydron_auprc"),
                "eval_moe_load": [
                    metrics.get(f"eval_moe_load_e{i}") for i in range(4)
                ],
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
                f"[full_spine] epoch={epoch} structure={row['pdb_id']}:{row['chain']} "
                f"eps={eps_t:.4f} loss={row['loss_total']:.4f} "
                f"eval_n_alive={row['eval_moe_n_alive']} "
                f"eval_load_min={row['eval_moe_load_min']:.3f} "
                f"eval_entropy_norm={row['eval_moe_entropy_norm']:.3f} "
                f"liveness_pass={row['eval_moe_liveness_pass']}"
            )

            if floor_epoch is not None and epoch >= floor_epoch + POST_FLOOR_HOLD_EPOCHS:
                break

        held = floor_epoch is not None and all(
            r["eval_moe_liveness_pass"] for r in post_floor_rows
        )
        collapsed_at = next(
            (r["epoch"] for r in post_floor_rows if not r["eval_moe_liveness_pass"]),
            None,
        )

        stamp = {
            "gate_id": GATE_ID,
            "display_lineage": "Tokyo Eye EQU",
            "status": "HOLD_CONFIRMED" if held else "COLLAPSED_POST_FLOOR",
            "execution_state": "DIAGNOSTIC_COMPLETE",
            "scope": (
                "full spine (TokyoEyesHyperbolicV8 + stub-but-live SE(3)-lite "
                "frontend, cold random-init), real 12-structure Stage-A-12 "
                "manifest round-robin (manifests/v8_stage_a_small_v1.json)"
            ),
            "addendum_clause": "freeze_reconciliation_addendum_2.6_check_3",
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "structures": structure_tags,
            "dehydron_wrap_max": get_dehydron_wrap_max(),
            "floor_epoch": floor_epoch,
            "post_floor_hold_epochs_observed": len(post_floor_rows),
            "collapsed_at_epoch": collapsed_at,
            "final_eval_load": epoch_rows[-1]["eval_moe_load"] if epoch_rows else None,
            "gates": {
                "decay_floor_hold": {
                    "pass": held,
                    "detail": (
                        "eval-mode utilization stayed healthy for "
                        f"{len(post_floor_rows)} epochs after eps floor (epoch {floor_epoch})"
                        if held
                        else f"eval-mode utilization collapsed at epoch {collapsed_at}, "
                        f"{collapsed_at - floor_epoch if collapsed_at is not None and floor_epoch is not None else '?'} "
                        "epochs after eps reached floor"
                    ),
                }
            },
            "note": (
                "Full-spine, real-corpus reproduction of the isolated-router "
                "decay-floor-hold diagnostic. diagnostic=true / do_not_promote=true "
                "on the MLflow run — this does NOT flip the wrap-threshold or any "
                "train/promote gate; it satisfies the addendum §2.6/§2.7 evidence "
                "standard for the decay-floor check specifically. Does not overwrite "
                "tokyo_eye_equ_epsilon_decay_floor_hold.json (isolated-router scope)."
            ),
            "mlflow_experiment": MLFLOW_EXPERIMENT,
            "mlflow_run_name": MLFLOW_RUN_NAME,
            "epoch_log": epoch_rows,
        }
        STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        STAMP_PATH.write_text(json.dumps(stamp, indent=2))
        mlflow.log_artifact(str(STAMP_PATH))
        mlflow.set_tag("final_status", stamp["status"])

        print(f"\nfull-spine epsilon decay-floor hold: {stamp['status']}")
        print(f"  structures: {', '.join(structure_tags)}")
        print(f"  eps first reached floor at epoch {floor_epoch}")
        print(f"  watched {len(post_floor_rows)} epochs after floor")
        if collapsed_at is not None:
            print(f"  COLLAPSE at epoch {collapsed_at}: {stamp['gates']['decay_floor_hold']['detail']}")
        print(f"  final eval load: {stamp['final_eval_load']}")
        print(f"  stamp written: {STAMP_PATH}")

    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
