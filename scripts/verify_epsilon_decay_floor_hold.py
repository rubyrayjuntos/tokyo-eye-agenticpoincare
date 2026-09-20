"""§2.6 check #3 — the epsilon-greedy decay-floor hold (currently PENDING).

Freeze reconciliation addendum §2.6 names three required checks before a real
eps_start=0.20 run is trusted:
  1. eps=0 inert            -> tests/v8/test_epsilon_greedy_moe.py (PASS, exists)
  2. eps=1 forced-uniform    -> tests/v8/test_epsilon_greedy_moe.py (PASS, exists)
  3. decay-floor hold        -> NOT covered by any existing test (this script)

Check #3, verbatim from the addendum: "after eps(t) first reaches eps_end (0),
continue training and log eval-mode moe_eval_utilization / load / n_alive on
the epochs immediately after the floor ... If eval collapses back to a
single-expert monopoly once the training wheels are off, epsilon-greedy
bought temporary appearance of health rather than a durable fix; that outcome
FAILS this AMEND's acceptance even if mid-schedule loads looked healthy."

Scope note: this trains the isolated TopologyAwareHardMoE router (matching
the standard the eps=0 / eps=1 unit tests already use — a fixed synthetic
batch, real gradient steps, no full spine), not the full
TokyoEyesHyperbolicV8 spine with real Stage-A-12 structures. That's a
deliberate scope cut for speed. The synthetic task below gives genuine
specialization pressure (four latent node clusters, one "natural" expert
each) so an eval-time monopoly is a real possible outcome, not a strawman.
Before this clears the AMEND as *binding* evidence, Ray should re-run the
same eval-mode-post-floor protocol on the real spine + corpus per the
addendum's stated evidence standard (same discipline as the wrap-threshold
and lr_hyperbolic decisions, which were both corpus/run-log grounded, not
isolated-module synthetic).

Writes data/gates/tokyo_eye_equ_epsilon_decay_floor_hold.json and logs an
MLflow run under experiment id 14 (tokyoeye/equiformer-v3-moe/geometric/
hyperbolic-spine) tagged do_not_promote=true / diagnostic=true — this is a
router stress test, not a spine checkpoint candidate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import torch
import torch.nn.functional as F

import mlflow

from science.tokyo_eye.v8.engine import EpsilonGreedySchedule
from science.tokyo_eye.v8.moe import TopologyAwareHardMoE
from science.tokyo_eye.v8.moe_eval_gates import eval_moe_utilization

HIDDEN_DIM = 128
GATE_HIDDEN = 16
NUM_EXPERTS = 4
N_NODES = 128
TOTAL_EPOCHS = 60
HALF_EPOCHS = 12
POST_FLOOR_HOLD_EPOCHS = 24  # epochs to watch *after* eps first hits floor
CV_COEFF = 1.0
QUOTA_COEFF = 5.0
HINGE_COEFF = 1.0
LR = 3e-3

MLFLOW_EXPERIMENT = "tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine"
MLFLOW_RUN_NAME = "tokyo_eye_equ_epsilon_decay_floor_hold"
GATE_ID = "tokyo_eye_equ_epsilon_decay_floor_hold"


def _clustered_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """N nodes in 4 latent clusters — a task that rewards 1 expert/cluster."""
    g = torch.Generator().manual_seed(seed)
    per = N_NODES // NUM_EXPERTS
    cluster_id = torch.arange(NUM_EXPERTS).repeat_interleave(per)
    centers = torch.eye(NUM_EXPERTS) * 0.06
    z = centers[cluster_id] + torch.randn(N_NODES, NUM_EXPERTS, generator=g) * 0.01
    if HIDDEN_DIM > NUM_EXPERTS:
        pad = torch.randn(N_NODES, HIDDEN_DIM - NUM_EXPERTS, generator=g) * 0.01
        z = torch.cat([z, pad], dim=-1)
    src = torch.randint(0, N_NODES, (4 * N_NODES,), generator=g)
    dst = torch.randint(0, N_NODES, (4 * N_NODES,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return z, edge_index, cluster_id


def task_loss(out: torch.Tensor, cluster_id: torch.Tensor) -> torch.Tensor:
    """Push each cluster's output norm along its own axis — cheap proxy for
    a real per-cluster supervised target (stands in for dehydron/theme AUPRC
    pressure without needing the real loader)."""
    target = F.one_hot(cluster_id, NUM_EXPERTS).float() * 0.05
    if out.shape[-1] > NUM_EXPERTS:
        target = F.pad(target, (0, out.shape[-1] - NUM_EXPERTS))
    return F.mse_loss(out, target)


def main() -> int:
    torch.manual_seed(0)
    moe = TopologyAwareHardMoE(
        HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        gate_hidden=GATE_HIDDEN,
        temperature=1.0,
        explore_epsilon=0.20,
    )
    opt = torch.optim.Adam(moe.parameters(), lr=LR)
    sched = EpsilonGreedySchedule(0.20, 0.0, half_epochs=HALF_EPOCHS)

    z, edge_index, cluster_id = _clustered_batch(seed=1)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    epoch_rows: list[dict] = []
    floor_epoch: int | None = None
    post_floor_rows: list[dict] = []

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.set_tags(
            {
                "gate_id": GATE_ID,
                "addendum_clause": "2.6_check_3_decay_floor_hold",
                "diagnostic": "true",
                "do_not_promote": "true",
                "scope": "isolated_moe_router_synthetic_task",
            }
        )
        mlflow.log_params(
            {
                "hidden_dim": HIDDEN_DIM,
                "num_experts": NUM_EXPERTS,
                "eps_start": sched.eps_start,
                "eps_end": sched.eps_end,
                "half_epochs": HALF_EPOCHS,
                "total_epochs": TOTAL_EPOCHS,
                "post_floor_hold_epochs": POST_FLOOR_HOLD_EPOCHS,
                "lr": LR,
            }
        )

        for epoch in range(TOTAL_EPOCHS):
            eps_t = sched.epsilon(epoch)
            moe.set_explore_epsilon(eps_t)

            moe.train()
            opt.zero_grad()
            out, aux = moe(z, edge_index)
            loss_task = task_loss(out, cluster_id)
            loss = (
                loss_task
                + CV_COEFF * aux["cv_loss"]
                + QUOTA_COEFF * aux["quota_loss"]
                + HINGE_COEFF * aux["majority_hinge_loss"]
            )
            loss.backward()
            opt.step()

            moe.eval()
            with torch.no_grad():
                _, eval_aux = moe(z, edge_index)
            eval_report = eval_moe_utilization(eval_aux["routing"])

            if eps_t <= sched.eps_end and floor_epoch is None:
                floor_epoch = epoch

            row = {
                "epoch": epoch,
                "eps_t": eps_t,
                "train_loss": float(loss.detach()),
                "train_load": [float(x) for x in aux["load"].tolist()],
                "eval_load": eval_report["load"],
                "eval_n_alive": eval_report["gates"]["moe_eval_n_alive"]["value"],
                "eval_min_load": eval_report["gates"]["moe_eval_min_load"]["value"],
                "eval_entropy_norm": eval_report["gates"]["moe_eval_entropy_norm"]["value"],
                "eval_all_pass": eval_report["all_pass"],
            }
            epoch_rows.append(row)
            if floor_epoch is not None and epoch >= floor_epoch:
                post_floor_rows.append(row)

            mlflow.log_metric("eps_t", eps_t, step=epoch)
            mlflow.log_metric("train_loss", row["train_loss"], step=epoch)
            mlflow.log_metric("eval_n_alive", row["eval_n_alive"], step=epoch)
            mlflow.log_metric("eval_min_load", row["eval_min_load"], step=epoch)
            mlflow.log_metric("eval_entropy_norm", row["eval_entropy_norm"], step=epoch)
            mlflow.log_metric("eval_all_pass", 1.0 if row["eval_all_pass"] else 0.0, step=epoch)

            if floor_epoch is not None and epoch >= floor_epoch + POST_FLOOR_HOLD_EPOCHS:
                break

        # Decisive criterion (addendum §2.6, verbatim): ANY post-floor eval
        # collapse fails the AMEND's acceptance, even if mid-schedule looked fine.
        held = floor_epoch is not None and all(r["eval_all_pass"] for r in post_floor_rows)
        collapsed_at = next(
            (r["epoch"] for r in post_floor_rows if not r["eval_all_pass"]), None
        )

        stamp = {
            "gate_id": GATE_ID,
            "display_lineage": "Tokyo Eye EQU",
            "status": "HOLD_CONFIRMED" if held else "COLLAPSED_POST_FLOOR",
            "execution_state": "DIAGNOSTIC_COMPLETE",
            "scope": "isolated TopologyAwareHardMoE router, synthetic 4-cluster task",
            "addendum_clause": "freeze_reconciliation_addendum_2.6_check_3",
            "floor_epoch": floor_epoch,
            "post_floor_hold_epochs_observed": len(post_floor_rows),
            "collapsed_at_epoch": collapsed_at,
            "final_eval_load": epoch_rows[-1]["eval_load"] if epoch_rows else None,
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
                "Router-level synthetic evidence only. Per addendum §2.6/§2.7 "
                "evidence standard, treat as PENDING for real sign-off until "
                "reproduced on the full spine against Stage-A-12 real batches."
            ),
            "epoch_log": epoch_rows,
        }
        out_path = Path(
            "/workspace/data/gates/tokyo_eye_equ_epsilon_decay_floor_hold.json"
        )
        out_path.write_text(json.dumps(stamp, indent=2))
        mlflow.log_artifact(str(out_path))
        mlflow.set_tag("final_status", stamp["status"])

        print(f"epsilon decay-floor hold: {stamp['status']}")
        print(f"  eps first reached floor at epoch {floor_epoch}")
        print(f"  watched {len(post_floor_rows)} epochs after floor")
        if collapsed_at is not None:
            print(f"  COLLAPSE at epoch {collapsed_at}: {stamp['gates']['decay_floor_hold']['detail']}")
        print(f"  final eval load: {stamp['final_eval_load']}")
        print(f"  stamp written: {out_path}")

    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
