"""wrap=1 biology re-score (SDRP + dehydron) on Stage-A-12, real spine harness.

Host agent: land as scripts/verify_wrap1_biology_rescore.py (scripts/ is EPERM for this account).

Protocol (diagnostic only; no pre-registered pass/fail card exists for this check):
  * build_system / run_epoch from experiments/training/v8/run_v8_experiment (unchanged),
    cold random-init stub-but-live SE(3)-lite frontend, cfg from the weight-map SSOT,
    schedules exactly as verify_epsilon_decay_floor_hold_full_spine.py (no retune).
  * manifests/v8_stage_a_small_v1.json, enabled:true only (12), round-robin one
    structure per epoch, 60 epochs (5 passes). wrap asserted == 1 (frozen SSOT).
  * After training: eval-mode score of every structure (no grad) -> dehydron
    (AUPRC, ROC-AUC, prevalence, lift) and SDRP 5-way (acc, macro-F1, majority acc).
  * Controls, because dehydron_labels and sdrp_target are deterministic functions of
    batch["edge_type"], which is ALSO a model input (frontend + attention) and the
    source of gate_chem:
      - init:   same seed, untrained spine, same eval (learned-vs-init delta)
      - blind:  second full train+eval arm with edge_type collapsed to one constant
                relation and gate_chem dropped (type information removed; edge
                topology still present, so this is an upper bound on non-leak signal)
  * Scores are in-sample (train == eval structures; no held-out split exists in
    Stage-A-12 round-robin). Not a generalization number.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

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
from science.tokyo_eye.v8.freeze_reconciliation import CANONICAL_MLFLOW_EXPERIMENT
from science.tokyo_eye.v8.loader import TokyoEyeCuratedDataset
from science.tokyo_eye.v8.metrics import binary_auprc
from science.tokyo_eye.v8.r0_r5_graph import (
    DEHYDRON_WRAP_MAX,
    R1_HBOND,
    R2_DEHYDRON,
    R5_LOCAL_NEIGHBORHOOD,
    get_dehydron_wrap_max,
)

REPO_ROOT = Path("/workspace")
MANIFEST_PATH = REPO_ROOT / "manifests" / "v8_stage_a_small_v1.json"
PDB_DIR = REPO_ROOT / "pdb_cache"
GRAPH_CACHE_DIR = PDB_DIR / "v8_graph_cache"

TOTAL_EPOCHS = 60
SEED = 0
GATE_ID = "tokyo_eye_equ_wrap1_biology_rescore"
STAMP_PATH = REPO_ROOT / "data" / "gates" / f"{GATE_ID}.json"
MLFLOW_EXPERIMENT = CANONICAL_MLFLOW_EXPERIMENT


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _blind(batch: dict) -> dict:
    b = dict(batch)
    b["edge_type"] = torch.full_like(batch["edge_type"], R5_LOCAL_NEIGHBORHOOD)
    b.pop("gate_chem", None)
    return b


def _hbond_incident(batch: dict) -> np.ndarray:
    n = int(batch["num_nodes"])
    ei = batch["edge_index"].cpu().numpy()
    et = batch["edge_type"].cpu().numpy()
    m = (et == R1_HBOND) | (et == R2_DEHYDRON)
    out = np.zeros(n, dtype=np.float64)
    if m.any():
        out[ei[0, m]] = 1.0
        out[ei[1, m]] = 1.0
    return out


def _labels_equal_r2_incidence(batch: dict) -> bool:
    n = int(batch["num_nodes"])
    ei = batch["edge_index"].cpu().numpy()
    et = batch["edge_type"].cpu().numpy()
    m = et == R2_DEHYDRON
    inc = np.zeros(n, dtype=np.float32)
    if m.any():
        inc[ei[0, m]] = 1.0
        inc[ei[1, m]] = 1.0
    return bool(np.array_equal(inc, batch["dehydron_labels"].cpu().numpy()))


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    present = sorted(set(int(v) for v in y.tolist()))
    return float(f1_score(y, p, labels=present, average="macro", zero_division=0))


def _score_struct(out: dict, batch: dict) -> dict:
    y = batch["dehydron_labels"].cpu().numpy().astype(np.float64)
    s = torch.sigmoid(out["mechanism_score"]).detach().cpu().numpy().astype(np.float64)
    prev = float(y.mean())
    auprc = binary_auprc(torch.from_numpy(s), torch.from_numpy(y))
    roc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    t = batch["sdrp_target"].cpu().numpy().astype(np.int64)
    p = out["sdrp_logits"].detach().argmax(dim=-1).cpu().numpy().astype(np.int64)
    maj = float(np.bincount(t, minlength=5).max() / len(t))
    hb = _hbond_incident(batch)
    hb_auprc = binary_auprc(torch.from_numpy(hb), torch.from_numpy(y))
    return {
        "n": int(len(y)),
        "dehydron_prevalence": prev,
        "dehydron_auprc": auprc,
        "dehydron_lift": auprc / prev if prev > 0 else float("nan"),
        "dehydron_roc_auc": roc,
        "hbond_incidence_baseline_auprc": hb_auprc,
        "sdrp_acc": float((p == t).mean()),
        "sdrp_macro_f1": _macro_f1(t, p),
        "sdrp_majority_acc": maj,
        "sdrp_target_hist": np.bincount(t, minlength=5).tolist(),
        "sdrp_pred_hist": np.bincount(p, minlength=5).tolist(),
        "_y": y, "_s": s, "_t": t, "_p": p,
    }


@torch.no_grad()
def eval_all(system, dataset, device, tau, blind: bool) -> dict:
    system.eval()
    per, ys, ss, ts, ps = {}, [], [], [], []
    for i in range(len(dataset)):
        batch = dataset.get_on_device(i, device)
        tag = f"{batch['pdb_id']}:{batch['chain']}"
        b = _blind(batch) if blind else batch
        out = system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=tau, chem=b.get("gate_chem"))
        r = _score_struct(out, batch)
        ys.append(r.pop("_y")); ss.append(r.pop("_s")); ts.append(r.pop("_t")); ps.append(r.pop("_p"))
        per[tag] = r
    y, s, t, p = map(np.concatenate, (ys, ss, ts, ps))
    prev = float(y.mean())
    auprc = binary_auprc(torch.from_numpy(s), torch.from_numpy(y))
    pooled = {
        "n": int(len(y)),
        "dehydron_prevalence": prev,
        "dehydron_auprc": auprc,
        "dehydron_lift": auprc / prev,
        "dehydron_roc_auc": float(roc_auc_score(y, s)),
        "sdrp_acc": float((p == t).mean()),
        "sdrp_macro_f1": _macro_f1(t, p),
        "sdrp_majority_acc": float(np.bincount(t, minlength=5).max() / len(t)),
        "sdrp_target_hist": np.bincount(t, minlength=5).tolist(),
        "sdrp_pred_hist": np.bincount(p, minlength=5).tolist(),
    }
    mean = {
        k: float(np.nanmean([v[k] for v in per.values()]))
        for k in ("dehydron_auprc", "dehydron_lift", "dehydron_roc_auc", "dehydron_prevalence",
                  "hbond_incidence_baseline_auprc", "sdrp_acc", "sdrp_macro_f1", "sdrp_majority_acc")
    }
    return {"pooled": pooled, "mean_per_structure": mean, "per_structure": per}


def build(cfg, device):
    torch.manual_seed(SEED)
    system, load_info = build_system(cfg, equiformer_ckpt=None, device=device, freeze_backbone=False)
    return system, load_info


def train_arm(cfg, dataset, device, blind: bool, arm: str, init_eval: dict | None):
    system, load_info = build(cfg, device)
    radius = CurriculumRadiusController(float(cfg["tau_start"]), float(cfg["tau_end"]), TOTAL_EPOCHS)
    tau_final = radius.tau_ceiling(TOTAL_EPOCHS - 1)
    if init_eval is None:
        init_eval = eval_all(system, dataset, device, tau_final, blind)
        system, load_info = build(cfg, device)  # re-seed: identical init for training
    groups = build_param_groups(
        system.frontend, system.spine,
        lr_backbone=float(cfg["lr_backbone"]), lr_hyperbolic=float(cfg["lr_hyperbolic"]),
        freeze_backbone=False,
    )
    optimizer = torch.optim.Adam(groups)
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]), float(cfg["gumbel_tau_end"]), TOTAL_EPOCHS,
        schedule=str(cfg.get("gumbel_schedule", "exponential")),
        alpha=cfg.get("gumbel_exp_alpha"), half_epochs=int(cfg.get("gumbel_exp_half_epochs", 12)),
    )
    eps_sched = EpsilonGreedySchedule(
        float(cfg.get("eps_start", 0.20)), float(cfg.get("eps_end", 0.0)),
        half_epochs=int(cfg.get("eps_half_epochs", 12)),
    )
    diagnostics = PoincareDiagnosticsEngine()
    telemetry = dict(cfg.get("telemetry") or {})
    n = len(dataset)
    log = []
    for epoch in range(TOTAL_EPOCHS):
        batch = dataset.get_on_device(epoch % n, device)
        if blind:
            batch = _blind(batch)
        eps_t = eps_sched.epsilon(epoch)
        m = run_epoch(
            system, optimizer, batch, epoch=epoch, radius=radius, gumbel=gumbel,
            diagnostics=diagnostics, cv_coeff=float(cfg["cv_coeff"]),
            moe_quota_coeff=float(cfg["moe_quota_coeff"]),
            eval_proxy_quota_coeff=float(cfg["eval_proxy_quota_coeff"]),
            explore_epsilon=eps_t, eps_at_floor=bool(eps_t <= eps_sched.eps_end + 1e-12),
            telemetry=telemetry, sdrp_coeff=float(SDRP_LOSS_COEFF),
        )
        if m.get("nan_abort", 0.0) >= 1.0:
            print(f"[{arm}] ABORT non-finite at epoch={epoch}")
            break
        for k, v in m.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                mlflow.log_metric(f"{arm}/{k}", float(v), step=epoch)
        log.append({
            "epoch": epoch, "structure": f"{batch['pdb_id']}:{batch['chain']}",
            "loss_dehydron": m["loss_dehydron"], "loss_sdrp": m["loss_sdrp"],
            "train_mode_auprc": m["val_dehydron_auprc"], "dehydron_frac": m["dehydron_frac"],
            "eval_moe_liveness_pass": bool(m.get("eval_moe_liveness_pass", 0.0) >= 1.0),
        })
        print(f"[{arm}] epoch={epoch} {log[-1]['structure']} ld={m['loss_dehydron']:.3f} "
              f"ls={m['loss_sdrp']:.3f} auprc={m['val_dehydron_auprc']:.3f} "
              f"live={log[-1]['eval_moe_liveness_pass']}", flush=True)
    final = eval_all(system, dataset, device, tau_final, blind)
    return {"init_eval": init_eval, "final_eval": final, "epoch_log": log,
            "tau_final": tau_final, "load_info": load_info}


def main() -> int:
    assert get_dehydron_wrap_max() == 1 == int(DEHYDRON_WRAP_MAX), "wrap SSOT must be 1"
    device = torch.device("cpu")
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    dataset = TokyoEyeCuratedDataset(
        manifest_path=MANIFEST_PATH, pdb_dir=PDB_DIR,
        use_graph_cache=True, graph_cache_dir=GRAPH_CACHE_DIR,
    )
    assert dataset.mode == "manifest" and len(dataset) == 12, len(dataset)
    tags = [f"{e['pdb_id']}:{e['chain']}" for e in dataset.entries]
    label_is_r2 = {
        f"{dataset.get_on_device(i, device)['pdb_id']}": _labels_equal_r2_incidence(dataset.get_on_device(i, device))
        for i in range(len(dataset))
    }

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=GATE_ID) as run:
        mlflow.set_tags({
            "gate_id": GATE_ID, "diagnostic": "true", "do_not_promote": "true",
            "scope": "full_spine_stage_a_12_wrap1_biology_rescore",
        })
        mlflow.log_params({
            "dehydron_wrap_max": get_dehydron_wrap_max(), "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "n_structures": len(dataset), "structures": ",".join(tags), "total_epochs": TOTAL_EPOCHS,
            "seed": SEED, "eval_scope": "in_sample", "arms": "typed,blind,init_controls",
            "hidden_dim": cfg["hidden_dim"], "lr_backbone": cfg["lr_backbone"],
            "lr_hyperbolic": cfg["lr_hyperbolic"], "eps_start": cfg.get("eps_start", 0.20),
            "eps_half_epochs": cfg.get("eps_half_epochs", 12),
        })
        typed = train_arm(cfg, dataset, device, blind=False, arm="typed", init_eval=None)
        blind = train_arm(cfg, dataset, device, blind=True, arm="blind", init_eval=None)

        def flat(arm, res):
            for phase in ("init_eval", "final_eval"):
                for k, v in res[phase]["pooled"].items():
                    if isinstance(v, float) and math.isfinite(v):
                        mlflow.log_metric(f"{arm}/{phase}/pooled/{k}", v)
                for k, v in res[phase]["mean_per_structure"].items():
                    if math.isfinite(v):
                        mlflow.log_metric(f"{arm}/{phase}/mean_struct/{k}", v)
        flat("typed", typed); flat("blind", blind)

        def beats(res):
            f = res["final_eval"]["pooled"]
            return {
                "dehydron_auprc_gt_prevalence": bool(f["dehydron_auprc"] > f["dehydron_prevalence"]),
                "sdrp_acc_gt_majority": bool(f["sdrp_acc"] > f["sdrp_majority_acc"]),
                "dehydron_auprc_gain_vs_init": f["dehydron_auprc"] - res["init_eval"]["pooled"]["dehydron_auprc"],
                "sdrp_acc_gain_vs_init": f["sdrp_acc"] - res["init_eval"]["pooled"]["sdrp_acc"],
            }

        stamp = {
            "gate_id": GATE_ID,
            "display_lineage": "Tokyo Eye EQU",
            "status": "DIAGNOSTIC_COMPLETE",
            "diagnostic": True, "do_not_promote": True,
            "promote_eligible": False,
            "scope": ("full spine (TokyoEyesHyperbolicV8 + stub-but-live SE(3)-lite frontend, cold random-init), "
                      "Stage-A-12 enabled:true, round-robin 60 epochs, in-sample eval-mode scoring"),
            "dehydron_wrap_max": get_dehydron_wrap_max(),
            "wrap_threshold_amend_stamp": "data/gates/tokyo_eye_equ_wrap_threshold.json",
            "wrap19_numbers_cited": False,
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "structures": tags,
            "seed": SEED, "total_epochs": TOTAL_EPOCHS,
            "leakage_audit": {
                "dehydron_labels_equal_r2_edge_incidence": label_is_r2,
                "statement": ("dehydron_labels == nodes incident to an R2 edge and sdrp_target is an argmax over "
                              "edge-type counts; both are functions of batch['edge_type'], which is fed to the "
                              "frontend and attention layers (and gate_chem). The typed arm is therefore a "
                              "label-leak-confounded score. The blind arm removes type information (constant "
                              "relation, gate_chem dropped) but keeps edge topology."),
            },
            "typed": {**{k: v for k, v in typed.items() if k != "load_info"}, "reads": beats(typed)},
            "blind": {**{k: v for k, v in blind.items() if k != "load_info"}, "reads": beats(blind)},
            "known_caveats": [
                "in-sample: train structures == scored structures; not generalization",
                "router is the parked COLLAPSED_POST_FLOOR eps config (no retune); eval_moe_liveness_pass logged per epoch",
                "no pre-registered biology pass threshold for this check; governed theme_biology thresholds are a different (held-out, 6-theme) protocol and are not applied",
            ],
            "supersedes_candidate": "data/gates/tokyo_eye_equ_wrap1_biology_rescore_full_spine.json (no backing MLflow run found; no baselines; left untouched)",
            "mlflow_experiment": MLFLOW_EXPERIMENT,
            "mlflow_run_name": GATE_ID,
            "mlflow_run_id": run.info.run_id,
        }
        STAMP_PATH.write_text(json.dumps(_clean(stamp), indent=2))
        mlflow.log_artifact(str(STAMP_PATH))
        for arm, res in (("typed", typed), ("blind", blind)):
            p = res["final_eval"]["pooled"]
            print(f"\n{arm}: dehydron AUPRC={p['dehydron_auprc']:.4f} (prev={p['dehydron_prevalence']:.4f}, "
                  f"lift={p['dehydron_lift']:.2f}, ROC={p['dehydron_roc_auc']:.4f}) | "
                  f"SDRP acc={p['sdrp_acc']:.4f} macroF1={p['sdrp_macro_f1']:.4f} maj={p['sdrp_majority_acc']:.4f}")
        print(f"\nstamp: {STAMP_PATH}\nrun: {run.info.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
