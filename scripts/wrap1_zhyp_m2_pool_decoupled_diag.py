"""Diagnostic held-out comparison for the decoupled M2 pool card (NOT a sealed run).

Runs scripts/wrap1_zhyp_m2_pool_decoupled.py on a SUBSET of folds with a different
``DEHYDRON_COEFF`` without editing that sealed script (its sha256 and the card pin stay valid).
It rebinds three module-level names of the sealed runner before calling its ``main()``:

  DEHYDRON_COEFF                 -> --dehydron-coeff
  RESULT_DIR                     -> data/gates/diag_heldout_dcoef<c>/   (the sealed run resumes from
                                    data/gates/wrap1_zhyp_m2_pool_decoupled/<arm>/seed*_hold_*.json,
                                    so diagnostic fold results must never be written there)
  CANONICAL_MLFLOW_EXPERIMENT    -> diag/heldout-dcoef<c>

WARNING (corrected 2026-09-26): the sealed runner scores and STAMPS the card whenever every REQUESTED fold
finished (it only prints "incomplete fold set" when fewer folds ran than were requested), so even a
single-fold request writes data/gates/tokyo_eye_equ_wrap1_zhyp_m2_pool_<arm>_result.json. This wrapper
therefore redirects the card's result path into the diagnostic results dir, and also refuses to run all 12 folds.

OPEN PRECONDITION for the eventual sealed 400-step launch: the verify_grad_flow gates ("min over all
steps > threshold") fail transiently at 400 steps in the near-fit regime (attn_layers, moe); a
phase-aware rule is still owed. Not needed for this comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
for p in (str(REPO_ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import wrap1_zhyp_m2_pool_decoupled as M  # noqa: E402
import wrap1_zhyp_g_fit as G  # noqa: E402

_ORIG_METRICS = G._sdrp_structure_metrics  # captured before the confusion logger replaces it


def _install_confusion_logger(out_file: Path) -> None:
    """Wrap G._sdrp_structure_metrics so every scored structure also saves its confusion matrix and
    per-node predictions (the sealed runner stores only summary metrics). Never raises into the run."""
    orig = G._sdrp_structure_metrics
    records: list[dict] = []

    def _shim(system, batch, *, tau_ceiling):
        res = orig(system, batch, tau_ceiling=tau_ceiling)
        try:
            was = system.training
            system.eval()
            with torch.no_grad():
                out = system(batch["x"], batch["edge_index"], batch["edge_type"],
                             tau_ceiling=tau_ceiling, chem=batch.get("gate_chem"))
                logits = out["sdrp_logits"]
                pred = logits.argmax(dim=-1).cpu()
                y = batch["sdrp_target"].long().cpu()
                k = int(logits.shape[-1])
            if was:
                system.train()
            cm = torch.zeros(k, k, dtype=torch.long)
            cm.index_put_((y, pred), torch.ones_like(y), accumulate=True)
            present = [c for c in range(k) if int((y == c).sum()) > 0]
            f1s = []
            for c in present:
                tp = int(cm[c, c]); fp = int(cm[:, c].sum()) - tp; fn = int(cm[c, :].sum()) - tp
                f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
            recomputed = sum(f1s) / len(f1s) if f1s else float("nan")
            records.append({
                "n": int(y.numel()), "macro_f1_logged": res.get("macro_f1"), "macro_f1_recomputed": recomputed,
                "confusion_true_rows_by_pred_cols": cm.tolist(), "y": y.tolist(), "pred": pred.tolist(),
            })
            out_file.write_text(json.dumps(records))
            if abs(recomputed - float(res.get("macro_f1", recomputed))) > 1e-9:
                print(f"[diag] WARNING: recomputed macro-F1 {recomputed} != logged {res.get('macro_f1')}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- diagnostics must never break the run
            print(f"[diag] confusion logger error (ignored): {type(exc).__name__}: {exc}", flush=True)
        return res

    G._sdrp_structure_metrics = _shim


def _diag_tags(tags: dict) -> dict:
    """The sealed runner tags its runs diagnostic=false, card=m2_pool_<arm>, gate_id=..._prereg. Rewrite those so a
    diagnostic run can never be mistaken for a sealed-card run by tag (the experiment name differs too)."""
    out = dict(tags)
    if "diagnostic" in out:
        out["diagnostic"] = "true"
        out["not_a_card_result"] = "true"
        out["diag_wrapper"] = "wrap1_zhyp_m2_pool_decoupled_diag.py"
    if "card" in out and not str(out["card"]).startswith("DIAG_"):
        out["card"] = "DIAG_" + str(out["card"])
    if "gate_id" in out and not str(out["gate_id"]).startswith("NOT_A_GATE"):
        out["gate_id"] = "NOT_A_GATE_diag_of_" + str(out["gate_id"])
    return out


def _install_tag_rewrite() -> None:
    orig_set_tags = mlflow.set_tags

    def _set_tags(tags, *args, **kwargs):
        return orig_set_tags(_diag_tags(tags), *args, **kwargs)

    mlflow.set_tags = _set_tags


def _install_interval_eval(hold: str, steps_sched: list[int], out_file: Path, smoke: bool) -> None:
    """Evaluate all 12 structures at chosen training steps WITHOUT perturbing training.

    Hooks G.run_step_sdrp_only (called once per step with epoch=step). An eval at step s runs BEFORE step s, i.e. on
    the model after s updates (s=0 is the untrained model). Step 400 = the runner's own end-of-run numbers, so it is
    not repeated. Evals use the same fixed tau (cfg['tau_end']) as the runner's end-of-run scoring.
    Measured on the real model: an eval forward changes the CUDA RNG state (CPU state is untouched), so CPU, CUDA,
    numpy and python RNG states are saved and restored around every eval; train mode is restored afterwards
    (eval->train cycle verified to bring back all 14 backbone regularization modules)."""
    orig_step = G.run_step_sdrp_only
    sched = sorted(set(int(s) for s in steps_sched))
    cfg = M.load_weight_map(M.REPO_ROOT / M.DEFAULT_WEIGHT_MAP)
    tau_eval = float(cfg["tau_end"])
    records: list[dict] = []
    cache: dict = {}

    def _batches(system):
        if not cache:
            dev = next(system.parameters()).device
            cache["hold"] = M._lb(hold, dev)
            cache["train"] = {t: M._lb(t, dev) for t in G._all_structure_tags() if t != hold}
        return cache["hold"], cache["train"]

    def _do_eval(system, step):
        cpu = torch.get_rng_state()
        cu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        nps, pys = np.random.get_state(), random.getstate()
        was = system.training
        try:
            hold_b, train_b = _batches(system)
            held = _ORIG_METRICS(system, hold_b, tau_ceiling=tau_eval)
            train_m = {t: _ORIG_METRICS(system, b, tau_ceiling=tau_eval) for t, b in train_b.items()}
            system.eval()
            with torch.no_grad():
                out = system(hold_b["x"], hold_b["edge_index"], hold_b["edge_type"],
                             tau_ceiling=tau_eval, chem=hold_b.get("gate_chem"))
                pred = out["sdrp_logits"].argmax(dim=-1).cpu()
            y = hold_b["sdrp_target"].long().cpu()
            k = int(out["sdrp_logits"].shape[-1])
            cm = torch.zeros(k, k, dtype=torch.long)
            cm.index_put_((y, pred), torch.ones_like(y), accumulate=True)
            f1s = [m["macro_f1"] for m in train_m.values()]
            rec = {"step": step, "held": held, "train": train_m,
                   "held_confusion_true_rows_by_pred_cols": cm.tolist(), "held_pred": pred.tolist(),
                   "train_macro_f1_mean": float(np.mean(f1s)), "train_macro_f1_min": float(np.min(f1s))}
            records.append(rec)
            out_file.write_text(json.dumps({"hold": hold, "tau_eval": tau_eval, "records": records}))
            if mlflow.active_run() is not None:
                payload = {"interval_heldout_macro_f1": held["macro_f1"], "interval_heldout_top1": held["sdrp_top1_acc"],
                           "interval_heldout_lift": held["lift"],
                           "interval_train_macro_f1_mean": rec["train_macro_f1_mean"],
                           "interval_train_macro_f1_min": rec["train_macro_f1_min"]}
                for tg, m in train_m.items():
                    payload[f"interval_train_f1_{tg.replace(':', '')}"] = m["macro_f1"]
                mlflow.log_metrics(payload, step=step)
            print(f"[diag] interval eval step={step} held_f1={held['macro_f1']:.4f} "
                  f"train_f1_mean={rec['train_macro_f1_mean']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- an eval problem must never break training
            print(f"[diag] interval eval ERROR at step {step} (ignored): {type(exc).__name__}: {exc}", flush=True)
        finally:
            torch.set_rng_state(cpu)
            if cu is not None:
                torch.cuda.set_rng_state_all(cu)
            np.random.set_state(nps)
            random.setstate(pys)
            if was:
                system.train()

    def _shim(system, optimizer, batches, **kw):
        if kw.get("epoch") in sched:
            _do_eval(system, int(kw["epoch"]))
        return orig_step(system, optimizer, batches, **kw)

    G.run_step_sdrp_only = _shim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dehydron-coeff", type=float, required=True)
    ap.add_argument("--folds", default="", help="comma-separated hold tags (fewer than all 12); required unless --smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 3 steps, no MLflow, writes nothing")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--interval-evals", default="", help="comma-separated training steps at which to evaluate all 12 structures (single fold only)")
    ap.add_argument("--dense-log", action="store_true", help="log every step for the whole run (sets DENSE_LOG_STEPS=400)")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    all_tags = G._all_structure_tags()
    if not a.smoke:
        folds = [t for t in a.folds.split(",") if t]
        if not folds:
            raise SystemExit("[diag] STOP: --folds is required (comma-separated hold tags) unless --smoke")
        bad = [t for t in folds if t not in all_tags]
        if bad:
            raise SystemExit(f"[diag] STOP: unknown fold tags {bad}; valid: {all_tags}")
        if len(set(folds)) >= len(all_tags):
            raise SystemExit("[diag] STOP: refusing to run all 12 folds from the diagnostic wrapper")

    tag = f"dcoef{a.dehydron_coeff:g}"
    M.DEHYDRON_COEFF = float(a.dehydron_coeff)
    M.RESULT_DIR = REPO_ROOT / "data" / "gates" / f"diag_heldout_{tag}"
    M.CANONICAL_MLFLOW_EXPERIMENT = f"diag/heldout-{tag}"

    conf_dir = Path("/tmp") if a.smoke else M.RESULT_DIR
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_name = "confusions_smoke.json" if a.smoke else f"confusions_seed{a.seed}_{'_'.join(folds).replace(':', '')}.json"
    _install_confusion_logger(conf_dir / conf_name)

    if a.dense_log:
        M.DENSE_LOG_STEPS = int(M.STEPS)  # per-step logging throughout; LOG_EVERY (clip_active_fraction definition) unchanged
    if a.interval_evals:
        sched = [int(s) for s in a.interval_evals.split(",") if s.strip() != ""]
        if not a.smoke and len(set(folds)) != 1:
            raise SystemExit("[diag] STOP: --interval-evals supports exactly one fold")
        hold_tag = "1MBN:A" if a.smoke else folds[0]
        iv_name = "interval_smoke.json" if a.smoke else f"interval_evals_seed{a.seed}_{hold_tag.replace(':', '')}.json"
        _install_interval_eval(hold_tag, sched, conf_dir / iv_name, a.smoke)
        print(f"[diag] interval evals at steps {sorted(set(sched))} (before-step semantics; step {M.STEPS} = runner's end-of-run numbers)", flush=True)

    _install_tag_rewrite()

    orig_card_paths = M._card_paths

    def _diag_card_paths(arm):  # CardPaths is a frozen dataclass; never let a diagnostic write the sealed card's result
        cp = orig_card_paths(arm)
        redirected = dataclasses.replace(cp, result=M.RESULT_DIR / f"NOT_A_CARD_RESULT_{tag}.json")
        print(f"[diag] card.result redirected: {cp.result.relative_to(REPO_ROOT)} -> "
              f"{redirected.result.relative_to(REPO_ROOT)}", flush=True)
        return redirected

    M._card_paths = _diag_card_paths

    orig = M.run_one_fold

    def _shim(*args, **kwargs):  # show what the sealed runner actually sees at call time
        print(f"[diag] run_one_fold sees DEHYDRON_COEFF={M.DEHYDRON_COEFF} RESULT_DIR={M.RESULT_DIR} "
              f"experiment={M.CANONICAL_MLFLOW_EXPERIMENT}", flush=True)
        return orig(*args, **kwargs)

    M.run_one_fold = _shim
    print("[diag] NOT a sealed run: subset of folds, results in "
          f"{M.RESULT_DIR.relative_to(REPO_ROOT)}, card never stamped. "
          "Open precondition for the sealed 400-step launch: phase-aware gate rule (see module docstring).", flush=True)

    argv = ["wrap1_zhyp_m2_pool_decoupled.py", "--device", a.device, "--seed", str(a.seed)]
    if a.smoke:
        argv.append("--smoke")
    else:
        argv += ["--folds", ",".join(folds)]
    if a.no_mlflow:
        argv.append("--no-mlflow")
    if a.allow_dirty:
        argv.append("--allow-dirty")
    sys.argv = argv
    return M.main()


if __name__ == "__main__":
    raise SystemExit(main())
