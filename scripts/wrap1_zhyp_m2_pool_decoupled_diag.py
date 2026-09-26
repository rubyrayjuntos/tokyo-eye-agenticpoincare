"""Diagnostic held-out comparison for the decoupled M2 pool card (NOT a sealed run).

Runs scripts/wrap1_zhyp_m2_pool_decoupled.py on a SUBSET of folds with a different
``DEHYDRON_COEFF`` without editing that sealed script (its sha256 and the card pin stay valid).
It rebinds three module-level names of the sealed runner before calling its ``main()``:

  DEHYDRON_COEFF                 -> --dehydron-coeff
  RESULT_DIR                     -> data/gates/diag_heldout_dcoef<c>/   (the sealed run resumes from
                                    data/gates/wrap1_zhyp_m2_pool_decoupled/<arm>/seed*_hold_*.json,
                                    so diagnostic fold results must never be written there)
  CANONICAL_MLFLOW_EXPERIMENT    -> diag/heldout-dcoef<c>

A subset of folds never stamps the card (the sealed runner prints "incomplete fold set"), and this
wrapper additionally refuses to run all 12 folds.

OPEN PRECONDITION for the eventual sealed 400-step launch: the verify_grad_flow gates ("min over all
steps > threshold") fail transiently at 400 steps in the near-fit regime (attn_layers, moe); a
phase-aware rule is still owed. Not needed for this comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
for p in (str(REPO_ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import wrap1_zhyp_m2_pool_decoupled as M  # noqa: E402
import wrap1_zhyp_g_fit as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dehydron-coeff", type=float, required=True)
    ap.add_argument("--folds", default="", help="comma-separated hold tags (fewer than all 12); required unless --smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 3 steps, no MLflow, writes nothing")
    ap.add_argument("--no-mlflow", action="store_true")
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
