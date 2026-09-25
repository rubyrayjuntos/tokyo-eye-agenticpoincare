# Git provenance guard: implemented (uncommitted) 2026-09-25

Supersedes the earlier draft. Applied by the MLflow assistant to the working tree; NOT committed.

## Why

Run 325ed1a2 recorded `mlflow.source.git.commit = addbcab...`, but ran a script that was not tracked at that commit. MLflow's automatic tag reports HEAD without checking the tree. Run 9352afc6 records 158491f, and no run records whether the tree was clean.

## What changed

- **New `scripts/git_provenance.py`** (shared by both runners):
  - `require_clean_code(repo_root, allow_dirty)`: exits if tracked files under `scripts/ science/ tests/ experiments/` differ from HEAD, unless `--allow-dirty`.
  - `provenance_params(repo_root)`: flat params `git_commit`, `git_dirty`, `git_dirty_files` (first 500 chars).
  - `log_dirty_diff(mlflow, repo_root)`: logs `git_diff_HEAD.patch` as an artifact when the tree is dirty.
  - Ignored on purpose: untracked files, `data/gates/` result JSONs (a baseline rerun rewrites a tracked one), and `*.md` notes.
- **`scripts/verify_grad_flow_decoupled.py`**: `--allow-dirty` flag, guard right after `parse_args`, provenance params and diff artifact in the MLflow block.
- **`scripts/wrap1_zhyp_m2_pool_decoupled.py`** (12-fold runner): same flag, guard in `main()`, provenance params and diff artifact in the per-fold MLflow block.

The guard runs in every mode, including `--no-mlflow` and `--smoke`.

## Verified

- Temporary git repo: clean, modified `data/gates` JSON, modified `.md` note and untracked new script all report clean; a modified tracked script reports dirty, aborts, and with `--allow-dirty` proceeds and produces the diff artifact.
- A bug found and fixed during testing: `.strip()` ate the leading space of the first porcelain line and clipped the first filename letter.
- Real tree: `verify_grad_flow_decoupled.py --mode verify-train-mode --no-mlflow` aborted on the dirty tree (naming both edited scripts); with `--allow-dirty` it printed `{"live": 14, "total": 14, "pass": true}`.

## Not yet verified

- A clean-tree run passing without `--allow-dirty` (needs a commit first).
- The MLflow side: that a run shows `git_commit`, `git_dirty=False`, empty `git_dirty_files`. Check after the first committed-tree run.
- The 12-fold runner's guard was compile-checked and reviewed, not executed.

## Next

1. Review `git diff` and commit `scripts/git_provenance.py` plus the two script edits.
2. Run the `--dehydron-coeff` arm on the committed tree, then `--mech-lr-scale`.
3. For each new run, confirm `git_commit` matches `git rev-parse HEAD` and `git_dirty=False`.
