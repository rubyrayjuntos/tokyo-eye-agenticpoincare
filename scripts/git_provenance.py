"""Git provenance for diagnostic / card runners.

MLflow's automatic ``mlflow.source.git.commit`` tag records HEAD without checking
that the tracked code matches it (run 325ed1a2 ran an uncommitted script while
tagged ``addbcab``). Runners call :func:`require_clean_code` at startup and log
:func:`provenance_params` with the run so every run is tied to the code that produced it.

Only code paths count as dirty; result JSONs under ``data/gates/`` and markdown
notes are ignored so a baseline rerun does not block the next run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

CODE_PATHS = ("scripts", "science", "tests", "experiments")
_EXCLUDE_MD = ":(exclude,glob)**/*.md"


def _git(repo_root: Path, *args: str) -> str:
    # rstrip newlines only: porcelain lines start with a significant space (" M path")
    return subprocess.check_output(["git", *args], cwd=str(repo_root), text=True).rstrip("\n")


def dirty_code_files(repo_root: Path) -> list[str]:
    """Modified or staged tracked files under CODE_PATHS (untracked files are ignored)."""
    out = _git(repo_root, "status", "--porcelain", "--untracked-files=no",
               "--", *CODE_PATHS, _EXCLUDE_MD)
    return [line[3:] for line in out.splitlines()]


def provenance_params(repo_root: Path) -> dict[str, str]:
    """Flat string params for ``mlflow.log_params``."""
    dirty = dirty_code_files(repo_root)
    return {
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": str(bool(dirty)),
        "git_dirty_files": ",".join(dirty)[:500],
    }


def require_clean_code(repo_root: Path, allow_dirty: bool) -> dict[str, str]:
    """Abort when tracked code differs from HEAD unless ``allow_dirty``."""
    prov = provenance_params(repo_root)
    if prov["git_dirty"] == "True" and not allow_dirty:
        raise SystemExit(
            "[git_provenance] STOP: tracked code differs from HEAD; commit or stash, "
            f"or pass --allow-dirty. dirty files: {prov['git_dirty_files']}"
        )
    if prov["git_dirty"] == "True":
        print(f"[git_provenance] WARNING: running with --allow-dirty on {prov['git_dirty_files']}")
    return prov


def log_dirty_diff(mlflow: Any, repo_root: Path) -> None:
    """Attach ``git diff HEAD`` for the code paths when the tree is dirty."""
    if not dirty_code_files(repo_root):
        return
    diff = _git(repo_root, "diff", "HEAD", "--", *CODE_PATHS, _EXCLUDE_MD)
    mlflow.log_text(diff + "\n", "git_diff_HEAD.patch")
