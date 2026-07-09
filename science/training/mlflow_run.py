"""MLflow run helpers — checkpoint resolution and lineage for v6 training."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from science.contracts.model_registry import resolve_checkpoint_file

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DEFAULT_EXPERIMENT = "tokyo-eyes-v6"
CHECKPOINT_ARTIFACT = "checkpoints/v6_best.pt"


def _import_mlflow() -> Any:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "false")
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required. Install with: uv pip install -e '.[science,training]'"
        ) from exc
    return mlflow


def _normalize_checkpoint_path(path: str | Path) -> str:
    return Path(path).as_posix()


def _path_lookup_candidates(path: str | Path) -> list[str]:
    """Paths that may appear in MLflow tags (container vs host-relative)."""
    p = Path(path)
    candidates = [_normalize_checkpoint_path(p)]
    parts = p.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        rel = Path(*parts[idx:]).as_posix()
        if rel not in candidates:
            candidates.append(rel)
    if str(p).startswith("/app/"):
        stripped = str(p)[5:]
        if stripped not in candidates:
            candidates.append(stripped)
    return candidates


def find_run_id_by_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> str | None:
    """Find the most recent MLflow run that saved this checkpoint path."""
    mlf = _import_mlflow()
    mlf.set_tracking_uri(tracking_uri)
    for candidate in _path_lookup_candidates(checkpoint_path):
        escaped = candidate.replace("'", "''")
        try:
            runs = mlf.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.checkpoint_path = '{escaped}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
        except Exception as exc:
            logger.debug("MLflow search failed for %s: %s", candidate, exc)
            continue
        if runs is not None and not runs.empty:
            return str(runs.iloc[0].run_id)
    return None


def resolve_run_checkpoint_path(
    run_id: str,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    download_dir: Path | None = None,
) -> Path:
    """
    Resolve v6_best.pt for an MLflow training run.

    Prefers the on-disk ``checkpoint_path`` tag when the file still exists,
    otherwise downloads ``checkpoints/v6_best.pt`` from run artifacts.
    """
    mlf = _import_mlflow()
    mlf.set_tracking_uri(tracking_uri)
    client = mlf.tracking.MlflowClient()
    run = client.get_run(run_id)
    ckpt_tag = run.data.tags.get("checkpoint_path")
    if ckpt_tag:
        direct = Path(ckpt_tag)
        if direct.is_file():
            return direct.resolve()
        resolved = resolve_checkpoint_file(ckpt_tag)
        if resolved is not None:
            return resolved.resolve()

    dest = download_dir or Path(tempfile.mkdtemp(prefix="mlflow_ckpt_"))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        local = client.download_artifacts(run_id, CHECKPOINT_ARTIFACT, dst_path=str(dest))
    except Exception as exc:
        raise FileNotFoundError(
            f"Run {run_id} has no resolvable checkpoint_path tag and no artifact "
            f"{CHECKPOINT_ARTIFACT}: {exc}"
        ) from exc
    path = Path(local)
    if not path.is_file():
        raise FileNotFoundError(
            f"Downloaded artifact for run {run_id} is not a file: {path}"
        )
    return path.resolve()


def run_summary(run_id: str, *, tracking_uri: str = DEFAULT_TRACKING_URI) -> dict[str, str | None]:
    """Lightweight run metadata for promotion logs."""
    mlf = _import_mlflow()
    mlf.set_tracking_uri(tracking_uri)
    run = mlf.tracking.MlflowClient().get_run(run_id)
    tags = run.data.tags
    metrics = run.data.metrics
    return {
        "run_id": run_id,
        "run_name": tags.get("mlflow.runName") or run.info.run_name,
        "phase_preset": tags.get("phase_preset"),
        "checkpoint_path": tags.get("checkpoint_path"),
        "checkpoint_sha256": tags.get("checkpoint_sha256"),
        "best_score": tags.get("best_score") or (
            str(metrics["score"]) if "score" in metrics else None
        ),
        "parent_run_id": tags.get("parent_run_id"),
        "resume_from": tags.get("resume_from"),
    }
