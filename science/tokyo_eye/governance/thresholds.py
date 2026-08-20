"""Threshold packs — MLflow artifact SSOT for evaluate/promote gates.

Going forward, packs live on MLflow runs as ``thresholds/thresholds.json``.
A local JSON path is accepted only as a legacy override (emits a warning).
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

from science.tokyo_eye.governance.evaluate import ThresholdSpec, thresholds_from_mapping
from science.tokyo_eye.governance.registry import ensure_tracking
from science.tokyo_eye.governance.taxonomy import (
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = "thresholds/thresholds.json"
TEMPLATE_TAG = "threshold_pack_template"
TEMPLATE_GOAL = "threshold_pack_ssot_v1"

# Default geometric/full-stack smoke pack (also used to seed the template run).
DEFAULT_SMOKE_PACK: dict[str, float] = {
    "smoke_metric": 0.5,
}


def log_threshold_pack(
    pack: dict[str, Any],
    *,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
) -> str:
    """Log a threshold pack onto the *active* MLflow run. Returns artifact path."""
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError("log_threshold_pack requires an active MLflow run")
    # Normalize then re-dump so the artifact is always valid ThresholdSpec JSON.
    specs = thresholds_from_mapping(pack)
    payload = {s.metric: s.minimum for s in specs}
    mlflow.log_dict(payload, artifact_path)
    return artifact_path


def seed_threshold_pack_template(
    *,
    domain: str = "geometric",
    subsystem: str = "full-stack",
    pack: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    package_revision: str | None = None,
) -> dict[str, Any]:
    """Create (or reuse) a tagged template run that holds the SSOT threshold pack.

    Does not touch registry aliases.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from science.tokyo_eye.governance.train_pipeline import ensure_taxonomy_experiment

    ensure_tracking(tracking_uri)
    ensure_taxonomy_experiment(domain, subsystem, tracking_uri=tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_path(domain, subsystem))
    assert exp is not None

    # Reuse existing template run if present.
    found = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.`{TEMPLATE_TAG}` = 'true'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    pack = dict(pack or DEFAULT_SMOKE_PACK)
    # Always create a new template run (latest tagged run wins as SSOT).
    goal = validate_run_name(TEMPLATE_GOAL)
    tags = mandatory_run_tags(
        domain=domain,
        subsystem=subsystem,
        capability_goal=goal,
        package_revision=package_revision,
    )
    with mlflow.start_run(run_name=goal) as run:
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        mlflow.set_tag(TEMPLATE_TAG, "true")
        mlflow.set_tag("governance_role", "threshold_pack_ssot")
        if found:
            mlflow.set_tag("supersedes_run_id", found[0].info.run_id)
        log_threshold_pack(pack)
        return {
            "run_id": run.info.run_id,
            "experiment": exp.name,
            "artifact_path": DEFAULT_ARTIFACT_PATH,
            "reused": False,
            "superseded": found[0].info.run_id if found else None,
            "pack": pack,
        }


def find_threshold_pack_template_run(
    *,
    domain: str = "geometric",
    subsystem: str = "full-stack",
    tracking_uri: str | None = None,
) -> str | None:
    """Return latest template run id, or None."""
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_path(domain, subsystem))
    if exp is None:
        return None
    found = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.`{TEMPLATE_TAG}` = 'true'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    return found[0].info.run_id if found else None


def load_thresholds_from_mlflow_artifact(
    *,
    run_id: str,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    tracking_uri: str | None = None,
) -> list[ThresholdSpec]:
    """Load threshold pack from a run artifact (SSOT)."""
    import mlflow

    ensure_tracking(tracking_uri)
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    path = Path(local)
    if path.is_dir():
        cand = path / Path(artifact_path).name
        path = cand if cand.is_file() else next(path.rglob("*.json"))
    return thresholds_from_mapping(json.loads(path.read_text()))


def resolve_thresholds(
    *,
    thresholds_run_id: str | None = None,
    thresholds_artifact: str = DEFAULT_ARTIFACT_PATH,
    thresholds_path: str | Path | None = None,
    thresholds_inline: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    domain: str | None = None,
    subsystem: str | None = None,
) -> tuple[list[ThresholdSpec], dict[str, Any]]:
    """Resolve threshold pack with MLflow-first priority.

    Order:
      1. ``thresholds_run_id`` (+ artifact path)
      2. latest ``threshold_pack_template`` run for domain/subsystem
      3. ``thresholds_inline`` dict
      4. local ``thresholds_path`` (legacy — warns)

    Returns ``(specs, provenance)``.
    """
    if thresholds_run_id:
        specs = load_thresholds_from_mlflow_artifact(
            run_id=thresholds_run_id,
            artifact_path=thresholds_artifact,
            tracking_uri=tracking_uri,
        )
        return specs, {
            "source": "mlflow_run_artifact",
            "run_id": thresholds_run_id,
            "artifact_path": thresholds_artifact,
        }

    if domain and subsystem:
        tid = find_threshold_pack_template_run(
            domain=domain, subsystem=subsystem, tracking_uri=tracking_uri
        )
        if tid:
            specs = load_thresholds_from_mlflow_artifact(
                run_id=tid,
                artifact_path=thresholds_artifact,
                tracking_uri=tracking_uri,
            )
            return specs, {
                "source": "mlflow_template_run",
                "run_id": tid,
                "artifact_path": thresholds_artifact,
            }

    if thresholds_inline is not None:
        return thresholds_from_mapping(thresholds_inline), {
            "source": "inline",
        }

    if thresholds_path:
        warnings.warn(
            "Local thresholds JSON is legacy; prefer MLflow artifact "
            f"({DEFAULT_ARTIFACT_PATH} on a template/train run).",
            DeprecationWarning,
            stacklevel=2,
        )
        raw = json.loads(Path(thresholds_path).read_text())
        return thresholds_from_mapping(raw), {
            "source": "legacy_local_json",
            "path": str(thresholds_path),
        }

    raise ValueError(
        "No threshold pack resolved. Seed one with seed_threshold_pack_template "
        "or pass --thresholds-run-id / inline metrics map."
    )
