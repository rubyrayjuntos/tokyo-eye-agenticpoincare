"""Make-free day-to-day train → evaluate → register → @experimental chain."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from science.tokyo_eye.governance.evaluate import (
    ThresholdSpec,
    assert_promotable,
    thresholds_from_mapping,
)
from science.tokyo_eye.governance.pyfunc_model import log_tokyoeye_pyfunc
from science.tokyo_eye.governance.registry import (
    ALIAS_EXPERIMENTAL,
    ensure_tokyoeye_model,
    ensure_tracking,
    register_run_model_version,
    set_model_alias,
)
from science.tokyo_eye.governance.taxonomy import (
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)
from science.tokyo_eye.governance.thresholds import log_threshold_pack

logger = logging.getLogger(__name__)

ARTIFACT_LOCATION = "mlflow-artifacts:/"


def _is_proxied_artifact_location(location: str | None) -> bool:
    """True when the experiment uses the HTTP artifact proxy root."""
    if not location:
        return False
    loc = str(location).strip()
    return loc.startswith("mlflow-artifacts:") or loc.startswith("mlflow-artifacts:/")


def ensure_taxonomy_experiment(
    domain: str,
    subsystem: str,
    *,
    tracking_uri: str | None = None,
) -> str:
    """Create or reuse taxonomy experiment (proxied artifacts on HTTP tracking).

    If a legacy experiment exists under the canonical name with a filesystem
    ``artifact_location`` (e.g. ``/app/mlflow-artifacts/N``), it is **renamed**
    to ``{name}.legacy_fs_{id}`` (runs preserved) and a new experiment is
    created with ``mlflow-artifacts:/``. Does not touch registry aliases.
    """
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    name = experiment_path(domain, subsystem)
    client = MlflowClient()
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
    http = str(uri).startswith(("http://", "https://"))

    exp = client.get_experiment_by_name(name)
    if exp is not None and http and not _is_proxied_artifact_location(exp.artifact_location):
        legacy_name = f"{name}.legacy_fs_{exp.experiment_id}"
        logger.warning(
            "Archiving legacy filesystem experiment %s (%s) → %s; "
            "creating proxied %s for forward writes",
            exp.experiment_id,
            exp.artifact_location,
            legacy_name,
            name,
        )
        client.rename_experiment(exp.experiment_id, legacy_name)
        exp = None

    if exp is None:
        if http:
            client.create_experiment(name, artifact_location=ARTIFACT_LOCATION)
        else:
            client.create_experiment(name)
    mlflow.set_experiment(name)
    return name


def run_train_evaluate_register_experimental(
    *,
    domain: str,
    subsystem: str,
    capability_goal: str,
    tracking_uri: str | None = None,
    package_revision: str | None = None,
    thresholds: list[ThresholdSpec] | dict[str, Any] | None = None,
    train_callable: Any | None = None,
    best_checkpoint: Path | str | None = None,
    skip_train: bool = False,
    set_experimental: bool = True,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Single API entry: taxonomy run → optional train → evaluate → register → @experimental.

    ``@champion`` is never set here (explicit approval only).

    Parameters
    ----------
    train_callable
        ``callable(run_id: str) -> Path`` returning best checkpoint path.
        Should log metrics on the active MLflow run before returning.
    metrics
        Optional metrics to log when ``skip_train`` (tests / import-style).
    thresholds
        MetricThreshold pack applied after train/metrics are on the run.
    """
    import mlflow

    goal = validate_run_name(capability_goal)
    tags = mandatory_run_tags(
        domain=domain,
        subsystem=subsystem,
        capability_goal=goal,
        package_revision=package_revision,
    )
    if isinstance(thresholds, dict):
        thr_list = thresholds_from_mapping(thresholds)
    else:
        thr_list = list(thresholds or [])

    ensure_tracking(tracking_uri)
    ensure_tokyoeye_model(tracking_uri=tracking_uri)
    exp = ensure_taxonomy_experiment(domain, subsystem, tracking_uri=tracking_uri)

    with mlflow.start_run(run_name=goal) as run:
        run_id = run.info.run_id
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        mlflow.set_tag("governance_pipeline", "train_evaluate_register_experimental")

        ckpt: Path
        if skip_train:
            if best_checkpoint is None:
                raise ValueError("best_checkpoint required when skip_train=True")
            ckpt = Path(best_checkpoint)
        else:
            if train_callable is None:
                raise ValueError("train_callable required unless skip_train=True")
            ckpt = Path(train_callable(run_id))
        if not ckpt.is_file():
            raise FileNotFoundError(f"Best checkpoint missing: {ckpt}")

        mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

        if thr_list:
            # Persist the pack used for this gate onto the train run (SSOT copy).
            log_threshold_pack({s.metric: s.minimum for s in thr_list})
            assert_promotable(
                run_id=run_id,
                thresholds=thr_list,
                tracking_uri=tracking_uri,
            )

        model_uri = log_tokyoeye_pyfunc(
            checkpoint_path=ckpt,
            config={
                "domain": domain,
                "subsystem": subsystem,
                "capability_goal": goal,
                "device": "cpu",
            },
        )
        reg = register_run_model_version(
            model_uri=model_uri,
            run_id=run_id,
            tags=tags,
            tracking_uri=tracking_uri,
        )
        alias_out = None
        if set_experimental:
            alias_out = set_model_alias(
                version=reg["version"],
                alias=ALIAS_EXPERIMENTAL,
                tracking_uri=tracking_uri,
            )

        result = {
            "experiment": exp,
            "run_id": run_id,
            "checkpoint": str(ckpt),
            "model_uri": model_uri,
            "register": reg,
            "alias": alias_out,
            "champion_unchanged": True,
            "note": "@champion requires explicit approval (set-alias); not set by this pipeline",
        }
        mlflow.log_dict(result, "governance_pipeline.json")
        return result
