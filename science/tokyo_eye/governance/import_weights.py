"""One-time import of weight files into MLflow (artifact store + registry).

Caller passes an explicit ``--checkpoint`` path. This module does not read
filesystem seal path constants. Source files are never deleted or modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from science.tokyo_eye.governance.evaluate import (
    ThresholdSpec,
    evaluate_metrics,
)
from science.tokyo_eye.governance.pyfunc_model import log_tokyoeye_pyfunc
from science.tokyo_eye.governance.registry import (
    ALIAS_CHAMPION,
    ALIAS_EXPERIMENTAL,
    ensure_tokyoeye_model,
    ensure_tracking,
    get_model_by_alias,
    register_run_model_version,
    set_model_alias,
)
from science.tokyo_eye.governance.taxonomy import (
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)


class ImportBlocked(RuntimeError):
    """Import blocked (e.g. alias already set and overwrite not requested)."""


def import_checkpoint_into_mlflow(
    *,
    checkpoint_path: Path | str,
    domain: str,
    subsystem: str,
    capability_goal: str,
    tracking_uri: str | None = None,
    alias: str | None = None,
    metrics: dict[str, float] | None = None,
    thresholds: list[ThresholdSpec] | None = None,
    package_revision: str | None = None,
    overwrite_alias: bool = False,
    log_pyfunc: bool = True,
    evidence_artifact: Path | str | None = None,
) -> dict[str, Any]:
    """Copy checkpoint bytes into a new MLflow run, register, optionally alias.

    - Does not delete or alter ``checkpoint_path``.
    - Does not import healthy_v8 / gate SSOTs.
    - If ``alias`` is set and already points at a version, refuses unless
      ``overwrite_alias=True`` (still does not delete the previous version).
    """
    import mlflow

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    goal = validate_run_name(capability_goal)
    tags = mandatory_run_tags(
        domain=domain,
        subsystem=subsystem,
        capability_goal=goal,
        package_revision=package_revision,
    )
    exp = experiment_path(domain, subsystem)

    ensure_tracking(tracking_uri)
    ensure_tokyoeye_model(tracking_uri=tracking_uri)

    if alias:
        existing = get_model_by_alias(alias=alias, tracking_uri=tracking_uri)
        if existing is not None and not overwrite_alias:
            raise ImportBlocked(
                f"Alias @{alias} already set to version {existing['version']}. "
                "Pass overwrite_alias=True to retarget (previous version is kept)."
            )

    if thresholds and metrics is not None:
        result = evaluate_metrics(metrics, thresholds)
        result.raise_if_failed()

    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=goal) as run:
        run_id = run.info.run_id
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        mlflow.set_tag("import_source_path", str(path))
        mlflow.set_tag("import_mode", "bytes_copy_into_mlflow")

        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

        # Own the bytes in the artifact store (do not register file:// as SSOT).
        art_name = path.name
        mlflow.log_artifact(str(path), artifact_path="checkpoints")

        if evidence_artifact is not None:
            ev = Path(evidence_artifact)
            if ev.is_file():
                mlflow.log_artifact(str(ev), artifact_path="evidence")

        if log_pyfunc:
            model_uri = log_tokyoeye_pyfunc(
                checkpoint_path=path,
                config={
                    "capability_goal": goal,
                    "domain": domain,
                    "subsystem": subsystem,
                },
            )
        else:
            model_uri = f"runs:/{run_id}/checkpoints"

        reg = register_run_model_version(
            model_uri=model_uri,
            run_id=run_id,
            tags={**tags, "checkpoint_filename": art_name},
            tracking_uri=tracking_uri,
        )

        alias_out = None
        if alias:
            if alias not in {ALIAS_CHAMPION, ALIAS_EXPERIMENTAL}:
                raise ValueError(f"Unsupported alias {alias!r}")
            alias_out = set_model_alias(
                version=reg["version"],
                alias=alias,
                tracking_uri=tracking_uri,
            )

        out = {
            "run_id": run_id,
            "experiment": exp,
            "register": reg,
            "alias": alias_out,
            "model_uri": model_uri,
            "source_checkpoint": str(path),
            "source_deleted": False,
            "note": "Source file left untouched; MLflow artifact store holds the copy.",
        }
        mlflow.log_dict(out, "governance_import.json")
        return out
