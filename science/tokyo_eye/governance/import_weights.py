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
    set_model_version_tags,
)
from science.tokyo_eye.governance.taxonomy import (
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)
from science.tokyo_eye.governance.vault import (
    GhRunner,
    VaultRequired,
    assert_vaulted,
    sha256_file,
    upload_checkpoint,
)


class ImportBlocked(RuntimeError):
    """Import blocked (e.g. alias already set and overwrite not requested)."""


class ArtifactVerifyError(RuntimeError):
    """MLflow artifact bytes do not match the source checkpoint digest."""


def verify_run_checkpoint(
    *,
    run_id: str,
    expected_sha256: str,
    tracking_uri: str | None = None,
) -> str:
    """Download ``checkpoints/`` from the run and require sha256 match."""
    import mlflow

    ensure_tracking(tracking_uri)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="checkpoints",
    )
    root = Path(downloaded)
    candidates = [root] if root.is_file() else sorted(root.rglob("*.pt"))
    if not candidates:
        raise ArtifactVerifyError(
            f"Run {run_id} has no checkpoint artifact under checkpoints/"
        )
    digest = sha256_file(candidates[0])
    if digest != expected_sha256:
        raise ArtifactVerifyError(
            f"MLflow artifact sha256 {digest} != source {expected_sha256}"
        )
    return digest


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
    vault_runner: GhRunner | None = None,
    vault_repo: str | None = None,
    confirm_champion: bool = False,
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
    digest = sha256_file(path)

    if alias == ALIAS_CHAMPION and not confirm_champion:
        raise VaultRequired(
            "@champion import requires confirm_champion=True "
            "(CLI: --confirm-champion) after evaluate Pass."
        )

    vault_record = None
    if alias == ALIAS_CHAMPION:
        vault_record = upload_checkpoint(
            checkpoint_path=path,
            alias=ALIAS_CHAMPION,
            capability_goal=capability_goal,
            repo=vault_repo,
            runner=vault_runner,
        )
        assert_vaulted(
            sha256=digest,
            alias=ALIAS_CHAMPION,
            repo=vault_repo,
            runner=vault_runner,
        )

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
        mlflow.set_tag("checkpoint_sha256", digest)
        mlflow.set_tag("checkpoint_sha256_16", digest[:16])
        if vault_record is not None:
            for key, value in vault_record.as_tags().items():
                mlflow.set_tag(key, value)

        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

        # Own the bytes in the artifact store (do not register file:// as SSOT).
        art_name = path.name
        mlflow.log_artifact(str(path), artifact_path="checkpoints")
        verify_run_checkpoint(
            run_id=run_id,
            expected_sha256=digest,
            tracking_uri=tracking_uri,
        )

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

        version_tags = {**tags, "checkpoint_filename": art_name, "checkpoint_sha256": digest}
        if vault_record is not None:
            version_tags.update(vault_record.as_tags())
        reg = register_run_model_version(
            model_uri=model_uri,
            run_id=run_id,
            tags=version_tags,
            tracking_uri=tracking_uri,
        )
        set_model_version_tags(
            version=reg["version"],
            tags=version_tags,
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
            "checkpoint_sha256": digest,
            "vault": vault_record.as_dict() if vault_record is not None else None,
            "note": "Source file left untouched; MLflow artifact store holds the copy.",
        }
        mlflow.log_dict(out, "governance_import.json")
        return out
