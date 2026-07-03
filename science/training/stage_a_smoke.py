"""Stage A one-epoch integration smoke — assembled stack validation before full curriculum."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from science.training.corpus_governance import LOCKED_MANIFEST
from science.training.mlflow_governance import (
    MANDATORY_METRICS,
    corpus_manifest_hash,
    validate_finished_run,
)

LOCKED_CORPUS_MANIFEST = LOCKED_MANIFEST


def collect_run_artifact_names(run_id: str, *, tracking_uri: str = "file:./mlruns") -> set[str]:
    """Collect artifact basenames from MLflow API, with file-store filesystem fallback."""
    import mlflow

    names: set[str] = set()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    for prefix in ("", "governance", "governance/flat"):
        try:
            for art in client.list_artifacts(run_id, path=prefix):
                names.add(Path(art.path).name)
        except (OSError, mlflow.MlflowException):
            continue
    if names:
        return names

    store_root = Path(tracking_uri.removeprefix("file:"))
    for meta in store_root.glob(f"*/{run_id}/meta.yaml"):
        art_dir = meta.parent / "artifacts"
        if not art_dir.is_dir():
            continue
        for path in art_dir.rglob("*"):
            if path.is_file():
                names.add(path.name)
        break
    return names


def validate_stage_a_smoke_run(
    params: dict[str, str],
    metric_keys: set[str],
    metrics: dict[str, float],
    artifact_names: set[str],
    *,
    manifest_path: Path = LOCKED_CORPUS_MANIFEST,
    min_per_fold_loss_keys: int = 2,
) -> list[str]:
    """Validate one-epoch smoke: full schema + per_fold_loss on CATH keys (subset ok)."""
    errors: list[str] = []

    if not manifest_path.is_file():
        errors.append(f"locked manifest missing: {manifest_path}")
        return errors

    expected_hash = corpus_manifest_hash(manifest_path)
    logged_hash = params.get("corpus_manifest_hash", "")
    if logged_hash != expected_hash:
        errors.append(
            f"corpus_manifest_hash mismatch: run logged {logged_hash[:16]}… "
            f"expected locked manifest {expected_hash[:16]}…"
        )

    family_keys = [k for k in metric_keys if k.startswith("per_family_loss.")]
    if family_keys:
        errors.append(f"legacy per_family_loss metrics present: {family_keys}")

    fold_keys = sorted(k for k in metric_keys if k.startswith("per_fold_loss."))
    if len(fold_keys) < min_per_fold_loss_keys:
        errors.append(
            f"expected at least {min_per_fold_loss_keys} per_fold_loss.* metrics, got {fold_keys}"
        )

    for key in fold_keys:
        suffix = key.removeprefix("per_fold_loss.")
        if "." in suffix:
            errors.append(f"per_fold_loss key must use underscores not dots: {key}")

    routing = ("effective_experts", "effective_experts_min", "min_routing_fraction")
    missing_routing = [k for k in routing if k not in metric_keys]
    if missing_routing:
        errors.append(f"routing gate metrics missing from run: {missing_routing}")

    mlflow_errors = validate_finished_run(
        params,
        metric_keys,
        artifact_names,
        manifest_path=manifest_path,
    )
    # Smoke may train a protein subset — relax per-fold requirement for folds not in epoch.
    mlflow_errors = [
        e
        for e in mlflow_errors
        if not e.startswith("missing metric per_fold_loss.")
    ]
    errors.extend(mlflow_errors)

    missing_core = MANDATORY_METRICS - metric_keys
    if missing_core:
        errors.append(f"missing mandatory metrics: {sorted(missing_core)}")

    if "stage_gate_passed" not in metric_keys:
        errors.append("stage_gate_passed not logged — routing gate not wired")

    return errors


def smoke_assertions_doc() -> dict[str, Any]:
    """Machine-readable checklist for Stage A one-epoch smoke (P_STAGE_A_SMOKE)."""
    return {
        "gate_id": "P_STAGE_A_SMOKE",
        "corpus_manifest": str(LOCKED_CORPUS_MANIFEST.relative_to(LOCKED_CORPUS_MANIFEST.parents[2])),
        "epochs": 1,
        "assertions": [
            "corpus_manifest_hash matches locked v6_corpus_stage_a.json",
            "no per_family_loss.* metrics",
            "≥2 per_fold_loss.{fold_id_underscored} metrics from CATH vocabulary",
            "effective_experts, effective_experts_min, min_routing_fraction logged",
            "stage_gate_passed logged (value may be 0 on 1-epoch subset)",
            "P_MLFLOW_01 params, core metrics, governance artifacts",
        ],
        "make_target": "train-v6-stage-a-smoke",
    }
