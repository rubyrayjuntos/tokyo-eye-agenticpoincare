"""Log lifecycle events (verify, CI, deploy) onto MLflow — the control pane."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from science.tokyo_eye.governance.registry import (
    ensure_tokyoeye_model,
    ensure_tracking,
    set_model_version_tags,
)
from science.tokyo_eye.governance.taxonomy import (
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)
from science.tokyo_eye.governance.train_pipeline import ensure_taxonomy_experiment


def log_lifecycle_event(
    *,
    kind: str,
    alias: str,
    payload: dict[str, Any],
    tracking_uri: str | None = None,
    domain: str = "geometric",
    subsystem: str = "full-stack",
    version: str | None = None,
) -> dict[str, Any]:
    """Write a tagged run + optional model-version tags so the UI has no silent steps.

    ``kind`` is a short verb: ``verify``, ``github_actions``, ``deploy``, ``compete``.
    Metrics use 1.0/0.0 for booleans so the MLflow UI charts them.
    """
    import mlflow

    ensure_tracking(tracking_uri)
    ensure_tokyoeye_model(tracking_uri=tracking_uri)
    exp = ensure_taxonomy_experiment(domain, subsystem, tracking_uri=tracking_uri)
    goal = validate_run_name(f"{kind}_{alias}_{_stamp()}")
    tags = mandatory_run_tags(
        domain=domain,
        subsystem=subsystem,
        capability_goal=goal,
    )
    tags.update(
        {
            "lifecycle_kind": kind,
            "lifecycle_alias": alias,
            "github_run_id": str(payload.get("github_run_id") or os.environ.get("GITHUB_RUN_ID") or ""),
            "github_sha": str(payload.get("github_sha") or os.environ.get("GITHUB_SHA") or ""),
        }
    )

    ok = bool(payload.get("ok"))
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=goal) as run:
        for key, value in tags.items():
            if value:
                mlflow.set_tag(key, value)
        mlflow.log_metric("lifecycle_ok", 1.0 if ok else 0.0)
        for metric_key in ("vault_ok", "mlflow_ok", "served_from_github"):
            if metric_key in payload:
                mlflow.log_metric(metric_key, 1.0 if payload[metric_key] else 0.0)
        if "status" in payload:
            mlflow.set_tag("ci_status", str(payload["status"]))
            mlflow.log_metric(
                "ci_success", 1.0 if str(payload["status"]).lower() == "success" else 0.0
            )
        mlflow.log_dict(_jsonable(payload), f"lifecycle/{kind}.json")
        run_id = run.info.run_id

    version_tags = {
        f"lifecycle_{kind}_ok": "true" if ok else "false",
        f"lifecycle_{kind}_run_id": run_id,
        f"lifecycle_{kind}_at": datetime.now(timezone.utc).isoformat(),
    }
    if version:
        set_model_version_tags(
            version=version,
            tags=version_tags,
            tracking_uri=tracking_uri,
        )

    return {
        "experiment": exp,
        "run_id": run_id,
        "kind": kind,
        "alias": alias,
        "ok": ok,
        "version_tags": version_tags if version else {},
    }


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if hasattr(value, "as_dict"):
            out[key] = value.as_dict()
        else:
            out[key] = value
    return out
