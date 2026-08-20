"""Model Registry helpers for registered model ``TokyoEye`` (aliases only)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from science.tokyo_eye.governance.taxonomy import REGISTERED_MODEL_NAME

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
ALIAS_CHAMPION = "champion"
ALIAS_EXPERIMENTAL = "experimental"
ALLOWED_ALIASES = frozenset({ALIAS_CHAMPION, ALIAS_EXPERIMENTAL})


def _import_mlflow() -> Any:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "false")
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required. Install with: uv pip install -e '.[science,training]'"
        ) from exc
    return mlflow


def ensure_tracking(tracking_uri: str | None = None) -> Any:
    mlf = _import_mlflow()
    uri = tracking_uri or DEFAULT_TRACKING_URI
    if uri.startswith("file:"):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlf.set_tracking_uri(uri)
    mlf.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", uri))
    return mlf


def ensure_tokyoeye_model(*, tracking_uri: str | None = None) -> str:
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    client = MlflowClient()
    name = REGISTERED_MODEL_NAME
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(
            name,
            description=(
                "Tokyo Eye product model (EquiformerV3 + MoE). "
                "Aliases: @champion (live), @experimental (staging)."
            ),
        )
    return name


def register_run_model_version(
    *,
    model_uri: str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Register ``model_uri`` (runs:/… or file:/…) as a new ``TokyoEye`` version."""
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    name = ensure_tokyoeye_model(tracking_uri=tracking_uri)
    client = MlflowClient()
    mv = client.create_model_version(
        name=name,
        source=model_uri,
        run_id=run_id,
        tags=tags or {},
    )
    return {
        "name": name,
        "version": str(mv.version),
        "source": model_uri,
        "run_id": run_id,
    }


def set_model_alias(
    *,
    version: str | int,
    alias: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    if alias not in ALLOWED_ALIASES:
        raise ValueError(
            f"Alias {alias!r} not allowed; use one of {sorted(ALLOWED_ALIASES)}"
        )
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    name = ensure_tokyoeye_model(tracking_uri=tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(name, alias, int(version))
    return {"name": name, "version": str(version), "alias": alias}


def get_model_by_alias(
    *,
    alias: str,
    tracking_uri: str | None = None,
) -> dict[str, Any] | None:
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    name = REGISTERED_MODEL_NAME
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except Exception:
        return None
    return {
        "name": name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": mv.run_id,
        "source": mv.source,
        "tags": dict(getattr(mv, "tags", {}) or {}),
    }


def resolve_alias_uri(alias: str) -> str:
    if alias not in ALLOWED_ALIASES:
        raise ValueError(
            f"Alias {alias!r} not allowed; use one of {sorted(ALLOWED_ALIASES)}"
        )
    return f"models:/{REGISTERED_MODEL_NAME}@{alias}"


def register_checkpoint_file(
    *,
    checkpoint_path: Path | str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Register a local checkpoint path as a model version (bootstrap / cache)."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return register_run_model_version(
        model_uri=path.as_uri(),
        run_id=run_id,
        tags={**(tags or {}), "checkpoint_path": str(path)},
        tracking_uri=tracking_uri,
    )
