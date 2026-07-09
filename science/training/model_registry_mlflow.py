"""MLflow Model Registry helpers — champion/challenger aliases per GNN lineage."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from science.training.gnn_lineage import GnnLineageId, get_lineage

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"


def _import_mlflow() -> Any:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "false")
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required. Install with: uv pip install -e '.[science,training]'"
        ) from exc
    return mlflow


def registered_model_name(lineage_id: str | GnnLineageId) -> str:
    return get_lineage(lineage_id).model_version


def ensure_tracking(tracking_uri: str | None = None) -> Any:
    mlf = _import_mlflow()
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlf.set_tracking_uri(uri)
    mlf.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", uri))
    return mlf


def ensure_registered_model(name: str, *, tracking_uri: str | None = None) -> None:
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    client = MlflowClient()
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name, description=f"Tokyo Eye GNN lineage model {name}")


def register_checkpoint_version(
    *,
    lineage_id: str,
    checkpoint_path: Path | str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Register a local .pt checkpoint as a new Model Registry version.

    Uses runs:/ URI when run_id is set and an artifact was logged; otherwise
    registers via a local file URI so hyperbolic custom checkpoints work.
    """
    from mlflow.tracking import MlflowClient

    mlf = ensure_tracking(tracking_uri)
    name = registered_model_name(lineage_id)
    ensure_registered_model(name, tracking_uri=tracking_uri)
    client = MlflowClient()
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    source = path.as_uri()
    if run_id:
        # Prefer run artifact if present
        try:
            arts = client.list_artifacts(run_id, path="checkpoints")
            if arts:
                source = f"runs:/{run_id}/checkpoints"
        except Exception as exc:
            logger.debug("list_artifacts failed for %s: %s", run_id, exc)

    mv = client.create_model_version(
        name=name,
        source=source,
        run_id=run_id,
        tags={
            "gnn_lineage": lineage_id,
            "checkpoint_path": str(path),
            **(tags or {}),
        },
    )
    return {
        "name": name,
        "version": str(mv.version),
        "source": source,
        "run_id": run_id,
        "lineage_id": lineage_id,
    }


def set_alias(
    *,
    lineage_id: str,
    version: str | int,
    alias: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    name = registered_model_name(lineage_id)
    client = MlflowClient()
    client.set_registered_model_alias(name, alias, int(version))
    return {"name": name, "version": str(version), "alias": alias, "lineage_id": lineage_id}


def get_version_by_alias(
    *,
    lineage_id: str,
    alias: str,
    tracking_uri: str | None = None,
) -> dict[str, Any] | None:
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    name = registered_model_name(lineage_id)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except Exception:
        return None
    tags = dict(getattr(mv, "tags", {}) or {})
    return {
        "name": name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": mv.run_id,
        "source": mv.source,
        "checkpoint_path": tags.get("checkpoint_path"),
        "tags": tags,
        "lineage_id": lineage_id,
    }


def list_model_versions(
    *,
    lineage_id: str,
    tracking_uri: str | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    name = registered_model_name(lineage_id)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{name}'", max_results=max_results)
    except Exception as exc:
        logger.warning("search_model_versions failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for mv in versions:
        tags = dict(getattr(mv, "tags", {}) or {})
        out.append(
            {
                "name": name,
                "version": str(mv.version),
                "run_id": mv.run_id,
                "source": mv.source,
                "checkpoint_path": tags.get("checkpoint_path"),
                "aliases": list(getattr(mv, "aliases", []) or []),
                "tags": tags,
            }
        )
    return out


def mlflow_server_health(tracking_uri: str | None = None) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    uri = (tracking_uri or DEFAULT_TRACKING_URI).rstrip("/")
    health_url = f"{uri}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            ok = 200 <= getattr(resp, "status", 200) < 300
        version = None
        try:
            mlf = _import_mlflow()
            version = getattr(mlf, "__version__", None)
        except ImportError:
            pass
        return {"available": ok, "tracking_uri": uri, "mlflow_version": version}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "available": False,
            "tracking_uri": uri,
            "error": str(exc),
            "mlflow_version": None,
        }
