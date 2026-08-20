"""GNN model registry helpers.

Active TokyoEye production restore is MLflow-native: ``models:/TokyoEye@champion``.
The contract catalog remains for API shape, legacy comparisons, and runner
metadata, but local checkpoint paths are not the production SSOT.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from science.tokyo_eye.governance.registry import ALIAS_CHAMPION, resolve_alias_uri
from science.tokyo_eye.governance.taxonomy import REGISTERED_MODEL_NAME

ModelStatus = Literal["production", "legacy", "candidate", "deprecated"]
CheckpointStatus = Literal["production", "legacy", "candidate", "deprecated"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_MODEL_URI_PREFIX = "models:/"
PRODUCTION_ALIAS_ENV = "TOKYOEYE_PRODUCTION_ALIAS"


def _production_alias() -> str:
    return os.environ.get(PRODUCTION_ALIAS_ENV, ALIAS_CHAMPION).strip() or ALIAS_CHAMPION

def _load_contract() -> dict[str, Any]:
    """Lazy contract load — avoids import cycle with onboard_contract ↔ runners."""
    from science.contracts.onboard_contract import load_contract

    return load_contract()


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    model_id: str
    path: str
    status: CheckpointStatus


@dataclass(frozen=True)
class GnnModelSpec:
    model_id: str
    model_version: str
    api_alias: str
    status: ModelStatus
    runner_module: str
    runner_class: str
    production_checkpoint_id: str


def _gnn_models_section() -> dict[str, Any]:
    section = _load_contract().get("gnn_models")
    if not isinstance(section, dict):
        raise ValueError("onboard_contract.yaml missing gnn_models section")
    return section


@lru_cache(maxsize=1)
def get_gnn_model_catalog() -> dict[str, GnnModelSpec]:
    section = _gnn_models_section()
    models_raw = section.get("models", {})
    if not isinstance(models_raw, dict):
        raise ValueError("gnn_models.models must be a mapping")
    catalog: dict[str, GnnModelSpec] = {}
    for model_id, spec in models_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"gnn_models.models.{model_id} must be a mapping")
        catalog[str(model_id)] = GnnModelSpec(
            model_id=str(model_id),
            model_version=str(spec["model_version"]),
            api_alias=str(spec["api_alias"]),
            status=spec.get("status", "legacy"),
            runner_module=str(spec["runner_module"]),
            runner_class=str(spec["runner_class"]),
            production_checkpoint_id=str(spec["production_checkpoint_id"]),
        )
    return catalog


@lru_cache(maxsize=1)
def get_checkpoint_catalog() -> dict[str, CheckpointSpec]:
    section = _gnn_models_section()
    checkpoints_raw = section.get("checkpoints", {})
    if not isinstance(checkpoints_raw, dict):
        raise ValueError("gnn_models.checkpoints must be a mapping")
    catalog: dict[str, CheckpointSpec] = {}
    for checkpoint_id, spec in checkpoints_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"gnn_models.checkpoints.{checkpoint_id} must be a mapping")
        catalog[str(checkpoint_id)] = CheckpointSpec(
            checkpoint_id=str(checkpoint_id),
            model_id=str(spec["model_id"]),
            path=str(spec["path"]),
            status=spec.get("status", "legacy"),
        )
    return catalog


def get_production_model_id() -> str:
    return str(_gnn_models_section()["production_model_id"])


def get_production_model() -> GnnModelSpec:
    model_id = get_production_model_id()
    catalog = get_gnn_model_catalog()
    if model_id not in catalog:
        raise ValueError(f"production_model_id {model_id!r} not in gnn_models.models")
    return catalog[model_id]


def get_production_model_version() -> str:
    return get_production_model().model_version


def get_production_api_alias() -> str:
    return _production_alias()


def get_checkpoint_spec(checkpoint_id: str) -> CheckpointSpec:
    catalog = get_checkpoint_catalog()
    if checkpoint_id not in catalog:
        raise KeyError(f"Unknown checkpoint_id {checkpoint_id!r}")
    return catalog[checkpoint_id]


def get_model_spec(model_id: str) -> GnnModelSpec:
    catalog = get_gnn_model_catalog()
    if model_id not in catalog:
        raise KeyError(f"Unknown model_id {model_id!r}")
    return catalog[model_id]


def get_production_checkpoint_id() -> str:
    model = get_production_model()
    return model.production_checkpoint_id


def get_production_checkpoint_path() -> str:
    """Logical MLflow restore URI for active production."""
    return resolve_alias_uri(_production_alias())


def get_production_restore_uri(alias: str | None = None) -> str:
    """Return the MLflow model URI for an active restore alias."""
    return resolve_alias_uri(alias or _production_alias())


def resolve_model_for_api_alias(api_alias: str) -> GnnModelSpec | None:
    for model in get_gnn_model_catalog().values():
        if model.api_alias == api_alias:
            return model
    return None


def resolve_model_version_for_checkpoint(path: str) -> str | None:
    normalized = path.strip()
    if normalized.startswith(f"{MLFLOW_MODEL_URI_PREFIX}{REGISTERED_MODEL_NAME}@"):
        alias = normalized.rsplit("@", 1)[-1]
        return f"{REGISTERED_MODEL_NAME}@{alias}"
    for checkpoint in get_checkpoint_catalog().values():
        if checkpoint.path == normalized:
            return get_model_spec(checkpoint.model_id).model_version
    return None


def checkpoint_search_roots() -> list[Path]:
    """Directories searched when resolving a contract checkpoint path to a file."""
    roots: list[Path] = []
    env_dir = os.environ.get("CHECKPOINT_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    roots.append(_REPO_ROOT / "checkpoints")
    roots.append(_REPO_ROOT)
    return roots


def resolve_checkpoint_file(path: str) -> Path | None:
    """Return the first existing filesystem path for a contract logical path."""
    if path.strip().startswith(MLFLOW_MODEL_URI_PREFIX):
        return None
    logical = Path(path)
    if logical.is_file():
        return logical

    stripped = path
    if stripped.startswith("checkpoints/"):
        stripped = stripped[len("checkpoints/") :]

    for root in checkpoint_search_roots():
        for candidate in (root / path, root / stripped, root / logical.name):
            if candidate.is_file():
                return candidate
    return None


def checkpoint_exists(path: str) -> bool:
    return resolve_checkpoint_file(path) is not None


def compute_checkpoint_sha256(path: str) -> str | None:
    """SHA-256 of checkpoint bytes; None if file not found."""
    if path.strip().startswith(MLFLOW_MODEL_URI_PREFIX):
        return None
    resolved = resolve_checkpoint_file(path)
    if resolved is None:
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_status(path: str) -> dict[str, Any]:
    if path.strip().startswith(MLFLOW_MODEL_URI_PREFIX):
        return {
            "path": path,
            "resolved_path": None,
            "exists": False,
            "sha256": None,
            "sha256_prefix": None,
            "uri": path,
        }
    resolved = resolve_checkpoint_file(path)
    sha256 = compute_checkpoint_sha256(path) if resolved is not None else None
    return {
        "path": path,
        "resolved_path": str(resolved) if resolved is not None else None,
        "exists": resolved is not None,
        "sha256": sha256,
        "sha256_prefix": sha256[:12] if sha256 else None,
    }


def get_production_mlflow_metadata(
    *,
    alias: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Return MLflow alias metadata without downloading artifacts."""
    active_alias = alias or _production_alias()
    uri = resolve_alias_uri(active_alias)
    try:
        from science.tokyo_eye.governance.registry import get_model_by_alias

        meta = get_model_by_alias(alias=active_alias, tracking_uri=tracking_uri)
    except Exception as exc:
        return {
            "ok": False,
            "registered_model": REGISTERED_MODEL_NAME,
            "alias": active_alias,
            "uri": uri,
            "error": str(exc),
        }
    if meta is None:
        return {
            "ok": False,
            "registered_model": REGISTERED_MODEL_NAME,
            "alias": active_alias,
            "uri": uri,
            "error": f"No model version for {uri}",
        }
    tags = dict(meta.get("tags") or {})
    return {
        "ok": True,
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": active_alias,
        "uri": uri,
        "mlflow_model_version": meta.get("version"),
        "run_id": meta.get("run_id"),
        "source": meta.get("source"),
        "tags": tags,
        "model_version": f"{REGISTERED_MODEL_NAME}@{active_alias}",
    }


def resolve_production_checkpoint(
    *,
    alias: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Resolve the active MLflow alias into a local cache path, failing closed."""
    active_alias = alias or _production_alias()
    from science.tokyo_eye.governance.resolve import resolve_alias_checkpoint

    resolved = resolve_alias_checkpoint(alias=active_alias, tracking_uri=tracking_uri)
    return {
        **resolved,
        "cache_path": resolved["path"],
        "registered_model": REGISTERED_MODEL_NAME,
        "model_id": REGISTERED_MODEL_NAME,
        "model_version": f"{REGISTERED_MODEL_NAME}@{active_alias}",
        "restore_source": "mlflow_alias",
    }


def get_production_restore_summary(
    *,
    alias: str | None = None,
    tracking_uri: str | None = None,
    resolve_cache: bool = False,
) -> dict[str, Any]:
    """MLflow production restore payload for API/health responses."""
    summary = get_production_mlflow_metadata(alias=alias, tracking_uri=tracking_uri)
    if not resolve_cache or not summary.get("ok"):
        return summary
    try:
        resolved = resolve_production_checkpoint(alias=summary["alias"], tracking_uri=tracking_uri)
    except Exception as exc:
        return {**summary, "ok": False, "cache_error": str(exc)}
    return {
        **summary,
        "cache_path": resolved.get("cache_path"),
        "checkpoint_sha256": resolved.get("sha256"),
        "checkpoint_sha256_16": resolved.get("sha256_16"),
    }


def build_models_api_payload() -> dict[str, Any]:
    contract = _load_contract()
    production_restore = get_production_restore_summary()
    models_out: list[dict[str, Any]] = []
    for model in get_gnn_model_catalog().values():
        checkpoint = get_checkpoint_spec(model.production_checkpoint_id)
        models_out.append(
            {
                "model_id": model.model_id,
                "model_version": model.model_version,
                "api_alias": model.api_alias,
                "status": model.status,
                "runner_module": model.runner_module,
                "runner_class": model.runner_class,
                "production_checkpoint_id": model.production_checkpoint_id,
                "production_checkpoint": {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    **checkpoint_status(checkpoint.path),
                },
            }
        )

    checkpoints_out: list[dict[str, Any]] = []
    for checkpoint in get_checkpoint_catalog().values():
        checkpoints_out.append(
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "model_id": checkpoint.model_id,
                "status": checkpoint.status,
                **checkpoint_status(checkpoint.path),
            }
        )

    return {
        "contract_version": str(contract.get("version", "")),
        "production_model_id": get_production_model_id(),
        "production_checkpoint_path": get_production_checkpoint_path(),
        "production_restore": production_restore,
        "models": models_out,
        "checkpoints": checkpoints_out,
    }


def validate_gnn_models_contract() -> list[str]:
    """Validate gnn_models section internal consistency."""
    errors: list[str] = []
    section = _load_contract().get("gnn_models")
    if not isinstance(section, dict):
        return ["missing gnn_models section"]

    models = get_gnn_model_catalog()
    checkpoints = get_checkpoint_catalog()
    production_model_id = section.get("production_model_id")
    if not production_model_id:
        errors.append("gnn_models.production_model_id is required")
    elif production_model_id not in models:
        errors.append(f"production_model_id {production_model_id!r} not in models")

    production_checkpoints = [
        ckpt_id for ckpt_id, ckpt in checkpoints.items() if ckpt.status == "production"
    ]
    if len(production_checkpoints) != 1:
        errors.append(
            "gnn_models.checkpoints must have exactly one status=production checkpoint; "
            f"found {production_checkpoints!r}"
        )

    production_models = [
        model_id for model_id, model in models.items() if model.status == "production"
    ]
    if len(production_models) != 1:
        errors.append(
            "gnn_models.models must have exactly one status=production model; "
            f"found {production_models!r}"
        )

    model_versions: dict[str, str] = {}
    api_aliases: dict[str, str] = {}
    for model_id, model in models.items():
        if model.model_version in model_versions:
            errors.append(
                f"duplicate model_version {model.model_version!r} "
                f"on {model_id!r} and {model_versions[model.model_version]!r}"
            )
        model_versions[model.model_version] = model_id
        if model.api_alias in api_aliases:
            errors.append(
                f"duplicate api_alias {model.api_alias!r} "
                f"on {model_id!r} and {api_aliases[model.api_alias]!r}"
            )
        api_aliases[model.api_alias] = model_id
        if model.production_checkpoint_id not in checkpoints:
            errors.append(
                f"model {model_id!r} references unknown checkpoint "
                f"{model.production_checkpoint_id!r}"
            )
        else:
            ckpt = checkpoints[model.production_checkpoint_id]
            if ckpt.model_id != model_id:
                errors.append(
                    f"model {model_id!r} production_checkpoint "
                    f"{model.production_checkpoint_id!r} belongs to {ckpt.model_id!r}"
                )

    for checkpoint_id, checkpoint in checkpoints.items():
        if checkpoint.model_id not in models:
            errors.append(
                f"checkpoint {checkpoint_id!r} references unknown model "
                f"{checkpoint.model_id!r}"
            )
        if not checkpoint.path:
            errors.append(f"checkpoint {checkpoint_id!r} missing path")

    if production_model_id in models:
        model = models[production_model_id]
        prod_ckpt = checkpoints.get(model.production_checkpoint_id)
        if prod_ckpt is not None and prod_ckpt.status != "production":
            errors.append(
                f"production model {production_model_id!r} checkpoint "
                f"{model.production_checkpoint_id!r} must have status=production"
            )

    return errors
