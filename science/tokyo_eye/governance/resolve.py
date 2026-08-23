"""Resolve ``models:/TokyoEye@alias`` to a local cache path from MLflow only.

This module must not import filesystem seal modules or gate-file SSOTs.
If the alias is missing or artifacts cannot be downloaded, it fails closed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from science.tokyo_eye.governance.registry import (
    ALLOWED_ALIASES,
    REGISTERED_MODEL_NAME,
    ensure_tracking,
    get_model_by_alias,
    resolve_alias_uri,
)
from science.tokyo_eye.governance.taxonomy import REGISTERED_MODEL_NAME as _MODEL
from science.tokyo_eye.governance.vault import (
    GhRunner,
    VaultError,
    download_checkpoint,
    sha256_file,
)

logger = logging.getLogger(__name__)

# Cache under cwd-relative dir by default; override with TOKYOEYE_MLFLOW_CACHE
_DEFAULT_CACHE = Path("mlflow-artifacts") / ".tokyoeye_alias_cache"


class ResolveError(RuntimeError):
    """Alias missing or artifact not downloadable from MLflow."""


def _cache_root() -> Path:
    raw = os.environ.get("TOKYOEYE_MLFLOW_CACHE", "").strip()
    return Path(raw) if raw else _DEFAULT_CACHE


def resolve_alias_checkpoint(
    *,
    alias: str = "champion",
    tracking_uri: str | None = None,
    prefer_filename_substrings: tuple[str, ...] = (".pt", "checkpoint", "weights"),
    vault_runner: GhRunner | None = None,
) -> dict[str, Any]:
    """Download checkpoint bytes for ``models:/TokyoEye@alias`` into an MLflow cache.

    Returns dict with ``path``, ``version``, ``run_id``, ``alias``, ``uri``.
    Never consults HEALTHY_* or other repo path seals.
    """
    if alias not in ALLOWED_ALIASES:
        raise ResolveError(
            f"Alias {alias!r} not allowed; use one of {sorted(ALLOWED_ALIASES)}"
        )

    ensure_tracking(tracking_uri)
    meta = get_model_by_alias(alias=alias, tracking_uri=tracking_uri)
    if meta is None:
        raise ResolveError(
            f"No model version for {resolve_alias_uri(alias)}. "
            "Import weights into MLflow and set the alias first "
            "(science.tokyo_eye.governance.entrypoints import-weights). "
            "This resolver does not fall back to filesystem seal paths."
        )

    version = meta["version"]
    run_id = meta.get("run_id")
    source = meta.get("source") or ""
    dest_dir = _cache_root() / f"{_MODEL}_{alias}_v{version}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    import mlflow

    local_model_dir = dest_dir / "model"
    if local_model_dir.exists():
        cached = _find_checkpoint(local_model_dir, prefer_filename_substrings)
        if cached is not None:
            digest = hashlib.sha256(cached.read_bytes()).hexdigest()
            return {
                "path": str(cached.resolve()),
                "version": version,
                "run_id": run_id,
                "alias": alias,
                "uri": resolve_alias_uri(alias),
                "name": REGISTERED_MODEL_NAME,
                "sha256": digest,
                "sha256_16": digest[:16],
                "source": source,
                "cache_hit": True,
                "restored_from_vault": False,
            }
        shutil.rmtree(local_model_dir)

    # Prefer registry URI so we always pull through MLflow, not a stale file:// tag.
    uri = resolve_alias_uri(alias)
    try:
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=uri,
            dst_path=str(local_model_dir),
        )
    except Exception as exc:
        # Fallback: runs:/ artifact or logged checkpoint tag only if still in MLflow
        if run_id:
            try:
                downloaded = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="checkpoints",
                    dst_path=str(dest_dir / "from_run"),
                )
            except Exception as exc2:
                vault_hit = _restore_from_vault(
                    meta,
                    dest_dir=dest_dir / "from_vault",
                    runner=vault_runner,
                )
                if vault_hit is not None:
                    return vault_hit
                raise ResolveError(
                    f"Failed to download {uri} (and run checkpoints); "
                    f"GitHub Release vault also unavailable: {exc}; {exc2}"
                ) from exc2
        else:
            vault_hit = _restore_from_vault(
                meta,
                dest_dir=dest_dir / "from_vault",
                runner=vault_runner,
            )
            if vault_hit is not None:
                return vault_hit
            raise ResolveError(f"Failed to download {uri}: {exc}") from exc

    ckpt = _find_checkpoint(Path(downloaded), prefer_filename_substrings)
    if ckpt is None and source.startswith("file:"):
        # Registered source was a raw file URI (bootstrap). Copy into cache so
        # callers still get an MLflow-managed cache path, not the original seal path.
        from urllib.parse import urlparse, unquote

        parsed = urlparse(source)
        src = Path(unquote(parsed.path))
        if src.is_file():
            ckpt = dest_dir / src.name
            if not ckpt.exists() or src.stat().st_mtime > ckpt.stat().st_mtime:
                shutil.copy2(src, ckpt)
            logger.warning(
                "Alias %s version %s still points at file:// source; "
                "re-import with import-weights so the registry owns the artifact.",
                alias,
                version,
            )
    if ckpt is None:
        vault_hit = _restore_from_vault(
            meta,
            dest_dir=dest_dir / "from_vault",
            runner=vault_runner,
        )
        if vault_hit is not None:
            return vault_hit
        raise ResolveError(
            f"Downloaded {uri} but found no checkpoint file under {downloaded}. "
            "No GitHub Release vault tag on this version."
        )

    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    return {
        "path": str(ckpt.resolve()),
        "version": version,
        "run_id": run_id,
        "alias": alias,
        "uri": uri,
        "name": REGISTERED_MODEL_NAME,
        "sha256": digest,
        "sha256_16": digest[:16],
        "source": source,
        "cache_hit": False,
        "restored_from_vault": False,
    }


def _restore_from_vault(
    meta: dict[str, Any],
    *,
    dest_dir: Path,
    runner: GhRunner | None,
) -> dict[str, Any] | None:
    tags = dict(meta.get("tags") or {})
    tag = (tags.get("vault_release_tag") or "").strip()
    expected = (tags.get("checkpoint_sha256") or "").strip()
    if not tag or not expected:
        return None
    try:
        ckpt = download_checkpoint(
            tag=tag,
            dest_dir=dest_dir,
            expected_sha256=expected,
            runner=runner,
            repo=tags.get("vault_repo") or None,
        )
    except VaultError:
        return None
    digest = sha256_file(ckpt)
    return {
        "path": str(ckpt.resolve()),
        "version": meta.get("version"),
        "run_id": meta.get("run_id"),
        "alias": meta.get("alias"),
        "uri": resolve_alias_uri(str(meta.get("alias") or "champion")),
        "name": REGISTERED_MODEL_NAME,
        "sha256": digest,
        "sha256_16": digest[:16],
        "source": f"github_release:{tag}",
        "cache_hit": False,
        "restored_from_vault": True,
    }


def _find_checkpoint(
    root: Path, prefer: tuple[str, ...]
) -> Path | None:
    if root.is_file() and root.suffix == ".pt":
        return root
    if not root.is_dir():
        return None
    pts = sorted(root.rglob("*.pt"))
    if not pts:
        # pyfunc often nests artifacts/checkpoint
        for p in root.rglob("*"):
            if p.is_file() and any(s in p.name.lower() for s in prefer):
                if p.suffix in {".pt", ".pth", ".bin", ""} or "checkpoint" in p.name:
                    return p
        return None
    # Prefer names that look like checkpoints
    for p in pts:
        low = p.name.lower()
        if any(s in low for s in ("best", "affinity", "checkpoint", "weights")):
            return p
    return pts[0]
