"""Runtime helpers for provenance completeness at governed write time."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path

from science.dtie.common.normalizer_payloads import RunType

logger = logging.getLogger(__name__)

_CHECKPOINT_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

_MODEL_BEARING_RUN_TYPES = frozenset({RunType.INFERENCE, RunType.TRAINING})


def is_provenance_strict() -> bool:
    """Return True when governed writes must satisfy full provenance contracts."""
    explicit = os.environ.get("PROVENANCE_STRICT")
    if explicit is not None:
        return explicit.strip().lower() not in {"0", "false", "no", "off"}
    environment = os.environ.get("ENVIRONMENT", "dev").strip().lower()
    return environment in {"prod", "staging"}


def resolve_code_version(explicit: str | None = None) -> str | None:
    """Resolve git commit / deploy revision for provenance."""
    if explicit and explicit.strip():
        return explicit.strip()

    for key in ("CODE_VERSION", "GIT_COMMIT", "GIT_SHA"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
            if revision:
                return revision
    except (OSError, subprocess.SubprocessError):
        pass

    return None


def sha256_file(path: Path) -> str:
    """Compute lowercase hex SHA-256 for a checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint_path(checkpoint_uri: str) -> Path | None:
    """Resolve a checkpoint URI/path to a readable local file."""
    candidate = Path(checkpoint_uri)
    if candidate.is_file():
        return candidate

    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir:
        nested = Path(checkpoint_dir) / candidate
        if nested.is_file():
            return nested

    repo_relative = Path.cwd() / candidate
    if repo_relative.is_file():
        return repo_relative

    return None


def resolve_checkpoint_sha256(
    checkpoint_uri: str | None,
    *,
    explicit: str | None = None,
) -> str | None:
    """Resolve checkpoint SHA-256 from explicit value or on-disk checkpoint."""
    if explicit and explicit.strip():
        return explicit.strip().lower()

    if not checkpoint_uri:
        return None

    path = _resolve_checkpoint_path(checkpoint_uri)
    if path is None:
        logger.warning("Checkpoint file not found for provenance hash: %s", checkpoint_uri)
        return None

    try:
        return sha256_file(path)
    except OSError as exc:
        logger.warning("Failed to hash checkpoint %s: %s", path, exc)
        return None


def validate_provenance_fields(
    *,
    run_type: RunType,
    code_version: str | None,
    checkpoint_sha256: str | None,
    strict: bool | None = None,
) -> list[str]:
    """Return validation error messages; empty list means provenance is acceptable."""
    if strict is None:
        strict = is_provenance_strict()
    if not strict:
        return []

    errors: list[str] = []

    if not code_version or not code_version.strip():
        errors.append("code_version is required for governed writes")

    if run_type in _MODEL_BEARING_RUN_TYPES:
        if not checkpoint_sha256:
            errors.append(
                f"checkpoint_sha256 is required for {run_type.value} runs"
            )
        elif not _CHECKPOINT_SHA256_RE.match(checkpoint_sha256.lower()):
            errors.append(
                "checkpoint_sha256 must be a 64-character lowercase hex string"
            )

    return errors
