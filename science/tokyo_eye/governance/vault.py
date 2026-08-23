"""GitHub Release durability replica for Tokyo Eye weights.

MLflow remains identity SSOT. This module only stores and retrieves bytes.
Talks to GitHub through ``gh`` so tests can inject a runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from science.tokyo_eye.governance.registry import ALLOWED_ALIASES
from science.tokyo_eye.governance.taxonomy import REGISTERED_MODEL_NAME

VAULT_BACKEND = "github_release"
LINEAGE_SLUG = "eqf"
DEFAULT_REPO_ENV = "TOKYOEYE_VAULT_REPO"
TAG_PREFIX = f"tokyoeye-{LINEAGE_SLUG}"

GhRunner = Callable[..., subprocess.CompletedProcess[str]]


class VaultError(RuntimeError):
    """GitHub Release vault operation failed."""


class VaultRequired(VaultError):
    """Champion promote/alias requires a matching Release."""


class VaultUnavailable(VaultError):
    """``gh`` is missing, unauthenticated, or the runner failed to start."""


@dataclass(frozen=True)
class VaultRecord:
    backend: str
    repo: str
    tag: str
    moving_tag: str | None
    asset: str
    sha256: str
    sha256_16: str
    alias: str
    capability_goal: str
    reused: bool

    def as_tags(self) -> dict[str, str]:
        tags = {
            "vault_backend": self.backend,
            "vault_repo": self.repo,
            "vault_release_tag": self.tag,
            "checkpoint_sha256": self.sha256,
            "checkpoint_sha256_16": self.sha256_16,
            "vault_asset": self.asset,
        }
        if self.moving_tag:
            tags["vault_moving_tag"] = self.moving_tag
        return tags

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_tag(sha256: str) -> str:
    hex_digest = _require_sha256(sha256)
    return f"{TAG_PREFIX}-{hex_digest[:16]}"


def moving_tag(alias: str) -> str:
    if alias not in ALLOWED_ALIASES:
        raise VaultError(f"Alias {alias!r} not allowed; use one of {sorted(ALLOWED_ALIASES)}")
    return f"{TAG_PREFIX}-{alias}"


def sanitize_asset_name(filename: str) -> str:
    name = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not cleaned.endswith(".pt"):
        cleaned = f"{cleaned}.pt" if "." not in cleaned else cleaned
    if not cleaned or cleaned in {".", ".."}:
        raise VaultError(f"Invalid asset name from {filename!r}")
    return cleaned


def default_repo(*, runner: GhRunner | None = None) -> str:
    env = os.environ.get(DEFAULT_REPO_ENV, "").strip()
    if env:
        return env
    proc = _run_gh(
        ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        runner=runner,
    )
    repo = (proc.stdout or "").strip()
    if proc.returncode != 0 or not repo:
        raise VaultUnavailable(
            "Cannot determine GitHub repo. Set TOKYOEYE_VAULT_REPO or run `gh auth login`."
        )
    return repo


def upload_checkpoint(
    *,
    checkpoint_path: Path | str,
    alias: str,
    capability_goal: str,
    repo: str | None = None,
    runner: GhRunner | None = None,
    retarget_moving: bool = True,
) -> VaultRecord:
    """Upload bytes to an immutable Release; optionally retarget the moving tag."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if alias not in ALLOWED_ALIASES:
        raise VaultError(f"Alias {alias!r} not allowed; use one of {sorted(ALLOWED_ALIASES)}")

    digest = sha256_file(path)
    tag = immutable_tag(digest)
    asset = sanitize_asset_name(path.name)
    resolved_repo = repo or default_repo(runner=runner)
    existing = describe_release(tag=tag, repo=resolved_repo, runner=runner)
    if existing is not None:
        existing_sha = str(existing.get("sha256") or "")
        if existing_sha and existing_sha != digest:
            raise VaultError(
                f"Immutable tag {tag} already holds different bytes "
                f"({existing_sha[:16]}… vs {digest[:16]}…)"
            )
        record = VaultRecord(
            backend=VAULT_BACKEND,
            repo=resolved_repo,
            tag=tag,
            moving_tag=moving_tag(alias) if retarget_moving else None,
            asset=str(existing.get("asset") or asset),
            sha256=digest,
            sha256_16=digest[:16],
            alias=alias,
            capability_goal=capability_goal,
            reused=True,
        )
        if retarget_moving:
            _retarget_moving(record, source=path, runner=runner)
        return record

    notes = {
        "backend": VAULT_BACKEND,
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": alias,
        "capability_goal": capability_goal,
        "sha256": digest,
        "asset": asset,
    }
    work = path.parent
    staged = work / asset
    sidecar = work / f"{asset}.sha256"
    copied = False
    if staged.resolve() != path:
        shutil.copy2(path, staged)
        copied = True
    sidecar.write_text(f"{digest}  {asset}\n", encoding="utf-8")
    try:
        proc = _run_gh(
            [
                "release",
                "create",
                tag,
                str(staged),
                str(sidecar),
                "--repo",
                resolved_repo,
                "--title",
                f"TokyoEye EQF {alias} {digest[:16]}",
                "--notes",
                json.dumps(notes, indent=2),
                "--latest=false",
            ],
            runner=runner,
        )
        if proc.returncode != 0:
            raise VaultError(
                f"gh release create {tag} failed: {(proc.stderr or proc.stdout or '').strip()}"
            )
    finally:
        if copied and staged.exists():
            staged.unlink()
        if sidecar.exists():
            sidecar.unlink()

    record = VaultRecord(
        backend=VAULT_BACKEND,
        repo=resolved_repo,
        tag=tag,
        moving_tag=moving_tag(alias) if retarget_moving else None,
        asset=asset,
        sha256=digest,
        sha256_16=digest[:16],
        alias=alias,
        capability_goal=capability_goal,
        reused=False,
    )
    if retarget_moving:
        _retarget_moving(record, source=path, runner=runner)
    return record


def download_checkpoint(
    *,
    tag: str,
    dest_dir: Path | str,
    repo: str | None = None,
    runner: GhRunner | None = None,
    expected_sha256: str | None = None,
    pattern: str = "*.pt",
) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    resolved_repo = repo or default_repo(runner=runner)
    proc = _run_gh(
        [
            "release",
            "download",
            tag,
            "--repo",
            resolved_repo,
            "--pattern",
            pattern,
            "--dir",
            str(dest),
            "--clobber",
        ],
        runner=runner,
    )
    if proc.returncode != 0:
        raise VaultError(
            f"gh release download {tag} failed: {(proc.stderr or proc.stdout or '').strip()}"
        )
    pts = sorted(p for p in dest.glob(pattern) if p.is_file())
    if not pts:
        raise VaultError(f"Release {tag} downloaded but no files matched {pattern}")
    path = pts[0]
    digest = sha256_file(path)
    if expected_sha256 and digest != _require_sha256(expected_sha256):
        raise VaultError(
            f"Downloaded {path.name} sha256 {digest} != expected {expected_sha256}"
        )
    return path


def describe_release(
    *,
    tag: str,
    repo: str | None = None,
    runner: GhRunner | None = None,
) -> dict[str, Any] | None:
    resolved_repo = repo or default_repo(runner=runner)
    proc = _run_gh(
        ["release", "view", tag, "--repo", resolved_repo, "--json", "tagName,body,assets"],
        runner=runner,
    )
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VaultError(f"Invalid gh release view JSON for {tag}") from exc
    notes = _parse_notes(payload.get("body"))
    assets = payload.get("assets") or []
    asset_name = notes.get("asset")
    if not asset_name:
        for item in assets:
            name = str(item.get("name") or "")
            if name.endswith(".pt"):
                asset_name = name
                break
    return {
        "tag": payload.get("tagName") or tag,
        "repo": resolved_repo,
        "sha256": notes.get("sha256"),
        "asset": asset_name,
        "alias": notes.get("alias"),
        "capability_goal": notes.get("capability_goal"),
        "assets": [str(item.get("name") or "") for item in assets],
        "notes": notes,
    }


def assert_vaulted(
    *,
    sha256: str,
    alias: str,
    repo: str | None = None,
    runner: GhRunner | None = None,
) -> VaultRecord:
    """Fail closed unless the immutable Release exists with this digest."""
    digest = _require_sha256(sha256)
    tag = immutable_tag(digest)
    info = describe_release(tag=tag, repo=repo, runner=runner)
    if info is None:
        raise VaultRequired(
            f"@champion requires GitHub Release {tag}. "
            "Upload first: python -m science.tokyo_eye.governance.entrypoints vault-upload"
        )
    found = str(info.get("sha256") or "")
    if found and found != digest:
        raise VaultError(
            f"Release {tag} sha256 {found} does not match checkpoint {digest}"
        )
    return VaultRecord(
        backend=VAULT_BACKEND,
        repo=str(info.get("repo") or repo or ""),
        tag=tag,
        moving_tag=moving_tag(alias),
        asset=str(info.get("asset") or ""),
        sha256=digest,
        sha256_16=digest[:16],
        alias=alias,
        capability_goal=str(info.get("capability_goal") or ""),
        reused=True,
    )


def _retarget_moving(record: VaultRecord, *, source: Path, runner: GhRunner | None) -> None:
    if not record.moving_tag:
        return
    notes = {
        "backend": record.backend,
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": record.alias,
        "capability_goal": record.capability_goal,
        "sha256": record.sha256,
        "asset": record.asset,
        "immutable_tag": record.tag,
    }
    existing = describe_release(tag=record.moving_tag, repo=record.repo, runner=runner)
    work = source.parent
    staged = work / record.asset
    sidecar = work / f"{record.asset}.sha256"
    copied = False
    if staged.resolve() != source:
        shutil.copy2(source, staged)
        copied = True
    sidecar.write_text(f"{record.sha256}  {record.asset}\n", encoding="utf-8")
    try:
        if existing is None:
            proc = _run_gh(
                [
                    "release",
                    "create",
                    record.moving_tag,
                    str(staged),
                    str(sidecar),
                    "--repo",
                    record.repo,
                    "--title",
                    f"TokyoEye EQF {record.alias} (moving)",
                    "--notes",
                    json.dumps(notes, indent=2),
                    "--latest=false",
                ],
                runner=runner,
            )
        else:
            proc = _run_gh(
                [
                    "release",
                    "upload",
                    record.moving_tag,
                    str(staged),
                    str(sidecar),
                    "--repo",
                    record.repo,
                    "--clobber",
                ],
                runner=runner,
            )
            if proc.returncode == 0:
                proc = _run_gh(
                    [
                        "release",
                        "edit",
                        record.moving_tag,
                        "--repo",
                        record.repo,
                        "--notes",
                        json.dumps(notes, indent=2),
                    ],
                    runner=runner,
                )
        if proc.returncode != 0:
            raise VaultError(
                f"Failed to retarget {record.moving_tag}: "
                f"{(proc.stderr or proc.stdout or '').strip()}"
            )
    finally:
        if copied and staged.exists():
            staged.unlink()
        if sidecar.exists():
            sidecar.unlink()


def _parse_notes(body: Any) -> dict[str, Any]:
    text = str(body or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _require_sha256(value: str) -> str:
    digest = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise VaultError(f"Invalid sha256: {value!r}")
    return digest


def _run_gh(
    args: list[str],
    *,
    runner: GhRunner | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["gh", *args]
    if runner is not None:
        return runner(command)
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **(env or {})},
        )
    except FileNotFoundError as exc:
        raise VaultUnavailable(
            "GitHub CLI `gh` is not on PATH. Install gh and run `gh auth login`."
        ) from exc
