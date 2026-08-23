"""GitHub Release vault — durability replica for TokyoEye weights."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from science.tokyo_eye.governance.vault import (
    VaultError,
    VaultRequired,
    assert_vaulted,
    download_checkpoint,
    immutable_tag,
    moving_tag,
    sha256_file,
    upload_checkpoint,
)


class FakeGh:
    """In-memory ``gh`` stand-in. Stores release assets under ``root``."""

    def __init__(self, root: Path, repo: str = "owner/tokyo-eye-agenticpoincare") -> None:
        self.root = root
        self.repo = repo
        self.releases: dict[str, dict] = {}
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        self.calls.append(list(argv))
        args = argv[1:] if argv and argv[0] == "gh" else list(argv)
        if args[:2] == ["repo", "view"]:
            return CompletedProcess(argv, 0, stdout=f"{self.repo}\n", stderr="")
        if args[:2] == ["release", "view"]:
            tag = args[2]
            rec = self.releases.get(tag)
            if rec is None:
                return CompletedProcess(argv, 1, stdout="", stderr="release not found\n")
            payload = {
                "tagName": tag,
                "body": json.dumps(rec["notes"]),
                "assets": [{"name": name, "size": len(data)} for name, data in rec["files"].items()],
            }
            return CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")
        if args[:2] == ["release", "create"]:
            return self._create(argv, args)
        if args[:2] == ["release", "download"]:
            return self._download(argv, args)
        if args[:2] == ["release", "upload"]:
            return self._upload(argv, args)
        if args[:2] == ["release", "edit"]:
            return self._edit(argv, args)
        return CompletedProcess(argv, 1, stdout="", stderr=f"unhandled: {args}\n")

    def _flag(self, args: list[str], name: str) -> str | None:
        if name in args:
            idx = args.index(name)
            if idx + 1 < len(args):
                return args[idx + 1]
        return None

    def _files(self, args: list[str]) -> list[Path]:
        skip_next = False
        flags_with_val = {
            "--repo",
            "--title",
            "--notes",
            "--pattern",
            "--dir",
            "--json",
            "-q",
        }
        out: list[Path] = []
        for i, tok in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if tok in flags_with_val:
                skip_next = True
                continue
            if tok.startswith("--"):
                continue
            if i < 3:
                continue
            path = Path(tok)
            if path.is_file():
                out.append(path)
        return out

    def _create(self, argv: list[str], args: list[str]) -> CompletedProcess[str]:
        tag = args[2]
        if tag in self.releases:
            return CompletedProcess(argv, 1, stdout="", stderr="already exists\n")
        notes_raw = self._flag(args, "--notes") or "{}"
        notes = json.loads(notes_raw)
        files = {p.name: p.read_bytes() for p in self._files(args)}
        self.releases[tag] = {"notes": notes, "files": files}
        return CompletedProcess(argv, 0, stdout="created\n", stderr="")

    def _download(self, argv: list[str], args: list[str]) -> CompletedProcess[str]:
        tag = args[2]
        rec = self.releases.get(tag)
        dest = Path(self._flag(args, "--dir") or ".")
        if rec is None:
            return CompletedProcess(argv, 1, stdout="", stderr="not found\n")
        dest.mkdir(parents=True, exist_ok=True)
        pattern = self._flag(args, "--pattern") or "*"
        for name, data in rec["files"].items():
            if Path(name).match(pattern) or (
                pattern == "*.pt" and name.endswith(".pt")
            ):
                (dest / name).write_bytes(data)
        return CompletedProcess(argv, 0, stdout="downloaded\n", stderr="")

    def _upload(self, argv: list[str], args: list[str]) -> CompletedProcess[str]:
        tag = args[2]
        rec = self.releases.setdefault(tag, {"notes": {}, "files": {}})
        for path in self._files(args):
            rec["files"][path.name] = path.read_bytes()
        return CompletedProcess(argv, 0, stdout="uploaded\n", stderr="")

    def _edit(self, argv: list[str], args: list[str]) -> CompletedProcess[str]:
        tag = args[2]
        rec = self.releases.get(tag)
        if rec is None:
            return CompletedProcess(argv, 1, stdout="", stderr="not found\n")
        notes_raw = self._flag(args, "--notes")
        if notes_raw:
            rec["notes"] = json.loads(notes_raw)
        return CompletedProcess(argv, 0, stdout="edited\n", stderr="")


def test_upload_creates_immutable_and_moving_tags(tmp_path: Path) -> None:
    ckpt = tmp_path / "tokyoeye_best.pt"
    ckpt.write_bytes(b"eqf-seal-bytes")
    digest = sha256_file(ckpt)
    gh = FakeGh(tmp_path)

    record = upload_checkpoint(
        checkpoint_path=ckpt,
        alias="champion",
        capability_goal="eqf_baseline_seal",
        runner=gh,
    )
    assert record.reused is False
    assert record.tag == immutable_tag(digest)
    assert record.moving_tag == moving_tag("champion")
    assert record.sha256 == digest
    assert record.tag in gh.releases
    assert moving_tag("champion") in gh.releases
    assert ckpt.read_bytes() == b"eqf-seal-bytes"


def test_upload_reuses_same_digest(tmp_path: Path) -> None:
    ckpt = tmp_path / "tokyoeye_best.pt"
    ckpt.write_bytes(b"same-bytes")
    gh = FakeGh(tmp_path)
    first = upload_checkpoint(
        checkpoint_path=ckpt,
        alias="champion",
        capability_goal="eqf_baseline_seal",
        runner=gh,
    )
    second = upload_checkpoint(
        checkpoint_path=ckpt,
        alias="champion",
        capability_goal="eqf_baseline_seal",
        runner=gh,
    )
    assert first.tag == second.tag
    assert second.reused is True


def test_upload_refuses_tag_with_different_sha(tmp_path: Path) -> None:
    ckpt = tmp_path / "tokyoeye_best.pt"
    ckpt.write_bytes(b"payload-a")
    digest = sha256_file(ckpt)
    tag = immutable_tag(digest)
    gh = FakeGh(tmp_path)
    other = "ab" * 32
    gh.releases[tag] = {
        "notes": {"sha256": other, "asset": "tokyoeye_best.pt"},
        "files": {"tokyoeye_best.pt": b"other"},
    }
    with pytest.raises(VaultError, match="different bytes"):
        upload_checkpoint(
            checkpoint_path=ckpt,
            alias="champion",
            capability_goal="eqf_baseline_seal",
            runner=gh,
        )


def test_download_verifies_sha256(tmp_path: Path) -> None:
    ckpt = tmp_path / "tokyoeye_best.pt"
    ckpt.write_bytes(b"restore-me")
    digest = sha256_file(ckpt)
    gh = FakeGh(tmp_path)
    record = upload_checkpoint(
        checkpoint_path=ckpt,
        alias="champion",
        capability_goal="eqf_baseline_seal",
        runner=gh,
    )
    dest = tmp_path / "out"
    got = download_checkpoint(
        tag=record.tag,
        dest_dir=dest,
        expected_sha256=digest,
        runner=gh,
    )
    assert got.read_bytes() == b"restore-me"


def test_assert_vaulted_missing_release(tmp_path: Path) -> None:
    gh = FakeGh(tmp_path)
    with pytest.raises(VaultRequired, match="vault-upload"):
        assert_vaulted(sha256="aa" * 32, alias="champion", runner=gh)


def test_set_alias_champion_blocked_without_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"no-vault")

    from science.tokyo_eye.governance.entrypoints import main
    from science.tokyo_eye.governance.import_weights import import_checkpoint_into_mlflow
    from science.tokyo_eye.governance.taxonomy import experiment_path

    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(tracking)
    mlflow.set_experiment(experiment_path("geometric", "full-stack"))
    imported = import_checkpoint_into_mlflow(
        checkpoint_path=ckpt,
        domain="geometric",
        subsystem="full-stack",
        capability_goal="eqf_baseline_seal",
        tracking_uri=tracking,
        alias=None,
        log_pyfunc=False,
    )
    rc = main(
        [
            "--tracking-uri",
            tracking,
            "set-alias",
            "--version",
            imported["register"]["version"],
            "--alias",
            "champion",
        ]
    )
    assert rc == 2


def test_import_champion_requires_vault_then_resolve_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking)
    monkeypatch.setenv("TOKYOEYE_MLFLOW_CACHE", str(tmp_path / "alias_cache"))
    monkeypatch.setenv("TOKYOEYE_VAULT_REPO", "owner/tokyo-eye-agenticpoincare")

    ckpt = tmp_path / "tokyoeye_best.pt"
    ckpt.write_bytes(b"champion-bytes-keep")
    gh = FakeGh(tmp_path)

    from science.tokyo_eye.governance.import_weights import import_checkpoint_into_mlflow
    from science.tokyo_eye.governance.resolve import resolve_alias_checkpoint

    imported = import_checkpoint_into_mlflow(
        checkpoint_path=ckpt,
        domain="geometric",
        subsystem="full-stack",
        capability_goal="eqf_baseline_seal",
        tracking_uri=tracking,
        alias="champion",
        overwrite_alias=True,
        log_pyfunc=False,
        vault_runner=gh,
        confirm_champion=True,
    )
    assert imported["alias"]["alias"] == "champion"
    assert imported["vault"]["tag"] in gh.releases

    # Wipe MLflow artifact store + local cache (the loss we just had).
    import shutil

    mlruns = tmp_path / "mlruns"
    for artifacts in mlruns.rglob("artifacts"):
        if artifacts.is_dir():
            shutil.rmtree(artifacts, ignore_errors=True)
    cache = tmp_path / "alias_cache"
    if cache.exists():
        shutil.rmtree(cache)

    resolved = resolve_alias_checkpoint(
        alias="champion",
        tracking_uri=tracking,
        vault_runner=gh,
    )
    assert Path(resolved["path"]).read_bytes() == b"champion-bytes-keep"
    assert resolved.get("restored_from_vault") is True
