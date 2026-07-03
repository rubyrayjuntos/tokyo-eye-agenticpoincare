"""Codegen hygiene gates for onboard_contract.yaml artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "science/contracts/onboard_contract.yaml"
JOB_SCHEMA_PATH = ROOT / "science/compute/job_schema.json"
ONBOARD_TS_PATH = (
    ROOT / "visualizer/frontend/src/lib/generated/onboard.ts"
)


def expected_job_schema_text() -> str:
    from science.compute.job_schema import build_job_catalog

    catalog = build_job_catalog()
    return json.dumps(catalog, indent=2) + "\n"


def expected_onboard_ts_text() -> str:
    from science.contracts.generate_typescript import generate_typescript

    return generate_typescript()


def audit_contract_sync() -> list[str]:
    """Fail when generated artifacts are stale relative to the contract."""
    errors: list[str] = []

    if not JOB_SCHEMA_PATH.is_file():
        errors.append(f"missing generated file: {JOB_SCHEMA_PATH.relative_to(ROOT)}")
    else:
        actual = JOB_SCHEMA_PATH.read_text(encoding="utf-8")
        expected = expected_job_schema_text()
        if actual != expected:
            errors.append(
                f"{JOB_SCHEMA_PATH.relative_to(ROOT)} is stale — run make contract-sync"
            )

    if not ONBOARD_TS_PATH.is_file():
        errors.append(f"missing generated file: {ONBOARD_TS_PATH.relative_to(ROOT)}")
    else:
        actual = ONBOARD_TS_PATH.read_text(encoding="utf-8")
        expected = expected_onboard_ts_text()
        if actual != expected:
            errors.append(
                f"{ONBOARD_TS_PATH.relative_to(ROOT)} is stale — run make contract-sync"
            )

    return errors


def _git_diff(ref: str, path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "diff", ref, "--", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    if proc.returncode not in (0, 1):
        return ""
    return proc.stdout


def audit_contract_version_bump(*, base_ref: str = "HEAD") -> list[str]:
    """Fail when contract changed vs *base_ref* without a version-line diff."""
    if not CONTRACT_PATH.is_file():
        return [f"missing contract file: {CONTRACT_PATH.relative_to(ROOT)}"]

    diff = _git_diff(base_ref, CONTRACT_PATH)
    if not diff.strip():
        return []

    if "version:" not in diff:
        return [
            "science/contracts/onboard_contract.yaml changed without bumping "
            f"version: (diff vs {base_ref})"
        ]
    return []
