#!/usr/bin/env python3
"""Non-mutating preflight for the containerized MLflow Codex Assistant."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    command: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


CHECK_TIMEOUT_SECONDS = 30


def build_checks() -> list[Check]:
    return [
        Check(
            "mlflow-health",
            "python -c \"import urllib.request; "
            "print(urllib.request.urlopen('http://localhost:5000/health').status)\"",
        ),
        Check(
            "mlflow-version",
            "python -c \"import mlflow; from packaging.version import Version; "
            "print(mlflow.__version__); assert Version(mlflow.__version__) >= Version('3.13')\"",
        ),
        Check("codex-installed", "codex --version"),
        Check("codex-auth", "codex login status"),
        Check(
            "assistant-config",
            "python -c \"import json,pathlib; "
            "p=pathlib.Path('/home/appuser/.mlflow/assistant/config.json'); "
            "d=json.loads(p.read_text()); "
            "assert d['providers']['codex']['selected'] is True; "
            "assert any(x['location']=='/workspace' for x in d['projects'].values()); "
            "print(p)\"",
        ),
    ]


def run_checks(checks: list[Check]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        try:
            proc = subprocess.run(
                ["docker", "compose", "exec", "-T", "mlflow", "sh", "-lc", check.command],
                check=False,
                capture_output=True,
                text=True,
                timeout=CHECK_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            results.append(
                CheckResult(
                    check.name,
                    False,
                    f"timed out after {CHECK_TIMEOUT_SECONDS}s",
                )
            )
            continue
        detail = (proc.stdout if proc.returncode == 0 else proc.stderr).strip()
        results.append(CheckResult(check.name, proc.returncode == 0, detail))
    return results


def main() -> int:
    results = run_checks(build_checks())
    for result in results:
        state = "PASS" if result.passed else "FAIL"
        print(f"{state:<5} {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
