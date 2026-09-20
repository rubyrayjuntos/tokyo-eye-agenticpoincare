#!/usr/bin/env python3
"""Non-mutating preflight for the containerized MLflow Assistant (Codex / Claude Code)."""

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

_CORE_CHECKS = [
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
    Check("claude-installed", "claude --version"),
    Check(
        "assistant-config",
        "python -c \"import json,pathlib; "
        "p=pathlib.Path('/home/appuser/.mlflow/assistant/config.json'); "
        "d=json.loads(p.read_text()); "
        "selected=[n for n,c in d.get('providers',{}).items() if c.get('selected')]; "
        "assert selected, 'no provider selected'; "
        "assert any(x.get('location')=='/workspace' for x in d.get('projects',{}).values()); "
        "print(','.join(selected), p)\"",
    ),
]

_CODEX_AUTH = Check("codex-auth", "codex login status")
_CLAUDE_AUTH = Check(
    "claude-auth",
    "claude -p hi --max-turns 1 --output-format json",
)
_CLAUDE_MOUNT = Check(
    "claude-home-mount",
    "python -c \"import pathlib; "
    "p=pathlib.Path('/home/appuser/.claude/.credentials.json'); "
    "assert p.is_file(), p; "
    "assert p.stat().st_size > 0; "
    "p.open('rb').read(1); "
    "print(p)\"",
)


def _selected_providers() -> list[str]:
    proc = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "mlflow",
            "sh",
            "-lc",
            "python -c \"import json,pathlib; "
            "d=json.loads(pathlib.Path('/home/appuser/.mlflow/assistant/config.json').read_text()); "
            "print(','.join(n for n,c in d.get('providers',{}).items() if c.get('selected')))\"",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=CHECK_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        return []
    line = (proc.stdout or "").strip()
    return [p for p in line.split(",") if p]


def build_checks(selected: list[str] | None = None) -> list[Check]:
    """Build checks. If selected is None, include both provider auth probes (tests)."""
    checks = list(_CORE_CHECKS)
    if selected is None:
        checks.extend([_CODEX_AUTH, _CLAUDE_AUTH, _CLAUDE_MOUNT])
        return checks
    if "codex" in selected:
        checks.append(_CODEX_AUTH)
    if "claude_code" in selected:
        checks.extend([_CLAUDE_MOUNT, _CLAUDE_AUTH])
    return checks


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
    selected = _selected_providers()
    checks = build_checks(selected if selected else None)
    results = run_checks(checks)
    for result in results:
        state = "PASS" if result.passed else "FAIL"
        print(f"{state:<5} {result.name}: {result.detail}")
    if selected:
        print(f"INFO  selected provider(s): {','.join(selected)}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
