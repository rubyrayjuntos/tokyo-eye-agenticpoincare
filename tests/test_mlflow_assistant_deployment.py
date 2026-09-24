import subprocess
from pathlib import Path

import yaml

from scripts.mlflow_assistant_preflight import Check, CheckResult, build_checks, main, run_checks

ROOT = Path(__file__).resolve().parents[1]


def test_mlflow_image_contains_pinned_codex() -> None:
    dockerfile = (ROOT / "Dockerfile.mlflow").read_text()
    assert "FROM node:22-bookworm-slim AS node" in dockerfile
    assert "FROM tokyoeye-science:latest" in dockerfile
    assert "npm install --global @openai/codex@0.146.0 @anthropic-ai/claude-code" in dockerfile
    assert "codex --version" in dockerfile
    assert "claude --version" in dockerfile
    assert "mlflow_assistant_localhost_gate.py" in dockerfile
    assert "scripts/mlflow_assistant_localhost_gate.py" in dockerfile
    assert dockerfile.rstrip().endswith("USER appuser")


def test_mlflow_compose_security_boundary() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    mlflow = compose["services"]["mlflow"]

    assert mlflow["build"]["dockerfile"] == "Dockerfile.mlflow"
    assert mlflow["image"] == "tokyoeye-mlflow:latest"
    assert mlflow["ports"] == ["127.0.0.1:5000:5000"]
    assert mlflow["environment"]["MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT"] == "1"
    assert "./:/workspace:rw" in mlflow["volumes"]
    assert "mlflow_codex_auth:/home/appuser/.codex" in mlflow["volumes"]
    assert "${HOME}/.claude:/home/appuser/.claude:rw" in mlflow["volumes"]
    assert "mlflow_assistant_cfg:/home/appuser/.mlflow/assistant" in mlflow["volumes"]
    assert "mlflow_codex_auth" in compose["volumes"]
    assert "mlflow_assistant_cfg" in compose["volumes"]


def test_mlflow_appuser_matches_host_uid_for_claude_credentials() -> None:
    dockerfile = (ROOT / "Dockerfile.mlflow").read_text()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    args = compose["services"]["mlflow"]["build"]["args"]

    assert "ARG APPUSER_UID=1000" in dockerfile
    assert "usermod -o -u" in dockerfile
    assert args["APPUSER_UID"] == "${HOST_UID:-1000}"
    assert args["APPUSER_GID"] == "${HOST_GID:-1000}"


def test_mlflow_governance_storage_is_preserved() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    mlflow = compose["services"]["mlflow"]
    command = "\n".join(mlflow["command"])

    assert "--serve-artifacts" in command
    assert "--artifacts-destination /app/mlflow-artifacts" in command
    assert "--default-artifact-root mlflow-artifacts:/" in command
    assert "./mlflow-artifacts:/app/mlflow-artifacts" in mlflow["volumes"]


def test_preflight_checks_cover_install_auth_config_and_health() -> None:
    checks = build_checks()
    names = [check.name for check in checks]
    commands = "\n".join(check.command for check in checks)

    assert names == [
        "mlflow-health",
        "mlflow-version",
        "codex-installed",
        "claude-installed",
        "assistant-config",
        "codex-auth",
        "claude-auth",
        "claude-home-mount",
    ]
    assert "http://localhost:5000/health" in commands
    assert "mlflow.__version__" in commands
    assert "codex --version" in commands
    assert "claude --version" in commands
    assert "codex login status" in commands
    assert "claude -p hi" in commands
    assert "config.json" in commands
    assert "/workspace" in commands
    assert ".credentials.json" in commands

    claude_only = build_checks(["claude_code"])
    assert [c.name for c in claude_only] == [
        "mlflow-health",
        "mlflow-version",
        "codex-installed",
        "claude-installed",
        "assistant-config",
        "claude-home-mount",
        "claude-auth",
    ]


def test_preflight_exit_status(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "scripts.mlflow_assistant_preflight.run_checks",
        lambda checks: [
            CheckResult(name="mlflow-health", passed=True, detail="ok"),
            CheckResult(name="codex-auth", passed=False, detail="not logged in"),
        ],
    )

    assert main() == 1
    output = capsys.readouterr().out
    assert "PASS  mlflow-health" in output
    assert "FAIL  codex-auth" in output


def test_preflight_check_timeout(monkeypatch) -> None:
    checks = [
        Check("slow-check", "sleep 999"),
        Check("fast-check", "echo ok"),
    ]
    call_count = 0

    def fake_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        assert kwargs["timeout"] == 30
        if call_count == 1:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=30)
        return subprocess.CompletedProcess(args[0], 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("scripts.mlflow_assistant_preflight.subprocess.run", fake_run)

    results = run_checks(checks)

    assert len(results) == 2
    assert results[0].name == "slow-check"
    assert results[0].passed is False
    assert results[0].detail == "timed out after 30s"
    assert results[1].name == "fast-check"
    assert results[1].passed is True
    assert results[1].detail == "ok"


def test_assistant_runbook_documents_bootstrap_and_security() -> None:
    runbook = (ROOT / "docs/training/MLFLOW_ASSISTANT.md").read_text()
    assert "docker compose exec mlflow codex login" in runbook
    assert "scripts/mlflow_claude_host_auth_acl.sh" in runbook
    assert "${HOME}/.claude" in runbook
    assert "MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT" in runbook
    assert "You do not have permission to access this resource" in runbook
    assert "docker compose exec mlflow mlflow assistant --configure" in runbook
    assert "python scripts/mlflow_assistant_preflight.py" in runbook
    assert "Claude Code" in runbook
    assert "read/write" in runbook
    assert "TokyoEye@champion" in runbook
