# MLflow Assistant with OpenAI Codex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run MLflow 3.14's in-app Assistant with its OpenAI Codex provider inside the existing `mlflow` Docker service, with persistent ChatGPT login and read/write repository access.

**Architecture:** Build a dedicated `tokyoeye-mlflow` image from the existing science image, adding Node.js and a pinned Codex CLI. The MLflow server invokes Codex in-process; named volumes persist Codex auth and Assistant configuration, while a localhost-only port and `/workspace` bind mount define the security boundary.

**Tech Stack:** Docker/Compose, MLflow 3.14, Node.js 22, `@openai/codex@0.146.0`, Python 3.11+, pytest, PyYAML.

## Global Constraints

- MLflow Assistant's Codex provider uses `--sandbox danger-full-access`; the container and mounts are the security boundary.
- Repository access is intentionally read/write at `/workspace`.
- Authentication is a persistent `codex login` using the user's ChatGPT account; do not add an OpenAI API key to `.env`, Compose, or git.
- Publish MLflow only on `127.0.0.1:5000`; Docker-network clients continue to use `http://mlflow:5000`.
- Do not change MLflow backend, artifact proxy, registry, or alias configuration.
- Setup and verification must not move `TokyoEye@champion` or `TokyoEye@experimental`.
- Do not commit unless the user explicitly requests a commit.

---

## File Map

- Create `Dockerfile.mlflow` — MLflow-only runtime image with Node.js and Codex CLI.
- Modify `docker-compose.yml` — use the dedicated image; add localhost binding, repository/auth/config mounts, and named volumes.
- Create `tests/test_mlflow_assistant_deployment.py` — static deployment-contract and preflight unit tests.
- Create `scripts/mlflow_assistant_preflight.py` — non-mutating host-side checker for the live container.
- Create `docs/training/MLFLOW_ASSISTANT.md` — bootstrap, security, recovery, and verification runbook.
- Modify `docs/training/GNN_LIFECYCLE.md` — link the Assistant runbook from lifecycle operations.

---

### Task 1: Dedicated Codex-enabled MLflow image and Compose boundary

**Files:**
- Create: `Dockerfile.mlflow`
- Modify: `docker-compose.yml:150-205`
- Create: `tests/test_mlflow_assistant_deployment.py`

**Interfaces:**
- Produces image `tokyoeye-mlflow:latest`.
- Produces container paths `/workspace`, `/home/appuser/.codex`, and `/home/appuser/.mlflow/assistant`.
- Preserves internal service endpoint `http://mlflow:5000`.

- [ ] **Step 1: Write deployment-contract tests**

Create `tests/test_mlflow_assistant_deployment.py`:

```python
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_mlflow_image_contains_pinned_codex() -> None:
    dockerfile = (ROOT / "Dockerfile.mlflow").read_text()
    assert "FROM node:22-bookworm-slim AS node" in dockerfile
    assert "FROM tokyoeye-science:latest" in dockerfile
    assert "npm install --global @openai/codex@0.146.0" in dockerfile
    assert "codex --version" in dockerfile
    assert dockerfile.rstrip().endswith("USER appuser")


def test_mlflow_compose_security_boundary() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    mlflow = compose["services"]["mlflow"]

    assert mlflow["build"]["dockerfile"] == "Dockerfile.mlflow"
    assert mlflow["image"] == "tokyoeye-mlflow:latest"
    assert mlflow["ports"] == ["127.0.0.1:5000:5000"]
    assert "./:/workspace:rw" in mlflow["volumes"]
    assert "mlflow_codex_auth:/home/appuser/.codex" in mlflow["volumes"]
    assert "mlflow_assistant_cfg:/home/appuser/.mlflow/assistant" in mlflow["volumes"]
    assert "mlflow_codex_auth" in compose["volumes"]
    assert "mlflow_assistant_cfg" in compose["volumes"]


def test_mlflow_governance_storage_is_preserved() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    mlflow = compose["services"]["mlflow"]
    command = "\n".join(mlflow["command"])

    assert "--serve-artifacts" in command
    assert "--artifacts-destination /app/mlflow-artifacts" in command
    assert "--default-artifact-root mlflow-artifacts:/" in command
    assert "./mlflow-artifacts:/app/mlflow-artifacts" in mlflow["volumes"]
```

- [ ] **Step 2: Run tests and verify the expected failure**

Run:

```bash
python -m pytest tests/test_mlflow_assistant_deployment.py -q
```

Expected: FAIL because `Dockerfile.mlflow` does not exist and Compose does not yet declare the Assistant mounts.

- [ ] **Step 3: Create the dedicated image**

Create `Dockerfile.mlflow`:

```dockerfile
# MLflow server image with the in-app OpenAI Codex Assistant provider.
# Codex runs as a subprocess of mlflow server, so it must be in this image.
FROM node:22-bookworm-slim AS node

FROM tokyoeye-science:latest

USER root

# Copy the official Node runtime rather than installing an unpinned distro package.
COPY --from=node /usr/local/ /usr/local/

RUN npm install --global @openai/codex@0.146.0 \
    && codex --version \
    && mkdir -p /home/appuser/.codex /home/appuser/.mlflow/assistant /workspace \
    && chown -R appuser:appuser \
        /home/appuser/.codex \
        /home/appuser/.mlflow \
        /workspace

USER appuser
```

- [ ] **Step 4: Update the `mlflow` Compose service**

In `docker-compose.yml`, change only the `mlflow` image/build, port, and mounts:

```yaml
  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
      cache_from:
        - tokyoeye-science:latest
    image: tokyoeye-mlflow:latest
    # ...existing environment, command, depends_on, network, and healthcheck...
    ports:
      - "127.0.0.1:5000:5000"
    volumes:
      - ./mlflow-artifacts:/app/mlflow-artifacts
      - ./checkpoints:/app/checkpoints
      - ./experiments:/app/experiments
      - ./:/workspace:rw
      - mlflow_codex_auth:/home/appuser/.codex
      - mlflow_assistant_cfg:/home/appuser/.mlflow/assistant
```

Add under the existing top-level `volumes:` block:

```yaml
  mlflow_codex_auth:
    driver: local
  mlflow_assistant_cfg:
    driver: local
```

- [ ] **Step 5: Run deployment tests and Compose validation**

Run:

```bash
python -m pytest tests/test_mlflow_assistant_deployment.py -q
docker compose config --quiet
```

Expected: tests PASS and Compose exits 0 with no output.

- [ ] **Step 6: Review the task diff**

Run:

```bash
git diff -- Dockerfile.mlflow docker-compose.yml tests/test_mlflow_assistant_deployment.py
```

Expected: only the dedicated image, localhost binding, Assistant mounts/volumes, and deployment tests are present. Do not commit without explicit authorization.

---

### Task 2: Non-mutating Assistant preflight

**Files:**
- Create: `scripts/mlflow_assistant_preflight.py`
- Modify: `tests/test_mlflow_assistant_deployment.py`

**Interfaces:**
- Produces `Check(name: str, command: str)` and `run_checks(...) -> list[CheckResult]`.
- CLI exits 0 only when MLflow health/version, Codex install/auth, and Assistant config all pass.
- Makes no registry or repository writes.

- [ ] **Step 1: Add failing unit tests for preflight behavior**

Append to `tests/test_mlflow_assistant_deployment.py`:

```python
from scripts.mlflow_assistant_preflight import CheckResult, build_checks, main


def test_preflight_checks_cover_install_auth_config_and_health() -> None:
    checks = build_checks()
    names = [check.name for check in checks]
    commands = "\n".join(check.command for check in checks)

    assert names == ["mlflow-health", "mlflow-version", "codex-installed", "codex-auth", "assistant-config"]
    assert "http://localhost:5000/health" in commands
    assert "mlflow.__version__" in commands
    assert "codex --version" in commands
    assert "codex login status" in commands
    assert "config.json" in commands
    assert "/workspace" in commands


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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_mlflow_assistant_deployment.py -q
```

Expected: FAIL with `ModuleNotFoundError: scripts.mlflow_assistant_preflight`.

- [ ] **Step 3: Implement the preflight**

Create `scripts/mlflow_assistant_preflight.py`:

```python
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
        proc = subprocess.run(
            ["docker", "compose", "exec", "-T", "mlflow", "sh", "-lc", check.command],
            check=False,
            capture_output=True,
            text=True,
        )
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
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_mlflow_assistant_deployment.py -q
python -m ruff check scripts/mlflow_assistant_preflight.py tests/test_mlflow_assistant_deployment.py
```

Expected: PASS.

- [ ] **Step 5: Review the task diff**

Run:

```bash
git diff -- scripts/mlflow_assistant_preflight.py tests/test_mlflow_assistant_deployment.py
```

Expected: preflight is read-only and contains no credentials or alias mutation. Do not commit without explicit authorization.

---

### Task 3: Operator runbook and lifecycle link

**Files:**
- Create: `docs/training/MLFLOW_ASSISTANT.md`
- Modify: `docs/training/GNN_LIFECYCLE.md`
- Modify: `tests/test_mlflow_assistant_deployment.py`

**Interfaces:**
- Produces a copy/paste bootstrap and recovery runbook.
- Explicitly documents the `danger-full-access` boundary and read/write repo mount.

- [ ] **Step 1: Add failing runbook contract test**

Append:

```python
def test_assistant_runbook_documents_bootstrap_and_security() -> None:
    runbook = (ROOT / "docs/training/MLFLOW_ASSISTANT.md").read_text()
    assert "docker compose exec mlflow codex login" in runbook
    assert "docker compose exec mlflow mlflow assistant --configure" in runbook
    assert "python scripts/mlflow_assistant_preflight.py" in runbook
    assert "danger-full-access" in runbook
    assert "read/write" in runbook
    assert "TokyoEye@champion" in runbook
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_mlflow_assistant_deployment.py::test_assistant_runbook_documents_bootstrap_and_security -q
```

Expected: FAIL because the runbook does not exist.

- [ ] **Step 3: Write the runbook**

Create `docs/training/MLFLOW_ASSISTANT.md` with these exact operational sections:

````markdown
# MLflow Assistant — OpenAI Codex

The Assistant runs Codex inside `tokyoeye_mlflow`. MLflow 3.14 invokes Codex
with `danger-full-access`; `/workspace` is intentionally read/write. Review
all changes in git before accepting them.

## Build and start

```bash
docker compose build mlflow
docker compose up -d mlflow
docker compose exec mlflow codex --version
```

## One-time ChatGPT login

```bash
docker compose exec mlflow codex login
```

Complete the browser/device flow. Credentials persist in the
`mlflow_codex_auth` Docker volume.

## Configure Assistant

```bash
docker compose exec mlflow mlflow assistant --configure
```

Select **OpenAI Codex**, configure the repository as `/workspace`, and enable
file editing/read docs. Config persists in `mlflow_assistant_cfg`.

## Verify

```bash
python scripts/mlflow_assistant_preflight.py
```

Open `http://localhost:5000`, select an experiment, and open **Assistant**.
First ask a read-only question. Then test a write only under `/workspace/tmp/`.
Confirm `TokyoEye@champion` and `TokyoEye@experimental` aliases are unchanged.

## Recovery

- Missing CLI: rebuild `mlflow`.
- Logged out: rerun `docker compose exec mlflow codex login`.
- Wrong provider/project: rerun `docker compose exec mlflow mlflow assistant --configure`.
- Reset config only: remove `mlflow_assistant_cfg` after explicit operator approval.
- Reset login only: remove `mlflow_codex_auth` after explicit operator approval.
````

- [ ] **Step 4: Link from lifecycle docs**

Add under `docs/training/GNN_LIFECYCLE.md`'s “MLflow in the workbench” section:

```markdown

For the in-app OpenAI Codex Assistant, see
[`MLFLOW_ASSISTANT.md`](MLFLOW_ASSISTANT.md).
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_mlflow_assistant_deployment.py -q
```

Expected: PASS.

- [ ] **Step 6: Review docs**

Run:

```bash
git diff -- docs/training/MLFLOW_ASSISTANT.md docs/training/GNN_LIFECYCLE.md
```

Expected: no API key instructions, no remote exposure instructions, and no implication that Assistant setup moves aliases.

---

### Task 4: Build, authenticate, configure, and smoke-test live

**Files:**
- No source changes expected.
- Runtime state: Docker images/containers and named volumes only.

**Interfaces:**
- Consumes image and Compose contract from Task 1.
- Consumes preflight from Task 2 and runbook from Task 3.
- Produces a working Assistant tab using Codex and persistent login/config.

- [ ] **Step 1: Capture alias baseline**

Run:

```bash
python - <<'PY'
from mlflow.tracking import MlflowClient

c = MlflowClient("http://localhost:5000")
for alias in ("champion", "experimental"):
    mv = c.get_model_version_by_alias("TokyoEye", alias)
    print(alias, mv.version, mv.run_id)
PY
```

Expected: print the current alias targets; save the values for Step 8.

- [ ] **Step 2: Validate and build**

Run:

```bash
docker compose config --quiet
docker compose build mlflow
docker compose up -d mlflow
docker compose ps mlflow
```

Expected: `tokyoeye_mlflow` is healthy and port output is bound to `127.0.0.1:5000`.

- [ ] **Step 3: Verify Codex installation**

Run:

```bash
docker compose exec -T mlflow codex --version
```

Expected: `codex-cli 0.146.0` (or equivalent output containing `0.146.0`).

- [ ] **Step 4: Authenticate interactively**

Run:

```bash
docker compose exec mlflow codex login
```

Complete the ChatGPT browser/device authorization. Do not paste an API key into chat, shell history, `.env`, or Compose.

- [ ] **Step 5: Confirm authentication**

Run:

```bash
docker compose exec -T mlflow codex login status
```

Expected: authenticated/logged-in status and exit 0.

- [ ] **Step 6: Configure MLflow Assistant**

Run:

```bash
docker compose exec mlflow mlflow assistant --configure
```

Choose:
- Provider: **OpenAI Codex**
- Model: **default**
- Project/repository location: `/workspace`
- Permissions: edit files **on**, read docs **on**, full access as required by provider

- [ ] **Step 7: Run preflight**

Run:

```bash
python scripts/mlflow_assistant_preflight.py
```

Expected: five `PASS` lines and exit 0.

- [ ] **Step 8: Verify aliases did not move**

Repeat Step 1 and compare exact version/run IDs.

Expected: `TokyoEye@champion` and `TokyoEye@experimental` targets are identical to baseline.

- [ ] **Step 9: Browser smoke**

At `http://localhost:5000`:
1. Open a Tokyo Eye experiment and the Assistant tab.
2. Ask: “List the registered `TokyoEye` model aliases and their run IDs. Do not modify anything.”
3. Confirm the answer matches Step 8.
4. Ask it to create `/workspace/tmp/mlflow-assistant-smoke.txt` containing `MLflow Codex Assistant online`.
5. Confirm the host file exists and has exactly that content.
6. Delete only the scratch file.

Expected: read and controlled write both succeed; no registry aliases move.

- [ ] **Step 10: Run final verification**

Run:

```bash
python -m pytest tests/test_mlflow_assistant_deployment.py -q
python -m ruff check scripts/mlflow_assistant_preflight.py tests/test_mlflow_assistant_deployment.py
docker compose config --quiet
git status --short
```

Expected: tests/lint/Compose pass. Git status shows only intended source/doc changes and no credentials. Do not commit without explicit authorization.
