# MLflow Assistant (OpenAI Codex) — Design

**Status:** APPROVED — 2026-07-30
**Date:** 2026-07-30
**Scope:** Bring the in-app **MLflow Assistant** online for Tokyo Eye, backed by the **OpenAI Codex CLI** provider, running inside the existing `mlflow` Docker service.
**Related:** MLflow governance SSOT — [`2026-07-23-tokyoeye-mlflow-governance-design.md`](2026-07-23-tokyoeye-mlflow-governance-design.md); lifecycle ops — [`docs/training/GNN_LIFECYCLE.md`](../../training/GNN_LIFECYCLE.md).

---

## 0. Purpose

Give operators an MLflow-UI-native assistant that can read the Tokyo Eye MLflow control plane (experiments, runs, metrics, registry/aliases, artifacts) and act in the repository, powered by OpenAI via the locally-installed Codex CLI. This complements the governance SSOT: the Assistant is a **reasoning/automation surface over MLflow**, not a new governance authority.

**Non-goals (v1):** MLflow AI Gateway proxy routing for Codex, Codex *tracing* hook (`@mlflow/codex` notify hook), multi-user/hosted access, replacing CI/CD.

---

## 1. Key facts (verified)

- Installed MLflow is **3.14.0**; `mlflow.assistant` and the `CodexProvider` are present (`mlflow/assistant/providers/codex.py`). Provider display name: **"OpenAI Codex"**.
- The Assistant runs the Codex CLI as a **subprocess of the MLflow server process** (`codex exec --json ...`). Therefore Codex must be installed and authenticated **in the same container/host that runs `mlflow server`** — for us, the `mlflow` service.
- **Security-critical:** the provider invokes Codex with `--sandbox danger-full-access` (and the connection check uses `--dangerously-bypass-approvals-and-sandbox`). Codex is *not* self-sandboxing here — **the container and its bind mounts are the security boundary.**
- Codex CLI is distributed as `@openai/codex` (npm, Node 18+). Auth is either `codex login` (ChatGPT account, persisted under `~/.codex`) or `OPENAI_API_KEY`.
- Assistant config lives at `~/.mlflow/assistant/config.json` (provider selection, model, permissions, project→experiment mapping). Configurable via `mlflow assistant --configure` or the UI gear/setup wizard.
- The MLflow Assistant API is **localhost-gated**; the browser reaches it via `http://localhost:5000`, which is already published from the `mlflow` container.

### Decisions (locked with user)
- Repository access: **read/write** mount into the Assistant container.
- Codex auth: **persistent Codex login volume** (ChatGPT account), not an API key.

---

## 2. Architecture

```text
Browser (host) ── http://localhost:5000 ──► mlflow service (container)
                                              │  mlflow server (UI + Tracking/Registry/Artifacts + Assistant API)
                                              │
                                              └─ Assistant selects provider=codex
                                                    └─ subprocess: codex exec --json --sandbox danger-full-access
                                                          reads/writes /workspace  (repo, rw)
                                                          auth from   /home/appuser/.codex        (persistent)
                                                          config from /home/appuser/.mlflow/assistant (persistent)
                                                          MLFLOW_TRACKING_URI=http://localhost:5000 (in-container)
```

- Internal services keep using `http://mlflow:5000`. The Assistant subprocess uses the in-container tracking URI the server passes it (`http://localhost:5000` inside the container resolves to the same server).

---

## 3. Components

### 3.1 `Dockerfile.mlflow` (new)
- `FROM tokyoeye-science:latest` (reuse the pinned science base; keep Codex out of the science runtime image).
- Add Node.js 18+ and `npm i -g @openai/codex` (pinned version).
- Keep non-root `appuser` (home `/home/appuser`) as the runtime user so auth/config volumes are owned correctly.
- Verify `codex --version` at build time.

**Boundary:** what it does — produce an MLflow server image that also has Codex on PATH. Depends on — the science base image + npm registry. Consumers — the `mlflow` compose service only.

### 3.2 `docker-compose.yml` — `mlflow` service changes
- `build.dockerfile: Dockerfile.mlflow` (image retag, e.g. `tokyoeye-mlflow:latest`, to avoid clobbering `tokyoeye-science:latest`).
- Add bind mounts:
  - `./:/workspace` (repository, **rw**)
  - `mlflow_codex_auth:/home/appuser/.codex` (named volume, persistent auth)
  - `mlflow_assistant_cfg:/home/appuser/.mlflow/assistant` (named volume, persistent Assistant config)
- Keep existing artifact/checkpoint/experiment mounts and server flags unchanged.
- Declare the two named volumes in the top-level `volumes:` block.

### 3.3 One-time bootstrap (documented, not automated in v1)
1. `docker compose build mlflow && docker compose up -d mlflow`
2. `docker compose exec mlflow codex login` (interactive; persists to the auth volume).
3. `docker compose exec mlflow mlflow assistant --configure` → provider **OpenAI Codex**, project `/workspace`, permissions **allow edit files** (read/write), read docs on.
4. Open `http://localhost:5000` → Assistant tab → confirm provider connected.

### 3.4 Preflight / health script (new, small)
A script (invoked manually or via a thin make/CLI wrapper — **not** part of MLflow governance ops) that asserts inside the `mlflow` container:
- `codex` on PATH and `codex --version` OK,
- Codex authenticated (non-interactive probe),
- `~/.mlflow/assistant/config.json` has `codex` selected,
- MLflow server `/health` OK and version ≥ 3.13.

Emit a clear pass/fail summary; do not mutate registry aliases.

---

## 4. Data flow (a turn)
1. User types in the Assistant tab (browser → `localhost:5000` → server Assistant API).
2. Server picks selected provider `codex`, spawns `codex exec --json --sandbox danger-full-access --skip-git-repo-check -` with `cwd=/workspace` and `MLFLOW_TRACKING_URI` set.
3. Codex streams JSONL; server parses `item.completed`/`agent_message` into SSE events to the UI.
4. File edits land in the repo bind mount (rw); MLflow reads/writes go to the same server.

---

## 5. Error handling
- **Codex missing:** provider raises `CLINotInstalledError` → surfaced in setup wizard; preflight catches earlier.
- **Not authenticated:** provider raises `NotAuthenticatedError` ("run: codex login"); fix by re-running `codex login` in the container (auth volume persists).
- **Server version too old in container:** preflight fails fast (container image must ship MLflow ≥ 3.13 with `mlflow.assistant`).
- **Write blast radius:** mitigated only by container isolation + the rw mount scope (`/workspace`). Documented explicitly; no in-tool approval gate exists in this MLflow version.

---

## 6. Testing / verification
- Build + compose up; preflight script passes.
- UI smoke: ask the Assistant a **read-only** MLflow question (e.g. "list registered model TokyoEye aliases") and confirm a sensible answer.
- Controlled write: ask it to create/modify a scratch file under `/workspace/tmp/` and confirm the change appears on the host, then delete it.
- Governance guard: confirm the Assistant path does **not** move `@champion`/`@experimental` as a side effect of setup (aliases unchanged before/after).

---

## 7. Security notes (LOCKED)
- Codex runs with full access **by MLflow's design**; treat the `mlflow` container + its mounts as the trust boundary.
- Repository mount is rw per user decision — acknowledge that the Assistant can modify tracked files. Keep the working tree under git so changes are reviewable/revertible.
- Auth is a ChatGPT-account Codex login stored in a named volume; **no API key** is committed or placed in `.env`.
- Assistant API stays localhost-gated; do not expose port 5000 beyond localhost without revisiting this design.

---

## 8. Out of scope / follow-ons
- MLflow AI Gateway routing for centralized token/budget governance of Codex.
- Codex conversation *tracing* via `@mlflow/codex` notify hook (separate from the Assistant provider).
- Automating the interactive `codex login` / `--configure` bootstrap.
