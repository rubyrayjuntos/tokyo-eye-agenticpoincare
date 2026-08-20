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

The container runs as `appuser` (uid 999). Host-owned repo directories may deny
create unless prepared. For the write smoke:

```bash
docker compose exec -u root mlflow \
  sh -lc 'mkdir -p /workspace/tmp && chown appuser:appuser /workspace/tmp'
```

## Recovery

- Missing CLI: rebuild `mlflow`.
- Logged out: rerun `docker compose exec mlflow codex login`.
- Wrong provider/project: rerun `docker compose exec mlflow mlflow assistant --configure`.
- Reset config only: remove `mlflow_assistant_cfg` after explicit operator approval.
- Reset login only: remove `mlflow_codex_auth` after explicit operator approval.
