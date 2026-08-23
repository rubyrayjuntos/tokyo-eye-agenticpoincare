# GNN Model Lifecycle — operator guide

Primary surface: MLflow HTTP API + UI at **http://localhost:5000**.

Backend: MLflow 3.x server with Postgres `mlflow` DB + artifact proxy. Local checkpoint files are cache/evidence only; production restore is `models:/TokyoEye@champion`. Durable weight bytes live on **GitHub Releases** (`tokyoeye-eqf-<sha256_16>`). See [`docs/superpowers/specs/2026-08-23-tokyoeye-github-release-vault-design.md`](../superpowers/specs/2026-08-23-tokyoeye-github-release-vault-design.md).

**Ops note:** Prefer the Python client / CLI against the tracking server. From the host, `http://localhost:5000` is both UI and HTTP API. Inside Docker science containers use `http://mlflow:5000`. Do **not** use Make targets for MLflow governance.

## Quick status

```bash
python -m science.tokyo_eye.governance.entrypoints \
  --tracking-uri http://localhost:5000 \
  resolve --alias champion
```

## Threshold packs (SSOT)

Gate packs live on MLflow runs as `thresholds/thresholds.json`. Seed a tagged template once per domain/subsystem:

```bash
python -m science.tokyo_eye.governance.entrypoints \
  --tracking-uri http://localhost:5000 \
  seed-thresholds --domain geometric --subsystem full-stack \
  --pack-json '{"smoke_metric": 0.5}'
```

Evaluate / promote / train `--pipeline` resolve packs in this order:

1. `--thresholds-run-id` (explicit run artifact)
2. latest `threshold_pack_template` run for `--domain` / `--subsystem`
3. legacy local `--thresholds path.json` (emits `DeprecationWarning`)

Repo files under `data/gates/` remain archaeology / probe stamps — not the production threshold SSOT.

If a taxonomy experiment still has a filesystem `artifact_location` (pre-proxy), `ensure_taxonomy_experiment` renames it to `{name}.legacy_fs_{id}` and recreates the canonical name with `mlflow-artifacts:/` so host API writes work. Registry aliases are untouched.

## Promote (CI / power user)

```bash
python -m science.tokyo_eye.governance.entrypoints \
  --tracking-uri http://localhost:5000 \
  promote \
  --run-id "$RUN_ID" \
  --domain geometric \
  --subsystem full-stack \
  --capability-goal "$CAPABILITY_GOAL" \
  --alias experimental \
  --model-uri "$MLFLOW_MODEL_URI"
```

(`--thresholds-run-id` optional when a template pack is already seeded.)

Alias moves must follow evaluate → register/import → set-alias. Automated pipelines never set `@champion` (explicit `--confirm-champion` only).

`@champion` also requires a GitHub Release whose sha256 matches the checkpoint. Upload first if needed:

```bash
python -m science.tokyo_eye.governance.entrypoints vault-upload \
  --checkpoint "$CKPT" --alias champion --capability-goal "$CAPABILITY_GOAL"

python -m science.tokyo_eye.governance.entrypoints verify --alias champion
```

Production inference serves **GitHub Release bytes** for a vault-tagged `@champion`. MLflow still chooses the version. `resolve` downloads the Release into a sha256 cache; it does not read `./mlflow-artifacts` for champion.

Verify and CI results are logged as MLflow runs (`lifecycle/verify.json`, `ci-report`).

**CI split:** `property-gates` runs on GitHub-hosted `ubuntu-latest` and is the code gate. `mlflow-report` runs on the self-hosted `tokyoeye` runner (`MLFLOW_TRACKING_URI=http://localhost:5000`), writes the pane row, and cannot fail the gate (`continue-on-error`, 5 minute timeout). Keep `~/actions-runner-tokyoeye` running so the report job is not queued.

## Train (MLflow pipeline)

```bash
python -m science.tokyo_eye.governance.entrypoints \
  --tracking-uri http://localhost:5000 \
  train --pipeline \
  --domain geometric --subsystem full-stack \
  --capability-goal geometric_full_stack_probe \
  --device cuda
```

Smoke / join without moving `@champion` (may update `@experimental` unless `--no-experimental-alias`):

```bash
python -m science.tokyo_eye.governance.entrypoints \
  --tracking-uri http://localhost:5000 \
  train --pipeline --skip-train \
  --checkpoint "$CKPT" \
  --domain geometric --subsystem full-stack \
  --capability-goal pipeline_smoke_do_not_champion \
  --metrics /tmp/smoke_metrics.json \
  --no-experimental-alias
```

See [`docs/audit/LEARNED_GNN_VS_SLIM_SSOT.md`](../audit/LEARNED_GNN_VS_SLIM_SSOT.md).

## RCSB → corpus

1. Structure picker → RCSB → Load & run (ingest).
2. Lifecycle → **Add current structure to corpus**.

## MLflow in the workbench

Vite proxies `/mlflow` → `:5000`. Open from the Lifecycle panel or spawn Dockview `mlflow-panel`.

For the in-app OpenAI Codex Assistant, see
[`MLFLOW_ASSISTANT.md`](MLFLOW_ASSISTANT.md).
