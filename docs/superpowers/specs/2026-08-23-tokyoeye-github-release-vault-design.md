# Tokyo Eye GitHub Release Vault

**Status:** APPROVED — 2026-08-23  
**Date:** 2026-08-23  
**Amends:** [`2026-07-23-tokyoeye-mlflow-governance-design.md`](2026-07-23-tokyoeye-mlflow-governance-design.md)  
**Trigger:** Champion import left a Postgres pointer (`models:/TokyoEye@champion`) whose artifact bytes lived only on a gitignored bind mount (`./mlflow-artifacts`). The catalog survived; the weights did not.

---

## 0. Purpose

Keep MLflow as the **identity and promotion** plane. Add a **durable byte replica** that does not cost extra cloud storage and cannot vanish when a local folder is emptied.

---

## 1. Split of authority (locked)

| Concern | Authority |
|---------|-----------|
| Registered model, aliases, metrics, thresholds, experiment path | **MLflow** (`TokyoEye`, `@champion`, `@experimental`) |
| Durable weight bytes + sha256 | **GitHub Release** on this private repo |
| Local `checkpoints/` and `mlflow-artifacts/` | Cache / working copy only |
| Equiformer pretrained backbone | Hugging Face URL already in the weight map — do not vault a third-party dump |

This is **not** a second model catalog. GitHub does not own aliases. MLflow does not own last-copy durability.

**Forbidden vaults:** git-tracked `.pt` files (GitHub blocks >100 MiB); Git LFS (bandwidth quota recreates “pointer without bytes”); S3 unless a later cost decision reopens it; GitHub Actions artifacts (they expire).

---

## 2. Release layout (locked)

Content-addressed immutable tag:

```text
tokyoeye-eqf-<sha256_16>
```

Optional moving pointer for humans / restore-without-MLflow:

```text
tokyoeye-eqf-champion
tokyoeye-eqf-experimental
```

The moving pointer may be retargeted. The immutable tag may **never** be overwritten with different bytes. Same sha256 → reuse. Different sha256 on an existing immutable tag → refuse.

Each release holds:

- the checkpoint asset (original sanitized filename)
- a `*.sha256` sidecar (`<hex>  <filename>`)
- JSON notes: `backend`, `registered_model`, `alias`, `capability_goal`, `sha256`, `asset`

Repo default: `TOKYOEYE_VAULT_REPO` or `gh repo view` (this repo is private).

---

## 3. Champion promote rule (locked)

`@champion` may move only after **all** of:

1. Evaluate Pass (existing threshold pack).
2. Bytes copied into MLflow and **read back** (sha256 match).
3. Immutable GitHub Release exists with the **same** sha256.
4. Explicit `--confirm-champion` on the CLI (pipelines never set champion implicitly).

`@experimental` does **not** require a Release (staging may stay MLflow-only). Vaulting experimental is allowed.

`set-alias --alias champion` without a matching Release **fails closed**.

Low-level `registry.set_model_alias` stays a thin MLflow wrapper for tests. Policy is enforced on CLI `set-alias` / `promote` / `import-weights` when the requested alias is `champion`.

---

## 4. Resolve rule (locked)

1. Resolve `models:/TokyoEye@alias` from MLflow artifacts (existing path).
2. If download fails **and** the version/run carries `vault_release_tag` + `checkpoint_sha256`, download from GitHub Release into the MLflow cache, verify sha256, then return that path.
3. Still no fallback to `HEALTHY_*` or other filesystem seals.

If both MLflow artifacts and the Release are missing, fail with a message that names both.

---

## 5. Ops (no Make SSOT)

```text
python -m science.tokyo_eye.governance.entrypoints vault-upload ...
python -m science.tokyo_eye.governance.entrypoints vault-restore ...
python -m science.tokyo_eye.governance.entrypoints verify --alias champion
```

Implementation talks to GitHub only through `gh` (injectable runner for tests).

---

## 6. Non-goals

- Renaming `science/tokyo_eye/v8` (separate labeling pass).
- Uploading every smoke run.
- Making GitHub Actions the model store.
- Changing registered model name `TokyoEye`.

---

## 7. Acceptance

1. Unit tests cover upload reuse, sha mismatch refuse, champion CLI blocked without vault, resolve restores from vault after MLflow artifacts are deleted.
2. `import-weights --alias champion` without `gh` / FakeGh fails; with a runner it writes vault tags on the model version.
3. Operator doc (`docs/training/GNN_LIFECYCLE.md`) and `data/gates/tokyoeye_mlflow_ssot.json` name the Release vault.
