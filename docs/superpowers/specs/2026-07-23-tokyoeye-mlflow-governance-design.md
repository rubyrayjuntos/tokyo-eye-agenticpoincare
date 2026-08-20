# TokyoEye MLflow-Native Governance Design

**Status:** APPROVED — 2026-07-23  
**Date:** 2026-07-23  
**Scope:** End-to-end model lifecycle for Tokyo Eye. **MLflow is the governance system** (dedicated Postgres DB `mlflow` + artifact store + UI).  
**Supersedes:** Dual-SSOT catalog JSON draft in earlier revisions of this file; ad-hoc `HEALTHY_*` path seals; Makefile/script-as-SSOT for train/eval/promote.  
**Companion cookbook:** [`docs/architecture/MLflow Transformer End-to-End Management.md`](../../architecture/MLflow%20Transformer%20End-to-End%20Management.md) (API patterns; product packaging is **pyfunc**, not HF Transformers).

---

## 0. Purpose

Govern Tokyo Eye **entirely inside MLflow** so that:

1. Nothing critical to lineage lives only as a loose file that can be lost or overlooked.  
2. The UI drill path is the institutional hierarchy.  
3. Train → evaluate → register → alias can be triggered without external orchestration SSOTs.  
4. Reusable tests/scorers live with the model control plane and are invoked from MLflow Projects.

**Non-goals for v1:** live traffic serving, Unity Catalog, auto Docker deploy (may use `build_docker` later).

---

## 1. Principle: governance *is* MLflow

| Concern | Authority |
|---------|-----------|
| Hierarchy, history, metrics, datasets, specs, thresholds | **MLflow Tracking + Model Registry** (DB + artifacts) |
| Live restore pointer | **`models:/TokyoEye@champion`** |
| Staging pointer | **`models:/TokyoEye@experimental`** |
| Code that *implements* train/infer | Repo package `science/tokyo_eye/…` + `MLproject` entry points |
| Local `.pt` under `checkpoints/` / `HEALTHY_*` | **Archaeology / untouched transition cache** — **not** consulted by governance resolve |

### 1.1 Separation rule (locked)

The MLflow governance path is **completely separate** from filesystem seals:

1. **`resolve` / production restore** may only use `models:/TokyoEye@alias` and MLflow artifact download. **No fallback** to `HEALTHY_*`, gate JSON paths, or `gnn_lineage` package ids.  
2. **`import-weights`** may accept an explicit local file path as a one-time **byte source** to copy into the MLflow artifact store. It does **not** import `healthy_v8` constants, does **not** delete or rewrite the source, and after import the registry must own `runs:/…` (or pyfunc) URIs — not `file://` as SSOT.  
3. Legacy files and scripts remain on disk for easy transition / archaeology; governance automation must not depend on them at runtime.

There is **no** separate model-catalog JSON SSOT. Manifests/gate JSON in the repo may still exist as *optional evidence files that get logged* into a run; after log, the run artifact + dataset digest are authoritative.

Legacy `science/training/gnn_lineage.py` package ids (`v6`/`v7`/`v8`) remain archaeology/compare helpers — **not** the product governance identity.

---

## 2. Hierarchy (UI drill path)

```text
Registered model: TokyoEye
  └── lineage (architecture stack): equiformer-v3-moe
        └── domain (finite, expandable): geometric | biologic | chemical | …
              └── subsystem / family (finite): equiformer-frontend | hyperbolic-spine | …
                    └── run (name = capability + goal)
                          └── registry version (integer, under the hood)
                                └── aliases @champion | @experimental  (model-level only)
```

### 2.1 Experiment path (locked)

```text
tokyoeye/equiformer-v3-moe/{domain}/{subsystem}
```

Example: `tokyoeye/equiformer-v3-moe/chemical/affinity-head`  
Run name example: `affinity_core_pearson_ge_0.40`

### 2.2 Domains (start set; may add)

| Domain | Intent |
|--------|--------|
| `geometric` | Geometry / hyperbolic / SE(3) structure of the representation |
| `biologic` | Biology-facing probes (teleconnections, epistasis, basins, …) |
| `chemical` | Ligand / affinity / chemistry heads and related gates |

### 2.3 Subsystems / families (start set; may add)

| Subsystem | Intent |
|-----------|--------|
| `equiformer-frontend` | SE(3) Equiformer frontend |
| `hyperbolic-spine` | Hyperbolic backbone |
| `moe-router` | Mixture-of-Experts routing |
| `affinity-head` | Affinity / binding head |
| `full-stack` | Joint stack; preferred path toward `@champion` |

### 2.4 Product version numbers

Package labels like `v8` are **optional run tags** (`package_revision`), not governance identity. Operators navigate by model → lineage → domain → subsystem → run → alias.

Registry version integers (1, 2, 3…) are MLflow-internal; aliases hide them for restore.

---

## 3. Aliases (model-level only)

| Alias | Meaning |
|-------|---------|
| `@champion` | Live / production restore |
| `@experimental` | Promoted-but-not-live staging |

No per-domain or per-subsystem aliases (avoids confusion about what is live). Domain/subsystem live only in the experiment path and run tags.

Resolve:

```text
models:/TokyoEye@champion
models:/TokyoEye@experimental
```

---

## 4. What is stored where (all in MLflow)

| Asset | MLflow surface |
|-------|----------------|
| Corpus / splits | `mlflow.data` + `log_input` (digest on the run) |
| Sprint / gate / prereg specs | Run or model-version **artifacts** |
| Threshold packs (VSD) | Artifacts + `MetricThreshold` / evaluate gate |
| Reusable scorers / diagnostic tests | Shared eval components logged once; invoked from Project `evaluate` |
| Weights + inference contract | **`mlflow.pyfunc`** with `load_context` (MoE / Equiformer must not pickle in `__init__`) |
| Orchestration | **`MLproject`** entry points |
| System telemetry | `mlflow.enable_system_metrics_logging()` where useful |
| CI hooks | Registry webhooks on version/alias events (optional follow-on) |

---

## 5. Orchestration: Projects + PyFunc (locked)

**(3) Both:**

1. **MLflow Project** entry points orchestrate the mandatory order:  
   `train` → `evaluate` → `register` → `set_alias`  
2. **PyFunc `PythonModel`** is what aliases load for inference.  
3. Shared scorers/tests are generalized and stored/attached via MLflow for reuse — not one-off post-run shell scripts.

**Hard rule:** Alias moves to `@experimental` or `@champion` only after Project `evaluate` Pass (thresholds). Fail → stop; no silent promote.

**Hard rule:** No Makefile / wrapper script as SSOT. Thin make targets may call `mlflow run` for DX only.

---

## 6. Run tags (mandatory)

Every governed run sets at least:

| Tag / param | Example |
|-------------|---------|
| `model` | `TokyoEye` |
| `lineage` | `equiformer-v3-moe` |
| `domain` | `chemical` |
| `subsystem` | `affinity-head` |
| `capability_goal` | `affinity_core_pearson_ge_0.40` |
| `package_revision` | optional, e.g. `v8` |
| `git_sha` | commit |

---

## 7. Packaging

- Use **`mlflow.pyfunc.log_model`** (and/or `mlflow.pytorch` for weights as artifacts consumed in `load_context`).  
- **Do not** use `mlflow.transformers` (Hugging Face) for Tokyo Eye.  
- **Do not** use deprecated registry Stages; aliases only.  
- Signatures via `infer_signature` on representative graph/ligand inputs when practical.

---

## 8. Migration

1. Ensure registered model `TokyoEye` exists on the `mlflow` DB.  
2. Backfill: register current healthy spine/affinity weights once; set `@experimental` then `@champion` after evaluate Pass where evidence already exists.  
3. Retire hub/AGENTS language that treats local `HEALTHY_*` paths as SSOT (point to aliases).  
4. Leave legacy GNN lineage registry for archaeology; new work uses this hierarchy only.  
5. Log existing prereg/closeout JSON as artifacts on backfill runs (optional but preferred).

---

## 9. Acceptance

1. Experiment path helper enforces domain ∈ start set ∪ registered extensions and subsystem ∈ start set ∪ extensions.  
2. Project entry points refuse `set_alias` without evaluate Pass.  
3. `models:/TokyoEye@champion` and `@experimental` resolve after a smoke register.  
4. PyFunc loads via `load_context` without pickling expert weights in `__init__`.  
5. Docs: AGENTS / hub cite MLflow aliases + experiment path, not sacred `.pt` paths.  
6. No dual catalog JSON required for automation.

---

## 10. Locked decisions

| # | Decision |
|---|----------|
| 1 | MLflow alone is governance SSOT — **LOCKED** |
| 2 | Hierarchy: model → lineage → domain → subsystem → run — **LOCKED** |
| 3 | Experiment path `tokyoeye/equiformer-v3-moe/{domain}/{subsystem}` — **LOCKED** |
| 4 | Run name = capability + goal — **LOCKED** |
| 5 | Aliases only on `TokyoEye`: `@champion`, `@experimental` — **LOCKED** |
| 6 | Projects + PyFunc (option 3) — **LOCKED** |
| 7 | No product version numbers as identity; package_revision optional tag — **LOCKED** |
| 8 | Separation: resolve/promote path never falls back to HEALTHY_*; import-weights copies bytes without deleting sources — **LOCKED** |
| 9 | **Ops via MLflow HTTP API** (`http://localhost:5000` from host / `http://mlflow:5000` in Docker network). The MLflow **service is the API** (UI is one client). Agent/governance use Python `MlflowClient` / fluent API — not Make, not docker-exec for routine ops — **LOCKED** |
| 10 | **Artifact proxy (`--serve-artifacts`)** so host clients can full read/write without sharing `/app` filesystem paths — **LOCKED** |
| 11 | **No Make for MLflow governance** — **LOCKED** |

---

## 11. Ops path vs UI

| Surface | Role |
|---------|------|
| `http://localhost:5000` | **Tracking + Registry + Artifact HTTP API** and the browser UI |
| Python `MlflowClient` / `mlflow.*` | Preferred agent and automation plane (request/response) |
| `tokyoeye_science` | Still valid for long training jobs; not required for registry/artifact API ops once serve-artifacts is on |
| Make / Makefile | **Not** used for MLflow lifecycle |

Server flags (compose): `--serve-artifacts`, `--artifacts-destination /app/mlflow-artifacts`, `--default-artifact-root mlflow-artifacts:/`.

**Note:** Experiments created *before* the proxy default keep filesystem `artifact_location` values; new experiments use `mlflow-artifacts:/`. Host full-write works for new experiments immediately; legacy experiment roots can be left as history or migrated separately without touching `@champion` / `@experimental`.
