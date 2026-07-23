# TokyoEye MLflow Governance Design

**Status:** DRAFT — awaiting user review before implementation  
**Date:** 2026-07-23  
**Scope:** Generalized model lifecycle for the **active trunk and all future lineages**. Archaeology lines are not migrated into this scaffold unless explicitly opted in.  
**Approach:** Config-driven train entry + MLflow identity SSOT  
**Supersedes:** ad-hoc `HEALTHY_*` path seals, Makefile “which `.pt`” tribal knowledge, lineage-hardcoded experiment/docs sprawl  
**Informs:** onboard contract runtime resolution (resolve **aliases**, not sacred paths)  
**Seed pattern:** `tokyoeye_governance_implementation_spec.py` (manifests, VSD, promotion gate)

---

## 0. Purpose

Cement a single process so model management is **auditable, reproducible, and controllable** — without baking a particular architecture revision into the language of the process.

| Classifier | Meaning | Example values (not the schema) |
|------------|---------|----------------------------------|
| **model** | Registered model name | `TokyoEye-<lineage>` |
| **lineage** | Architecture family id | whatever the active trunk declares |
| **version** | MLflow Model Registry version | `1`, `2`, … |
| **experiment** | MLflow experiment path | `tokyoeye/<lineage>/<family>` |
| **run** | MLflow training run id | UUID |
| **alias** | Mutable pointer on a registered model | `@champion`, `@candidate`, family seals |

**SSOT = MLflow Model Registry + run lineage.** Local filesystem checkpoints are **cache mirrors only**, never authority for “what is production / sealed.”

Two failure modes this design prevents:

1. **Wild-west paths** — sacred local filenames / symlinks with no registry link.  
2. **Non-self-describing trains** — runs that cannot be reproduced because config, corpus manifest, and gate thresholds were not pinned on the run.  
3. **Lineage-hardcoded process docs** — treating one revision id as if it were the governance system itself.

---

## 1. Frozen policy decisions

1. **Process is lineage-parameterized** — every API, JSON schema, and path template takes `lineage` (and `family`) as data, not as prose baked into titles and tables.  
2. **Aliases replace `HEALTHY_*` as SSOT** — no institutional reliance on local symlinks; local paths are optional caches after download from MLflow.  
3. **Promotion is alias-only** — after gate Pass: `register_model` → `set_registered_model_alias`. No “copy this file to HEALTHY.”  
4. **Generic trainer + JSON configs** — scaffolding imports a versioned train config; model-specific weights/hparams live in that config, not in the trainer’s identity.  
5. **Checkpoint on-disk name embeds short run id + role** — human convenience; **full `mlflow_run_id` lives in registry tags / run metadata** (authoritative).  
6. **Within-run roles** — `best` / `last` / `improve_*` are artifacts of that run; only one role (usually `best` after Pass) is registered.  
7. **Rollout** — implement and enforce first on the **active trunk**; do not require archaeology migration.

---

## 2. Identity taxonomy

### 2.1 Experiment paths (template)

```
tokyoeye/{lineage}/{family}
```

Examples of `{family}`: `spine`, `affinity`, `biology`, … — defined per lineage config, not in this spec’s identity table.

Do not invent parallel experiment naming schemes in Makefile without updating the lineage’s config pack.

### 2.2 Registered model (template)

- Name: `TokyoEye-{lineage}` (one registered model per lineage unless an architecture fork requires a new registered model name).  
- Each **registry version** tags at minimum:

| Tag / field | Required |
|-------------|----------|
| `lineage` | architecture family id |
| `family` | train family within lineage |
| `mlflow_run_id` | full run UUID |
| `artifact_role` | `best` \| `last` \| … |
| `git_commit` | SHA |
| `train_config_id` + hash | JSON train config |
| `dataset_manifest_id` + hash | corpus/split manifest |
| `vsd_id` + hash | validation / gate spec |

### 2.3 Aliases (mutable; names are convention)

| Alias | Intent |
|-------|--------|
| `@champion` | Default production / onboard restore for that lineage |
| `@candidate` | Latest gate-eligible build awaiting promotion |
| `@staging` | Optional pre-prod |
| `@<family>-seal` | Optional claim-scoped seal (e.g. affinity ranking Pass) — **family-defined**, not process-hardcoded |

Alias moves **do not** rewrite history: old registry versions remain immutable.

### 2.4 Resolving “what to load”

Order for tooling / onboard:

1. Explicit `--model-uri models:/TokyoEye-{lineage}@{alias}` or registry version  
2. Else config `init.alias` (+ `lineage` from config)  
3. Else **fail** — do not silently fall back to a hardcoded `checkpoints/...` path except as an explicit `cache_path` after download from MLflow

---

## 3. Config + manifest layers

Layout is **under the active package root**, parameterized by lineage id in filenames/fields — not a separate governance system per revision:

```
science/tokyo_eye/{lineage}/configs/
  train/           # hyperparams, init alias, epochs, family
  vsd/             # gate thresholds + required metrics
manifests/         # corpus / split JSON (shared or lineage-scoped)
science/tokyo_eye/governance/   # shared typed helpers (lineage-agnostic)
```

If the active code package is nested (e.g. `science/tokyo_eye/<lineage>/`), configs live beside that package; **governance code stays shared** so the next lineage does not fork the process.

### 3.1 Train config (JSON)

Minimum fields:

- `schema_version`, `lineage`, `family`, `model_name` (`TokyoEye-{lineage}`)  
- `experiment` path (`tokyoeye/{lineage}/{family}`)  
- `init`: `{ "alias": "champion" }` or `{ "model_uri": "..." }`  
- `data`: `{ "manifest": "manifests/...", ... }`  
- `vsd`: path to VSD JSON  
- `hparams`, `seed`, `device` policy  
- `artifacts_to_log`: list of roles (`best`, `last`, …)

Trainer computes hashes at start and logs them; config file itself is logged as an MLflow artifact.

### 3.2 Dataset / corpus manifest

Existing manifests remain valid inputs. Governance wrapper records:

- `dataset_id`, `version`, path, content hash  
- split keys present  
- leak / identity policy assertion results logged as metric/tag  

Typed shape may follow `DatasetManifest` / `CorpusSpecification`.

### 3.3 VSD (validation specification)

JSON gate file:

- `required_metrics`  
- `thresholds` (e.g. `{ "metric": { "ge": ... } }`)  
- optional `geometry_version` / `graph_version` tokens  

Promotion evaluates VSD **before** alias update (`PipelineAutomator` + `ValidationSpecification` pattern).

---

## 4. Run lifecycle

```text
load train JSON + manifests + VSD
    → start MLflow run under tokyoeye/{lineage}/{family}
    → tag identity + hashes + git commit
    → train; log metrics each epoch
    → log artifacts: best / last / (optional improve_*)
    → evaluate VSD
    → if Pass: register artifact → new registry version
         → optionally set @candidate; human or CI sets @champion / @<family>-seal
    → if Fail: no alias move; run remains audit trail
```

### 4.1 On-disk cache naming

```text
{lineage}_{family}_{run_id_short8}_{artifact_role}.pt
```

- Full run id in MLflow tags / registry metadata (authoritative).  
- Short id in filename for grep-ability.  
- **Do not** treat filename as SSOT.

### 4.2 Within-run “best”

- `best` = training loop’s selection rule, logged on the run.  
- Registration chooses which role to promote (default `best` after VSD Pass).  
- Epoch `improve_*` files are optional diagnostics.

---

## 5. Code layout (implementation target)

| Path | Role |
|------|------|
| `science/tokyo_eye/governance/` | Lineage-agnostic: manifests, VSD eval, identity helpers, promotion |
| `science/tokyo_eye/{lineage}/configs/train/*.json` | Train configs for that lineage |
| `science/tokyo_eye/{lineage}/configs/vsd/*.json` | Gates for that lineage |
| `experiments/training/train_from_config.py` | Generic entry: load JSON → run → register |
| `experiments/training/promote_alias.py` | CLI: set alias after VSD check / audited override |

Active-trunk harnesses emit governance tags + register path first; “train only via `train_from_config`” follows in the same program.

Legacy `science/training/mlflow_governance.py` (older schema) is **not** the SSOT for this process — do not extend it as the institutional core.

---

## 6. Contract / onboard

- Production restore resolves `models:/TokyoEye-{lineage}@{champion}` (or documented staging alias).  
- Runner accepts model URI / alias; downloads to cache if needed.  
- Full Normalizer `run_inference` parity remains a separate adapter TODO where unfinished.

---

## 7. Migration of existing local seals

| Current habit | Target |
|---------------|--------|
| Local “healthy” / seal symlinks | Register the producing run’s chosen artifact → set the appropriate **alias**; record `mlflow_run_id` + registry version on the closeout gate JSON |
| Hub/AGENTS language naming sacred paths | Point to **alias** (+ how to resolve URI) |

Closeout stamps gain `mlflow_run_id`, `registry_version`, `alias` when migrated.

---

## 8. Out of scope (later)

- S3 tier storage automation  
- Live production drift monitors  
- Full biophysical plugin registry (stub OK)  
- Sprint governance automation  
- Forced migration of archaeology trainers  

---

## 9. Acceptance (when implemented)

1. Unit tests: identity tag schema; VSD pass/fail; alias set mocked via `MlflowClient` — **parameterized by lineage string**, not a single hard-coded revision.  
2. Smoke: `train_from_config` on a tiny JSON → run tags present → artifact logged with naming convention.  
3. Docs: active-trunk README points here; AGENTS/hub say **alias SSOT**, not HEALTHY paths.  
4. No new code path that promotes by copying a file to a sacred path.  
5. Spec and code comments do not treat one lineage id as synonymous with “governance.”

---

## 10. Open points (locked unless reopened)

| # | Decision |
|---|----------|
| 1 | Registered model template `TokyoEye-{lineage}` + aliases — **LOCKED** |
| 2 | Aliases replace HEALTHY as SSOT — **LOCKED** |
| 3 | Filename = `{lineage}_{family}_{run_short}_{role}.pt`; full run id in metadata — **LOCKED** |
| 4 | Shared `science/tokyo_eye/governance/`; per-lineage JSON configs — **LOCKED** |
| 5 | Process docs stay lineage-agnostic — **LOCKED** |

---

## 11. Next step

User reviews this file. On approval → implementation plan, then code.
