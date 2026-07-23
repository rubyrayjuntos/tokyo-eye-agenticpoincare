# TokyoEye-v8 MLflow Governance Design

**Status:** DRAFT — awaiting user review before implementation  
**Date:** 2026-07-23  
**Scope:** **v8 and forward only** (archaeology `v6` / `v66` / `v7` out of scope)  
**Approach:** Config-driven train entry + MLflow identity SSOT (Approach B)  
**Supersedes (for v8+):** ad-hoc `HEALTHY_V8_*` path seals, Makefile one-liner “which `.pt`” tribal knowledge  
**Informs / does not replace:** onboard contract runtime resolution (will be updated to resolve aliases)  
**Example seed:** user-provided `tokyoeye_governance_implementation_spec.py` (dataclasses + promotion gate pattern)

---

## 0. Purpose

Cement a single process so model management is **auditable, reproducible, and controllable**:

| Classifier | Meaning |
|------------|---------|
| **model** | Registered model name (e.g. `TokyoEye-v8`) |
| **lineage** | Architecture family (`v8`) |
| **version** | MLflow Model Registry version (`1`, `2`, …) |
| **experiment** | MLflow experiment path |
| **run** | MLflow training run id |
| **alias** | Mutable pointer (`@champion`, `@affinity-seal`, `@candidate`, …) |

**SSOT = MLflow Model Registry + run lineage.** Local filesystem checkpoints are **cache mirrors only**, never authority for “what is production / sealed.”

Two failure modes this design prevents:

1. **Wild-west paths** — multiple `HEALTHY_*` / `v8_best.pt` copies with no registry link.  
2. **Non-self-describing trains** — runs that cannot be reproduced because config, corpus manifest, and gate thresholds were not pinned on the run.

---

## 1. Frozen policy decisions

1. **v8+ only** — no requirement to wrap archaeology lines in this scaffold.  
2. **Aliases replace `HEALTHY_*` as SSOT** — no new institutional reliance on local symlinks; existing local pointers become optional caches or are retired in a follow-up.  
3. **Promotion is alias-only** — after gate Pass: `register_model` → `set_registered_model_alias`. No “copy this file to HEALTHY.”  
4. **Generic trainer + JSON configs** — model-specific behavior lives in versioned JSON under `science/tokyo_eye/v8/configs/`; scaffolding imports config, not the reverse.  
5. **Checkpoint on-disk name embeds short run id + role** — human convenience; **full `mlflow_run_id` lives in registry tags / run metadata** (authoritative).  
6. **Within-run roles** — `best` / `last` / `improve_*` are artifacts of that run; only one role (usually `best` after Pass) is registered.

---

## 2. Identity taxonomy

### 2.1 Experiment paths

```
tokyoeye/v8/spine
tokyoeye/v8/affinity
tokyoeye/v8/biology   # future B0/B1 probes
```

Convention: `tokyoeye/v8/<family>`. Do not invent parallel experiment names in Makefile without updating this list.

### 2.2 Registered model

- Name: `TokyoEye-v8` (single registered model for the lineage; family is distinguished by tags + aliases, not separate model names — unless a future architecture fork requires a new registered model).  
- Each **registry version** tags at minimum:

| Tag / field | Required |
|-------------|----------|
| `lineage` | `v8` |
| `mlflow_run_id` | full run UUID |
| `artifact_role` | `best` \| `last` \| … |
| `git_commit` | SHA |
| `train_config_id` + hash | JSON train config |
| `dataset_manifest_id` + hash | corpus/split manifest |
| `vsd_id` + hash | validation / gate spec |
| `family` | `spine` \| `affinity` \| … |

### 2.3 Aliases (mutable)

| Alias | Intent |
|-------|--------|
| `@champion` | Default production / onboard restore for v8 spine (or whole-line if single) |
| `@affinity-seal` | CASF affinity gate Pass snapshot (ranking claims only) |
| `@candidate` | Latest gate-eligible build awaiting promotion |
| `@staging` | Optional pre-prod |

Alias moves **do not** rewrite history: old registry versions remain immutable.

### 2.4 Resolving “what to load”

Order for tooling / onboard (implementation):

1. Explicit `--model-uri models:/TokyoEye-v8@<alias>` or registry version  
2. Else config `init.alias`  
3. Else **fail** — do not silently fall back to a hardcoded `checkpoints/v8/...` path except as an explicit `cache_path` after download from MLflow

---

## 3. Config + manifest layers

```
science/tokyo_eye/v8/configs/
  train/           # hyperparams, init alias, epochs, family
  vsd/             # gate thresholds + required metrics
manifests/         # existing corpus / split JSON (reuse cluster30 etc.)
```

### 3.1 Train config (JSON)

Minimum fields:

- `schema_version`, `lineage` (`v8`), `family`, `model_name` (`TokyoEye-v8`)  
- `experiment` path  
- `init`: `{ "alias": "champion" }` or `{ "model_uri": "..." }` (not bare orphan paths as SSOT)  
- `data`: `{ "manifest": "manifests/...", "manifest_hash": "optional-at-write" }`  
- `vsd`: path to VSD JSON  
- `hparams`, `seed`, `device` policy  
- `artifacts_to_log`: list of roles (`best`, `last`, …)

Trainer computes hashes at start and logs them; config file itself is logged as an MLflow artifact.

### 3.2 Dataset / corpus manifest

Reuse existing manifests (e.g. `manifests/v8_pdbbind_refined_cluster30_v1.json`). Governance wrapper records:

- `dataset_id`, `version`, path, content hash  
- split keys present (`train` / `val` / `core_test`)  
- `NO_LEAK` assertion result logged as metric/tag

Typed shape may follow the example `DatasetManifest` / `CorpusSpecification` in code under `science/tokyo_eye/v8/governance/`.

### 3.3 VSD (validation specification)

JSON gate file, e.g. affinity:

- `required_metrics`: [`core_pearson`, …]  
- `thresholds`: `{ "core_pearson": { "ge": 0.40 } }`  
- `geometry_version` / `graph_version` tokens when relevant  

Promotion code evaluates VSD **before** alias update (pattern from example `PipelineAutomator` + `ValidationSpecification`).

---

## 4. Run lifecycle

```text
load train JSON + manifests + VSD
    → start MLflow run under tokyoeye/v8/<family>
    → tag identity + hashes + git commit
    → train; log metrics each epoch
    → log artifacts: best / last / (optional improve_*)
    → evaluate VSD on held-out / Core as specified
    → if Pass: register artifact → new registry version
         → optionally set @candidate; human or CI sets @champion / @affinity-seal
    → if Fail: no alias move; run remains audit trail
```

### 4.1 On-disk cache naming

When materializing an artifact locally:

```text
v8_{family}_{run_id_short8}_{artifact_role}.pt
```

Example: `v8_affinity_a1b2c3d4_best.pt`

- Full run id in MLflow tags / registry metadata (authoritative).  
- Short id in filename for grep-ability.  
- **Do not** treat filename as SSOT.

### 4.2 Within-run “best”

- `best` = training loop’s selection rule (e.g. best val Pearson), logged on the run.  
- Registration chooses which role to promote (default `best` after VSD Pass).  
- Epoch `improve_*` files are optional diagnostics, not separate registry versions unless explicitly registered.

---

## 5. Code layout (implementation target)

| Path | Role |
|------|------|
| `science/tokyo_eye/v8/governance/` | Typed manifests, VSD eval, MLflow identity helpers, promotion |
| `science/tokyo_eye/v8/configs/train/*.json` | Train configs |
| `science/tokyo_eye/v8/configs/vsd/*.json` | Gates |
| `experiments/training/v8/train_from_config.py` | Generic entry: load JSON → run → register |
| `experiments/training/v8/promote_alias.py` | CLI: set alias after VSD check / manual override with audit tag |

Wire existing harnesses (`run_affinity_s10.py`, `run_v8_experiment.py`) to **emit** governance tags + register path first; full “only train_from_config” can follow in the same plan.

**Do not** extend `science/training/mlflow_governance.py` (v6-era schema) for v8 SSOT — keep v8 package isolated; optionally deprecate pointers in docs.

---

## 6. Contract / onboard

Follow-up (same program, ordered after registry helpers):

- `onboard_contract.yaml` production checkpoint resolves via `models:/TokyoEye-v8@champion` (or documented staging alias).  
- `TokyoEyeV8Runner` accepts model URI / alias; downloads to cache if needed.  
- `run_inference` adapter remains a separate TODO; governance does not block on full Normalizer parity.

---

## 7. Migration of existing seals

| Current | Target |
|---------|--------|
| Affinity closeout Core R≈0.407 + local `HEALTHY_V8_AFFINITY_CKPT` | Register that run’s `best` (or re-log artifact) → `@affinity-seal`; document run id in closeout JSON |
| Mode C spine `v8_best.pt` | Register → `@champion` (or `@candidate` until biology B0 smoke) |
| Local `HEALTHY_V8_*` symlinks | Optional cache; remove from hub/AGENTS as SSOT language |

Closeout gate JSON gains `mlflow_run_id` + `registry_version` + `alias` fields when migration runs.

---

## 8. Out of scope (later)

- S3 tier storage layout from example `StorageRegistry`  
- Live `DriftMonitor` against production traffic  
- Full `BiophysicalGateRegistry` plugin set (stub interface OK in governance package)  
- Sprint governance manager automation  
- Wrapping v7/v6 trainers  

---

## 9. Acceptance (when implemented)

1. Unit tests: identity tag schema; VSD pass/fail; alias set mocked via `MlflowClient`.  
2. Smoke: `train_from_config` on a tiny JSON → run tags present → artifact logged with naming convention.  
3. Docs: `docs/specs/tokyo-eye-v8/README.md` points here; AGENTS/hub say **alias SSOT**, not HEALTHY paths.  
4. No new code path that promotes by copying a file to a sacred path.

---

## 10. Open points (locked unless reopened)

| # | Decision |
|---|----------|
| 1 | Single registered model `TokyoEye-v8` + aliases by family — **LOCKED** |
| 2 | Aliases replace HEALTHY as SSOT — **LOCKED** |
| 3 | Filename = `v8_{family}_{run_short}_{role}.pt`; full run id in metadata — **LOCKED** |
| 4 | v8+ only — **LOCKED** |
| 5 | New governance package under `science/tokyo_eye/v8/governance/` — **LOCKED** |

---

## 11. Next step

User reviews this file. On approval → implementation plan (`docs/superpowers/plans/2026-07-23-tokyo-eye-v8-mlflow-governance.md`) then code.
