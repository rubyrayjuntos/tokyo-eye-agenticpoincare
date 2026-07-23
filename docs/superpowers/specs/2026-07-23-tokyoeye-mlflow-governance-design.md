# TokyoEye MLflow Governance Design

**Status:** DRAFT — awaiting user review before implementation  
**Date:** 2026-07-23  
**Scope:** Generalized **definition + automation** for the model lifecycle (active trunk and all future lineages). Archaeology may appear in the catalog as `status: archived` history without requiring trainer migration.  
**Approach:** One hierarchical **model catalog JSON** drives all automations; **MLflow** holds runs, metrics, artifacts, registry versions, and aliases.  
**Supersedes:** ad-hoc `HEALTHY_*` seals, manual “open a lineage” checklists, lineage-hardcoded process docs  
**Seed pattern:** `tokyoeye_governance_implementation_spec.py` (manifests, VSD, promotion gate)

---

## 0. Purpose

Stop lifecycle steps from being left to chance. Every new lineage / family / train / promote / resolve path must be:

1. **Defined** in one hierarchical catalog file  
2. **Executed** by automation that reads that file  
3. **Recorded** in MLflow (and mirrored as history entries in the catalog)

### Classifiers (data, not prose)

| Classifier | Meaning |
|------------|---------|
| **model** | Product model id (root of the catalog) |
| **lineage** | Architecture family under that model |
| **version** | MLflow Model Registry version |
| **experiment** | MLflow experiment path |
| **run** | MLflow training run id |
| **alias** | Mutable registry pointer |

**Dual SSOT (clear split):**

| Store | Owns |
|-------|------|
| **Model catalog JSON** | Hierarchy, lifecycle stage definitions, automation bindings, pointers to configs/manifests/VSD, **historical index** (run ids, registry versions, alias moves, gate stamps) |
| **MLflow** | Actual runs, metrics, logged artifacts, registered model versions, live alias targets |

Local `.pt` files are **cache only** after download from MLflow.

Failure modes prevented:

1. Dropped steps when opening a lineage (manual ritual)  
2. Sacred filesystem paths as “production”  
3. Process docs that hardcode one revision id as if it were governance  
4. History scattered across chat / hub / random gate files with no single index  

---

## 1. Frozen policy decisions

1. **Single catalog file** — one hierarchical JSON is the operational definition of the model and its lineages/families/history/automation hooks.  
2. **Automations key only off the catalog (+ MLflow)** — no Makefile one-liner that bypasses catalog stage machine.  
3. **Lifecycle stages are enumerated and mandatory** — automation refuses to skip or reorder unless the catalog marks a stage `optional` (default: required).  
4. **Aliases replace `HEALTHY_*` as restore SSOT** — resolve `models:/{registered_model}@{alias}`.  
5. **Promotion is automated after VSD Pass** — register → update catalog history → set alias per catalog rules (human approval only where catalog requires `approval: required`).  
6. **Process is lineage-parameterized** — `{lineage}` / `{family}` are fields in the catalog, never the name of the governance system.  
7. **Checkpoint cache names** — `{lineage}_{family}_{run_short}_{role}.pt`; full `mlflow_run_id` in MLflow + catalog history.  

---

## 2. The model catalog (single hierarchical JSON)

**Path (proposed):** `data/model_catalog/tokyoeye.model.json`  
**Schema id:** `tokyoeye.model_catalog` / `schema_version: 1`

### 2.1 Hierarchy

```text
model
 └── lineages{}
      └── families{}
           ├── definition (experiment, package, configs, vsd, manifests)
           ├── lifecycle[]          # ordered stages + automation
           ├── aliases{}            # intended alias policy + current pointer summary
           └── history[]            # append-only index into MLflow
```

### 2.2 Sketch (illustrative — values are placeholders)

```json
{
  "schema_version": 1,
  "schema_id": "tokyoeye.model_catalog",
  "model": {
    "id": "TokyoEye",
    "registered_model_template": "TokyoEye-{lineage}",
    "active_lineage": "<lineage_id>",
    "lineages": {
      "<lineage_id>": {
        "status": "active",
        "opened_at": "ISO-8601",
        "package": "science.tokyo_eye.<…>",
        "registered_model": "TokyoEye-<lineage_id>",
        "families": {
          "<family_id>": {
            "experiment": "tokyoeye/<lineage_id>/<family_id>",
            "train_config": "science/tokyo_eye/<…>/configs/train/<family>.json",
            "vsd": "science/tokyo_eye/<…>/configs/vsd/<family>.json",
            "data_manifest": "manifests/<…>.json",
            "lifecycle": [
              {
                "stage": "validate_definition",
                "automation": "governance.stages.validate_definition",
                "required": true
              },
              {
                "stage": "train",
                "automation": "governance.stages.train_from_config",
                "required": true
              },
              {
                "stage": "evaluate_vsd",
                "automation": "governance.stages.evaluate_vsd",
                "required": true
              },
              {
                "stage": "register",
                "automation": "governance.stages.register_best",
                "required": true,
                "on_fail": "stop"
              },
              {
                "stage": "alias",
                "automation": "governance.stages.set_alias",
                "required": true,
                "alias": "candidate",
                "approval": "none"
              },
              {
                "stage": "promote_champion",
                "automation": "governance.stages.set_alias",
                "required": false,
                "alias": "champion",
                "approval": "required"
              }
            ],
            "aliases": {
              "champion": { "registry_version": null, "mlflow_run_id": null },
              "candidate": { "registry_version": null, "mlflow_run_id": null }
            },
            "history": [
              {
                "at": "ISO-8601",
                "event": "register",
                "mlflow_run_id": "<uuid>",
                "registry_version": 1,
                "artifact_role": "best",
                "git_commit": "<sha>",
                "train_config_hash": "<hash>",
                "vsd_id": "<id>",
                "gate_status": "pass",
                "aliases_set": ["candidate"]
              }
            ]
          }
        }
      }
    }
  }
}
```

### 2.3 What “includes historical” means

- **Deep history** (metrics, plots, full artifact bytes) stays in **MLflow**.  
- **Catalog `history[]`** is the **append-only institutional index**: every register / alias move / gate Pass|Fail with `mlflow_run_id` + `registry_version` + config/VSD hashes + git commit.  
- Closing a lineage sets `status: archived` but **retains** its families and history (no delete).  
- Opening a lineage is a **catalog transaction** (add node + required families + lifecycle templates), not a wiki edit.

### 2.4 Definition docs

Human-readable docs (README, sprint notes) **must cite catalog paths** (`model.lineages.<id>.families.<id>`). They are narrative, not SSOT. Automation never reads prose for stage order.

---

## 3. Lifecycle stages (locked set)

Every family lifecycle is an ordered list. Default stages:

| Stage | Purpose | Automation |
|-------|---------|------------|
| `validate_definition` | Configs/manifests/VSD exist; hashes recordable; experiment path legal | Fail if missing |
| `ensure_mlflow` | Experiment created; tracking URI set | Idempotent |
| `train` | Run training from family’s `train_config` | Logs run under catalog experiment |
| `evaluate_vsd` | Apply family’s VSD to run metrics | Fail → no register |
| `register` | Log artifact role → Model Registry version | Writes catalog history entry |
| `alias` | Set alias per stage config (`candidate`, seals, …) | MLflow alias + catalog `aliases` + history |
| `promote_champion` | Optional; may require approval | Same as alias with `approval: required` |
| `resolve_check` | Smoke: load `models:/…@alias` | CI / pre-deploy |

**Rule:** CLI `governance run --lineage L --family F` executes stages in order and **stops on first required failure**. No silent skip.

**Opening a new lineage:**  
`governance lineage open --id <id> --from-template default` → inserts lineage + required families + default lifecycle from a **template in the catalog schema** (or `templates/` referenced by catalog). No hand-built Makefile forest.

---

## 4. Identity + MLflow mapping

### 4.1 Experiment path

From catalog: `families.<family>.experiment`  
Template convention: `tokyoeye/{lineage}/{family}`

### 4.2 Registered model

From catalog: `lineages.<id>.registered_model`  
Template: `TokyoEye-{lineage}`

### 4.3 Resolve order

1. Explicit model URI  
2. Catalog `aliases.<name>` → registry version / run id  
3. Fail  

### 4.4 Cache naming

```text
{lineage}_{family}_{run_id_short8}_{artifact_role}.pt
```

---

## 5. Config packs (referenced by catalog, not parallel SSOTs)

```
science/tokyo_eye/governance/     # shared stage machine + catalog IO
science/tokyo_eye/{lineage}/configs/train/*.json
science/tokyo_eye/{lineage}/configs/vsd/*.json
manifests/                        # data splits
data/model_catalog/tokyoeye.model.json
```

Train/VSD JSON remain the **payload** for a stage; the **catalog** decides *which* payload and *which* stage runs next.

---

## 6. Automation surface

| Entry | Behavior |
|-------|----------|
| `experiments/training/governance_cli.py` (name TBD) | `lineage open`, `run`, `promote`, `status`, `sync-history` |
| `governance run -l L -f F` | Execute family’s `lifecycle[]` via MLflow |
| `governance promote -l L -f F --alias champion` | VSD+approval gates from catalog, then alias |
| `governance status -l L` | Print catalog aliases + last history vs live MLflow |
| `governance sync-history` | Reconcile catalog history with MLflow registry (detect drift) |

CI: on train PRs / scheduled jobs, invoke `governance run` / `status` — not ad-hoc train scripts that skip register/alias.

Makefile targets become **thin wrappers** around the CLI with lineage/family args from the catalog’s `active_lineage`.

---

## 7. Contract / onboard

- Onboard production pointer reads catalog `active_lineage` + family alias policy (default `@champion`).  
- Runner resolves MLflow URI; cache optional.  
- Normalizer `run_inference` adapter remains a separate completion item where unfinished.

---

## 8. Migration

1. Create catalog with `active_lineage` + families for current trunk work.  
2. Backfill `history[]` from known seals (affinity / spine) with `mlflow_run_id` when available; if a seal never had a run id, **register once** then record.  
3. Retire hub/AGENTS language that treats local HEALTHY paths as SSOT.  
4. Mark older lineages `archived` in catalog with history stubs as needed (optional).

---

## 9. Out of scope (later)

- Object store tiering beyond MLflow artifact store  
- Live traffic drift monitors  
- Rich biophysical plugin registry (stub hooks OK on `evaluate_vsd`)  
- Auto-writing sprint prose from catalog  

---

## 10. Acceptance

1. Catalog schema validated by unit tests (hierarchy + required lifecycle stages).  
2. `governance lineage open` creates a legal node; `governance run` cannot skip a required stage.  
3. Train smoke: catalog-driven run → MLflow metrics/artifacts → register → history append → alias.  
4. `governance status` detects catalog vs MLflow alias drift.  
5. Docs: AGENTS/hub point to **catalog + aliases**, not sacred paths; process language stays lineage-agnostic.  
6. No promote-by-file-copy path.

---

## 11. Locked decisions

| # | Decision |
|---|----------|
| 1 | Single hierarchical `tokyoeye.model.json` catalog — **LOCKED** |
| 2 | Automations key off catalog; MLflow holds deep run/registry state — **LOCKED** |
| 3 | Mandatory ordered lifecycle stages; no silent skip — **LOCKED** |
| 4 | Aliases replace HEALTHY restore SSOT — **LOCKED** |
| 5 | History index in catalog; bytes/metrics in MLflow — **LOCKED** |
| 6 | Shared `science/tokyo_eye/governance/` — **LOCKED** |

---

## 12. Next step

User reviews this file. On approval → implementation plan, then code (catalog schema + CLI stage machine first).
