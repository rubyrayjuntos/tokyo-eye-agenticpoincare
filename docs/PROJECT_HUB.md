# Tokyo Eye — Project Hub

**Purpose:** One place to hold the roadmap, active trunk, and the docs that actually matter.  
**Not:** a second source of truth. Machine status lives in gate JSON; this page stitches it.

| Role | Path |
|------|------|
| **This hub** | [`docs/PROJECT_HUB.md`](PROJECT_HUB.md) |
| **Architecture SSOT (front door)** | [`TOKYOEYE_ARCHITECTURE_SSOT.md`](TOKYOEYE_ARCHITECTURE_SSOT.md) |
| **Freeze addendum (numbers)** | [`superpowers/specs/2026-09-16-tokyo-eye-v8-freeze-addendum.md`](superpowers/specs/2026-09-16-tokyo-eye-v8-freeze-addendum.md) |
| **Live SSOT stamp (JSON)** | [`data/gates/tokyoeye_mlflow_ssot.json`](../data/gates/tokyoeye_mlflow_ssot.json) |
| **Print live ledger + next** | `make project-hub` |
| **Platform identity** | [`AGENTS.md`](../AGENTS.md) |
| **Human onboarding** | [`DEVELOPER_ONBOARDING.md`](DEVELOPER_ONBOARDING.md) |
| **AI contract** | [`AI_DEVELOPER_CONTRACT.md`](AI_DEVELOPER_CONTRACT.md) |

---

## 1. Where we are (TokyoEye active trunk)

| Item | Pointer |
|------|---------|
| **Production model** | Registered model `TokyoEye` — package under [`science/tokyo_eye/`](../science/tokyo_eye/) |
| **Restore SSOT (live)** | `models:/TokyoEye@champion` via MLflow (`http://localhost:5000`) — **v5 dispositioned lesson-only for EQU geometry reboot** ([disposition stamp](../data/gates/tokyo_eye_equ_champion_disposition.json)) |
| **Staging alias** | `models:/TokyoEye@experimental` |
| **Governance** | MLflow HTTP API + Python client (UI at same URL). Artifact proxy on. Not Make. Durable bytes: GitHub Releases (`tokyoeye-eqf-<sha16>`). · [`superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md`](superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md) · [`2026-08-23 vault`](superpowers/specs/2026-08-23-tokyoeye-github-release-vault-design.md) |
| **Cache (not SSOT)** | Local `.pt` under `checkpoints/` after alias/run download — e.g. prior spine/affinity seals |
| **SSOT stamp** | [`tokyoeye_mlflow_ssot.json`](../data/gates/tokyoeye_mlflow_ssot.json) |
| **Biology roadmap** | Active biology work should reference `TokyoEye@experimental` / `@champion`, not a package revision label |
| **Historical design specs** | Dated sprint/design specs remain evidence only; do not use them as restore pointers |

**Hard rule:** older package lineages are archaeology. Do not recommend old checkpoint constants, old grade sheets, or dated package specs as an open trunk.

**Do not claim:** affinity CASF Pass = cryptic-pocket / allostery Pass; free-energy from embedding density; zero-shot ligands; old sealed health = TokyoEye biology champion.

Refresh:

```bash
make project-hub
```

---

## 1b. Archaeology (compare-only — do not open)

| Lineage | Pointer |
|---------|---------|
| **Hyp-MP package line** | `checkpoints/v7/` · [`specs/tokyo-eye-v7/`](specs/tokyo-eye-v7/) |
| **v6.6 Fix-1** | `FIX1_SPARSITY_CHAMPION_CKPT` · [`fix1_sparsity_biology_phase_status.json`](../data/gates/fix1_sparsity_biology_phase_status.json) |

---

## 2. Priority roadmap

Edit the active MLflow run/evidence, then update [`tokyoeye_mlflow_ssot.json`](../data/gates/tokyoeye_mlflow_ssot.json) when restore identity changes.

**Active operator trunk (2026-09-15):** **Tokyo Eye EQU** — geometry-first; freeze amended for pure hyp.  

**EQU SSOT = MLflow.** Naming stem: `tokyo_eye_equ_<slug>` (same string for gate file, run name, `gate_id` tag). Experiment charter (purpose/goals/metrics) on experiment Notes + [`tokyo_eye_equ_experiment_charter.json`](../data/gates/tokyo_eye_equ_experiment_charter.json).  
Resolve: experiment `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine` tag `tokyo_eye_equ_ssot_run_id` → run `tokyo_eye_equ_ssot` (`f0e706526c2c481fa7c3112c89a5aeca`). Thin pointer: [`data/gates/tokyo_eye_equ_ssot.json`](../data/gates/tokyo_eye_equ_ssot.json). Convention: [`tokyo_eye_equ_naming_convention.json`](../data/gates/tokyo_eye_equ_naming_convention.json).

`models:/TokyoEye@champion` **v5** remains the registry pointer but is **dispositioned lesson-only / non-trunk** for EQU geometry.

0. **EQU governance standard** — ACTIVE ([spec](superpowers/specs/2026-09-15-tokyoeye-equ-governance-design.md) · [stamp](../data/gates/tokyo_eye_equ_governance.json))  
1. **EQU disposition** — APPROVED (logged on SSOT run)  
2. **EQU cold boot** — hygiene **Fail** sealed (`f217ade3…`)  
3. **Pure-hyp freeze amendment** — APPROVED (`tokyo_eye_equ_pure_hyp_v1`)  
4. **Next:** `tokyo_eye_equ_correct_start` — DRAFT ([spec](superpowers/specs/2026-09-15-tokyoeye-equ-correct-start-design.md) · [stamp](../data/gates/tokyo_eye_equ_correct_start.json)) — awaiting APPROVE; not remediation  
5. **SIGNED AMEND:** `tokyo_eye_equ_wrap_threshold` — `DEHYDRON_WRAP_MAX` 19 → **1** (Stage-A-12 median-then-descend; [spec](superpowers/specs/2026-09-17-tokyoeye-equ-wrap-threshold-decision.md) · [stamp](../data/gates/tokyo_eye_equ_wrap_threshold.json) · addendum §2.7). Caveat on REVERT-era wrap=19 probes: [`WRAP19_LABEL_SATURATION_CAVEAT.json`](../checkpoints/v8/runs/freeze_recon_wrap_threshold_amend/WRAP19_LABEL_SATURATION_CAVEAT.json) (theme_biology / theme_restore AUPRC already wrap=1 — out of scope)  
6. **ACTIVE path:** Assembly ENFORCED → resolve LOSO `PENDING_G_FIT` first → then only cards in [`tokyo_eye_equ_next_experiment_sealed.json`](../data/gates/tokyo_eye_equ_next_experiment_sealed.json) (MoE on pool, or non-leaking hyp biology). Front door: [`TOKYOEYE_ARCHITECTURE_SSOT.md`](TOKYOEYE_ARCHITECTURE_SSOT.md). Operator brief: [`CLAUDE.md`](../CLAUDE.md). **Do not** treat PROJECT_HUB history above as the open roadmap.
7. Historical Mode / affinity / B1 / SDRP Stage-2 — parked

---

## 3. Doc map (by job)

### Platform contracts (always)
| Doc | Use when |
|-----|----------|
| [`science/contracts/onboard_contract.yaml`](../science/contracts/onboard_contract.yaml) | Artifacts, jobs, geometry, readiness |
| [`specs/ingest-compute-contract/`](specs/ingest-compute-contract/) | Agent vs ingest compute boundary |
| [`ENFORCEMENT_MATRIX.md`](ENFORCEMENT_MATRIX.md) | What CI actually gates |
| [`audit/PIPELINE_AUDIT.md`](audit/PIPELINE_AUDIT.md) | Runtime audit |

### Active GNN (TokyoEye)
| Doc | Use when |
|-----|----------|
| [`tokyo_eye_equ_ssot.json`](../data/gates/tokyo_eye_equ_ssot.json) | **EQU operator SSOT** → MLflow run (ladder / law / next card) |
| [`superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md`](superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md) | Lifecycle and restore identity |
| [`tokyoeye_mlflow_ssot.json`](../data/gates/tokyoeye_mlflow_ssot.json) | Registry restore stamp (`@champion` / `@experimental`) |
| [`TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md`](TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md) | MLflow params/metrics |

### Product pathways
| Doc | Use when |
|-----|----------|
| [`specs/cryptic-site-discovery/`](specs/cryptic-site-discovery/) | Mapper → MD → fragments |
| [`specs/discovery-story-pathway/`](specs/discovery-story-pathway/) | Discovery Story acts |
| [`specs/chem-mvp-reengage/`](specs/chem-mvp-reengage/) | Chem-MVP **PARKED** |

### Diagnostics / gates
Gate stamps under `data/gates/`. The active restore stamp is above; dated package stamps are evidence only.

---

## 4. How to update the hub

1. Close or open a probe → update MLflow evidence and **`data/gates/tokyoeye_mlflow_ssot.json`** when restore identity changes.  
2. If the active trunk changed → update **`AGENTS.md`** + §1 here.  
3. Run `make project-hub` and sanity-check.  
4. Do **not** invent Pass claims that contradict the JSON.  
5. Do **not** reopen old package lineages as trunk.
