# Tokyo Eye — Project Hub

**Purpose:** One place to hold the roadmap, active trunk, and the docs that actually matter.  
**Not:** a second source of truth. Machine status lives in gate JSON; this page stitches it.

| Role | Path |
|------|------|
| **This hub** | [`docs/PROJECT_HUB.md`](PROJECT_HUB.md) |
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
| **Restore SSOT (live)** | `models:/TokyoEye@champion` via MLflow API (`http://localhost:5000`) |
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

Edit the active MLflow run/evidence, then update [`tokyoeye_mlflow_ssot.json`](../data/gates/tokyoeye_mlflow_ssot.json). Current active: **B0 prep on TokyoEye**.

1. **B0** — alias lock + forward smoke on TokyoEye  
2. **B1** — teleconnections re-prereg on current architecture  
3. Affinity **10.2** — only if explicitly prioritized (PRE-REG frozen)

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
| [`superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md`](superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md) | Lifecycle and restore identity |
| [`tokyoeye_mlflow_ssot.json`](../data/gates/tokyoeye_mlflow_ssot.json) | Current machine-readable restore stamp |
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
