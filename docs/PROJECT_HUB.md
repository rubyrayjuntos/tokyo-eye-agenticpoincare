# Tokyo Eye — Project Hub

**Purpose:** One place to hold the roadmap, active trunk, and the docs that actually matter.  
**Not:** a second source of truth. Machine status lives in gate JSON; this page stitches it.

| Role | Path |
|------|------|
| **This hub** | [`docs/PROJECT_HUB.md`](PROJECT_HUB.md) |
| **Live biology roadmap (JSON)** | [`data/gates/tokyo_eye_v8_biology_roadmap.json`](../data/gates/tokyo_eye_v8_biology_roadmap.json) |
| **Print live ledger + next** | `make project-hub` |
| **Platform identity** | [`AGENTS.md`](../AGENTS.md) |
| **Human onboarding** | [`DEVELOPER_ONBOARDING.md`](DEVELOPER_ONBOARDING.md) |
| **AI contract** | [`AI_DEVELOPER_CONTRACT.md`](AI_DEVELOPER_CONTRACT.md) |

---

## 1. Where we are (Tokyo Eye **v8** trunk)

| Item | Pointer |
|------|---------|
| **Production GNN** | `TokyoEye-v8` — [`science/tokyo_eye/v8/`](../science/tokyo_eye/v8/) · contract `tokyo_eye_v8` |
| **Spine restore** | `HEALTHY_V8_SPINE_CKPT` — `checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt` |
| **Affinity seal (CASF ≥0.40)** | `HEALTHY_V8_AFFINITY_CKPT` — gate [`tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json`](../data/gates/tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json) |
| **Trunk cutover stamp** | [`tokyo_eye_v8_trunk_ssot.json`](../data/gates/tokyo_eye_v8_trunk_ssot.json) |
| **Biology roadmap (v8 TODOs)** | [`specs/tokyo-eye-v8/biology-roadmap.md`](specs/tokyo-eye-v8/biology-roadmap.md) · B0 reset → B1 re-prereg |
| **Spec** | [`specs/tokyo-eye-v8/README.md`](specs/tokyo-eye-v8/README.md) |
| **Sprint 10.2** | PRE-REG only — [`superpowers/specs/2026-07-23-tokyo-eye-v8-sprint10-2-multisite-ligand-prereg.md`](superpowers/specs/2026-07-23-tokyo-eye-v8-sprint10-2-multisite-ligand-prereg.md) |

**Hard rule:** **v7 is dead.** Different architecture; was never production. Do not recommend `HEALTHY_V7_CKPT`, v7 B1 grades, or `docs/specs/tokyo-eye-v7/` as an open trunk. Archaeology only.

**Do not claim:** affinity CASF Pass = cryptic-pocket / allostery Pass; free-energy from embedding density; zero-shot ligands; v7 sealed health = v8 biology champion.

Refresh:

```bash
make project-hub
```

---

## 1b. Archaeology (compare-only — do not open)

| Lineage | Pointer |
|---------|---------|
| **v7 Hyp-MP** | `checkpoints/v7/` · [`specs/tokyo-eye-v7/`](specs/tokyo-eye-v7/) · `HEALTHY_V7_CKPT` |
| **v6.6 Fix-1** | `FIX1_SPARSITY_CHAMPION_CKPT` · [`fix1_sparsity_biology_phase_status.json`](../data/gates/fix1_sparsity_biology_phase_status.json) |

---

## 2. Priority roadmap

Edit [`tokyo_eye_v8_biology_roadmap.json`](../data/gates/tokyo_eye_v8_biology_roadmap.json). Current active: **B0 prep on v8**.

1. **B0** — Θ lock + forward smoke on v8 spine  
2. **B1** — teleconnections re-prereg (v8 architecture)  
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

### Active GNN (v8)
| Doc | Use when |
|-----|----------|
| [`specs/tokyo-eye-v8/README.md`](specs/tokyo-eye-v8/README.md) | Trunk identity |
| [`specs/tokyo-eye-v8/biology-roadmap.md`](specs/tokyo-eye-v8/biology-roadmap.md) | B0→B1→B2 TODOs |
| [`specs/tokyo-eye-v8/investigation-allele-epistasis-metrics.md`](specs/tokyo-eye-v8/investigation-allele-epistasis-metrics.md) | AlleleSens / Epistasis |
| [`TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md`](TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md) | MLflow params/metrics |

### Product pathways
| Doc | Use when |
|-----|----------|
| [`specs/cryptic-site-discovery/`](specs/cryptic-site-discovery/) | Mapper → MD → fragments |
| [`specs/discovery-story-pathway/`](specs/discovery-story-pathway/) | Discovery Story acts |
| [`specs/chem-mvp-reengage/`](specs/chem-mvp-reengage/) | Chem-MVP **PARKED** |

### Diagnostics / gates
Gate stamps under `data/gates/`. v8 affinity + trunk stamps above.

---

## 4. How to update the hub

1. Close or open a probe → update **`data/gates/tokyo_eye_v8_biology_roadmap.json`**.  
2. If the active trunk changed → update **`AGENTS.md`** + §1 here.  
3. Run `make project-hub` and sanity-check.  
4. Do **not** invent Pass claims that contradict the JSON.  
5. Do **not** reopen v7 as trunk.
