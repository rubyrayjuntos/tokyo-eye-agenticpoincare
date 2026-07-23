# Tokyo Eye — Project Hub

**Purpose:** One place to hold the roadmap, active trunk, and the docs that actually matter.  
**Not:** a second source of truth. Machine status lives in gate JSON; this page stitches it.

| Role | Path |
|------|------|
| **This hub** | [`docs/PROJECT_HUB.md`](PROJECT_HUB.md) |
| **Live biology roadmap (JSON)** | [`data/gates/fix1_sparsity_biology_phase_status.json`](../data/gates/fix1_sparsity_biology_phase_status.json) |
| **Print live ledger + next** | `make project-hub` |
| **Platform identity** | [`AGENTS.md`](../AGENTS.md) |
| **Human onboarding** | [`DEVELOPER_ONBOARDING.md`](DEVELOPER_ONBOARDING.md) |
| **AI contract** | [`AI_DEVELOPER_CONTRACT.md`](AI_DEVELOPER_CONTRACT.md) |

---

## 1. Where we are (Tokyo Eye v7 cutover)

| Item | Pointer |
|------|---------|
| **Production GNN** | `TokyoEye-v7` — `science/tokyo_eye/TokyoEye.py` · contract `tokyo_eye_v7` |
| **Healthy B′ restore (disc Pass)** | `HEALTHY_V7_CKPT` — `checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt` |
| **Biology roadmap (v7)** | [`specs/tokyo-eye-v7/biology-roadmap.md`](specs/tokyo-eye-v7/biology-roadmap.md) · stamp [`data/gates/tokyo_eye_v7_biology_roadmap.json`](../data/gates/tokyo_eye_v7_biology_roadmap.json) — **B0 PASS → B1 FAIL (scramble>conduit) → B2 blocked** · `make grade-v7-b1-teleconnections` |
| **Uncertainty** | PARKED — [`specs/tokyo-eye-v7/bprime-uncertainty-unpark.md`](specs/tokyo-eye-v7/bprime-uncertainty-unpark.md) |
| **Cutover scaffold ckpt** | `checkpoints/v7/tokyo_eye_v7_cutover_scaffold.pt` (init; **not** biology-graded) |
| **Cold Hyp MP funnel (open)** | `tokyo_eye_v7_cold_hyp_mp_funnel_v1` — cold init + Cα graph + funnel/disc · [`cold-hyp-mp-funnel-prereg.md`](specs/tokyo-eye-v7/cold-hyp-mp-funnel-prereg.md) · `make train-v7-cold-hyp-mp-funnel` |
| **MLflow** | experiment `tokyo-eyes-v7` · registered model `TokyoEye-v7` · `make mlflow-register-v7-lineage` |
| **Spec** | [`specs/tokyo-eye-v7/README.md`](specs/tokyo-eye-v7/README.md) |
| **Frozen biology baseline (compare-only)** | `FIX1_SPARSITY_CHAMPION_CKPT` under `checkpoints/v66/` — Phase 4c claims retained; not default trunk for new work |
| **Sealed v66 restore** | `HEALTHY_FIX1_CKPT` |
| **Sealed v8 affinity (CASF Core ≥0.40)** | `HEALTHY_V8_AFFINITY_CKPT` → `checkpoints/v8/runs/tokyo_eye_v8_affinity_s1011_finetune_all/v8_affinity_best.pt` · gate [`tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json`](../data/gates/tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json) — **compare-only affinity trunk; not Mol* / onboard production** |
| **v8 Sprint 10.2** | PRE-REG outline only — [`superpowers/specs/2026-07-23-tokyo-eye-v8-sprint10-2-multisite-ligand-prereg.md`](superpowers/specs/2026-07-23-tokyo-eye-v8-sprint10-2-multisite-ligand-prereg.md) |

**Cutover note:** Production inference identity is Tokyo Eye. Formal Pass vs Fix-1 champion is a **parallel** pack — not required to start B0/B1 investigation probes on sealed health.

**Do not claim:** scaffold or sealed health = biology champion; Ledger B latch Pass; unconstrained OOD; free-energy from embedding density; zero-shot ligands; v8 affinity Core \(R\approx0.407\) = cryptic-pocket discovery.


Refresh the live table anytime:

```bash
make project-hub
```

---

## 1b. Frozen v6.6 Fix-1 ledger (compare-only)

| Item | Pointer |
|------|---------|
| **Champion gate** | [`data/gates/fix1_sparsity_champion.json`](../data/gates/fix1_sparsity_champion.json) |
| **Phase status / roadmap** | [`data/gates/fix1_sparsity_biology_phase_status.json`](../data/gates/fix1_sparsity_biology_phase_status.json) |
| **Narrative next-phase** | [`specs/routing-entropy-sparsity/next-phase.md`](specs/routing-entropy-sparsity/next-phase.md) |

**Allowed claim (Phase 4c, frozen trunk):** governed, state-aware transport on the sparsity champion for pre-registered SHP2 inactive→open (`2SHP→6CRF`) and monopoly reduction vs sealed baseline; classical KRAS G12D four-quadrant Cα edge-ΔE rewiring (`4LPK`/`6GOD`/`5US4`/`6GOF`); honest Ledger B interface Fail + Phase 4b explanation; honest platform CB concordance **hard Fail**.

**Do not claim:** Ledger B latch/tunnel interface Pass; silent Phase 4b→\(I\) adoption; generic CB concordance Pass; lowered Spearman bar; settled gradient/Jacobian interpretation; historical `4OBE`/`4DSO`/`5VQ2` complementarity as rewiring proof; `6MCF` as open SHP2; unconstrained OOD runtime.
---

## 2. Priority roadmap (edit the JSON)

Edit [`fix1_sparsity_biology_phase_status.json`](../data/gates/fix1_sparsity_biology_phase_status.json) → `ledger` / `next` / **`workstreams`**. Then update this hub’s “where we are” if the trunk changes.

**Workstreams (schema v3):** parent programs with ordered children capture *progression* (e.g. `gnn_perturbation_boundary` → single-site graft → neighborhood conduit graft). Each node carries **`meta`** (dates, model/lineage, inputs, outputs). Flat `ledger` entries remain the grade SSOT; children point at `ledger_key`. Schema note: [`specs/routing-entropy-sparsity/workstream-metadata.md`](specs/routing-entropy-sparsity/workstream-metadata.md).

Current priority order (from JSON):

1. **`gnn_perturbation_boundary` next child** — decide ON-arm neighborhood, multi-hop paste, or G12V/C (new pre-reg)  

---

## 3. Doc map (by job)

### Platform contracts (always)
| Doc | Use when |
|-----|----------|
| [`science/contracts/onboard_contract.yaml`](../science/contracts/onboard_contract.yaml) | Artifacts, jobs, geometry, readiness |
| [`specs/ingest-compute-contract/`](specs/ingest-compute-contract/) | Agent vs ingest compute boundary |
| [`ENFORCEMENT_MATRIX.md`](ENFORCEMENT_MATRIX.md) | What CI actually gates |
| [`audit/PIPELINE_AUDIT.md`](audit/PIPELINE_AUDIT.md) | Runtime audit |
| [`audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md) | Construction vs inference SSOT |

### Active GNN / Fix-1 biology
| Doc | Use when |
|-----|----------|
| [`specs/tokyo-eye-v7/biology-roadmap.md`](specs/tokyo-eye-v7/biology-roadmap.md) | **v7 hub/teleconnection biology order (B0→B1→B2)** |
| [`specs/tokyo-eye-v7/investigation-allele-epistasis-metrics.md`](specs/tokyo-eye-v7/investigation-allele-epistasis-metrics.md) | AlleleSens / Epistasis defs (≠ NIG) |
| [`specs/routing-entropy-sparsity/`](specs/routing-entropy-sparsity/) | Sparsity champion design + next phase (v66 compare-only) |
| [`specs/fix1-s4-restore/`](specs/fix1-s4-restore/) | Sealed restore / expand ladder |
| [`specs/kras-topo-structural-inference/`](specs/kras-topo-structural-inference/) | KRAS topo matrix + Ledger B (interface Fail retained; [policy closeout](specs/kras-topo-structural-inference/ledger-b-interface-policy-closeout.md)) |
| [`specs/shp2-2shp-6mcf-ood/`](specs/shp2-2shp-6mcf-ood/) | SHP2 inactive→open OOD (active PDB = **6CRF**) |
| [`TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md`](TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md) | MLflow params/metrics |

### Product pathways (wired; V66 not fully plugged in)
| Doc | Use when |
|-----|----------|
| [`specs/cryptic-site-discovery/`](specs/cryptic-site-discovery/) | Mapper → MD → fragments (V5 backend) |
| [`specs/cryptic-site-calibration/`](specs/cryptic-site-calibration/) | Benchmark / thresholds / provenance honesty |
| [`specs/discovery-story-pathway/`](specs/discovery-story-pathway/) | Discovery Story acts |
| [`specs/chem-mvp-reengage/`](specs/chem-mvp-reengage/) | Chem-MVP **PARKED** (compare-only) |

### Diagnostics corpus
Grade / probe outputs typically land under:

`checkpoints/v66/diagnostics/routing_sparsity/`

Gate stamps under:

`data/gates/`

---

## 4. How to update the hub (discipline)

1. Close or open a probe → update **`data/gates/fix1_sparsity_biology_phase_status.json`** (`ledger` + `next`).  
2. If the story changed → update **`specs/routing-entropy-sparsity/next-phase.md`**.  
3. If the active trunk changed → update **`AGENTS.md`** key decision + §1 of this hub.  
4. Run `make project-hub` and sanity-check the printed ledger.  
5. Do **not** invent Pass claims that contradict the JSON.

Optional later: attach the status JSON to the champion MLflow run as `ssot/roadmap`; sync `next[]` to Linear. The hub stays the stitch point either way.

---

## 5. Quick commands

```bash
make project-hub                          # live ledger + next from status JSON
make grade-v66-kras-topo-edge-four-quadrant   # KRAS G12D four-quadrant Cα ΔE
make grade-v66-fix1-sparsity-kras-topo-matrix   # model triad (edge report-only)
make grade-v66-fix1-gini-reduction-analysis
make grade-v66-fix1-shp2-2shp-6crf-ood
```
