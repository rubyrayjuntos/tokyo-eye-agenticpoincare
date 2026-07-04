# Training Governance & MLflow Schema

## Tokyo Eye / DTIE · Corpus Expansion + Multiscale Branch · Eidetix Bio


|                  |                                                                              |
| ---------------- | ---------------------------------------------------------------------------- |
| **Status**       | GOVERNANCE — active before corpus expansion begins                           |
| **Author**       | Ray Swan (Eidetix Bio)                                                       |
| **Date**         | 2026-07-02                                                                   |
| **Governs**      | Residue-scale corpus expansion (production) + multiscale experiment (branch) |
| **Supersedes**   | Ad-hoc training run tracking                                                 |
| **Prerequisite** | Phase 1a curvature SSOT fix (COMPLETE, P_CURV_01–04 passed)                  |


---

## 0. Purpose

This document governs how training runs are tracked, how the two branches relate, and what every run must record so that branch comparison and geometry reproduction are valid. It is the single source of truth for training governance during corpus expansion and the multiscale experiment.

Two failure modes this document exists to prevent:

1. **Branch divergence into a fork.** Two branches that edit shared code become two codebases that cannot be reconciled. The branch contract (§2) prevents this via an additive-only constraint against a frozen interface.
2. **Non-self-describing runs.** A checkpoint whose geometry cannot be reproduced because the curvature at write-time was not recorded. The MLflow schema (§3) prevents this — with curvature as the highest-priority field, because it floats during expansion.

---



## 1. Current State — MLflow Integration

**What exists:** MLflow is integrated into the training loop with the governance schema in `science/training/mlflow_governance.py`. A 1-epoch lever_a resume smoke (`make train-v6-mlflow-governance-smoke`) verifies params, per-epoch metrics, and end-of-run artifacts; `tests/test_mlflow_governance.py` enforces P_MLFLOW_01.

**What "single epoch run" proves and does not prove:**

- ✅ Proves: mandatory schema fields emit end-to-end (params, `log_c` + `curvature_final`, disc overlay, angular stats, probe JSON).
- ❌ Does not prove: P_MLFLOW_02 lineage across multi-stage corpus expansion (needs Stage A→B parent/child runs).

**Treat the current integration as: schema verified on residue-only smoke; lineage verification pending.**

**Known gaps (confirmed by Ray, 2026-07-02):**


| Field                                | Status        | Why it matters                                                             |
| ------------------------------------ | ------------- | -------------------------------------------------------------------------- |
| `k` / `log_c` (curvature)            | **[EMITTED]** | `log_c` per epoch + `curvature_final` at run end                           |
| Poincaré disc plot + biology overlay | **[EMITTED]** | `poincare_disc_overlay.png` + `angular_distribution_stats.json` at run end |


All other schema fields are marked `[VERIFY]` in §3 — P_MLFLOW_01 on a finished run converts them to `[EMITTED]`.

---



## 2. Branch Contract



### 2.1 Branch structure

```
master (production)
  └─ Residue-scale corpus expansion (Stage A → B → C)
     → THE MVP. This is what ships and backs the SBIR.
     → Full primary effort.

feature/multiscale-experiment (branched from master)
  └─ Step 0 δ-hyperbolicity analysis (read-only, no model changes)
  └─ Patch + meta layers OR product-manifold coupling maps
     (which one is decided by Step 0 output)
     → Secondary effort. Begins real build ONLY after Stage A
       gate passes on master.
     → Rebases from master regularly to inherit corpus/checkpoint progress.
```



### 2.2 Frozen interface — neither branch may edit these

The following are the shared substrate. The experimental branch **consumes** them and never modifies them. If the multiscale work requires a change to any of these, that change goes to **master first**, tested at residue scale, then inherited by the experiment via rebase.


| Frozen component                          | File(s)                                                     |
| ----------------------------------------- | ----------------------------------------------------------- |
| Curvature SSOT loader                     | `science/dtie/common/curvature_loader.py`                   |
| `embedding_space` schema + curvature_hash | migration `051` + table                                     |
| Core GNN encoder                          | `science/dtie/v6/gnn/` (encoder, radial/angular heads)      |
| Training loop + curriculum                | `train_loop.py`, `stage_runner.py`, `config.py` PhaseConfig |
| Loss module                               | `science/dtie/v6/loss.py`                                   |
| Möbius / Lorentz geometric ops            | (verified isometric — do not touch)                         |




### 2.3 Additive-only rule

The experimental branch contributes **new files only**. It never edits a frozen-interface file. New modules (patch pooling, meta-graph embedding, coupling maps) sit **on top of** the frozen residue GNN.

**Enforcement:** P_COUPLE_03 (§4) asserts residue GNN behavior is unchanged when experimental layers are added. If the residue-scale metrics shift, the experimental branch violated the frozen interface.

### 2.4 Rebase cadence

The experimental branch rebases from master **after every master Stage gate** (A, B, C) and **before** beginning any new experimental layer. This keeps the branch on the current corpus/checkpoint and prevents drift. A branch that hasn't rebased in more than one Stage is out of contract.

### 2.5 Resourcing

Not two equal branches. Production (residue-scale MVP) gets primary effort. The experimental branch gets Step 0 now (cheap, decisive) and real build effort only after Stage A proves the foundation holds. The MVP ships regardless of the experiment's outcome.

---



## 3. MLflow Schema — Mandatory Fields

Every training run MUST emit the following. Fields are tagged:

- `[EMITTED]` — confirmed present in current integration
- `[GAP]` — confirmed missing, must be added before corpus work
- `[VERIFY]` — Ray confirms against repo; P_MLFLOW_01 proves it



### 3.1 Params (logged once, at run start)


| Param                             | Tag        | Purpose                                                                                              |
| --------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------- |
| `branch`                          | `[VERIFY]` | residue-only | multiscale — the comparison axis                                                      |
| `parent_run_id`                   | `[VERIFY]` | MLflow run ID of warm-start source — **lineage tree**                                                |
| `corpus_manifest_hash`            | `[VERIFY]` | SHA256 of exact protein set — **reproducibility**                                                    |
| `corpus_size`                     | `[VERIFY]` | 25 | 60 | 120                                                                                        |
| `curvature_mode`                  | `[VERIFY]` | pinned:0.7026… | free                                                                                |
| `curvature_final` (converged `c`) | **[GAP]**  | The value pinned + hashed into embedding_space for this run's space_name. **The geometry artifact.** |
| `scale`                           | `[VERIFY]` | micro | micro+macro | all                                                                            |
| `feature_set`                     | `[VERIFY]` | dehydron-only | dehydron+ESM                                                                         |
| `curriculum_schedule`             | `[VERIFY]` | Phase 1 ramp params (JSON)                                                                           |
| `git_commit`                      | `[VERIFY]` | repo SHA at training time                                                                            |
| `spec_version`                    | `[VERIFY]` | which spec doc governs this run                                                                      |
| `space_name`                      | `[VERIFY]` | target embedding_space name (e.g., poincare_v7)                                                      |


**Two params do the heavy lifting:** `parent_run_id` makes warm-start lineage a traceable tree (wrong-parent warm-start becomes visible, not silent), and `corpus_manifest_hash` makes "diverse corpus" a reproducible fact rather than a description.

### 3.2 Metrics (logged per epoch)


| Metric                         | Tag           | Purpose                                                                                                              |
| ------------------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------- |
| `log_c` (curvature trajectory) | **[EMITTED]** | The path `c` takes as it floats. **Stabilization signal.**                                                           |
| `effective_experts`            | **[EMITTED]** | `exp(H(routing))` — entropy-derived expert count (uniform over N → N; collapse → 1)                                  |
| `effective_experts_min`        | **[EMITTED]** | Worst protein in the epoch; Stage gate floor                                                                         |
| `min_routing_fraction`         | **[EMITTED]** | `min(p_i)` across experts — fraction-scale collapse tell                                                             |
| `sigma2_sigma1`                | **[EMITTED]** | Geometry health (target ~0.665 on **full-run** eval; see caveat below)                                               |
| `disc_thick`                   | **[EMITTED]** | Geometry health (target ~0.219 on full-run eval)                                                                     |
| `r_d_s`                        | **[EMITTED]** | Radial/depth decoupling (target ~0.730 on full-run eval)                                                             |
| `r_e_s`                        | **[EMITTED]** | Epistemic/SASA relationship (~0.780 on full-run eval)                                                                |
| `per_fold_loss.{fold_id}`      | **[EMITTED]** | Imbalance detection by CATH topology (`fold_id` dots → underscores in MLflow keys, e.g. `per_fold_loss.3_40_50_300`) |
| `stage_gate_passed` (0|1)      | **[EMITTED]** | The gate verdict, logged as metric (see §5)                                                                          |


**Geometry baseline caveat:** σ₂/σ₁, disc_thick, r(d,s), r(e,s) targets in §5 were originally set from the **3-protein training-eval baseline** (11QE+4OBE+1IVO). **Corpus-transfer check (2026-07-03):** σ₂/σ₁ ≈ 0.665 **transfers** to the 25-fold Stage A corpus (lever_a@25 → 0.676). r(d,s) ≈ 0.730 does **not** transfer (lever_a@25 → 0.784). See §5.1 and the r(d,s) re-pin open item. Do not read a 1-epoch smoke or single-phase σ₂/σ₁ in isolation as regression from 0.665 without corpus context.

**Routing metric caveat:** Do **not** gate on mean per-expert routing fraction — it is always `1/N` by construction and cannot detect collapse. The Stage gate uses `effective_experts = exp(entropy)` and `min_routing_fraction`.

**Curvature logs as both:** `log_c` per-epoch metric (the trajectory, to watch stabilization) AND `curvature_final` param (the converged value that gets pinned). One is the path, one is the artifact. Both required.

**Per-fold loss is not optional.** Without it, you won't know if the model learns one CATH fold well while starving another until final eval — the motivated-reasoning trap reproduced at training time. Keys use sanitized CATH topology codes from the corpus manifest (resolved via PDBe cache when `fold_id` is absent on entries).

### 3.3 Artifacts (logged at run end, and per-checkpoint)


| Artifact                                   | Tag        | Purpose                                                                                                    |
| ------------------------------------------ | ---------- | ---------------------------------------------------------------------------------------------------------- |
| Poincaré disc plot + biology overlay (PNG) | **[GAP]**  | Visual audit: did the crescent hold structure, do dehydrons still separate angularly, did geometry degrade |
| `angular_distribution_stats.json`          | **[GAP]**  | Tier 2 gate numbers (KS, within-Q perm) — travels **with** the disc plot                                   |
| `probe_curvature_sources` output           | `[VERIFY]` | SSOT consistency check for this run                                                                        |
| The checkpoint                             | `[VERIFY]` | (S3 in Layer 2, not the DB)                                                                                |


**Disc plot + stats travel together.** The visual and the statistic attach to the run that produced them — geometry health, biology overlay, and gate verdict all on one run. Satisfies single-location documentation at the run level.

---



## 4. Property Tests — Enforcing the Schema

Discipline is a test, not a habit. These convert `[VERIFY]` fields to proven `[EMITTED]`.

### P_MLFLOW_01 — mandatory schema completeness

```python
def test_run_logs_mandatory_schema(finished_run):
    required_params = {
        'branch', 'parent_run_id', 'corpus_manifest_hash', 'corpus_size',
        'curvature_mode', 'curvature_final', 'scale', 'feature_set',
        'curriculum_schedule', 'git_commit', 'spec_version', 'space_name',
    }
    required_metrics = {
        'log_c', 'effective_experts', 'effective_experts_min', 'min_routing_fraction',
        'sigma2_sigma1', 'disc_thick', 'r_d_s', 'r_e_s', 'stage_gate_passed',
    }
    required_artifacts = {
        'poincare_disc_overlay.png', 'angular_distribution_stats.json',
    }
    assert required_params <= set(finished_run.params)
    assert required_metrics <= set(finished_run.metrics)
    assert required_artifacts <= set(finished_run.artifacts)
    # per-fold loss present for every fold_id in the corpus manifest
    for fid in corpus_fold_ids(finished_run.params['corpus_manifest_hash']):
        assert f'per_fold_loss.{fold_id_to_mlflow_key(fid)}' in finished_run.metrics
```



### P_MLFLOW_02 — lineage integrity

```python
def test_warm_start_lineage(finished_run):
    # If this run warm-started, parent_run_id must resolve to a real run
    if finished_run.params.get('curvature_mode') != 'cold_start':
        parent = mlflow.get_run(finished_run.params['parent_run_id'])
        assert parent is not None
        # child corpus must be superset of parent (expansion, not swap)
        assert corpus_superset(finished_run, parent)
```



### P_COUPLE_03 — frozen interface (experimental branch only)

```python
def test_residue_gnn_unchanged_with_experimental_layers():
    # Residue-scale metrics must be identical with/without patch+meta layers
    base = run_residue_only(structure='11QE')
    with_layers = run_multiscale(structure='11QE')
    assert abs(base['sigma2_sigma1'] - with_layers['sigma2_sigma1']) < 1e-3
    assert abs(base['r_d_s'] - with_layers['r_d_s']) < 5e-3
    # if these shift, the experimental branch edited the frozen interface
```

---



### P_ROUTING_01 — effective-experts metric semantics

```python
def test_expert_load_metric_semantics():
    # uniform routing over N experts → effective_experts ≈ N
    uniform = torch.ones(N) / N
    assert abs(effective_experts(uniform) - N) < 0.1
    # collapsed routing (all mass on one) → effective_experts ≈ 1
    collapsed = torch.zeros(N); collapsed[0] = 1.0
    assert abs(effective_experts(collapsed) - 1.0) < 0.1
    # mean fraction is constant 1/N — NOT a collapse signal
    assert uniform.mean() == collapsed.mean()
    # gate must distinguish them via effective count
    assert effective_experts(uniform) != effective_experts(collapsed)
```

Implemented in `tests/test_routing_metrics.py`.

### P_STOP_ENFORCEMENT — inference routing stop halts the loop

Pre-registered **stop-and-diagnose** criteria for Stage A Phase 2 (locked `v6_corpus_stage_a.json`). Evaluated on **inference-mode** routing each epoch (`inference_mode_routing_metrics`). Training-mode `min_routing_fraction=0` from `expert_dropout_p` does **not** trip the stop.

**Trips (raises, halts training) when any:**
- `inference min_routing_fraction` < 0.05 (any structure; worst PDB logged)
- canary `1PGB` inference `min_r` < 0.05
- `inference effective_experts_min` ≤ 2.5

**Does not trip:** geometry-only `stage_gate_passed=0`, training-mode routing metrics.

Implementation: `science/training/stage_a_stop.py`; wired in `experiments/training/v6/stage_runner.py` after inference routing each epoch. Disable for debug: `--no-stage-a-stop`. Property tests: `tests/test_stage_a_stop.py`.

**Archived checkpoint verification (belt-and-suspenders):** On `manifests/v6_corpus_disc_target.json` (3 proteins), `lever_a_clean_slate_v1` yields `effective_experts ≈ 3.97` and `stage_gate_passed = 1`. `full_hyp_moe_test` yields `effective_experts ≈ 2.19`, `min_routing_fraction = 0.0`, and `stage_gate_passed = 0` — the gate **correctly refuses** sub-threshold routing (dead expert at 0% load). That checkpoint is a **test fixture** for gate behavior, not a verdict that the run was pathological; partial concentration on a 3-protein eval can be a legitimate intermediate state. P_ROUTING_01 synthetic collapse covers `effective_experts ≈ 1.0`; `full_hyp_moe_test` covers realistic partial degeneracy. Integration test: `test_collapsed_checkpoint_trips_stage_gate` (skips if checkpoints absent).

---



## 5. Gate vs. Track — The Critical Separation

**MLflow tracks what happened. Property tests gate what proceeds. These are different jobs.**

- MLflow shows routing entropy / effective experts. It does NOT stop Stage B from starting by itself.
- The Stage A→B gate is a **pre-registered stop condition** in the property test suite (`stage_a_gate_passed`).
- The test's verdict logs INTO MLflow as `stage_gate_passed`, so the audit trail shows the metric AND the decision in one place.
- **The dashboard is not the gate.** Instruments report; tests gate.

**CI vs MLflow (load-bearing separation):** GitHub Actions (`.github/workflows/gates.yml`) runs property tests on every push/PR. Postgres service container + `tests/fixtures/seed_curvature_ssot.sql` enables **P_CURV_01** (curvature SSOT four-source probe). `curvature_hash` uses IEEE 754 bytes (migration 052), not `repr()`, for cross-runner stability. Unit gates without DB: P_CURV_02/03, P_MLFLOW_01, P_ROUTING_01, enforcement-matrix claim freshness. Collapsed-checkpoint trip test stays `@pytest.mark.integration` (on-demand, not per-commit). No GPU, no training. **P_CURV_01 in CI is** `[VERIFY]` **until the first green** `gates.yml` **run** on a PR with the fixture committed — wired is not passed. Experimental branch CI should add P_COUPLE_03 when multiscale layers exist.

**Routing semantics (corrected 2026-07-02):** The Stage gate governs **effective expert count** `exp(H(p))`, not mean routing fraction. Mean fraction `(Σpᵢ)/N` is always `1/N` regardless of collapse — comparing it to `[3.0, 4.5]` was a units mismatch that made `stage_gate_passed` always fail. Implementation: `science/training/routing_metrics.py`; gate: `stage_a_gate_passed()` in `mlflow_governance.py`.

**Stage gate definitions (pre-registered):**

```
Stage A → B gate:
  effective_experts       ∈ [3.0, 4.5]   # exp(entropy); ~4 uniform, ~1 collapsed
  effective_experts_min   > 2.5          # worst protein in epoch
  min_routing_fraction    ≥ 0.05       # no expert starved below 5%
  |sigma2_sigma1 - 0.665| / 0.665 < 0.10   # full-run eval baseline only
  |r_d_s - 0.730| < 0.05
  per_fold_loss max/min ratio < 3.0   (no fold starved)

Stage B → C gate: same thresholds, corpus_size=60
Stage C completion: same + curvature stabilization (§6)
```

**Geometry baseline corpus-transfer (2026-07-03, lever_a@25 inference):** The σ₂/σ₁ pin (0.665) **transfers** to the locked 25-fold Stage A corpus — `lever_a_clean_slate_v1` evaluated on all 25 train structures yields σ₂/σ₁ ≈ **0.676** (within ±10%). The r(d,s) pin (0.730) does **not** transfer — the same checkpoint yields r(d,s) ≈ **0.784**, which fails the current upper band (0.780). **Pending governance correction (separate from routing stop):** re-derive r(d,s) target from lever_a@25 (~0.784 ± band). Do not relax the threshold to make a run pass. σ pin stays at 0.665 until re-derived on evidence.

### 5.1 Stage A P2 stop-and-diagnose — `stage_a_curriculum_v1` (2026-07-03)

**Status:** RUN STOPPED — pre-registered inference-routing criterion tripped at global epoch **236**. Training continued to ~259 before manual kill; **`P_STOP_ENFORCEMENT` now wired** so future trips halt the loop automatically (see §4).

#### Trigger

| Field | Value |
| ----- | ----- |
| Run | `checkpoints/v6/runs/stage_a_curriculum_v1` |
| Corpus | Locked `manifests/v6_corpus_stage_a.json` (25 train / 16 holdout) |
| Stop epoch | Global **236** (Phase 2) |
| Criterion | Pre-registered **inference-mode** `min_routing_fraction` < 0.05 |
| Canary | **1PGB** (GB1, 56 residues, fold `3.10.20.10`) inference `min_r` = **0.0491** |
| Train-mode `min_r` at ep236 | 0.000 — **dropout artifact** (`expert_dropout_p=0.15`); not the stop signal |

**Honored:** criterion tripped → stop-and-diagnose. No continuation past the pre-registered line.

#### Prior investigation context (same run)

1. **Training `min_r=0` on 36/40 P2 epochs** — falsified as routing collapse. Per-structure inference audit: **0/25** structures below 0.05 when dropout off; training zeros were **rotating dropout**, not starvation.
2. **`stage_gate_passed=0`** — after inference gate wiring, routing **passes** on clean input; refusal is **geometry** (σ₂/σ₁ ~0.55–0.58 vs transferable 0.665 target), not routing artifact.
3. **Ep236→259 gap** — stop detected but did not halt until manual kill. Fixed by `P_STOP_ENFORCEMENT`.

#### 1PGB inference `min_r` trajectory

**Long arc (P2 ep149+, inference audit):** 1PGB became stable worst structure; monotone slide ~0.14 → ~0.062 by global 180.

**ep200–236 (37 epoch snapshots, inference):**

```
Range: [0.0328, 0.0698]   ep200→ep236: 0.0582 → 0.0491 (Δ −0.0092)
OLS slope: +0.00006/epoch (flat over window)
Epochs below 0.05 in window: 17/37
Trough: 0.0328 (g211); recoveries to ~0.07 then re-cross
```

**Shape verdict:** Neither pure monotone abandonment nor a single bounce. **Contested-expert oscillation** — router repeatedly nearly drops 1PGB's expert, partial recovery, drops again. Abandonment would descend monotonically; clean service would hold above floor; **contested** structures oscillate at the routing boundary.

#### Next-smallest check (size cliff?)

| Structure | Residues | ep236 inference `min_r` |
| --------- | -------- | ----------------------- |
| **1CRN** | **46** (smallest in corpus) | **0.206** |
| **1PGB** | 56 | **0.049** |
| 1R69 | 63 | 0.183 |
| Next-worst overall | 1A2P (108 res) | 0.140 |

**No size gradient.** A smaller structure routes four times healthier than 1PGB. The reflexive "raise small-structure exclusion threshold from 36 to ~65 residues" is **falsified** — 1CRN proves 56 residues is not inherently too small to route. Failure is **1PGB-specific** (topology / contact graph), not size-regime class.

#### Diagnosis (complete)

**Contested-expert oscillation on a single topologically distinct structure** (`3.10.20.10`, sole corpus representative). Not size-regime degeneracy. Not progressive abandonment. Not dropout artifact.

#### Ruled out

| Hypothesis | Evidence against |
| ---------- | ---------------- |
| Widespread routing starvation | Inference audit: 0/25 below 0.05 until late P2; train zeros = dropout |
| Exclude 1PGB on size grounds | 1CRN (46 res) healthy at 0.206 |
| "0.0491 is fine, one reading above floor" | 17/37 epochs below 0.05 in ep200–236; pre-registered line is 0.05 |
| Bundle geometry stall into routing stop | Separate issues; σ may re-read after routing fix |

#### Chosen fix path

**Primary: load-balancing floor** — auxiliary loss penalizing per-structure min expert load below 0.05 (matches inference `min_r` stop line). **Conservative λ=10** (`--routing-load-floor`); verification must satisfy **both**:

- 1PGB inference `min_r` > 0.05 on all epochs, and
- Batch `effective_experts` ∈ [3.0, 4.5] (no flattening specialization)

Implementation: `routing_load_floor_penalty()` in `science/training/routing_metrics.py`; wired in `gosp_loss_v6` via `routing_load_floor_coeff` / `routing_load_floor_min` on Phase 2 (`apply_routing_load_floor_phase2`). Differentiable hinge uses soft gate probabilities (not hard Gumbel scores). Run verification with `P_STOP_ENFORCEMENT` live.

**Excluded for now: corpus exclusion of 1PGB** — would delete sole `3.10.20.10` fold and export the class problem to Stage B without addressing mechanism. Revisit only if floor verification fails and topology-degeneracy is argued on evidence beyond size.

#### Geometry baseline corrections (separate track — do not bundle with routing fix)

| Pin | lever_a@25 (inference) | Verdict | Action |
| --- | ---------------------- | ------- | ------ |
| σ₂/σ₁ = 0.665 | ≈ 0.676 | **Transferable** | Keep pin; σ stall below 0.665 is real not-converged — **re-read after routing fix**, not diagnosed in isolation |
| r(d,s) = 0.730 | ≈ 0.784 | **Miscalibrated** (above current band) | Re-derive pin from lever_a@25; record in this doc; do not tune until green |

#### Next actions (ordered)

1. ~~Wire `P_STOP_ENFORCEMENT`~~ **DONE** (`stage_a_stop.py`, `tests/test_stage_a_stop.py`)
2. ~~Implement load-balancing floor~~ **DONE** (`routing_load_floor_penalty`, `--routing-load-floor`)
3. ~~Short P2 verification run~~ **PASS** (`stage_a_floor_verify_v1`, ep87–116, λ=10): 1PGB inference min_r ∈ [0.100, 0.200] (0/30 below 0.05); `effective_experts` ∈ [3.58, 3.97]; no stop enforced.
4. **Full P2 retrain** (`stage_a_p2_floor_v1`, ep87–107): **stop at ep107** — `P_STOP_ENFORCEMENT` tripped on **1R69** inference min_r=0.0492 (not 1PGB). **1PGB held** [0.112, 0.196], 0/21 below 0.05; `effective_experts` ∈ [3.62, 3.96]. Floor fixed contested 1PGB; new boundary structure is 1R69.

#### 5.1.1 1R69 stop diagnosis (`stage_a_p2_floor_v1`)

| Field | Value |
| ----- | ----- |
| Structure | **1R69** (63 res, fold `1.10.260.40`, sole **train** rep; holdout `2CRO` disabled) |
| Stop | ep107 inference min_r=**0.0492** (single breach; 1/21 P2 epochs below 0.05) |
| Trajectory ep102–107 | **Monotone abandonment** 0.098 → 0.087 → 0.057 → **0.049** (not 1PGB-style contested oscillation) |
| Worst-structure handoff | ep96–99: 1PGB; ep100+: **1R69** takes over as batch worst |
| Size cliff | **Falsified** — 1CRN (46 res) stayed ≥0.102 through ep107 |

**Verdict (revised — interaction data pulled post-v2 mistake):** λ=10 fixed 1PGB contested oscillation; **1R69 monotone abandonment is a different mechanism**. v2 (λ=15) launched without this analysis — **killed** (self-stopped ep107, 1R69=0.0442, worse than v1). **Do not resume v2.**

#### 5.1.2 v1 interaction data (pre-λ decision)

**Audit-script boundary:** The ep102–107 monotone finding comes from `metrics.json` (`stage_runner` inference pass), **not** the audit script. Old audit logic (`ge≥137`) mis-counted P2 epochs (denominator 0); it did **not** shift the 1R69 trajectory numbers. New inference: P2 start = **87** from `metrics.json`.

**1R69 full-v1 inference `min_r` (λ=10):**

| Window | mean | min | Notes |
| ------ | ---- | --- | ----- |
| ep87–95 | 0.194 | 0.156 | Healthy; floor nonzero from ep88 but 1R69 still ~0.19–0.21 |
| ep96–99 | 0.136 | 0.126 | **Descent begins** — coincides with 1PGB dip and `effective_experts` drop |
| ep100–107 | 0.094 | 0.049 | Monotone to breach; 1PGB recovers/stabilizes [0.112, 0.148] |

**`effective_experts` (inference batch mean):** [3.624, 3.964] — **stayed in [3.0, 4.5] entire run**. Drift −0.31 from ep87→107; no flatten-to-collapse, but downward trend correlates with floor loss (r≈−0.47).

**Floor loss vs outcomes (Pearson on v1):** floor_loss ↔ 1R69 min_r **r≈−0.51**; floor_loss ↔ effective_experts **r≈−0.47**. Stronger floor pressure co-occurs with lower 1R69 and lower specialization — consistent with redistribution / wrong-direction for λ-up.

**Three-way read:**

| Hypothesis | Evidence | λ=15? |
| ---------- | -------- | ----- |
| Floor *caused* 1R69 from onset | **Weak** — healthy ep87–95 while floor active ep88+ | — |
| Floor *redistributed* after 1PGB stabilized | **Partial** — ep96 joint dip (P2 pressure); ep100+ 1PGB holds while 1R69 monotone slides | **Wrong direction** (v2 confirmed: 0.0442 < 0.0492) |
| Floor *too weak* to catch second structure | **Partial** — 1R69 drifts even in `floor_verify` (ep116 1R69=0.083) | Global λ-up not the shape |
| Floor *flattened* MoE | **No** — effective_experts stayed in band | — |

**Working hypothesis (revised §5.1.3):** Two-part — floor wrong *shape* (redistribution) **and** P2 specialization pressure too high (`effective_experts` drifts on `floor_verify` without 1PGB crisis). **Not** option-1-vs-option-2; **P2 LR tail first** (single variable), per-structure floor only if breaches persist.

#### 5.1.3 `floor_verify` pressure diagnostic (existing run, no training)

`stage_a_floor_verify_v1` (λ=10, ep87–116):

| Signal | Result |
| ------ | ------ |
| `effective_experts` | **Drifts down** 3.934 → 3.700 (Δ=**−0.234**, slope **−0.011/ep**) |
| Buckets | ep87–95 **3.937** → ep96–105 **3.778** → ep106+ **3.708** |
| 1R69 `min_r` | 0.200 → 0.083; ep105 **0.0515**, ep107 **0.0521** (boundary, not pure monotone) |
| 1PGB `min_r` | Also drifts 0.200 → 0.100 over 30ep on verify |
| Pearson(`effective_experts`, 1R69 `min_r`) | **+0.95** — joint descent |

**Read:** Pressure abandons boundary structures independent of 1PGB redistribution. Option 2 needed at minimum; per-structure floor (option 1) is additive guardrail after pressure fix, not substitute.

#### 5.1.4 No-floor comparison (`stage_a_curriculum_v1`)

Best available: **25-corpus, no `routing_load_floor`**, full curriculum P2 ep137–259. **Gap:** not P2-only from lever_a@87 (P1 precedes P2); no identical warm-start control.

| Run | Mode | Window | eff drift | Slope |
| --- | ---- | ------ | --------- | ----- |
| `curriculum_v1` | train | P2 ep137–259 (123ep) | 3.874→3.715 (Δ−0.159) | **−0.0006/ep** |
| `curriculum_v1` | inference (snapshots) | P2 ep137–166 sample | 3.921→3.808 (Δ−0.113) | **−0.0047/ep** |
| `floor_verify` | inference | ep87–116 | 3.934→3.700 (Δ−0.234) | **−0.0108/ep** |
| `floor_verify` | train | ep87–116 | 3.981→3.510 (Δ−0.470) | **−0.0165/ep** |

**Read:** `effective_experts` **drifts without the floor** (intrinsic P2 pressure confirmed). **With floor, drift ~3× steeper** (inference slopes −0.005 vs −0.011; train Pearson(floor_loss, eff)≈**−0.83**). Floor is **not innocent** — it amplifies concentration; fix is **pressure reduction + floor-shape rethink**, not λ-up.

**Pre-registered acceptance (four clauses):** (1) all 25 inference `min_r` > 0.05 every epoch; (2) `effective_experts` ∈ [3.0, 4.5]; (3) slope ≥ −0.003/ep; (4) **specialization occurred** — eff demonstrably below ~4.0 (not uniform ceiling).

**Sequence:** property-test per-structure floor (parallel) → **bounded P2 pressure sweep** (LR tail and/or angular/dropout — one knob per run) → per-structure floor only if breaches persist.

5. ~~Resume P2 @ λ=15~~ **KILLED**
6. No-floor eff diagnostic — **§5.1.4** (pressure + floor amplification)
7. Property-test per-structure floor → bounded pressure sweep (not LR-only assumption)
7. Re-read σ₂/σ₁ trajectory on clean routing run before independent geometry diagnosis
8. Governance PR: r(d,s) pin re-derivation (separate from floor work)

---



## 6. Curvature Stabilization Protocol (Phase 2 / corpus_120)

Curvature floats during expansion. The stabilization criterion is pre-registered BEFORE the run, not observed after.

```
log_c gradient norm logged per epoch (add to §3.2 metrics)
Stable = gradient norm < 1e-4 for 10 consecutive epochs
Minimum 20 epochs before declaring stable (avoid early false plateau)
Record curvature_final at stabilization → pin as new space_name
  If c ∈ [0.65, 0.72]: expected range (between 0.605 and 0.7026), proceed
  If c < 0.60 or c > 0.75: anomalous, investigate before pinning
```

The new stable `c` becomes a new `embedding_space` version (new space_name, new curvature_hash, Phase 1b Secrets Manager key). lever_a embeddings under the old space_name remain valid for comparison. No coordinate mixing between space_names.

---



## 7. Phased Buildout

Same phased discipline as the SSOT fix — prove the schema before investing in backend infra.

```
Layer 1 — Tracking wrapper + schema (MVP — mostly done)
  Existing MLflow integration + close the two [GAP] fields (curvature, disc overlay)
  Add P_MLFLOW_01/02 property tests
  Gate: a residue-only run passes P_MLFLOW_01 end to end

Layer 2 — Backend (after Layer 1 stable)
  Postgres backend store (reuse Aurora pattern)
  S3 artifact store (checkpoints are large — not in the DB)
  Deferred because: file-store MVP proves schema correctness first

Layer 3 — Comparison tooling (when both branches have runs)
  Parent/child run nesting for Stage A→B→C lineage
  Branch-tagged run groups (residue-only vs multiscale filterable)
  Saved comparison query: four health metrics across branches
```

---



## 8. What Not To Do

**Do not replace the training loop with a framework.** MLflow is a tracking layer, not a trainer. The Phase 1 curriculum, `epistemic_uncertainty_only_train` mode, Riemannian/AdamW switching, and freeze mechanics are load-bearing and hard-won. Re-encoding them into a framework's abstraction is a rewrite that risks regression right before corpus expansion. Adopt MLflow's tracking API; keep the loop.

**Do not let the experimental branch edit frozen-interface files.** Changes to shared code go to master first. (Enforced by P_COUPLE_03.)

**Do not gate on the MLflow dashboard.** Log metrics there; gate in property tests; log the verdict back as a metric.

**Do not start corpus expansion with the two [GAP] fields unclosed.** A run that doesn't log curvature is not self-describing — its geometry is unreproducible. Close the gaps first.

**Do not skip Phase 1 curriculum on warm-start.** The MoE routing has never seen most expansion proteins. It needs the curriculum again — faster, because geometry is initialized, but not skipped. Skipping is how expert collapse returns.

---



### P_CORPUS_01 — locked corpus vs frozen TM-align report

See `docs/specs/stage-a-corpus-selection/design.md` §6. CI: `tests/test_corpus_redundancy_gate.py` — frozen JSON only, SHA256 pins in `corpus_governance.py`, proxy metric regression fails explicitly.

### P_STAGE_A_SMOKE — one-epoch assembled stack (before full Stage A curriculum)

**Purpose:** Prove corpus lock + MLflow schema + `per_fold_loss` + routing metrics compose on the **locked** manifest — not full curriculum.

```bash
make train-v6-stage-a-smoke          # 1 epoch, v6_corpus_stage_a.json, lever_a resume, CPU ok
STAGE_A_SMOKE_RUN_ID=<id> make test-stage-a-smoke
```

**Asserts (via** `science/training/stage_a_smoke.py`**):**


| Check                     | Required                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------ |
| `corpus_manifest_hash`    | Matches locked `v6_corpus_stage_a.json`                                                          |
| `per_fold_loss.{fold_id}` | ≥2 keys, underscores not dots; **no** `per_family_loss.`*                                        |
| Routing gate live         | `effective_experts`, `effective_experts_min`, `min_routing_fraction`, `stage_gate_passed` logged |
| P_MLFLOW_01 core          | Params, mandatory metrics, governance artifacts                                                  |


`stage_gate_passed` value may be 0 on a 1-epoch subset — smoke verifies the gate is **wired**, not that training has converged.

**Regeneration pins:** `make sync-corpus-pins` → update `corpus_governance.py` in the **same commit** as report/manifest JSON.

---



## 9. Open Items Before Corpus Expansion


| Item                                                                  | Status                                                             | Blocks                           |
| --------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------- |
| Close curvature (`log_c` metric + `curvature_final` param) in MLflow  | **[EMITTED]**                                                      | Corpus expansion                 |
| Close Poincaré disc overlay + stats artifact in MLflow                | **[EMITTED]**                                                      | Corpus expansion                 |
| Verify `[VERIFY]` fields against repo                                 | Done (P_MLFLOW_01 smoke)                                           | Schema enforcement               |
| Add P_MLFLOW_01/02/ROUTING_01 to property suite                       | P_MLFLOW_01 + P_ROUTING_01 done; P_MLFLOW_02 pending               | Schema enforcement               |
| Wire property gates into GitHub Actions CI                            | **[EMITTED]** `.github/workflows/gates.yml`                        | Governance self-enforcement      |
| P_CURV_01 SSOT probe in CI (Postgres service)                         | **[VERIFY]** wired; first green `gates.yml` run pending            | Merge-time SSOT guard            |
| `curvature_hash` IEEE754 format (migration 052)                       | **[EMITTED]** replaces repr for CI cross-env                       | False-red guard risk             |
| P_CURV_01 fixture checkpoint in git (`lever_a_v6_best_disc.pt`)       | **[EMITTED]** MVP; migrate to LFS/S3 if refreshed >2×              | Repo size / binary rot           |
| Step 0 δ-hyperbolicity analysis (defines experimental branch)         | Pending                                                            | Multiscale branch spec           |
| Define Stage A corpus (fold-topology selection via redundancy script) | **[EMITTED]** locked `v6_corpus_stage_a.json` (25 train, TM-align) | Stage A start                    |
| `per_fold_loss` rename (CATH `fold_id` vocabulary)                    | **[EMITTED]**                                                      | Wrong imbalance metric corrected |
| P_CORPUS_01 corpus redundancy gate in CI                              | **[EMITTED]** frozen TM-align report vs locked manifest            | Corpus drift                     |
| Staging/prod `embedding_space.curvature` check                        | Pending                                                            | Phase 2 re-ingest                |
| P_STOP_ENFORCEMENT (inference stop halts loop)                        | **[EMITTED]** `stage_a_stop.py`, ep236 memo §5.1                   | Trustworthy verification runs    |
| r(d,s) Stage gate pin re-derivation (0.730 → ~0.784 lever_a@25)       | Pending                                                            | Correct geometry gate on 25-corpus |
| Load-balancing floor (1PGB contested routing)                         | **Half-pass MVP**; pressure+shape (§5.1.3)                         | P2 LR tail first; then floor shape |


---



## 10. Audit Trail


| Event                                                    | Date       | Artifact                                                                          |
| -------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------- |
| Phase 1a curvature SSOT fix complete                     | 2026-07-02 | 5 commits, P_CURV_01–04 passed                                                    |
| MLflow integrated (single-epoch pipe test at lever_a)    | 2026-07-02 | lever_a run                                                                       |
| Two schema gaps identified (curvature, disc overlay)     | 2026-07-02 | This document §1                                                                  |
| This governance document written                         | 2026-07-02 | This document                                                                     |
| Two [GAP] fields closed                                  | 2026-07-02 | `science/training/mlflow_governance.py`, smoke run `mlflow_governance_smoke2`     |
| Routing gate units mismatch fixed (`effective_experts`)  | 2026-07-02 | `routing_metrics.py`, §5 correction, P_ROUTING_01                                 |
| P_MLFLOW_01 passing on residue-only smoke                | 2026-07-02 | run `dec3c7f528074a318ec4ec4aff805d3f`                                            |
| Collapsed checkpoint fails routing gate (`full_hyp_moe`) | 2026-07-02 | `effective_experts ≈ 2.19`, P_ROUTING_01 integration                              |
| Property gates CI workflow                               | 2026-07-02 | `.github/workflows/gates.yml`                                                     |
| P_CURV_01 CI enforcement (Postgres + seed)               | 2026-07-02 | `seed_curvature_ssot.sql`, fixture checkpoint — **[VERIFY]** until first CI green |
| `curvature_hash` → IEEE754 bytes (migration 052)         | 2026-07-03 | `curvature_loader.py`, CI cross-env stability                                     |
| Stage A corpus defined + started                         | 2026-07-03 | `v6_corpus_stage_a.json`, `stage_a_curriculum_v1`                                 |
| Inference routing gate wiring (`stage_gate_passed`)    | 2026-07-03 | `inference_mode_routing_metrics`, P_ROUTING_EVAL_MODE                             |
| Stage A P2 stop — 1PGB inference `min_r` < 0.05        | 2026-07-03 | Global ep236; memo §5.1; contested-oscillation diagnosis                          |
| P_STOP_ENFORCEMENT wired                               | 2026-07-03 | `stage_a_stop.py`, `tests/test_stage_a_stop.py`                                   |
| r(d,s) pin miscalibration noted (lever_a@25 = 0.784)   | 2026-07-03 | §5.1; re-pin pending                                                              |
| Step 0 δ-hyperbolicity complete                          | —          | To be filled                                                                      |


