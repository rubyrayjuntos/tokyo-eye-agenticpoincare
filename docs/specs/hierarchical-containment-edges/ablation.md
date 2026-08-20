# Hierarchical Containment Edges — Ablation / Gates

**Date:** 2026-07-17
**Purpose:** Pre-registered physics + diameter-stratified flow-influence acceptance for containment v2.
**Design:** [`design.md`](design.md) · Hard lock: [`depth-collision.md`](depth-collision.md)

---

## Matched arms (to lock at launch)

| Arm | Intent |
|-----|--------|
| **Baseline** | Same parent as Chem-MVP feeler stack (role ± chem as in the Stage A-12 flow baseline), **no** containment edges |
| **Containment** | Same + `contain_down` / `contain_up` (+2 radial MLPs), parent nodes mean-pooled (Option A) |

Identity checklist before cold run: isolated-seed prototype identity (`max |Δ|=0` except new radial MLP slots); `node_dim` / lineage / role / chem flags matched.

---

## Gate order

1. Isolated-seed check (§5.3 design) — plan Task 4, `max |Δ|=0`
2. Physics non-regression (rim / cone·τ / MoE)
3. Oversmoothing-at-root (T1c-style on parent nodes) — plan Task 8, design §5.2
4. **Primary:** diameter-stratified asymmetry via `jacobian_flow_influence.py` — plan Task 9

Score is sanity only — not the verdict.

---

## Primary acceptance (locked before results)

Baseline numbers from `checkpoints/v66/diagnostics/learned_flow_influence/stage_a12.json`
and diameters from `asymmetry_vs_diameter.json`. Asymmetry floor = **0.05**
(`ASYMMETRY_FLOOR` in the flow probe). Pass-group regression tolerance = **10% relative**
below that structure's own baseline asymmetry.

| Group | Structures | Baseline asym | Requirement |
|-------|------------|---------------|-------------|
| Fail (diam ≥14) | 1BG1, 2Z6H, 1IVO, 2SHP, 1F88 | 0.023–0.032 | Individually clear ≥0.05 on **≥3 / 5** |
| Pass (diam ≤9) | 1UBQ, 1TEN, 1HHP, 1LYZ, 1MBN, 4OBE, 1TIM | 0.058–0.148 | No structure below `baseline × 0.90` |
| Hub-tracking | all 12 | ρ ≈ 0.44–0.69 | No structure below its own baseline Spearman(betweenness) |

### Outcomes

| Outcome | Criteria |
|---------|----------|
| **Win** | Fail-group ≥3/5 clear 0.05; pass-group holds; hub-tracking holds corpus-wide |
| **Partial** | Fail-group any rise but &lt;3/5 clear floor, **or** pass-group mild regression within 10% |
| **Fail** | No fail-group movement, **or** pass-group regression &gt;10%, **or** any hub-tracking drop below own baseline |

No pooling. Per-structure only. Disc layer reported but **not** deciding
([`DISC_PROJECTION_NOT_TRUNK_PROXY`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)).

---

## What this does *not* decide

- Path 2 (explicit directionality reward on diam ≤9) — separate registration after this closes
- Chem-Full — independent, not gated
- Path A ingest extension / train–app hierarchy unification — **explicitly deferred**; Path B is a stopgap only (design §9.2)
- Whether biotite `sse_code` should remain the app SSOT for `ss` features — separate fidelity question (design §9.1)
- Cross-domain parent shortcuts — out of scope for v1
- Dual-loading residue radius — **forbidden** (depth-collision lock)

---

## Status

| Item | Status |
|------|--------|
| Design v2 promotion | **Adopted** 2026-07-17 |
| Acceptance table | **Pre-registered** |
| Hierarchy data coverage | **Audited 2026-07-17** — source SSE 12/12; no governed hierarchy tables |
| Materialization path | **Path B locked** (train-side HELIX/SHEET stopgap; Path A supersedes later) — see design §9.2 |
| SSE fidelity (biotite vs deposited) | **Flagged** independently — design §9.1; side-check when docker is up |
| Implementation / first train | **Done** — Path B cold arms landed 2026-07-17/18 (`containment_baseline_chem_stage_a12_cold_v1`, `containment_pathb_stage_a12_cold_v1`) |
| Matched cold Makefile arms | **Registered** — `train-v66-containment-baseline` / `train-v66-containment-pathb` (Stage A-12 chem parent stack) |
| Checkpoint reload (`num_relations=9`) | **Tested** — `radial_mlps.7/8` sniff + metadata in `test_load_checkpoint_restores_containment_num_relations` |
| Stage-runner liveness wiring | **Wired** — `liveness_containment_*` MLflow metrics on `--containment-edge-mp` runs |
| Oversmoothing-at-root (Task 8 T1c parent) | **PASS** 2026-07-18 — `MP_LOSS_CONSISTENT_WITH_NULL` (parent drop −0.024 vs null ≈0.289; excess −0.313). Artifact: `checkpoints/v66/diagnostics/t1c_containment_parent_oversmooth/report.json` |
| Chem-MVP probe provenance (substrate) | **Closed** 2026-07-18 — re-ran `jacobian_flow_influence` on `chem_mvp_stage_a12_cold_v1` / 4OBE under post-containment code: asym **0.08549355** and CV **0.63016853** bit-identical to `pilot_4obe.json`; ρ(betweenness) **0.6909 → 0.6929** (bootstrap CI endpoints resample). Artifact: `pilot_4obe_reverify_post_containment.json` |
| Flow-influence primary (Task 9) | **FAIL on diameter claim** 2026-07-18 — see §Results; pass-group “tax” is **not** seed-stable |

---

## Results (2026-07-18 cold arms)

**Score sanity (not verdict):** seed1 baseline best ≈3.618; Path B best ≈3.640. Containment liveness alive (~118 contain_down edges).

**Task 8:** parents do **not** oversmooth beyond fan-in-matched null — gate clears; diameter asymmetry may be interpreted.

### Provenance (before trusting the fail)

Re-ran the flow probe against the **original chem-MVP** checkpoint with current committed code (after the untracked-substrate discovery). Anchors that justified this effort still hold: asym=0.0855, CV=0.6302, ρ≈+0.69. Containment baseline fail-group (~0.02–0.04) sitting next to the original chem-MVP Stage A-12 fail-group range is therefore not an informal eyeball — the probe substrate reproduces.

### Task 9 primary (trunk `encoder_h`, seed1 matched arms)

| Group | Seed1 result |
|-------|--------------|
| Fail (diam ≥14): clear ≥0.05 on ≥3/5 | **0/5** — only 1F88 rose (+0.0048 → 0.037); all still below floor |
| Pass (diam ≤9): no structure &lt; baseline×0.90 | Seed1: 1LYZ / 1TEN / 1UBQ below — **see seed2 check** |
| Hub-tracking Spearman(betweenness) | Small corpus-level dip — **see bootstrap** |

Artifacts (seed1 full Stage A-12):
- Baseline: `checkpoints/v66/diagnostics/learned_flow_influence/containment_baseline_stage_a12.json`
- Path B: `checkpoints/v66/diagnostics/learned_flow_influence/containment_pathb_stage_a12.json`

### Pass-group regression: seed2 check (not single-seed noise dismissal)

Matched cold arms retrained with `SEED=2` (`…_cold_seed2_v1`), then flow probe on the 7 pass-group PDBs only.

| Seed | Median relative Δ (pathb/baseline − 1) | # below baseline×0.90 | Structures |
|------|------------------------------------------|------------------------|------------|
| 1 | **−10.0%** (boot 95% CI [−17.2%, −6.4%], excludes 0) | 3/7 | 1UBQ, 1TEN, 1LYZ |
| 2 | **+22.5%** (boot 95% CI [+19.4%, +26.3%], excludes 0) | **0/7** | — |
| Both seeds below floor | — | **0/7** | — |

**Read:** the seed1 pass-group double-digit drops are **not** a stable cost of SSE parents. Seed2 reverses the sign. Do not frame Path B as “no diameter unlock + real tax on working structures.” Frame as **no diameter unlock**; pass-group movement is seed-dependent.

Artifacts: `passgroup_seed{1,2}_{baseline,pathb}.json` under `learned_flow_influence/`.

### Hub ρ dip: bootstrap, not eyeball

Per-structure pathb−baseline Δρ: every structure’s Δ sits inside a conservative combined probe-CI (each dip compatible with 0 alone).  
Across-structure **median Δρ = −0.0099**, bootstrap 95% CI over the 12 structures **[−0.0193, −0.0049]** — **excludes 0**.

**Read (sized, not dismissed or inflated):** this is **not** “no effect,” and **not** hub-tracking collapse. It is a **real, small, directionally-consistent** corpus-level dip (~0.01) — consistent with SSE parents adding mild pooling/averaging pressure that slightly dampens the symmetric hub signal without meaningfully disrupting it. Too small to decide the hub-tracking gate as a scientific veto; worth remembering if Path 2 / Path A later reuses a containment-style parent mechanism (may inherit a similar tiny ρ tax). Strict table language (“no structure below own baseline”) fails on seed1; that is bookkeeping, not “tracking died.”

### Ablation outcome (filed)

| Claim | Verdict |
|-------|---------|
| Diameter unlock (fail-group ≥3/5 clear 0.05) | **Fail** — useful negative: SSE-level parents as built do not unlock long-range directional flow |
| Pass-group regression as real mechanism cost | **Not supported** after seed2 (−10% → +22% median flip) — single-seed artifact |
| Hub tracking destroyed | **No** — but also **not a clean null**: real ~0.01 median dip (CI excludes 0); per-structure within probe noise |

**Filed as stated:** Path B is a **viable stopgap**; the **primary diameter claim fails** on reproduced numbers; the next bet (Path 2, Path A, or a different parent grain) gets **registered fresh** rather than reusing this exact SSE-parent mechanism or digging for a soft win in these numbers.
