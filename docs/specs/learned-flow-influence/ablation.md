# Learned flow influence — ablation / gates

**Date:** 2026-07-17 (standing conclusion locked 2026-07-18)  
**Purpose:** Pre-registered liveness + informativeness criteria for the Jacobian
flow-influence probe before any correlation is interpreted.

See [`design.md`](design.md) for method and numerical-safety audit.

---

## Standing probe defects (do not rediscover)

### Jacobian flow-influence under input z-norm — Workaround locked (2026-07-18)

**Full write-up:** [`JACOBIAN_ZNORM_DEFECT.md`](JACOBIAN_ZNORM_DEFECT.md)

On **z-norm-trained** trunks, Jacobian flow-centrality anti-correlates with
classical betweenness (~−0.3, 0/12) while **direct PC1** on the same embeddings
correlates (+0.46, 11/12). Survives `--score-mode pc1_sq`. Move 1 grad-site
sweep (`raw_x` / `post_zscore` / `post_node_emb`) on 4OBE: z-norm-on stays
≈−0.47…−0.50; hist z-norm-off stays ≈+0.69. Affine z-score alone does **not**
flip gradient sign (`tests/test_jacobian_znorm_gradient_repro.py`).

**Locked recipe:** Jacobian L2 influence **forbidden** on
`input_feature_zscore=True` (any `--grad-site`). Trusted causal substitute =
**forward knockout**; trusted geometry = **direct PC1**. Z-norm-off chem-MVP
triangulation below is **not** directly contaminated.

**Layer-tap smoke (4OBE):**
`experiments/diagnostics/jacobian_znorm_layer_grad_sign.py` — trunk DIV clean;
B-local raw_x cos ≈+0.05 is **wash-out not a flip**; pool shape under z-norm is
`attenuation_compressed_near_zero`. Grad-site artifacts:
`checkpoints/v66/diagnostics/learned_flow_influence/jacobian_znorm_grad_site_sweep/`.

---

## Standing conclusion (2026-07-18) — triangulated

**Claim (chem-MVP lineage trunk, Stage A / KRAS OOD as tested):**

> The trunk, as currently trained, does **not** carry the specific long-range
> **directional / causal** allosteric signal that classical physics methods
> (λ₂ conductance, ρ-based dehydron propagation) detect for KRAS G12D
> (hub migration GLY151 → ILE163; diameter-scale asymmetry on large graphs).

This is stronger than any single sub-result. Three independent negatives, same direction:

| # | Test | Method | Result |
|---|------|--------|--------|
| 1 | Diameter-stratified global asymmetry | Containment Path B vs matched chem baseline; Jacobian asymmetry | Fail — large graphs stay ~0.02–0.04; 0/5 clear floor 0.05 |
| 2 | Hub migration into 163 (4OBE→4DSO) | Jacobian in-centrality | Raw `in(163)` flat; normalized R “Pass” was **median drift** |
| 3 | Hub migration / causal out of 163 | Forward-pass input knockout (no gradients) | **Fail** — raw `out(163)` **drops** under G12D; rank worsens |

Knockout and Jacobian share **no** failure mode (forward causal vs backward sensitivity). Knockout raw decline rules out “tanh saturation deflated gradients” as the explanation for (2).

**Provenance note:** the Jacobian legs above used **z-norm-off** chem-MVP. They
remain the causal SSOT for that lineage. They do **not** license Jacobian use on
z-norm-on models (see standing probe defect).

**What still works (do not over-generalize the negative):** symmetric hub-tracking — trunk flow-centrality vs classical betweenness ρ≈0.69, holds on most Stage A-12 structures — is real **on z-norm-off chem-MVP**. The absence is **directionality / mutation-causal long-range influence**, not “the trunk is dead.”

**What this is not:** a soft reopen of containment Path B, doorway fishing, or more undirected centrality GT (degree / eigenvector / k-core / assortativity) — those re-probe the passing half or the graph, not the failed claim.

**Decision point (explicit):** further narrow probes that re-measure directionality with the same trunk (including edge-removal scored by directional tools) are unlikely to move the claim. Prefer **registering a real change** before more confirmation cycles:

| Open question (T1a-style) | If true, next bet looks like… |
|---------------------------|-------------------------------|
| Capacity / competition | **Partial (A Confirmed; B passive Partial; B causal knockout Partial):** [`../rho-tau-abs-dist-swap/ablation.md`](../rho-tau-abs-dist-swap/ablation.md) — knockout Spearman 0.380→0.457 (Δ+0.077 **below** +0.10 clear bar); holds 7→9 telemetry only. Soft agent “Pass” **withdrawn**. Jacobian still forbidden on z-norm-on ([`JACOBIAN_ZNORM_DEFECT.md`](JACOBIAN_ZNORM_DEFECT.md)) |
| No directionality objective | Path 2 diam≤9 agent pilot **ORPHANED** ([`PATH2_DIRECTIONALITY.md`](PATH2_DIRECTIONALITY.md)) — not authorized. Any directionality bet must be **user-registered** fresh; grade z-norm-on via knockout, not Jacobian |
| Depth / reach | Deeper MP or long-range architecture — **not** reusing failed SSE-parent Path B as-is — after a user-owned registration |

**Cleanup (2026-07-18):** Agent Move 3 Path 2 train orphaned; do not warm-start
`path2_dir_diam9_swap_warm_v1`. Resume from locked `|ρ−TAU|` / matched z-norm
trunk only.

Edge-removal remains useful **only** if scoped to the **symmetric** hub-tracking capability (does removing a specific disulfide / role edge change betweenness-aligned flow?) — not as a fourth directionality hunt.


---

## Sequencing (cheap pilot first)

1. **Pilot:** one structure, one late checkpoint.
   - Structure: `4OBE` (most-inspected this session; zero chem edges → clean
     physics-only reading on the chem-MVP feeler stack).
   - Checkpoint: `checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt`
   - Optional control: near-ep0 / reconstructed ep0 for known-degenerate
     reference (guard trip-rate baseline).
2. **Liveness gate** (below) must clear before any Stage A-12 scale-up.
3. If alive: all 12 Stage A structures, **per-structure only** (no pooling —
   1IVO-style dominance risk applies to gradient magnitudes too).
4. Correlate flow-centrality vs classical betweenness / current-flow / Fiedler /
   ANM per structure with bootstrap CIs (small-N skepticism).

---

## Pre-registered liveness gate (check before anything else)

| Check | Floor / rule | Fail behavior |
|-------|--------------|---------------|
| Flow-centrality CV | `std/mean ≥ 0.02` on total centrality (`FLOW_CV_FLOOR`, same shape as `NU_CV_FLOOR`) | Probe has nothing to correlate — stop |
| Asymmetry index | mean ≥ `0.05` (`ASYMMETRY_FLOOR`) for "directional" claim | Near-zero is **informative** (symmetric smoothing), not a crash |
| NaN/Inf gradients | exclude pair/column; count in report | Never zero-fill |
| Near-origin nodes | `‖layer[i]‖ < EPS` flagged | Unmeasurable — do not read as "not a hub" |
| Excluded columns | must be 0 for `alive=true` | Partial columns ⇒ not alive |

`alive` requires: non-degenerate CV **and** zero excluded columns **and** zero
near-origin nodes. Asymmetry is reported separately
(`symmetric_but_nondegenerate` when CV clears but asymmetry does not).

---

## Pre-registered outcomes (pilot / per-structure class)

| Outcome | Criteria | Next move |
|---------|----------|-----------|
| **Real signal** | asymmetry clears floor; flow-centrality non-degenerate; correlates (even weakly, consistently across structures) with ≥1 classical metric | Pursue Option B further (mutation Δ-flow, PPI) |
| **Symmetric-but-nondegenerate** | CV clears; asymmetry near zero | Informative: distance-based propagation without directionality |
| **Flat** | CV below floor | Capacity-ceiling / oversmoothing — next move is representation capacity (`|ρ−TAU|` swap or widening), **not** Option B productization |

Score / rim enrichment are **not** decision inputs for this probe.

---

## Ep1 attribution lock (4OBE — pinned before Stage A-12)

| Checkpoint | Trunk CV | Asymmetry | Spearman vs betweenness (95% CI) | vs ANM MSF |
|------------|---------:|----------:|---------------------------------:|-----------:|
| `epoch_001.pt` | 0.639 | 0.115 | **+0.689** [0.60, 0.76] | −0.697 |
| `v66_best.pt` (≈ep20) | 0.630 | 0.085 | **+0.691** [0.60, 0.76] | −0.702 |
| Δ (best − ep1) | −0.009 | −0.030 | **+0.002** | −0.005 |

Ep1 already tracks classical hubs at the same magnitude as the late checkpoint
(CIs fully overlap). The 4OBE pilot signal is therefore **role-edge graph
scaffolding** (packing / dehydron / spoke / ribbon topology present from init),
**not** "training discovered hub structure." That is a genuine positive for
Option B — the structured graph is doing real communicative work — but it is the
**more modest claim**. Do not market Stage A-12 as learned-hub discovery unless
late-epoch correlations systematically beat ep1 (see attribution row below).

---

## Stage A-12 per-structure floors (LOCKED before scale-up — 2026-07-17)

Same discipline as chem-MVP's ≤40% dominance guard and "win needs all 3
individually." **No pooled / median win.** Decision layer = trunk only.

### Per-structure "holds" predicate (trunk)

A structure **holds** iff all of:

1. Liveness: `alive` and `asymmetry_non_zero` on `encoder_h`
2. Spearman(trunk flow-total, classical betweenness) **≥ 0.30**
3. Bootstrap 95% CI lower bound on that Spearman **> 0** (same-sign, nontrivial)

Secondary consistency (reported, not required for hold): Spearman vs ANM MSF
**≤ −0.30** with CI upper bound **< 0**.

### Corpus outcomes (count of holds among 12 enabled Stage A structures)

| Outcome | Criteria | Claim framing |
|---------|----------|---------------|
| **Win (graph-scaffolded flow)** | **≥ 10 / 12** structures hold | Structured role-edge propagation recovers classical hubs consistently; Option B pursued further under the modest attribution |
| **Partial** | **6 – 9 / 12** hold | Real on a subset; do not promote; inspect failures individually (same skepticism as 1F88) |
| **Fail** | **≤ 5 / 12** hold, or trunk liveness fails on ≥ 6 / 12 | Flat / non-relational at corpus scale |

### Attribution row (does training add anything?)

On each structure where both ep1 (or earliest snapshot) and best are scored:

- Δρ = ρ_best − ρ_ep1 for trunk↔betweenness
- **Training-additive** only if median Δρ ≥ **+0.05** and ≥ 6 / 12 structures have Δρ > 0 with non-overlapping CIs favoring best
- Otherwise the Stage A-12 claim stays **graph-scaffolded, training-neutral** (still a Win under the table above if ≥10/12 hold)

Default Stage A-12 run scores **best only**; ep1 attribution is already locked on
4OBE. Optional multi-structure ep1 sweep is a follow-up, not a blocker for the
best-checkpoint scale-up.

Dominance: refuse any narrative that quotes a single structure's ρ as the corpus
result. Report all 12.

---

## Status

| Item | Status |
|------|--------|
| Option B pivot (flow over disc reshape) | **Adopted** 2026-07-17 |
| Hyperbolic ops epsilon audit | **Done** — geoopt already guards exp/log/Möbius/atanh; probe adds float64 + NaN policy + origin flag |
| Classical metrics extract | **Implemented** (`classical_network_metrics.py`) |
| Jacobian probe instrument | **Implemented** (`jacobian_flow_influence.py`) |
| Unit tests | **Pass** (`tests/test_jacobian_flow_influence.py`, 7/7) |
| Pilot 4OBE (chem-MVP best) | **COMPLETE** 2026-07-17 — trunk **Real signal**; disc **symmetric-but-nondegenerate** |
| Pilot reverify on committed code | **PASS** 2026-07-17 — CV / asym / ρ(betweenness) / ρ(ANM) **exact** vs `pilot_4obe.json`; CI endpoints bootstrap-only drift (`pilot_4obe_reverify_committed.json`) |
| Near-ep1 control | **COMPLETE** — trunk↔betweenness already **+0.689** at ep1 (≈ identical to best); disc flat |
| Attribution | **LOCKED** — 4OBE signal is graph-scaffolded, training-neutral (Δρ≈0) |
| Stage A-12 floors | **LOCKED** before scale-up — ≥10/12 hold for win; see table above |
| Stage A-12 scale-up | **COMPLETE** — **`partial` (6 / 12 holds)** |

### Pilot 4OBE numbers (chem-MVP `v66_best.pt`)

| Layer | CV (total) | Asymmetry mean | Liveness class | Spearman vs betweenness (95% CI) | vs current-flow | vs ANM MSF |
|-------|-----------:|---------------:|----------------|----------------------------------:|----------------:|-----------:|
| `encoder_h` (trunk) | **0.630** | **0.085** | **Real signal** | **+0.691** [0.60, 0.76] | +0.695 | **−0.702** |
| `hyp_projections_2d` | 0.036 | 0.0009 | symmetric-but-nondegenerate | −0.561 [−0.65, −0.44] | −0.533 | +0.549 |

Artifacts:
- `checkpoints/v66/diagnostics/learned_flow_influence/pilot_4obe.json`
- `checkpoints/v66/diagnostics/learned_flow_influence/pilot_4obe_ep001_control.json`

### Committed-code re-verification (2026-07-17, before containment cold)

**Why:** `jacobian_flow_influence.py` (and the Chem-MVP v66 stack it sat on) ran from
untracked working-tree files when the pilot was produced; committing them later is
not the same claim as “bit-identical to what generated ρ≈+0.69.”

**Method:** re-ran committed probe on the same chem-MVP `v66_best.pt` + same corpus
cache (`graphs_38a6993d7a439aa4.pt`), `--pdb-id 4OBE`, CPU float64.

| Quantity | Original pilot | Reverify (committed) | Match |
|----------|---------------:|---------------------:|:-----:|
| Trunk CV (total) | 0.630168530074803 | 0.630168530074803 | **exact** |
| Trunk asymmetry mean | 0.0854935523597222 | 0.0854935523597222 | **exact** |
| Spearman vs betweenness | 0.690937795236438 | 0.690937795236438 | **exact** |
| Spearman vs ANM MSF | −0.702235095221521 | −0.702235095221521 | **exact** |
| Disc Spearman vs betweenness | −0.5608845905225995 | −0.5608845905225995 | **exact** |
| Betweenness 95% CI | [0.604, 0.764] | [0.612, 0.763] | bootstrap resample only |

Artifact: `checkpoints/v66/diagnostics/learned_flow_influence/pilot_4obe_reverify_committed.json`.

**Verdict:** point estimates are bit-identical under committed code → no detectable
drift between “when the pilot ran” and “what is now in git.” Bootstrap CI endpoints
differ (expected; resampling). Containment cold arms are **unblocked** on this gate.

**Reading:** trunk recovers classical hubs with external validation and a
physically sensible ANM anti-correlation — but ep1 matches best, so the claim is
**role-edge scaffolding**, not training-discovered hubs. Disc sign-flips the same
ranking → disc rule escalated in
[`DISC_PROJECTION_NOT_TRUNK_PROXY`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md).

### Stage A-12 result (best checkpoint — floors applied as locked)

| Outcome | `partial` — **6 / 12** holds (need ≥10 for win) |
|---------|--------------------------------------------------|
| Artifact | `checkpoints/v66/diagnostics/learned_flow_influence/stage_a12.json` |

| PDB | Hold? | Fail reason | ρ betweenness | asym | ANM ρ |
|-----|:-----:|-------------|--------------:|-----:|------:|
| 1MBN | yes | — | +0.705 | 0.064 | −0.615 |
| 1LYZ | yes | — | +0.521 | 0.093 | −0.422 |
| 1BG1 | no | asymmetry 0.023 < 0.05 | +0.443 | 0.023 | −0.652 |
| 1F88 | no | asymmetry 0.032 < 0.05 | +0.450 | 0.032 | −0.329 |
| 2Z6H | no | asymmetry 0.024 < 0.05 | +0.494 | 0.024 | −0.602 |
| 1HHP | yes | — | +0.633 | 0.138 | −0.644 |
| 1TEN | no | Spearman weak / CI includes 0 | +0.156 | 0.148 | −0.345 |
| 1UBQ | yes | — | +0.621 | 0.145 | −0.529 |
| 1TIM | yes | — | +0.607 | 0.058 | −0.669 |
| 4OBE | yes | — | +0.691 | 0.085 | −0.702 |
| 1IVO | no | asymmetry 0.029 < 0.05 | +0.494 | 0.029 | −0.543 |
| 2SHP | no | asymmetry 0.029 < 0.05 | +0.468 | 0.029 | −0.594 |

**Do not move the floors post-hoc.** Honest split under the locked predicate:

- **Hub tracking** (Spearman ≥ 0.30, CI_lo > 0): **11 / 12** (only 1TEN fails).
- **Directionality** (asymmetry ≥ 0.05): **7 / 12**; five structures are
  *symmetric-but-correlating* — they recover classical hubs with near-symmetric
  influence.
- **Intersection (holds):** **6 / 12** → `partial`.

Claim framing stays **graph-scaffolded, training-neutral** (ep1 lock on 4OBE).
Option B is real as a hub-recovery signal on most of the corpus, but not a
directional-flow win at the pre-registered bar. No coefficient tweak / seed fish.

### Asymmetry-failure autopsy (cheap check — 2026-07-17)

Clarification: **5** structures fail the asymmetry floor (not 6). The sixth non-hold
(`1TEN`) clears asymmetry (0.148) and fails Spearman only.

| Group | PDBs | n residues |
|-------|------|------------|
| Asym fail | 1BG1, 2Z6H, 1IVO, 2SHP, 1F88 | **338–559** (median **511**) |
| Asym pass | 1UBQ, 1TEN, 1HHP, 1LYZ, 1MBN, 4OBE, 1TIM | **76–247** (median **129**) |

**Perfect size separation** — largest passer `1TIM` (247, asym 0.058 just clears);
smallest failer `1F88` (338). Spearman(n, asymmetry) = **−0.986** (p ≈ 4×10⁻⁹).

What they do **not** share as a discriminating factor:

| Candidate | Fail vs pass | Discriminates? |
|-----------|--------------|----------------|
| Active role types | 4 vs 4 | no |
| Mean role-graph degree | ~16.2 vs ~15.3 | no |
| Degree CV | ~0.23 vs ~0.25 | no |
| Role mix (pack/dehy/spoke/ribbon) | similar fractions; coupling=0 both | no |
| Chem bond count | mixed (0–52) vs mostly 0 | no — chem-rich 1IVO fails; chem-poor passers exist |
| Fold family | STAT3 / armadillo / EGFR / SHP2 / rhodopsin — unrelated | no single fold story |

**Reading:** directionality collapses with **structure size / graph scale**, not with
missing role vocabulary, chem coverage, or degree heterogeneity. Hub-tracking
survives on the same large structures (ρ ≈ +0.44–0.49), so size is washing out
*asymmetric* influence specifically — consistent with longer-range averaging /
oversmoothing diluting A→B vs B→A, while scalar hubness (near-symmetric
propagation into high-centrality nodes) remains intact.

That narrows Option B’s next question: not “does the trunk know structure?”
(yes, 11/12) but **whether directional asymmetry is suppressed as N grows, or
never rewarded** — size-stratified, not a blanket decorrelation loss yet.

### Diameter vs n (cheap check — 2026-07-17)

Role-edge graph diameter / mean shortest-path, vs asymmetry on the same Stage A-12
best-checkpoint trunk readings. Model MP depth = **6** layers.

| PDB | n | diameter | mean SPL | asym | asym OK? |
|-----|--:|---------:|---------:|-----:|:--------:|
| 1UBQ | 76 | 5 | 2.72 | 0.145 | yes |
| 1TEN | 89 | 6 | 2.84 | 0.148 | yes |
| 1HHP | 99 | 7 | 3.02 | 0.138 | yes |
| 1LYZ | 129 | 7 | 3.50 | 0.093 | yes |
| 1MBN | 153 | 9 | 4.04 | 0.064 | yes |
| 4OBE | 169 | 7 | 3.50 | 0.086 | yes |
| 1TIM | 247 | 8 | 4.02 | 0.058 | yes |
| 1F88 | 338 | **14** | 5.32 | 0.032 | no |
| 2SHP | 491 | **14** | 6.00 | 0.029 | no |
| 1IVO | 511 | **16** | 6.60 | 0.029 | no |
| 2Z6H | 533 | **25** | 8.58 | 0.024 | no |
| 1BG1 | 559 | **23** | 7.62 | 0.023 | no |

| Correlation with asymmetry | Spearman | p |
|----------------------------|---------:|--:|
| n residues | −0.986 | 4×10⁻⁹ |
| diameter | −0.970 | 2×10⁻⁷ |
| mean shortest-path | −0.979 | 3×10⁻⁸ |

n and diameter are nearly collinear on this corpus (cannot fully partial out with
N=12). Both separate the same way: pass diameter **5–9** (≤ ~1.5× MP depth);
fail diameter **14–25** (**≫ 6 hops**). That is the receptive-field-depth story
in concrete numbers — a fixed 6-layer stack covers small graphs and cannot span
asymmetric paths across large ones — not merely a residual capacity confound
independent of geometry. Artifact:
`checkpoints/v66/diagnostics/learned_flow_influence/asymmetry_vs_diameter.json`.

### Ep1 vs best asymmetry on large structures (cheap check — 2026-07-17)

| PDB | diam | ep1 asym | best asym | Δ | tag |
|-----|-----:|---------:|----------:|--:|-----|
| 1BG1 | 23 | 0.033 | 0.023 | −0.010 | **never_rewarded** |
| 2Z6H | 25 | 0.032 | 0.024 | −0.008 | **never_rewarded** |
| 1IVO | 16 | 0.041 | 0.029 | −0.012 | **never_rewarded** |
| 2SHP | 14 | 0.043 | 0.029 | −0.014 | **never_rewarded** |
| 1F88 | 14 | 0.043 | 0.032 | −0.010 | **never_rewarded** |
| 1TIM (boundary) | 8 | 0.083 | 0.058 | −0.025 | mild decay, still above floor |
| 4OBE (small) | 7 | 0.115 | 0.086 | −0.030 | mild decay, still above floor |

All five asymmetry failures were **already near-floor at ep1** and stayed there.
Hub-tracking ρ was already at final magnitude at ep1 too. That is **never
rewarded**, not actively suppressed — training is not erasing a large-graph
directionality signal that existed at init; the init role-edge stack never
produced one under a 6-hop receptive field.

Small/boundary graphs show mild asymmetry decay through training but remain
above floor — a secondary note, not the large-graph failure mode.

Artifact:
`checkpoints/v66/diagnostics/learned_flow_influence/ep1_vs_best_asymmetry.json`.

### Option B implication (before any new loss)

| Hypothesis | Supported? |
|------------|:----------:|
| Large graphs inherently oversmooth via neighbor pooling alone | Weak — diameter/SPL track as tightly as n; hop budget fits better |
| Fixed MP depth cannot span large diameter (receptive field) | **Strong** — fail diam ≫ 6; pass diam ≤ 9 |
| Training actively suppresses directionality on large graphs | **No** — never present at ep1 |
| Directionality never rewarded (no loss pushing asymmetry) | **Yes** on large graphs; mild erosion even on small ones |

Next Option B moves, if any — pick among these, do not invent a blanket
size-conditional decorrelation yet:

1. **Receptive-field / depth:** more MP layers, or long-range shortcuts
   (hierarchical containment parent nodes as hyperbolic shortcuts become
   directly relevant here).
2. **Explicit directionality reward:** only if we want asymmetry above the
   graph-scaffolded baseline on graphs the hop budget can already cover.
3. **Not yet:** fighting a suppression mechanism that the ep1 check says is
   not the large-structure story.

### Promotion (2026-07-17)

Containment design v2 is now **next buildable**, with diameter-stratified
asymmetry as the primary acceptance criterion reused from this probe:

- [`../hierarchical-containment-edges/design.md`](../hierarchical-containment-edges/design.md)
- [`../hierarchical-containment-edges/ablation.md`](../hierarchical-containment-edges/ablation.md)

Path 2 (explicit directionality reward on diam ≤9) was **agent-orphaned**
(2026-07-18 cleanup — [`PATH2_DIRECTIONALITY.md`](PATH2_DIRECTIONALITY.md)).
Any directionality bet now requires a **fresh user registration**, sequenced
only after standing flow-influence / feeler methodology is respected — not as
an automatic follow-on from soft knockout telemetry.

---

## Pre-registration: KRAS G12D conductance hub migration (4OBE → 4DSO)

**Date locked:** 2026-07-18  
**Status:** Pre-registered and **scored** — Pass on R (narrow; see §Result)  
**Scope:** One sharp external-physics test of trunk flow-influence. No new training.

### External source (what we borrow)

Independent structural-physics analysis (ρ / wrapping changes + spectral
conductance) on KRAS WT vs G12D — **not** GNN-derived:

| Role | Residues (physics-native) |
|------|---------------------------|
| Mutation / source | 12 (GLY in 4OBE WT; ASP in 4DSO G12D) |
| Sink candidates (context only) | 17 (P-loop), 68 (Switch II), 85 (α3), 163 (C-terminal, ~24.9Å) |
| **Primary claim** | Conductance hub migrates **GLY151 (WT) → ILE163 (G12D)** |

**Explicitly excluded from this gate:**

- Any “state-selective doorway” table filtered by `epistemic > median`
- Checkpoint `robust_experts.pt` (different generation; evidential head not
  trusted in this lineage)
- Model uncertainty scores as target selectors

We borrow the **biology / physics targets**, not the document’s model outputs.

### Structures + identity lock

| PDB | Role | Manifest | Local identity check (CA / chain A) |
|-----|------|----------|--------------------------------------|
| 4OBE | WT KRAS (Stage A-12 enabled) | in corpus | 12=GLY, 151=GLY, 163=ILE |
| 4DSO | G12D GDP (`enabled: false`) | Stage A manifests; OOD from coupled-lock | 12=ASP, 151=GLY, 163=ILE |

**Refusal gate (reuse coupled-lock discipline):** before scoring, map deposited
resseq → graph row and assert residue identities for `{12, 151, 163}` match the
table above. Numbering/identity mismatch ⇒ refuse (do not remap silently).

Infrastructure already exists: `experiments/diagnostics/kras_coupled_lock_ood.py`
identity/alignment pattern; 4DSO load path via PDB cache / legacy graph loader.

### Checkpoint (trusted lineage only)

| Allowed | Forbidden |
|---------|-----------|
| `checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt` (**preferred** — physics-only reading, 4OBE pilot parent) | `robust_experts.pt` |
| `checkpoints/v66/runs/containment_baseline_chem_stage_a12_cold_v1/v66_best.pt` (matched chem stack, no containment) | Any evidential-doorway checkpoint |

Containment Path B checkpoint is **out of scope** for this first shot (primary
claim already failed; do not confound external validation with SSE parents).

### Probe + primary metric

Instrument: `experiments.diagnostics.jacobian_flow_influence`  
Layer: **trunk only** (`encoder_h`) — disc reported but not deciding
([`DISC_PROJECTION_NOT_TRUNK_PROXY`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)).

From the influence matrix \(I(A\to B)\):

- **In-centrality** \(\mathrm{in}(B) = \sum_A I(A\to B)\) = directional influence **into** B.

**Primary (falsifiable) prediction — locked before any 4DSO run:**

> Trunk in-centrality at residue **163** is higher in **4DSO** than in **4OBE**,
> after within-structure scale normalization:
>
> \[
> R = \frac{\mathrm{in}(163)}{\mathrm{median}_i\,\mathrm{in}(i)}
> \qquad
> R_{4\mathrm{DSO}} > R_{4\mathrm{OBE}}
> \]

**Secondary (reported, not required for pass):**

1. Hub-shift ratio \(H = \mathrm{in}(163)/\mathrm{in}(151)\): \(H_{4\mathrm{DSO}} > H_{4\mathrm{OBE}}\)
2. Rank of 163 among in-centralities improves in 4DSO vs 4OBE (lower rank number = more hub-like)
3. Disc-layer repeats of (1)–(2) — consistency only

### Outcomes

| Outcome | Criteria | Read |
|---------|----------|------|
| **Pass** | Primary \(R_{4\mathrm{DSO}} > R_{4\mathrm{OBE}}\) | Flow probe recovers the physics hub migration on a trusted checkpoint |
| **Fail** | Primary false | Probe does not see the 151→163 migration under this checkpoint/stack — do not expand to sink-set fishing |
| **Refuse** | Identity/alignment gate fails | Fix numbering; do not score |

No multi-pair doorway sweep until this single test is filed. Sink set
`{17,68,85,163}` is reserved for a **separately registered** follow-up if Pass.

### Non-goals

- Reinterpreting containment Path B diameter fail via this test
- Using epistemic/doorway residues as targets
- Training a new checkpoint for this comparison
- Pooling trunk + disc

### Run recipe (when executed)

```bash
# Preferred checkpoint; 4OBE from Stage A cache, 4DSO via explicit PDB load
GNN_INPUT_MODE=topology_three_vector docker compose run --rm -e GNN_INPUT_MODE \
  --user "$(id -u):$(id -g)" science python -m experiments.diagnostics.kras_hub_migration_ood \
  --checkpoint /app/checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt \
  --corpus-cache /tmp/dtie_pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt \
  --pdb-dir /tmp/dtie_pdb_cache \
  --output /app/checkpoints/v66/diagnostics/learned_flow_influence/kras_hub_migration_4obe_4dso.json
```

Diagnostic: `experiments/diagnostics/kras_hub_migration_ood.py` — refuses missing 4DSO PDB;
logs **both** raw `in(163)` and `R` (pass criterion unchanged).

### Result (2026-07-18 — chem-MVP)

| Quantity | 4OBE | 4DSO | Δ |
|----------|-----:|-----:|--:|
| **R_163** (locked primary) | 1.0029 | 1.0337 | **+0.0308** |
| `in(163)` raw | 310.06 | 310.88 | +0.83 (~+0.3%) |
| `median(in)` | 309.16 | 300.74 | **−8.41 (~−2.7%)** |
| H = in(163)/in(151) | 1.0096 | 1.0192 | +0.0095 |
| Rank of 163 (1=strongest) | 84 | 85 | slightly worse |

**Outcome: Pass** on locked criterion \(R_{4\mathrm{DSO}} > R_{4\mathrm{OBE}}\).

**Attribution (why logging both mattered):** the Pass is **narrow** and driven mainly by a **lower `median(in)` on 4DSO**, not by a large absolute rise into 163. Raw `in(163)` is nearly flat (+0.3%). That is consistent with the physics note that mutant vs WT networks differ in overall conductance (λ₂ 0.241 vs 0.476) — the within-structure normalized score can clear while the absolute sink strength barely moves. Rank of 163 does **not** improve.

**Do not read as:** “probe recovered a strong 151→163 hub migration.”  
**Do read as:** locked R-test passes by baseline-shift; expand to sink-set fishing only with eyes open — or register a raw/absolute primary if that is the scientific claim you actually want.

Artifact: `checkpoints/v66/diagnostics/learned_flow_influence/kras_hub_migration_4obe_4dso.json`

---

## Pre-registration: Forward-pass knockout cross-check (4OBE / 4DSO / res 163)

**Date locked:** 2026-07-18  
**Status:** Pre-registered and **scored** — Fail (see §Result)  
**Why:** Jacobian and knockout are independent mechanisms. Two flats ⇒ stronger
“trunk lacks directional causal hubs.” Knockout-only signal ⇒ revisit Jacobian
saturation before trusting prior negatives.

### Method

For each residue \(i\): zero ``data.x[i]``, forward once, capture trunk
``encoder_h``. No gradients.

| Quantity | Definition |
|----------|------------|
| Out-effect \(o(i)\) | \(\mathrm{mean}_{j\neq i}\|h'(j)-h(j)\|\) after knocking \(i\) |
| \(R^{\mathrm{out}}_{163}\) | \(o(163)/\mathrm{median}_i\,o(i)\) |
| In-to-163 from 12 | \(\|h'(163)-h(163)\|\) after knocking residue 12 (secondary) |

Log **raw** \(o(163)\) and **median** \(o\) alongside \(R^{\mathrm{out}}\) (same
attribution discipline as the Jacobian R test).

### Primary (locked)

> \(R^{\mathrm{out}}_{163}(4\mathrm{DSO}) > R^{\mathrm{out}}_{163}(4\mathrm{OBE})\)

Same structures, identities, and chem-MVP checkpoint as the Jacobian hub-
migration test. Disc secondary only.

### Outcomes

| Outcome | Read |
|---------|------|
| **Fail / flat** (aligned with narrow Jacobian Pass attribution) | Triangulated: trunk does not carry a strong causal 163 hub under G12D |
| **Pass with large raw \(o(163)\) rise** | Jacobian may underestimate; re-open probe trust before more Option B bets |
| **Pass only via median shift** | Same story as Jacobian R — normalized Pass, not absolute hub recovery |

Diagnostic: `experiments/diagnostics/kras_knockout_causal.py`

### Result (2026-07-18 — chem-MVP)

| Quantity | 4OBE | 4DSO | Δ |
|----------|-----:|-----:|--:|
| **R_out_163** (locked primary) | 1.834 | 1.782 | **−0.052** |
| `out(163)` raw | 0.138 | 0.110 | **−20%** |
| `median_out` | 0.0752 | 0.0619 | −18% |
| Rank of 163 by out-effect (1=strongest) | 26 | 43 | worse |
| in(163)←knock(12) raw | 3.8e−6 | 6.4e−5 | tiny absolute |

**Outcome: Fail** on locked \(R^{\mathrm{out}}_{163}(4\mathrm{DSO}) > R^{\mathrm{out}}_{163}(4\mathrm{OBE})\).

**Triangulation vs Jacobian hub-migration:**

| Method | Locked R Pass? | Raw into/of 163 |
|--------|:--------------:|-----------------|
| Jacobian in-centrality | Yes (narrow) | raw `in(163)` flat; Pass via median drop |
| Forward knockout out-effect | **No** | raw `out(163)` **falls** on 4DSO |

**Read:** knockout does **not** show a causal 163 hub strengthening under G12D, and raw out-effect moves the wrong way. That supports “trunk lacks a recoverable directional/causal 163 hub under this stack” over “Jacobian gradients were merely saturating.” The earlier Jacobian R Pass remains a within-structure normalization artifact, not confirmed by an independent forward-pass method.

**Rolled into SSOT:** see §Standing conclusion at top of this file (triangulated claim).

Artifact: `checkpoints/v66/diagnostics/learned_flow_influence/kras_knockout_4obe_4dso.json`

**Deprioritized:** more undirected centrality GT (degree / eigenvector / k-core), assortativity, and any further directionality re-probe of the current trunk. **Edge-removal** only if scoped to symmetric hub-tracking (specific chem/role edge), not as another directional hunt. **Next decision:** register capacity / directionality-reward / reach change — do not spend another probe-cycle confirming the same absence.
