# GNNv7 success criteria — pre-retrain gates and Phase 4 uncertainty

**Status:** Routing track **LOCKED under decision (a)** (2026-07-16) — sensitivity + commitment verified; purity/Gram closed as not-solved-at-this-scale; uncertainty parked (symptoms patched).  
**Scope boundary:** See `docs/audit/GNNV6_DIAGNOSTIC_SYNTHESIS.md` — uncertainty
diagnostics sit **next to** routing. Passing P7 does not mean GNNv6 is healthier.

Related: `docs/audit/EVIDENTIAL_UNCERTAINTY.md`, `docs/audit/VIEWER_INVESTIGATION_CORRECTNESS.md`, `science/training/nig_identifiability.py`

---

## WHERE WE ARE — milestone checklist (reorient here)

**Goal (achieved in part):** MoE routing that **commits** to different experts on real structural differences — not one-expert collapse, not flat uncommitted shares. Diversification is the *prerequisite* for specialization. It is **not** “GNN finds new dehydrons / uncertainty beyond physics.”

**North star (still ahead, not this lock):** committed multi-expert routing that is *also* purity-clean within each structure, then — separately — uncertainty that adds information beyond ρ. The purity half is **closed as unsolved at this architecture/scale**, not left as the next lever.

### Locked verdict (decision **a**, 2026-07-16)

| Track | Status | Honest one-line |
| ----- | ------ | --------------- |
| **Sensitivity** | **Solved and verified** | T1a z-norm breaks input-scale domination; Fix-1 companions held under controlled width ablation; matches predicted ~1.7–1.8 trunk/disc ceiling. |
| **Commitment** | **Solved and verified** | Nearest-pair repulsion × elevated logit_scale fires on both seeds and both widths; magnitude is **seed-dependent** (~0.32–0.54 frac max-p≥0.60 in controlled ablation), not a fixed number — do not quote legacy ~0.75–0.86 as the standing claim. |
| **Purity / monopole / Gram** | **Closed — not solved at this scale** | Investigated with real rigor (composition reframe, prototype-proximity, repulsion, quotas, agnostic/core hinges, Gram-hinge, width control). Gram collapses to the **same mid-30s/low-40s** band on all four controlled runs regardless of seed or width. Not a coefficient away from resolution. |
| **Uncertainty (evidential heads)** | **Parked** | G5b live and corpus-confirmed (ale↔epi≈0.994 on 4-D seed1 Stage A-12). Deep-fix bar **not** met. Product symptoms patched: viewer Investigation → physics default; agent `get_source_leaks` → `physics_rim`. |

**Purity/Gram reopen condition** (same pattern as uncertainty — not investigative curiosity):

Reopen **only if**:
1. a **downstream deliverable** specifically requires clean multi-expert purity *within* a structure, **or**
2. a **materially different architectural idea** lands (more experts, different prototype-init family entirely, different bank geometry — **not** another loss-term / hinge / quota / width variant on the current 4-expert bank).

**Do not:** generate an eighth loss-term variant; treat conditioned-init as the next lead; cite accidental-4D seed2 12/12 as a clean baseline; leave purity as “still open.”

### Investigation log (historical — all rows closed)

| | Milestone | Status | Resolution (one line) |
| - | --------- | ------ | --------------------- |
| ☑ | Diagnose “collapse” vs invisible niche | **done** | route_v1 already had a thin dehydron-committed niche (~6%); mean H hid it |
| ☑ | Rule out false causes (corpus, balance coeffs, expert count, …) | **done** | Negative results; capacity/balance terms inert at IBU |
| ☑ | Fix trunk near-rank-1 (T1a z-score) | **done → sensitivity solved** | Input scale domination broken; effective rank → ~1.7–1.8 ceiling |
| ☑ | Show board has signal but commitment locked | **done** | Gate on topology board; `logit_scale` does not train up — must set/floor |
| ☑ | Scale alone raises max-p without hard niche growth | **done** | L2: axis held soft, Hᵢ\<0.5 stayed 0% (`AXIS_HELD_SOFT_NO_HARD_GROWTH`) |
| ☑ | Add orthogonal board cue (SASA) | **done** | Gate uses SASA; still no hard-commit growth — not the bottleneck |
| ☑ | Pin twin attractor = prototype proximity | **done** | Underwrap twins near-collinear; degree sensitivity dead until separated |
| ☑ | Prototype nearest-pair repulsion | **done → commitment path** | `PROTO_SEP_L1` — distance + re-measured sensitivity moved together |
| ☑ | Stack repulsion × elevated scale | **done → commitment solved** | `STACK_WIN_L1/L2`; reproduces under controlled 3-D/4-D × seed1/2 (magnitude seed-dependent) |
| ☑ | Catch local (within-structure) monopoles | **done** | Absolute share gate later shown composition-confounded |
| ☑ | Show minority is real dehydron/surface axis | **done** | Perfect dehydron partition on fails; sign-flip global vs local kNN explained |
| ☑ | Majority-share hinge (λ=0.5) | **done — failed** | `AXIS_SCRAMBLED_BY_DIVERSITY` |
| ☑ | Core-only majority hinge (λ=0.25) | **done — failed** | Commit+τ held; purity blurred |
| ☑ | Confirm gate grads ≠ trunk; monopole onset timing | **done** | Absolute share rise = late equilibrium, not early lock-in |
| ☑ | Core capacity quotas (STE) | **done — failed worse** | Unlearned STACK_WIN; park quota/hinge/STE-output family |
| ☑ | Re-score / relative purity floor | **done** | Relative polarity is standing measure; fixed maj_dh≤0.05 retired |
| ☑ | Monopole ↔ prototype proximity / bias / board-poor / persistence | **done** | Twin-style capture, bias-tip, persistence enrichment all killed or superseded |
| ☑ | Absolute monopole vs corpus composition | **done — resolved** | `share ≈ max(frac_core, frac_dh)` under clean bipartition — measurement artifact |
| ☑ | Full-bank Gram logdet hinge | **done — failed** | `AXIS_SCRAMBLED_BY_GRAM`; no λ rematch |
| ☑ | Accidental 4-D `node_emb` / stale cache | **done** | Foundation rewrite; solved-vs-open *numbers* from 4-D do not transfer |
| ☑ | Three-vector + controlled width×seed ablation | **done** | `NO_CONSISTENT_FOUR_D_WIN`; Gram mid-30s/40s **all four**; purity blur **all four**; early Gram seed gap width-independent |
| ☑ | Conditioned-init as dilution fix | **closed — not the lever** | Motived by wrong “seed2 clean / seed1 sick” foundation; under matched init, width barely moves Gram/purity |
| ☑ | Lock purity / `ROUTING_HEALTHY_ENOUGH` (#1/#2 bars) | **superseded by decision (a)** | Bars remain below as historical pre-reg; **do not** pursue as standing open work |
| ☑ | Viewer Investigation + agent source-leak consumers | **done — product patched** | Physics default; evidential experimental; see `VIEWER_INVESTIGATION_CORRECTNESS.md` |
| ☑ | Informative uncertainty beyond ρ | **parked** | Deep-fix bar not met; symptoms patched without fixing heads |

**You are here — plain state of play (2026-07-16, decision a locked):**

| | |
| --- | --- |
| **Banked wins** | Sensitivity (z-norm / Fix-1) and commitment (repulsion × elevated scale) — verified across controlled width×seed runs. |
| **Named limitation** | Purity/Gram on a 4-prototype bank at this corpus/training length — **not solved at this scale**. Six/seven distinct interventions failed or were retracted under control. |
| **Standing stack (keep)** | Fix-1 + S4 + T1a z-norm + SASA-on-gate-board + prototype nearest-pair repulsion + elevated logit_scale (~6.612). Gate-soft scoring. Relative purity = **monitor only**. |
| **Do not resume chasing** | Conditioned-init, Gram-hinge rematch, quota/STE family, eighth loss-term variant, accidental-4D purity numbers as baseline. |
| **Uncertainty** | Parked. Viewer + agent source-leak no longer rank on raw epistemic by default. |
| **Handoff** | GNN routing thread is in an honest closed state. Next work is outside this chase (parked barcode / S2F / frontend, or a reopen that meets the bar above). |

Controlled-ablation score artifact: `checkpoints/v66/diagnostics/three_vector_stack_battery_reverify/controlled_width_ablation_score.json`.

##### Early Gram / width control (closed into decision a)

Under matched gate/prototype init (`isolated_init`), seed1 vs seed2 early Gram tracks almost identically by width; late Gram collapses to mid-30s/low-40s on **all four** controlled runs. The early gap that motivated conditioned-init was small, width-independent, and never explained the purity story. Tagged **`NO_CONSISTENT_FOUR_D_WIN`** + **`GRAM_COLLAPSE_WIDTH_INDEPENDENT`**.

##### Phase 2 / cone / per-structure depth↔τ (keep for readers)

**Disc ceiling:** Phase-2 three-vector end-state matches T1a predicted ~1.7–1.8 (`PHASE2_EFF_RANK_MATCHES_T1A_CEILING`).

**`P_DEHYDRON_CONE_01` r≈1.000:** plumbing / loss-wiring pass (`CONE_GATE_SANITY_WIRING_OK`), not emergent discovery. On feeler-stack 4-D seed1 Stage A-12 viewers, per-structure `r(cone_depth,τ)` mean **0.56** (range 0.37–0.67) — objective partially satisfied with real per-structure slack. See `VIEWER_INVESTIGATION_CORRECTNESS.md`.

#### Historical pre-reg — “lock routing healthy enough” (#1/#2) — **SUPERSEDED 2026-07-16**

> **Superseded by decision (a).** The bars below were frozen 2026-07-15 assuming purity was still an open lever (conditioned-init → three-seed lock). That sequencing is **retired**. Kept verbatim for audit only — do **not** treat clearing `#1/#2` as standing work, and do **not** unlock “next science” by pursuing `LOCK_PURITY`.

**Why these bars existed:** residual was believed init/seed-trajectory sensitive; two-seed pass after fixing “sick” seed1 was easy to over-read.

**Stack under test (then):** Fix-1 + S4 + T1a z-norm + SASA + prototype nearest-pair repulsion + elevated logit_scale + conditioned-init.

##### What #1 was to clear first (conditioned init / dilution) — historical

Judge primarily at **ge30**, cold Stage A-12:

| Claim | Floor |
| ----- | ----- |
| **Seed1 relative purity** | ≥ **10/12** eligible (target **12/12**) |
| **Seed2 relative purity** | hold **12/12** |
| **Commit L2 (both seeds)** | frac max-p≥0.60 ≥ **15%** **and** rival soft ≤ **0.20** |
| **Axis (both seeds)** | best τ contrast ≥ **0.40** |
| **Early bank (seed1)** | Gram condition / eig_min inside accepted-init band |

##### What #2 was to require (lock) — historical

Three cold seeds; `LOCK_COMMIT` / `LOCK_PURITY` (each ≥10/12 AND ≥2 of 3 at 12/12) / `LOCK_AXIS` / `LOCK_SENS` / `LOCK_SPREAD`. Fail tags: `TWO_SEED_ONLY`, `LOCK_PURITY_FRAGILE`, `LOCK_COMMIT_REGRESS`, `INIT_RULE_OVERFIT`. Pass was to be `ROUTING_HEALTHY_ENOUGH`.

**Replacement pass verdict under decision (a):** `ROUTING_SENS_COMMIT_LOCKED` — sensitivity + commitment banked; purity/Gram named limitation with reopen bar above; uncertainty parked. No purity lock required to hand off this thread.

Detail and gates below; do not re-derive the chain from RESULT sections unless auditing a specific historical lever.

---

## HEADLINE — route_v1 dehydron niche (2026-07-14)

**The clearest signal this sprint produced:** `slim_moe_route_v1` — the checkpoint that anchored two months of “routing collapse” language — already specialized on a **real, corpus-wide dehydron axis** in a thin committed slice that **H(mean load) is structurally incapable of seeing**.

| Fact | Value |
| ---- | ----- |
| Committed tail | **204 / 3393 (6.0%)** with Hᵢ\<0.5 |
| Spread | **11/12** Stage-A proteins (not one-structure noise) |
| Axis (committed) | **e0:** τ=0.97, ρ_med=5, depth≈0.24 — underwrapped. **e2:** τ=0.00, ρ_med=29, depth≈0.49 — wrapped |
| Masking | **`NOT_CAPACITY_MASK_ARTIFACT`** (eval: dropout/ban off; capacity overload flipped 0/204) |
| Soft structure (full argmax) | Same e0/e2 axis, τ contrast ≈0.76, r(soft_e0, τ)≈0.61 |

Artifacts: `checkpoints/v66/diagnostics/route_v1_committed_tail/`, `checkpoints/v66/diagnostics/dehydron_axis_soft_structure/`.

**What this means:** “Collapsed” and “never differentiated” were **not the same failure**. Corpus-mean H averaged a real biophysical specialization into invisibility. Fix-1, T1a, and the scale ladder remain valid standalone findings — but the framing that organized *why* they mattered must update: the gate **can** specialize meaningfully; the open problem is **growing that niche**, not creating commitment from a blank slate.

### Soft-structure check — did L1→L2 keep or abandon the axis?

| Run | Soft axis label | Best τ contrast | Axis hard-mass | frac Hᵢ\<0.5 | mean max-p |
| --- | --------------- | --------------- | -------------- | ----------- | ---------- |
| route_v1 | STRONG | 0.76 (e0/e2) | 0.74 | **6.0%** | 0.58 |
| fix1_s4 | STRONG | 0.77 (e2/e1) | 0.95 | **0%** | 0.36 |
| L1 | STRONG | 0.81 (e2/e1) | 0.92 | **0%** | 0.37 |
| L2 | MODERATE\* | **1.00** (e3/e2) | 0.48 pair / **~1.0 twin-partition** | **0%** | 0.59 |

\*L2 does **not** erase the dehydron axis — it **twin-splits** it (underwrap e0+e3 τ≈1.0; wrap e1+e2 τ≈0). Soft axis held; hard commitment did not grow.

**Scale trajectory read: `AXIS_HELD_IN_SOFT_BUT_NO_HARD_COMMIT_GROWTH`.** More scale alone raised mean max-p into semi-commitment without recovering any Hᵢ\<0.5 mass. That is the wrong direction for expanding route_v1’s committed niche.

### L2 twin-split — duplicate dehydron, secondary exposure/SS (2026-07-14)

Cheap check: are e0≈e3 and e1≈e2 functionally identical, or is there a second dimension? Artifact: `checkpoints/v66/diagnostics/l2_twin_pair_differentiation/`.

| Within-twin | Underwrap e0 vs e3 | Wrap e1 vs e2 |
| ----------- | ------------------ | ------------- |
| ρ / depth | near-identical (\|d\| ≤ 0.26 / 0.12) | near-identical (\|d\| ≈ 0 / 0.04) |
| **SASA** | **\|d\| ≈ 1.57** (med 0.63 vs 0.44) | **\|d\| ≈ 1.29** (med 0.27 vs 0.45) |
| SS composition | L1≈0.24 | L1≈0.43 |
| Soft mass on twin | ~0.19 | ~0.22 |

**Verdict: `TWIN_SPLIT_SECONDARY_IS_EXPOSURE_SS_NOT_MORE_DEHYDRON`.** τ-contrast→1.00 is **redundant fragmentation of the one dehydron axis** plus an incidental **exposure/SS** correlate within each half — not evidence that dehydron has a second dimension. Topology-only gate board is `{clustering, cone_depth, degree, ρ, τ, ss}` — **SASA is not an explicit gate channel**, so the exposure split is indirect.

**Enrichment implication:** refining τ/ρ alone will not make twins earn distinct dehydron niches. To grow real multi-expert structure, the board needs a **genuinely orthogonal** signal (explicit exposure / local geometry), not more resolution on underwrap vs wrap.

### SASA vs SS orthogonality (required before enrichment reg — 2026-07-14)

Artifact: `checkpoints/v66/diagnostics/sasa_ss_redundancy_stage_a12/`.

| Check | Result |
| ----- | ------ |
| R²(SASA ~ SS dummies) | **0.036** |
| η² (ANOVA) | **0.036** |
| Twin SASA \|d\| raw | e0/e3 **1.57**, e1/e2 **1.29** |
| Twin SASA \|d\| after SS residual | e0/e3 **1.45**, e1/e2 **1.11** |
| Within each SS class | twin \|d\| still **1.18–1.83** |

**Verdict: `SASA_MOSTLY_ORTHOGONAL_TO_SS`.** The twin exposure split is **not** recoverable from the existing `ss` channel. Explicit SASA is a cleanly orthogonal board addition; full effect-size expectations apply (no “partial redundancy” tempering).

### Pre-registered gate — explicit-SASA board enrichment (frozen 2026-07-14, **before** train)

**Hypothesis:** topology-only gate already routes on an exposure correlate it cannot see; putting **normalized SASA on the gate board** (alongside existing `{clustering, cone_depth, degree, ρ, τ, ss}`) lets twins differentiate on an explicit channel and can grow hard commitment on the real dehydron × exposure structure.

**Config under test (single controlled run unless ladder pre-authorized below):** Fix-1 + S4 Stage A-12 cold; topology gate; **SASA concatenated into gate topo features** (declare norm: same train-set z-score / running stats discipline as ρ/degree). **No** `|ρ−TAU|` as the primary lever. **No** logit_scale L3. Declare z-norm on/off in run README. Baseline comparators: L2 ge20 soft structure + route_v1 committed tail.

Judge at **ge20** on locked Stage A-12.

**Standing metrics (always report):** H(mean load), median Hᵢ, frac Hᵢ\<0.5, mean max-p, per-expert τ/ρ/SASA/depth profiles, best dehydron-pair τ contrast, within-twin SASA \|d\|, soft r(expert_weight, SASA), committed-set protein spread.

#### Frozen quantitative floors (locked with this gate — do not soften post-hoc)

| Quantity | Floor / definition |
| -------- | ------------------ |
| **Commit growth** | frac Hᵢ\<0.5 **≥ 0.03** = **≥ 3 percentage points of the locked Stage A-12 corpus** (~102 / 3393 residues). Measured on the **same** Stage A-12 eval set as L2 / route_v1 retrospectives — not a different corpus slice. |
| **Distribution (both `GROWS_REAL_NICHE_*`)** | Committed residues must appear in **≥ 8 of 12** Stage-A proteins **and** no single protein may hold **\> 35%** of the committed set. (route_v1 reference: 11/12 proteins, mild enrichment; blocks one-/two-structure “exciting” noise.) Failure of this check → **`AMBIGUOUS`**, not success. |
| **Committed-set SASA separation (`TWIN_COMMIT`)** | On residues with Hᵢ\<0.5, within the twin (or successor exposure pair) under test: Cohen’s \|d\| on SASA **≥ 1.10**. Reference: L2 post-SS-residual twin \|d\| ≈ **1.11–1.45**. Do **not** accept a marginal \|d\| (e.g. 0.3–0.5) as “some separation.” If the twin collapses to one exposure-specialized expert, require that expert’s committed SASA median sit **≥ 0.15** away from the opposing dehydron half’s committed median (same units as Stage A-12 normalized SASA). |
| **Dehydron retention** | Underwrap vs wrap τ contrast on committed mass (or on hard argmax of the inheriting experts) **≥ 0.70**. |

#### Outcome buckets (SASA-specific companions to the headline labels)

| Verdict | Required |
| ------- | -------- |
| **`GROWS_REAL_NICHE_TWIN_COMMIT`** | Commit growth **and** distribution floors **and** dehydron retention **and** committed-set SASA \|d\| ≥ 1.10 (or single-expert exposure identity rule above). Twin / dehydron×exposure partition preserved — not abandoned. |
| **`GROWS_REAL_NICHE_CLEAN_EXPOSURE_SPLIT`** | Commit growth **and** distribution floors **and** a **cleaner** exposure-based hard split that **supersedes** the L2 twin map (stable 2×2 dehydron×exposure commitment, or one underwrap+high-SASA vs underwrap+low-SASA expert with τ contrast ≥ 0.70 on the underwrap side) **and** committed-set SASA \|d\| ≥ 1.10 between the exposure poles — not diffuse 4-way moderate entropy. Report the new expert map explicitly. |
| **`AXIS_HELD_SOFT_NO_HARD_GROWTH`** | Soft dehydron×SASA structure holds or strengthens (within-twin SASA \|d\| ≥ L2 baseline 1.2 **or** r(soft, SASA) improves) but frac Hᵢ\<0.5 stays **\< 0.01**. Same limited bucket as scale L2. |
| **`DIFFERENT_AXIS_OR_DIFFUSE`** | Commitment grows **or** mean max-p rises into semi-commitment, but dehydron τ contrast drops below 0.40 **or** exposure structure is scrambled relative to L2 — sharpening abandoned the proven axes. |
| **`SASA_REDUNDANT_NO_MOVE`** | Soft and hard metrics indistinguishable from L2 / fix1_s4 IBU band on the standing metrics (no twin cleanup, no commit growth). Explicit SASA added nothing the gate could use. |
| **`AMBIGUOUS`** | Anything else — including commit ≥ 0.03 that **fails** the ≥8/12 protein / ≤35% concentration check, or commit with SASA \|d\| in (0.5, 1.10), or near-miss commit 0.01–0.03. |

**Success claim:** only **`GROWS_REAL_NICHE_TWIN_COMMIT`** or **`GROWS_REAL_NICHE_CLEAN_EXPOSURE_SPLIT`**. Do **not** re-score `AXIS_HELD_SOFT_NO_HARD_GROWTH` or distribution-failed commit as a win.

**Escalation (frozen with this gate):** one run. If `SASA_REDUNDANT_NO_MOVE` or `AMBIGUOUS`, stop — do not add `|ρ−TAU|` or scale L3 without a **new** pre-registration. If `AXIS_HELD_SOFT_NO_HARD_GROWTH`, prefer gate-input policy (encoder_h → gate) or a second orthogonal channel over another SASA tweak. If either `GROWS_REAL_NICHE_*` hits, stop and claim — then plan corpus expand, not immediate loss stacking.

### RESULT — SASA board enrichment ge20 (2026-07-14, scored against frozen floors)

Run: `checkpoints/v66/runs/fix1_s4_gate_sasa_stage_a12_cold_v1/` (Fix-1+S4, T1a z-norm, `--gate-include-sasa`, Stage A-12 cold). Artifact: `checkpoints/v66/diagnostics/sasa_gate_enrichment_stage_a12/`.

| Standing | Value | Floor / comparator |
| -------- | ----- | ------------------ |
| H(mean load) | **1.382** | ≈ L2 / fix1_s4 |
| median Hᵢ | **1.343** | fix1_s4 1.30; L2 1.01 — still soft |
| frac Hᵢ\<0.5 | **0%** (0/3393) | need ≥ **3%** → **fail commit growth** |
| mean max-p | **0.322** | fix1_s4 0.36; L2 0.59 — IBU band, not semi-commit |
| Best soft τ contrast | **0.72** (e1 τ=0.95 / e2 τ=0.23) | dehydron axis **held** (≥0.70) |
| Underwrap twin e1 vs e3 SASA \|d\| | **1.19** | L2 baseline ≥1.2 — holds |
| soft r(w, SASA) | **±0.74–0.75** | explicit SASA **is used** (not `SASA_REDUNDANT_NO_MOVE`) |

**Verdict: `AXIS_HELD_SOFT_NO_HARD_GROWTH`.** Not a win. Soft dehydron × exposure structure is real and the gate clearly routes on the new SASA channel, but hard commitment did not grow — same limited bucket as scale L2. Do **not** re-score as `GROWS_REAL_NICHE_*`. Per frozen escalation: next lever is **gate-input policy (encoder_h → gate)** or a **new orthogonal channel**, not another SASA tweak / `|ρ−TAU|` / L3 without a fresh pre-reg.

### Flat-zero pattern — max-p distribution + decision-ceiling audit (2026-07-14)

Three consecutive registered runs (Fix-1+S4, logit scale L1/L2, SASA enrichment) all leave **frac Hᵢ\<0.5 = 0%**. Cheap checks before another input-side registration:

Artifacts: `checkpoints/v66/diagnostics/maxp_distribution_cross_runs/`, `checkpoints/v66/diagnostics/decision_ceiling_audit/`.

#### Check 1 — where does max soft-weight mass sit vs 0.5?

| Run | mean max-p | max max-p | frac max-p ≥ 0.5 | frac in [0.40, 0.50) | frac Hᵢ\<0.5 |
| --- | ---------- | --------- | ---------------- | -------------------- | ------------ |
| fix1_s4 | 0.36 | 0.38 | **0%** | **0%** | 0% |
| L1 | 0.37 | 0.39 | **0%** | **0%** | 0% |
| L2 | **0.59** | **0.73** | **81.5%** | 18.4% | **0%** |
| SASA | 0.32 | 0.35 | **0%** | **0%** | 0% |

**Read:** SASA / Fix-1 / L1 are **not** “nearly there” — mass sits well below 0.40, not piled in 0.40–0.49. L2 already put most residues **over** max-p 0.5 and still got **zero** Hᵢ\<0.5. So the flat-zero pattern is on the **Hᵢ\<0.5 bar**, not a universal hard stop at max-p=0.5.

**Theory (E=4 equal-tail):** H(max-p=0.50)≈1.24; H(0.70)≈0.94; H(0.80)≈0.72; Hᵢ\<0.5 needs **max-p ≳ 0.88**. L2’s ceiling (max 0.73) is still far below that regime.

#### Check 2 — softmax / loss construction ceiling?

| Candidate | Verdict |
| --------- | ------- |
| Label smoothing | **Not present** |
| Entropy term in loss | **Monitor-only** (`routing_entropy` not in sum) — previously ruled out |
| `balance_coeff` / `capacity_loss` | Starvation-only; **0 at IBU**; still inert for Fix-1/SASA |
| `routing_load_floor/ceiling` | **0.0** on feeler geom P12 |
| In-forward overload (`capacity_threshold=0.4`) | **Inert** even at L2 (mean max soft-load ≈0.31; 0 batches overload; Δmax-p raw→adj = 0) |
| Expert timeout @45% | Train-only; never engages while max share ≪0.45 |
| Grad clip `max_norm=1.0` | Present; soft training limit, not a softmax construction cap |
| Weight decay on main path | No AdamW WD on MoE train (AdamW WD only on Phase-4 uncertainty head) |

**No boring config smoking gun** that hard-caps max-p at 0.5.

#### What actually caps L2 peakedness: twin soft-share

At L2 ge20, when a residue hard-picks expert *e*, mean soft on *e* ≈ **0.58–0.60** and mean soft on the runner-up ≈ **0.24–0.28** (≥0.20 on ~62% of residues). Top-2 mass ≈ **0.85**. Runner-up is the **exposure twin** (e0↔e3, e1↔e2), not cross-axis.

**Verdict: `TWIN_SOFT_SHARE_CAPS_MAXP_BELOW_HI_FLOOR`.** Softmax can and does exceed 0.5; the attractor is **two near-equidistant prototypes on the same dehydron half**, which structurally keeps max-p in the ~0.55–0.73 band — enough for semi-commitment, never enough for Hᵢ\<0.5. At L2, the runner-up is the exposure twin on **63%** of residues (dominant pairs e1↔e2, e0↔e3). SASA enrichment strengthened soft r(w,SASA) without lifting peakedness (max max-p 0.35).

**Implication for next registration:** do **not** spend the next run on another pure input-richness lever hoping mass will creep past 0.5 — Fix-1/SASA never approached it, and L2 already cleared it without touching Hᵢ\<0.5. Prefer a lever that either (a) **breaks twin soft-share** (force one underwrap niche, not two), or (b) adds **commitment pressure past the twin band**, with encoder_h→gate only if pre-registered as a twin-equidistance breaker (not as “more orthogonal input”).

### Commitment metric calibration + within-twin residual (2026-07-14)

Artifact: `checkpoints/v66/diagnostics/commitment_metric_and_twin_residual/`.

#### 1. Is Hᵢ\<0.5 the right bar?

**Origin:** entered as a round-number companion during L1/L2 per-residue entropy histograms to separate “confident” from `SEMI_COMMITTED_MODERATE_ENTROPY`; then used to cut route_v1’s 6% tail. The SASA pre-reg floor (≥3%) locked to growing that niche. **Not** tied to a measured downstream MoE functional threshold.

**Equal-tail math (E=4):** max-p 0.60 → H≈1.11; 0.73 → H≈0.85; Hᵢ\<0.5 needs max-p ≳ **0.88**. So the metric demands near-one-hot routing that MoE practice rarely needs.

**Companion standing metrics (same Stage A-12):**

| Run | frac Hᵢ\<0.5 | frac Hᵢ\<1.0 | frac max-p≥0.60 | mean \|\|soft−uniform\|\| (hyp) |
| --- | ------------ | ------------ | --------------- | ------------------------------ |
| SASA | 0% | 0% | 0% | **0.10** |
| L2 | 0% | **47%** | **53%** | **0.36** (band 0.50–0.65: 0.35) |

corr(max-p, soft−uniform Δ) ≈ **0.95** on both runs. L2-style max-p≈0.6 already moves expert-mixture geometry ~3.5× farther from uniform than SASA/IBU soft — **functional specialization without clearing Hᵢ\<0.5**.

**Verdict: `HI_0_5_IS_ROUTE_V1_TAIL_CONVENTION`.** Keep it for historical niche-growth continuity if desired, but **do not** treat flat-zero Hᵢ\<0.5 alone as “no commitment.” Standing companions: **frac max-p≥0.60**, **median Hᵢ**, and (when cheap) soft−uniform mixture Δ.

#### 2. Twin residual on SASA-enriched ge20 — missing third channel?

Underwrap twins on SASA ckpt: **e1 vs e3**. Within-twin Cohen \|d\|:

| Feature | On board? | \|d\| |
| ------- | --------- | ---- |
| **degree** | yes | **2.02** |
| **SASA** / SASA⊥SS | yes | **1.19 / 1.10** |
| clustering | yes | 0.82 |
| depth / ρ | yes | ≤0.17 |
| b-factor | **no** | 0.34 |
| x_hyp norm | (trunk) | 0.17 |
| hydrophobicity | no | ~0 |

**Verdict: `NO_MISSING_THIRD_CHANNEL`.** Twins are already strongly separated by **board** features (degree + SASA); soft-share persists anyway. Twin equidistance is **not** “SASA wasn’t enough, invent a third physics channel” — it is a **decision/prototype attractor** with separable inputs. Do **not** register another orthogonal board channel on this evidence. encoder_h→gate remains optional only as a twin-breaking / readout-policy lever, not as “missing input.”

### Prototype proximity vs board-axis sensitivity (2026-07-14)

Before registering encoder_h→gate, pinned which “decision/prototype attractor” mechanism is live on SASA ge20 underwrap twins **e1/e3**. Artifact: `checkpoints/v66/diagnostics/prototype_attractor_vs_sensitivity/`.

#### Check 1 — are twin prototypes unusually close?

| Pair | hyp dist | tangent cos |
| ---- | -------- | ----------- |
| **e1–e3 (twin)** | **0.051** | **0.984** |
| e0–e2 | 0.158 | 0.920 |
| other pairs | 0.49–0.61 | ≈ −0.95 |
| non-twin mean | **0.473** | — |

Twin distance is **~11%** of non-twin mean and the **closest** of 6 pairs. At residues hard-assigned to the twin: mean \|d₁−d₃\| ≈ **0.010** (100% \< 0.05); mean soft on rival twin ≈ **0.30**.

**Verdict: `TWIN_PROTOTYPES_UNUSUALLY_CLOSE`.**

#### Check 2 — does the gate respond to degree/SASA/clustering?

Swap each twin residue’s board feature(s) to the **rival twin’s median** and re-forward the gate (x_hyp fixed):

| Perturbation | mean \|Δ logit gap\| | mean \|Δ soft self\| |
| ------------ | ------------------- | -------------------- |
| SASA → rival median | **6×10⁻⁵** | **0.0002** |
| degree → rival median | 0.016 | 0.024 |
| clustering → rival median | (≪ degree) | (≪ degree) |
| all three → rival median | 0.016 | 0.024 |

**Verdict: `MODERATE_SENSITIVITY_INSUFFICIENT_TO_BREAK_SOFT_SHARE`.** Degree moves the gate a little; **SASA almost does not** (despite being on the board and having large within-twin \|d\|). Soft-share cannot break while prototypes sit on top of each other.

#### Mechanism + next lever

**`PROTOTYPE_PROXIMITY_DOMINANT`** (with only partial axis sensitivity). Primary registration: **prototype repulsion / diversity regularization** — contained, training-side, does not reopen trunk-rank or encoder_h scope. encoder_h→gate is **not** justified while twin prototypes remain ~collinear; fix prototype geometry first.

### Prototype trajectory — init vs dynamics (2026-07-14)

Same SASA run, pairwise hyp dist over epochs. Artifact: `checkpoints/v66/diagnostics/prototype_trajectory_sasa_stage_a12/`.

| Epoch | twin e1–e3 | other-pair mean | twin / other | twin cos |
| ----- | ---------- | --------------- | ------------ | -------- |
| 1 | 0.035 | 0.039 | 0.90 | 0.22 |
| 5 | 0.045 | 0.172 | 0.26 | 0.92 |
| 10 | 0.048 | 0.329 | 0.15 | 0.97 |
| 20 | 0.051 | 0.473 | 0.11 | 0.98 |

**Verdict: `INIT_NEAR_ORIGIN_THEN_SELECTIVE_NONSEPARATION`.** `init_scale=1e-3` parks all prototypes near the origin (ep1: every pair ~0.03–0.04). Non-twin pairs then expand ~12×; the twin pair barely moves and **becomes more collinear** (cos 0.22→0.98). This is **not** “started healthy, training pulled them together” — it is “nothing ever pushed this pair apart while others did separate.”

**Registration shape implication:** prefer an **ongoing repulsion / diversity loss through training**, not init-only. Orthogonal/larger init may be a companion (L0), not the sole fix — other pairs already leave the origin without special init.

### Pre-registered gate — prototype nearest-pair repulsion (frozen 2026-07-14, **before** train)

**Hypothesis:** selective non-separation leaves underwrap twins (e1/e3 on SASA baseline) near-collinear while other prototype pairs expand; an **ongoing** nearest-pair repulsion term through training spreads that bottleneck enough for already-strong board axes (degree \|d\|≈2.0) to move routing decisions, without adding inputs or reopening encoder_h→gate.

**Config under test (single controlled run):** Fix-1 + S4 + T1a z-norm + `--gate-include-sasa` Stage A-12 cold (same lineage as SASA enrichment baseline). **Add** training-time prototype repulsion (see loss form). **No** encoder_h→gate. **No** new board channel. **No** logit_scale L3. Optional L0 companion: larger/orthogonal prototype init — declare on/off in run README; not a substitute for the repulsion term.

**Run structure (confirm intent):** **one training run**, one repulsion coefficient. **L1 / L2 / L3 are escalating evaluation floors checkpointed within that run** (e.g. ge5 / ge10 / ge20 or whenever nearest-pair clears the next bar) — **not** separate repulsion-strength retrainings. Do not start a second coeff ladder unless this run fails all floors or hits a fail condition.

**Loss form (declare exactly in run README):** soft hinge on the **current nearest** prototype pair in hyp distance each step — `relu(m − min_{i<j} d(p_i,p_j))` with **`m=0.25`** (L2 nearest-pair floor) and **`coeff=1.0`** (single declared strength; no anneal). Nearest-pair-only (not all-six equal push). L0 orthogonal/larger init: **off**. Make target: `train-v66-fix1-s4-proto-repulsion-stage-a12`.

**Standing metrics (always log — all six pairs every scored epoch):**

| Metric | Notes |
| ------ | ----- |
| All 6 pairwise hyp dists + tangent cos | Full matrix; identify current nearest pair (may rotate) |
| Historical twin (SASA baseline e1/e3) hyp dist + cos | Continuity with mechanism pin |
| e0–e2 hyp dist | Current second-tight pair on baseline |
| Nearest-pair id + dist | Whatever pair is tightest *now* |
| Degree→rival-median swap: mean \|Δ logit gap\|, mean \|Δ soft self\| | **Re-measure** on this checkpoint (same protocol as `prototype_attractor_vs_sensitivity`) — do **not** infer from distance alone |
| Mean soft on rival twin (hard∈{e1,e3} or successor twin map) | Soft-share |
| Best dehydron τ contrast; soft load H / median Hᵢ / frac max-p≥0.60 | Collateral guards |
| Soft−uniform mixture Δ (optional but preferred at L2) | Functional companion |

#### Frozen quantitative floors (one run; escalate within-run)

Baselines from SASA ge20: twin dist ≈0.051; degree-swap \|Δ logit gap\| ≈0.016; rival soft ≈0.30; non-twin mean dist ≈0.473; historical twin floor for collateral = **0.05**.

| Ladder | Cleared when (all must hold at the scored epoch) |
| ------ | ----------------------------------------------- |
| **L1 (modest)** | (a) Historical twin hyp dist ≥ **0.15** **or** current nearest-pair ≥ **0.15** if the twin is no longer nearest; (b) **re-measured** degree→rival-median mean \|Δ logit gap\| ≥ **0.05**; (c) mean soft on rival twin **\< 0.30** (strict improvement vs baseline); (d) **no** pairwise hyp dist \< **0.05** among the other five pairs (collateral: do not create a new near-collinear bottleneck while fixing the twin). |
| **L2 (functional)** | L1 **and** nearest-pair ≥ **0.25** **and** rival soft ≤ **0.20** **and** (soft−uniform mixture Δ up vs SASA baseline **or** frac max-p≥0.60 up vs SASA baseline 0%). |
| **L3 (stretch)** | L2 **and** historical-twin / non-twin-mean ratio ≥ **0.50**. **Not required** to claim success. |

Judge primarily at **ge20**; earlier epochs may claim L1/L2 if floors hold and fail conditions are clean — log the epoch.

#### Fail conditions (any → stop / no success claim)

| Fail | Definition |
| ---- | ---------- |
| **`REPULSION_NO_MOVE`** | ge20 nearest-pair still \< **0.10** (and historical twin \< 0.10). |
| **`COLLATERAL_NEW_COLLINEAR`** | While pushing the historical twin, **any other** pair’s hyp dist drops below **0.05**. |
| **`LOAD_COLLAPSE`** | Expert soft-load max share \> **0.45** sustained, or effective experts \< 3 on Stage A-12. |
| **`DEHYDRON_CONTRAST_LOST`** | Best underwrap/wrap τ contrast \< **0.40** (abandons proven axis). |
| **`SENSITIVITY_STILL_DEAD`** | Nearest-pair ≥ 0.15 but **re-measured** degree-swap \|Δ logit gap\| still \< **0.03** — distance moved, decisions did not (do not claim L1). |

#### Outcome buckets

| Verdict | Required |
| ------- | -------- |
| **`PROTO_SEP_L1`** | L1 floors + no fail |
| **`PROTO_SEP_L2`** | L2 floors + no fail |
| **`PROTO_SEP_L3`** | L3 floors + no fail (bonus) |
| **`REPULSION_NO_MOVE`** | Fail: distance never cleared 0.10 |
| **`COLLATERAL_DAMAGE`** | Fail: new collinear pair / load collapse / dehydron contrast lost |
| **`DISTANCE_WITHOUT_SENSITIVITY`** | Pair separated but re-measured degree-swap still dead — mechanism incomplete |
| **`AMBIGUOUS`** | Anything else |

**Success claim:** **`PROTO_SEP_L1`** or **`PROTO_SEP_L2`** only. L3 is stretch. Do **not** re-score distance-up-alone without the re-measured sensitivity check.

**Escalation (frozen):** one run, one coeff. If `REPULSION_NO_MOVE` or `COLLATERAL_DAMAGE`, stop — retune coeff only under a **new** pre-reg. If `DISTANCE_WITHOUT_SENSITIVITY`, prefer axis-aware gate readout / scale on twin axes before encoder_h→gate. If L1/L2 hits, stop and claim — then plan soft-share / niche growth under companion metrics (frac max-p≥0.60, mixture Δ); do not jump to encoder_h.

#### RESULT — prototype nearest-pair repulsion (ge20, 2026-07-14)

| | |
| --- | --- |
| **Run** | `fix1_s4_proto_repulsion_stage_a12_cold_v1` (Stage A-12 cold; Fix-1+S4+T1a+SASA + repulsion) |
| **Declared** | `coeff=1.0`, `m=0.25`, L0 init **off**; Make: `train-v66-fix1-s4-proto-repulsion-stage-a12` |
| **Artifact** | `checkpoints/v66/diagnostics/prototype_nearest_pair_repulsion/` |

| Metric (ge20) | SASA baseline | This run |
| ------------- | ------------- | -------- |
| Historical twin (e1/e3) hyp dist | ≈0.051 | **0.291** |
| Nearest-pair hyp dist | ≈0.051 (twin) | **0.270** ([0,3]) |
| Degree→rival-median \|Δ logit gap\| | ≈0.016 | **0.429** (re-measured) |
| Mean soft on rival twin | ≈0.30 | **0.228** |
| Other-pair min / mean | — / ≈0.473 | **0.270** / 0.284 |
| Twin / other-mean ratio | ≈0.11 | **1.02** (all six pairs now ~0.27–0.30) |
| frac max-p≥0.60 | 0% | **0%** |

**Ladder:** L1 cleared from **ge8**; holds at **ge20**. L2 dist yes (nearest≥0.25) but **rival soft 0.228 > 0.20** and frac max-p≥0.60 still 0% → L2 not claimed. No fail conditions (`REPULSION_NO_MOVE` / collateral / sensitivity-dead / load collapse).

**Verdict: `PROTO_SEP_L1`** (success claim). Mechanism pin held: distance and re-measured degree sensitivity moved together. Next (per frozen escalation): soft-share / niche growth under companion metrics — **not** encoder_h→gate, not a new coeff ladder unless a new pre-reg.

#### Follow-up shape check (post-RESULT, before next registration) — 2026-07-14

Artifact: `checkpoints/v66/diagnostics/prototype_nearest_pair_repulsion/max_p_and_sensitivity_shape.json`.

| Check | Finding |
| ----- | ------- |
| **Max-p histogram (repulsion ge20)** | **Flat bulk, no tail:** 61% in 0.25–0.30, 39% in 0.30–0.35, **n(≥0.40)=0**, max=**0.320**. Not a route_v1-style hidden committed fraction. SASA baseline was similarly soft (max 0.353); mean max-p did not rise under repulsion (0.294 vs 0.322). |
| **Distance → sensitivity** | **Convex / accelerating:** dist ×5.7, sens ×26.8 (gain_ratio ≈4.7); slope dS/dd ge1–8 ≈0.95 → ge8–20 ≈2.93; sens/twin 0.23→0.76→1.48. More distance would likely buy more sensitivity. |

**Implication for next registration:** do **not** treat a repulsion-coeff ladder as the primary path to L2’s `frac max-p≥0.60` / committed niche — the gap is **commitment**, not “not enough of the same separation.” Prefer a **structurally different soft-share / commitment lever** (new pre-reg). Optional separate registration: higher coeff only to chase rival soft ≤0.20, explicitly *not* sold as the commitment path.

#### Max-p ceiling vs learned `logit_scale` (cheap check, 2026-07-14)

Artifact: `checkpoints/v66/diagnostics/max_p_ceiling_logit_scale/report.json`.

**Hypothesis tested:** training actively prefers a low effective scale (softplus drifts down from init), which would explain the shared ~0.32–0.35 max-p ceiling across Fix-1 / L1 / SASA / repulsion.

| Run | softplus ge1 → ge20 | drift | mean dist_range | mean logit_range | max(max-p) |
| --- | ------------------- | ----- | --------------- | ---------------- | ---------- |
| Fix-1+S4 | 1.313 → 1.327 | **+0.014** | 0.67 | 0.90 | 0.378 |
| T1a z-norm | 1.313 → 1.323 | **+0.010** | 0.43 | 0.57 | 0.403 |
| Scale L1 (init≈2.65) | 2.645 → 2.627 | −0.018 | 0.39 | 1.02 | 0.388 |
| Scale L2 (init≈6.6) | 6.612 → 6.592 | −0.020 | 0.38 | 2.49 | **0.727** |
| SASA | 1.313 → 1.322 | **+0.010** | 0.43 | 0.57 | 0.353 |
| Proto repulsion | 1.313 → **1.329** | **+0.017** | **0.23** | **0.31** | **0.320** |

Default init = `logit_scale` param 1.0 → softplus ≈ **1.313**. `expert_bias` \|b\| ≤ 0.013 on all runs (not the cap). No evidence of a special clip on the distance→logit path beyond ordinary grad clip / AdamW wd=1e-5.

**Verdict: `SCALE_COLLAPSE_BY_TRAINING = false`.** Default-init runs never leave ~1.32 (tiny *upward* drift). L1/L2 keep their high inits; L2’s softplus≈6.6 survives and **breaks** the ceiling (max-p 0.73). The common ~0.35 band is **init-stuck scale × small residue-level dist_range**, not an active preference for flatness. Proto-repulsion is the sharp illustration: pairwise prototypes separated (sensitivity ↑) while mean residue→prototype dist_range *fell* (0.43→0.23) — more uniform distances to an evenly spread bank → flatter softmax despite a working decision boundary under axis swap.

**Next-registration implication:** do **not** register “why training prefers low scale.” Discriminate with a pre-reg that jointly tracks softplus + residue dist_range + max-p. Live options remain scale-hold (L2 already showed peakedness is achievable; Hᵢ\<0.5 calibration still applies), residue-level distance asymmetry / twin soft-share break, or a commitment lever that does not assume more pairwise prototype separation alone.

#### Dist-range shape confirmation (SASA vs repulsion ge20) — 2026-07-14

Artifact: `checkpoints/v66/diagnostics/dist_range_shape_sasa_vs_repulsion/report.json`.

| Metric | SASA ge20 | Repulsion ge20 |
| ------ | --------- | -------------- |
| mean residue→proto range (max−min) | 0.433 | **0.234** |
| mean CV across 4 distances | 0.080 | **0.041** |
| mean nearest / other3-mean | 0.887 | **0.926** |
| frac(nearest/other3 ≥ 0.95) | 4.6% | **30.8%** |
| mean gap(2nd − 1st) | 0.042 | **0.107** (widened) |
| frac(top2 gap \< 0.05) | 52% | **37%** |

**Refined read (`BANK_SYMMETRIZED_RANGE_COMPRESSED`):** repulsion makes the bank more symmetric (range↓, CV↓, nearest/other3 → 1) by pulling the *farthest* experts in (2.92→2.61), not by re-tying the top-2. Softmax flatness is **scale × compressed full-span**, not “nearest became average at cost of a new top-2 twin.” Plain finding: **logit_scale never trains upward from default init**; elevated scale (L2 ≈6.6) only appears when set by the ladder.

### Pre-registered gate — repulsion × elevated scale stack (frozen 2026-07-14, **before** train)

**Hypothesis:** sensitivity (repulsion) and commitment (elevated scale) are independently validated levers that hit the same ~0.32–0.40 max-p band through different pathways; stacking them on one Fix-1+S4+SASA Stage A-12 cold run produces both held prototype separation / degree sensitivity **and** genuine peakedness (frac max-p≥0.60), without requiring encoder_h→gate.

**Config under test (single controlled run):** same lineage as SASA / proto-repulsion baselines (Fix-1 + S4 + T1a z-norm + `--gate-include-sasa`). **Add both:**
- nearest-pair repulsion: `coeff=1.0`, `m=0.25` (PROTO_SEP validated)
- gate softplus init **and** floor at L2-validated value **≈6.612** (same as `train-v66-fix1-s4-scale-l2-stage-a12`; floor prevents silent drift below working band)

**No** encoder_h→gate. **No** new board channel. **No** second coeff ladder inside the run.

**Standing metrics (every scored epoch — joint log required):**

| Metric | Why |
| ------ | --- |
| All 6 pairwise proto hyp dists + nearest-pair id/dist | Detect twin re-collapse or collateral |
| Residue mean dist_range + mean CV across 4 | Interaction: does scale amplify a compressed bank? |
| softplus(logit_scale) + floor | Confirm scale holds ≥5 |
| frac max-p≥0.60, mean/median/max max-p, max-p hist bands | Commitment |
| Degree→rival-median \|Δ logit gap\| + rival soft | Sensitivity still live |
| Best dehydron τ / soft underwrap contrast | Axis must not collapse |

#### Frozen success floors (judge primarily at ge20)

| Claim | All must hold |
| ----- | ------------- |
| **`STACK_SENSITIVITY_HELD`** | nearest-pair ≥ **0.25** **and** re-measured degree-swap \|Δ logit gap\| ≥ **0.05** |
| **`STACK_SCALE_HELD`** | softplus(effective) ≥ **5.0** throughout scored epochs (floor is the enforcement; log confirms) |
| **`STACK_COMMIT_L1`** | frac max-p≥0.60 ≥ **5%**. **Anchor:** deliberately set just under route_v1’s **~6%** Stage A-12 committed niche (headline table) for comparability — “recover a real niche fraction with a known mechanism,” not an arbitrary round number. Metric is the standing companion **frac max-p≥0.60** (not Hᵢ\<0.5). A 5% result is a near-tie to that prior niche size; it is not a claim of exceeding route_v1. |
| **`STACK_COMMIT_L2`** | frac max-p≥0.60 ≥ **15%** **and** rival soft ≤ **0.20** |
| **`STACK_AXIS_HELD`** | best underwrap/wrap τ contrast ≥ **0.40** (same fail bar as PROTO_SEP) |

**Success claim:** **`STACK_SENSITIVITY_HELD` ∧ `STACK_SCALE_HELD` ∧ `STACK_COMMIT_L1` ∧ `STACK_AXIS_HELD`**. `STACK_COMMIT_L2` is stretch. Do **not** claim success from max-p alone if nearest-pair or degree-swap regresses.

#### Fail conditions

| Fail | Definition |
| ---- | ---------- |
| **`SCALE_LOST`** | softplus effective \< 5.0 at any scored epoch despite floor (misconfig) |
| **`REPULSION_REGRESSED`** | ge20 nearest-pair \< 0.15 **or** degree-swap \|Δ logit gap\| \< 0.03 |
| **`COMMIT_STILL_DEAD`** | ge20 frac max-p≥0.60 = 0% (stack failed to move commitment) |
| **`DIST_RANGE_KILLS_SCALE`** | **Concrete trigger (checked every scored epoch):** softplus(effective) ≥ **5.0** **and** mean residue→proto dist_range (max−min across 4 experts) ≤ **0.15** **and** frac max-p≥0.60 \< **2%**. Rationale: 0.15 is ~½ of repulsion-solo ge20 range (0.234) and well below L2-solo (0.38) — bank compression nullifies scale amplification. Log mean dist_range + mean CV every epoch; CV ≤ **0.025** with the same softplus/commit conditions is the same fail bucket (companion). |
| **`AXIS_COLLAPSE`** | τ contrast \< 0.40 |
| **`COLLATERAL_NEW_COLLINEAR`** | any non-nearest pair hyp dist \< 0.05 |

#### Outcome buckets

| Verdict | Required |
| ------- | -------- |
| **`STACK_WIN_L1`** | success claim above |
| **`STACK_WIN_L2`** | L1 + `STACK_COMMIT_L2` |
| **`SCALE_WINS_REPULSION_LOST`** | commit floors hit but sensitivity/nearest regress |
| **`SENS_WINS_COMMIT_DEAD`** | sensitivity held, commit still dead (interaction fail) |
| **`DIST_RANGE_NULLIFIES`** | `DIST_RANGE_KILLS_SCALE` |
| **`AMBIGUOUS`** | else |

**Escalation (frozen):** one stack run. If `STACK_WIN_*`, stop and plan niche growth under companion metrics. If `SENS_WINS_COMMIT_DEAD` or `DIST_RANGE_NULLIFIES`, do **not** raise repulsion coeff next — prefer a commitment lever that does not further symmetrize the bank (or a targeted dist_range floor). If `SCALE_WINS_REPULSION_LOST`, re-check floor/coeff interaction under a new pre-reg.

**Make target (when implementing):** `train-v66-fix1-s4-proto-repulsion-scale-l2-stage-a12` — not launched until this gate is acknowledged.

#### RESULT — repulsion × elevated scale stack (ge20, 2026-07-14)

| | |
| --- | --- |
| **Run** | `fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1` |
| **Artifact** | `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/` |

| Metric (ge20) | Floor | Result |
| ------------- | ----- | ------ |
| nearest-pair | ≥0.25 | **0.302** |
| degree \|Δ logit gap\| | ≥0.05 | **1.268** |
| softplus | ≥5.0 | **6.612** (held) |
| mean dist_range | fail if ≤0.15 w/ commit dead | **0.261** (cleared) |
| frac max-p≥0.60 | ≥5% (L1) / ≥15% (L2) | **14.8%** |
| rival soft | ≤0.20 for L2 | **0.118** |

Early epochs correctly flagged `DIST_RANGE_NULLIFIES` until range expanded; ge15 was `SENS_WINS_COMMIT_DEAD`; **ge20 = `STACK_WIN_L1`**. L2 stretch missed by 0.2 pp on frac max-p (14.8% vs 15%) despite rival soft already under 0.20. No fail conditions at ge20.

**Verdict: `STACK_WIN_L1`** (success claim). Stack hypothesis held: sensitivity + commitment together without encoder_h→gate.

#### Pre-lock scrutiny (2026-07-14) — trajectory, protein spread, +10 epochs

Before treating ge20 as locked, three checks (same thread):

**1. ge15→ge20 trajectory (in-train jsonl SSOT)** — continuous threshold-crossing ramp, not a last-epoch spike:

| ge | frac max-p≥0.60 | mean max-p | max max-p | verdict |
| -- | --------------- | ---------- | --------- | ------- |
| 15 | 0.00% | 0.462 | 0.598 | `SENS_WINS_COMMIT_DEAD` |
| 16 | 0.06% | 0.469 | 0.602 | `SENS_WINS_COMMIT_DEAD` |
| 17 | 1.03% | 0.476 | 0.614 | `SENS_WINS_COMMIT_DEAD` |
| 18 | 5.30% | 0.483 | 0.627 | `STACK_WIN_L1` |
| 19 | 10.22% | 0.491 | 0.637 | `STACK_WIN_L1` |
| 20 | 14.76% | 0.500 | 0.646 | `STACK_WIN_L1` |

Mean max-p climbs smoothly; frac≥0.60 is the mass finally crossing a fixed threshold (~+4–5 pp/epoch once past the edge).

**2. Protein spread of committed set (SASA-style discipline: ≥8/12 proteins, no protein >35%)** — **PASS** at ge20. Artifacts: `committed_protein_spread_ge20*.json`. Under thr=0.60 offline (hotter reload) and under a calibrated thr matching the logged **14.8%** mass: **12/12 proteins**, max share **~17%** (1IVO). Not a one-/two-protein artifact.

**3. Extend past ge20 (resume `epoch_020.pt`, +10 epochs → ge30)** — commitment **kept climbing**; no reverse by ge30. Softplus floor held at 6.612; dist_range **expanded** (0.261→0.508); rival soft fell further (0.118→0.055). First `STACK_WIN_L2` at **ge21** (25.2%); ge30 = **74.9%** frac max-p≥0.60.

| ge | frac max-p≥0.60 | dist_range | rival soft | τ contrast | verdict |
| -- | --------------- | ---------- | ---------- | ---------- | ------- |
| 20 | 14.8% | 0.261 | 0.118 | 0.966 | `STACK_WIN_L1` |
| 21 | 25.2% | 0.286 | 0.109 | 0.970 | `STACK_WIN_L2` |
| 25 | 53.4% | 0.382 | 0.077 | 0.923 | `STACK_WIN_L2` |
| 30 | 74.9% | 0.508 | 0.055 | 0.922 | `STACK_WIN_L2` |

Protein spread still **PASS** at ge25/ge30 (12/12; max share ≤17%). Note: ge21 shows a steeper step (+10.4 pp) right after resume — possible optimizer-state discontinuity; thereafter the climb continues monotonically with decelerating pp/epoch. τ contrast softens slightly (0.97→0.92) but stays far above the axis-fail floor (0.40).

**Honest scope:** still **one seed**. ge30 commitment is past “niche” into majority hard routing — L2 stretch cleared, then kept growing. Do **not** replace the collapse postmortem with a locked “the fix” narrative until a second seed; do treat the stack as the first confirmed break of the collapse pattern under the pre-registered gate.

**Extended verdict:** `STACK_WIN_L1` at ge20 confirmed stable under scrutiny; `STACK_WIN_L2` reached by ge21 and held through ge30 on this seed.

#### Pre-seed checks — monopole mirror + ge21 seam (2026-07-14)

**Soft dominant-expert share (in-train SSOT: jsonl + `metrics.json`)** — **not** the historical monopole’s twin:

| ge | soft_load | soft_load_max | eff. experts (soft) | infer max/min routing frac |
| -- | --------- | ------------- | ------------------- | -------------------------- |
| 20 | [0.240, 0.258, 0.261, 0.241] | 0.261 | 3.997 | 0.338 / 0.195 |
| 25 | [0.229, 0.257, 0.262, 0.252] | 0.262 | 3.995 | 0.369 / 0.162 |
| 30 | [0.204, 0.267, 0.252, 0.278] | 0.278 | 3.973 | 0.386 / 0.122 |

Historical monopole band was **56–72%** on one expert. Corpus soft max stays **~26–28%**; all four experts remain in the mid-20%s. `expert_tau_means` populated for experts 0–3 ⇒ all four receive hard argmax mass (not a single-expert hard wipeout). Exact **hard_share / committed_hard_share** fractions were **not** logged on seed-1; offline ckpt reload does not reproduce soft_load (soft L1 vs jsonl ~0.35–0.48 — `load_model` drops `input_feat_*` buffers; residual mismatch likely train-time vs working-tree code), so offline hard shares are **not** used for a verdict.

**Seed-2 instrumentation (required, 2026-07-14):** `measure_routing_companions` / `prototype_repulsion_per_epoch.jsonl` now logs, every scored epoch:

| Companion | Why |
| --------- | --- |
| `hard_share` / `hard_share_max` / `effective_experts_hard` | residue-weighted argmax — not soft-average |
| `committed_hard_share` / `committed_hard_share_max` / `effective_experts_committed` / `n_committed` | among max-p≥0.60 only — answers “did 74.9% distribute or concentrate?” |
| `n_proteins_with_committed` / `committed_protein_share_max` | SASA-style spread on the **committed** population (≥8/12, ≤35%) |
| `per_structure_soft_max` / `per_structure_soft_min` | same window that climbed to 38.6% on seed-1; catches emerging concentration before corpus soft moves |
| `per_structure_hard_max` / `per_structure_committed_hard_max` | per-protein hard / committed hard peak (floor: <0.56) |
| `per_structure[]` | pdb-level detail (soft/hard/committed shares) |
| `committed_distribution` | scored pass/fail vs frozen floors below |

Soft balance alone must **not** stand in for these. Seed 2 is testing multi-expert hard commitment under this pairing, not re-running soft-only ambiguity.

#### Seed-2 pre-registered distribution floors (frozen 2026-07-14, before launch)

Scored every epoch via `score_committed_distribution` → jsonl key `committed_distribution`. Soft corpus balance is **not** sufficient for pass.

| Gate | Metric | Floor (clear if) | Fail name |
| ---- | ------ | ---------------- | --------- |
| **Committed protein spread** | `n_proteins_with_committed` / `committed_protein_share_max` | ≥ **8/12** proteins with max-p≥0.60 mass **and** no protein > **35%** of the corpus committed set | `COMMITTED_PROTEIN_CONCENTRATED` |
| **Corpus committed-hard** | `committed_hard_share_max` | **< 0.56** (below historical monopole lower band) | `COMMITTED_HARD_MONOPOLE` |
| **Per-structure committed-hard** | `per_structure_committed_hard_max` | **< 0.56** — no single protein’s *committed* residues may themselves be a local monopole | `PER_STRUCTURE_COMMITTED_HARD_MONOPOLE` |

> **Superseded as a fail gate (2026-07-15):** under clean relative purity, absolute `per_structure_committed_hard_max < 0.56` is composition-confounded (`share ≈ max(frac_core, frac_dh)`). Keep as a **monitor**; standing concentration score is **excess over corpus majority fraction**. See § “Absolute monopole gate vs corpus composition”.

Pass verdict: `COMMITTED_DISTRIBUTION_PASS`. These gates apply to the **committed subset** (max-p≥0.60), not the full-corpus soft average. Also watch `per_structure_soft_max` (seed-1 climbed to 0.386 by ge30) — same window; not a hard fail floor yet, but must not be ignored if it approaches 0.56.

#### Seed-2 training protocol (frozen — deliberate, not convenience)

| | Seed-1 | **Seed-2** |
| --- | ------ | ---------- |
| Launch | cold → ge20, then **resume** +10 → ge30 | **cold continuous ge1→ge30** (one run) |
| Resume | used (`epoch_020` → ge30) | **forbidden** for the confirmatory run |
| Epochs | 20 + 10 | **30** |
| `RUN_ID` | `fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1` | `fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1` |
| `SEED` | unset / default | **`2`** |

**Why cold-to-30:** seed-1’s ge21 +10.4 pp step was a real resume discontinuity (~3.5× pre-seam rate). Seed-2 must test trajectory shape *without* that seam so a win is not confounded with optimizer re-init. If a future run resumes anyway, treat a one-epoch jump at the resume boundary as an **expected resume confound**, not evidence for or against the stack — do not score “smooth phase transition” across a resume seam.

**Make:** `make train-v66-fix1-s4-proto-repulsion-scale-l2-stage-a12-seed2` (EPOCHS=30, SEED=2, no RESUME).

#### RESULT — seed-2 cold continuous (ge30, 2026-07-14)

| | |
| --- | --- |
| **Run** | `fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1` (SEED=2, EPOCHS=30, no resume) |
| **Artifact** | `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/seed2_result_ge30.json` |

| Metric (ge30) | Floor | Result |
| ------------- | ----- | ------ |
| frac max-p≥0.60 | ≥15% (L2) | **85.6%** |
| rival soft | ≤0.20 | **held** (path to L2 from ge19) |
| soft_load_max | (watch) | **0.372** |
| `n_proteins_with_committed` / `committed_protein_share_max` | ≥8/12, ≤35% | **12/12, 16.5%** PASS |
| `committed_hard_share_max` | <0.56 | **0.544** PASS |
| `per_structure_committed_hard_max` | <0.56 | **0.745** FAIL |
| `per_structure_soft_max` | watch →0.56 | **0.519** (rising) |
| `committed_distribution` | `COMMITTED_DISTRIBUTION_PASS` | **`COMMITTED_HARD_CONCENTRATED`** (fail: `PER_STRUCTURE_COMMITTED_HARD_MONOPOLE`) |

Trajectory: first `STACK_WIN_L1` at ge17, `STACK_WIN_L2` at ge19; continuous climb with large mid-run steps (ge20/21 ~+10 pp) **without** a resume boundary — seed-1’s seam was not required for acceleration. No `COMMITTED_DISTRIBUTION_PASS` at any scored epoch.

**Verdict (contemporaneous 2026-07-14):** stack commitment mass **reproduces** on a second seed under cold-to-30 (`STACK_WIN_L2`). Pre-registered absolute per-structure gate **did not pass** (`per_structure_committed_hard_max` 0.745). Soft balance / corpus hard-share alone would have mis-called concentration risk — the companion was right to exist as a monitor.

**Verdict (composition reframe 2026-07-15):** that absolute fail was **mostly measurement artifact** under clean relative purity — 1F88/2Z6H excess over corpus majority ≈ +0.02; see § “Absolute monopole gate vs corpus composition”. Seed-2 **does** lock sensitivity + commitment + relative purity; absolute share ≥0.56 is no longer a standing fail.

#### Post-seed-2 cheap audits — 1F88 distinctiveness + seed-1 retrospective (2026-07-14)

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/f88_local_monopole_audit.json`.

**(2) Holdout role:** G4 thin protein-holdout (seed=42) **is** `1F88` — documented as the Stage A-12 single held-out example. **This stack lineage does not use** `--p4-v3-aleatoric-shaping`, so 1F88 is **fully in training** here and is also the Fix-1 epoch anchor. The seed-2 local monopole is **not** an OOD/holdout artifact of the G4 mask; it is an in-sample concentration on a structure that this sprint already treats as geometrically privileged.

**(1) Is 1F88 structurally distinctive?** **Yes**, vs the other 11 on Stage A-12 feature proxies (unique fold `1.20.1070.10`):

| Proxy | 1F88 | z vs other-11 |
| ----- | ---- | ------------- |
| n_residues | 338 | +0.32 (ordinary size) |
| degree / ρ mean | high | **+3.44** |
| degree / ρ std | high | **+2.26** |
| SASA mean | high | **+1.98** |
| dehydron / τ frac | **low** (~0.30) | **−3.04** |

So a single-expert concentration on 1F88 is **not** an arbitrary “could have been anyone” pick in feature space — rhodopsin-like profile (high connectivity/exposure, dehydron-poor relative to corpus). That does **not** excuse the gate fail (committed residues still monopole-local), but it reframes the failure as possibly **composition-conditioned** rather than pure RNG landing.

**(3) Seed-1 retrospective (no in-train hard-share log):** offline reload still mismatches soft_load (L1 ~0.36–0.48) — absolute shares untrusted. Ordinal signal only, partially validated because seed-2 offline still ranks **1F88** worst (0.72 vs in-train 0.745). Under that caveat, seed-1 ge20/25/30 offline shows **`per_structure_committed_hard_max` ~0.74–0.88** with **multiple** proteins ≥0.56 (e.g. 1TEN/1UBQ/1MBN/1LYZ — **not** 1F88 as the peak). Interpretation: local committed-hard concentration looks like a **systematic stack tendency**; **which** protein absorbs it is seed-dependent. Seed-1 was not “clean under the floor” — the floor simply was not logged then.

**Implication:** mechanisms (prototype sep + elevated scale) still stand; third failure mode is real and likely systematic. Next step is characterization / a per-structure diversity lever if desired — not a third full seed first unless a cheap in-train seed-1 re-score (impossible without retrain) is needed for absolute numbers.

#### N-vs-local-monopole check (seed-2 in-train SSOT, 2026-07-14)

Artifact: `n_vs_local_monopole_correlation.json`.

**Question:** is `per_structure_committed_hard_max` mostly small-N sampling variance, or within-structure homogenization?

| Evidence | Result |
| -------- | ------ |
| spearman(N, hard_max) at ge20/25/30 | **+0.41 to +0.50** (wrong sign for small-N story) |
| ge30 fails | **1F88 N=302 (0.745), 2Z6H N=478 (0.655), 1MBN N=135 (0.563)** — large committed subsets |
| ge30 small-N bin (51–100) | mean hard_max **0.47**, **0/3** fails |
| excess vs 4-expert random multinomial E[max] | **+0.16 to +0.35** in all late bins; largest excess in mid/large-N |

**Verdict: favors (2) training-dynamics / within-structure homogenization**, not (1) small-sample artifact. A min-N reporting floor alone will not clear the gate; a per-structure diversity term is the better-targeted lever if intervening (with the usual risk of fighting real signal). Composition (1F88-like dehydron-poor / high-degree) may still modulate *where* homogenization bites.

#### encoder_h local similarity vs local monopole (seed-2 ge30, 2026-07-14)

Artifact: `encoder_similarity_vs_local_monopole_seed2_ge30.json` (train-matched `max_residues=1200`).

| Metric vs `committed_hard_share_max` | spearman |
| ------------------------------------ | -------- |
| encoder_h mean pairwise cos (committed) | **−0.21** |
| encoder_h kNN-8 mean cos (committed) | **+0.34** |
| pre_mp kNN-8 cos (committed) | **+0.43** |
| kNN cosine range across 12 proteins | **~0.025** (0.963–0.988) — near ceiling |

Size-matched: 1F88 knn=0.981 vs 1TIM/2SHP ~0.979–0.984; 2Z6H knn=0.988 vs 1BG1/1IVO ~0.983–0.987 — **fails are not distinctively tighter**.

**Verdict: `LOCAL_TIGHT_EVERYWHERE_GATE_SIDE_MORE_LIKELY`.** Local neighborhoods are already homogenized corpus-wide; local monopoles are not “these proteins uniquely collapsed in encoder_h.” Pattern fits sharpened gate assignment locking a within-structure majority (with a representationally distinct minority keeping global pairwise from rising). If intervening, prefer a **within-structure assignment / routing-diversity** lever over a generic global encoder decorrelation loss.

#### Committed minority coherence on fails (seed-2 ge30, 2026-07-14)

Artifact: `committed_minority_coherence_seed2_ge30.json` (1F88 / 2Z6H).

| Protein | maj expert | maj share of committed | minority | dehydron partition |
| ------- | ---------- | ---------------------- | -------- | ------------------ |
| 1F88 | e3 | **0.745** (225) | e0=35, e2=42, e1=0 (77) | maj dehydron rate **0.0**; min **1.0** |
| 2Z6H | e3 | **0.655** (313) | e0=72, e2=93, e1=0 (165) | same perfect binary split |

Also: maj high-ρ / low-SASA vs min low-ρ / high-SASA (|d|≈1–3); many sequence runs (34 / 77) — **axis-coherent, not a single fold domain**. Explains the −0.21 global vs +0.34 local kNN sign flip (homogeneous core majority + distinct dehydron minority).

**Reframe (plain):** seed-2’s “83% clears / 17% fails” was never “sensitivity mostly works.” The fix **correctly preserves a real minority axis** (dehydron/surface on e0+e2); the failure is **core/non-dehydron majority over-consolidating onto one expert** (e3), with e1 empty on that side. Diversity pressure should target that majority pile — not the dehydron partition.

**ge21 resume seam (seed-1 only):** +10.4 pp in one epoch (~3.5× mean pre-seam rate ge16–20). Share of ge20→30 climb: **17%**. Post-seam mean rate ge22–30 is **5.5 pp/ep** (higher than pre-seam 3.0) — climb is not a seam artifact; counterfactual replacing ge21 with pre-mean rate still ~67% at ge30. Caveat the “smooth phase transition” framing at the resume boundary; do not discard the post-ge21 continuation. Seed-2 shows similar mid-run acceleration **without** resume — treat large pp/epoch steps as a feature of this recipe’s late phase, not solely an optimizer re-init artifact.

Artifacts: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/monopole_mirror_audit_ge25_ge30.json`, `ge21_resume_seam_attribution.json`, `seed2_result_ge30.json`, `f88_local_monopole_audit.json`, `n_vs_local_monopole_correlation.json`, `encoder_similarity_vs_local_monopole_seed2_ge30.json`, `committed_minority_coherence_seed2_ge30.json`.

### Pre-registered gate — majority-conditional committed-share hinge (frozen 2026-07-14, **before** train)

**Hypothesis:** once repulsion+scale unlocks commitment, local monopoles arise because the **within-structure committed majority** (mechanism-agnostic: whichever expert holds the largest share of max-p≥0.60 residues) has no counterpressure. A hinge on that majority share recruits unused capacity (e.g. empty e1) into the majority pile **without** conditioning the loss on dehydron=0. The dehydron axis is protected by a **monitor**, not by baking biology into the loss.

**Config under test (single controlled run):** same stack lineage (Fix-1 + S4 + T1a z-norm + `--gate-include-sasa` + nearest-pair repulsion `coeff=1.0`/`m=0.25` + softplus init/floor ≈6.612). **Add one term:**

#### Loss identity (pinned — mechanism-agnostic)

| | |
| --- | --- |
| **Name** | `majority_committed_soft_share_hinge` (implemented as STE hard-share) |
| **Trigger / condition** | Per structure: among residues with **max-p ≥ 0.60** (committed mask; stopgrad OK for membership). **Not** conditioned on dehydron / τ / SASA. |
| **Majority definition** | `e* = argmax_e` hard-count among committed (stopgrad). |
| **Penalty** | Straight-through estimator: forward value = hard majority share; grads via soft `w` on stopgrad maj-mask only. `L_s = ReLU(share − τ)²` with **τ = 0.56**. Skip if `n_committed < 20`. |
| **Why not soft mean share** | Soft `max_e mean(w\|commit)` is **below τ** at known seed-2 fails (e.g. 1F88 soft≈0.55 vs hard≈0.74) — would be inert at the operating point the gate is meant to fix. |
| **Coeff** | **λ = 0.5** (proxy-chosen; see Coefficient policy below) |
| **Explicitly out of scope** | Per-residue entropy floor; dehydron=0 mask in the loss; encoder_h→gate; raising repulsion coeff; in-run coeff ladder. |

**Dehydron purity = monitor only** (jsonl `dehydron_partition_purity` via `score_dehydron_partition_purity`). Partition maj/min the same way as the loss (committed hard majority), then measure dehydron rates on those sets.

#### Frozen floors (judge primarily at ge30; log every scored epoch)

| Claim | All must hold |
| ----- | ------------- |
| **`MAJ_LOCAL_CLEARED`** | `per_structure_committed_hard_max` **< 0.56** **and** `COMMITTED_DISTRIBUTION_PASS` |
| **`MAJ_COMMIT_HELD`** | frac max-p≥0.60 ≥ **15%** (do not trade away `STACK_COMMIT_L2`) |
| **`MAJ_AXIS_HELD`** | best underwrap/wrap τ contrast ≥ **0.40** |
| **`MAJ_PURITY_HELD`** | for every eligible structure (`n_committed≥20` and `n_minority≥5`): majority dehydron rate **≤ 0.05** **and** minority dehydron rate **≥ 0.90** → verdict `DEHYDRON_PARTITION_PURE` |

Baseline reference (seed-2 ge30 fails): maj rate **0.0**, min rate **1.0**. The 0.05 / 0.90 bands catch blurring (e.g. maj→5%, min→90%) immediately.

#### Fail conditions

| Fail | Definition |
| ---- | ---------- |
| **`AXIS_SCRAMBLED_BY_DIVERSITY`** | `DEHYDRON_PARTITION_BLURRED` (purity floors missed) |
| **`COMMIT_KILLED_BY_DIVERSITY`** | frac max-p≥0.60 \< **15%** |
| **`LOCAL_MONOPOLE_PERSISTS`** | purity+commit held but `per_structure_committed_hard_max` ≥ 0.56 |
| **`AXIS_COLLAPSE`** | τ contrast \< 0.40 |

#### Outcome buckets

| Verdict | Required |
| ------- | -------- |
| **`MAJORITY_SPLIT_WIN`** | all four claims above |
| **`MAJORITY_SPLIT_NO_MOVE`** | `LOCAL_MONOPOLE_PERSISTS` at the **registered** coeff |
| **`MAJORITY_SPLIT_COEFF_INCONCLUSIVE`** | same as NO_MOVE — **do not read as “lever ruled out.”** First-coeff miss is ambiguous between wrong magnitude and wrong mechanism (see coeff policy below). |
| **`AXIS_SCRAMBLED_BY_DIVERSITY`** | purity fail |
| **`COMMIT_KILLED_BY_DIVERSITY`** | commit regressed |
| **`AMBIGUOUS`** | else |

#### Coefficient policy (frozen with this gate — proxy-chosen, not a ladder)

**Proxy artifact:** `majority_hinge_coeff_proxy.json` (seed-2 ge30 operating point).

| Finding | |
| ------- | --- |
| Soft `mean(w\|commit)` hinge | **inert** at known fails (soft max ~0.43–0.55 while hard share 0.65–0.74) — rejected |
| STE hard-share + maj-mask hinge | fires on 1F88/2Z6H; raw hinge ~0.008–0.033 |
| Loss-matched ~10% of total | coeff ≈ **45–90** → gate-grad ~**100×** primary — rejected (COMMIT_KILLED risk) |
| Grad-matched ~O(1)× primary on fail structures | coeff ≈ **0.4** |

**Registered first coeff: `0.5`.** Rationale: same order as unit-hinge gate pressure on fail proteins (~2× primary on 1F88 at λ=0.5), audible without dominating; loss contribution small in the sum but grads land on the gate. Form: STE hard committed majority share, maj-mask restricted, `L = λ · ReLU(share − 0.56)²`.

**Pre-authorized one-step coeff rematch (same registration, not a free ladder):**
- After **`MAJORITY_SPLIT_NO_MOVE` / `COEFF_INCONCLUSIVE`** at 0.5 → one rerun at **`2.5`** (5×). Second NO_MOVE ⇒ lever needs a **new** pre-reg (mechanism, not magnitude).
- After **`COMMIT_KILLED_BY_DIVERSITY`** at 0.5 → one rerun at **`0.1`**. Second kill ⇒ new pre-reg.

Do **not** treat the first NO_MOVE as ruling out majority-conditional balancing.

**Escalation (frozen):** one primary run at λ=0.5 (+ at most one pre-authorized rematch above). If `MAJORITY_SPLIT_WIN`, stop and claim. If `AXIS_SCRAMBLED_*`, do **not** raise λ — redesign under a new pre-reg. If rematch still NO_MOVE/KILL, new pre-reg required.

**Scoring helpers (landed with this pre-reg):** `committed_majority_partition`, `score_dehydron_partition_purity`, `score_majority_conditional_ladder` in `experiments/diagnostics/prototype_repulsion_epoch.py`; jsonl keys `dehydron_partition_purity`, `majority_conditional`.

**Make:** `make train-v66-fix1-s4-proto-repulsion-scale-l2-majority-hinge-stage-a12` (EPOCHS=30, λ=0.5).

#### RESULT — majority-conditional hinge λ=0.5 (ge30, 2026-07-14)

| | |
| --- | --- |
| **Run** | `fix1_s4_stack_majority_hinge_stage_a12_cold_v1` |
| **Artifact** | `prototype_repulsion_per_epoch.jsonl` + proxy `majority_hinge_coeff_proxy.json` |

| Metric (ge30) | Floor | Result |
| ------------- | ----- | ------ |
| frac max-p≥0.60 | ≥15% | **40.9%** PASS |
| τ contrast | ≥0.40 | **0.75** PASS |
| `per_structure_committed_hard_max` | <0.56 | **0.625** FAIL (1HHP; was 0.745 on seed-2) |
| dehydron partition purity | maj≤0.05, min≥0.90 | **BLURRED on 12/12** eligible |
| `majority_conditional` | `MAJORITY_SPLIT_WIN` | **`AXIS_SCRAMBLED_BY_DIVERSITY`** |

Commitment L2 held; local monopole eased but did not clear. Purity monitor caught the real failure: maj/min dehydron rates mixed (~0.6/0.5 typical) — hinge pressure scrambled the dehydron axis rather than only splitting the core majority.

**Verdict: `AXIS_SCRAMBLED_BY_DIVERSITY`.** Per frozen escalation: **do not** rematch at 2.5 / 0.1 — those rematches are only for COEFF_INCONCLUSIVE / COMMIT_KILLED. Next step needs a **new pre-reg** (core-only hinge below) — not a coefficient ladder on this term.

### Pre-registered gate — core-only majority hinge (frozen 2026-07-14, **before** train)

**Hypothesis:** the agnostic majority hinge scrambled purity because it could push any committed majority residue. Restricting the eligible set to **committed ∧ dehydron=0** splits the core pile that over-consolidates, while excluding direct loss pressure on dehydron=1 residues.

**Guarantee vs monitor (honest):** Direct pressure on dh=1 residues is structurally excluded (zero grad from this term — unit-tested). That does **not** make purity failure impossible: core overflow can still redistribute onto minority-held experts (e0/e2) and dilute min dehydron rate. **Purity floors still monitor for that indirect dilution** and correctly land `AXIS_SCRAMBLED_*` if it happens.

**Config under test:** same stack lineage (Fix-1 + S4 + T1a + SASA + repulsion `1.0`/`0.25` + softplus ≈6.612). **Add** `core_majority_committed_share_coeff` (distinct name). **Agnostic** `majority_committed_share_coeff=0` for this run.

#### Loss identity

```
C0 = stopgrad(committed ∧ dehydron==0)
e* = argmax hard-count among C0
STE maj-mask share of e* among C0
L = λ · ReLU(share − 0.56)²   # skip if |C0| < 20
```

#### Eligible structure for purity (restated verbatim — self-contained)

A structure is **eligible** for the dehydron-partition purity check iff:

- `n_committed ≥ 20`, **and**
- `n_minority ≥ 5`

(Same values as the prior purity monitor; inlined here so this pre-reg does not require cross-referencing.)

On every eligible structure: majority dehydron rate **≤ 0.05** and minority dehydron rate **≥ 0.90**.

#### Frozen claims (ge30)

| Claim | Floor |
| ----- | ----- |
| Commit L2 | frac max-p≥0.60 ≥ **15%** |
| Axis | τ contrast ≥ **0.40** |
| Local cleared | `per_structure_committed_hard_max` **< 0.56** + `COMMITTED_DISTRIBUTION_PASS` |
| Purity | eligible structures pass maj≤0.05 / min≥0.90 |

| Verdict | Meaning |
| ------- | ------- |
| **`CORE_MAJORITY_SPLIT_WIN`** | all four claims |
| **`CORE_MAJORITY_SPLIT_COEFF_INCONCLUSIVE`** | local monopole persists (do not read as lever ruled out) |
| **`AXIS_SCRAMBLED_BY_DIVERSITY`** | purity fail (incl. indirect dilution) — **no coeff rematch** |
| **`COMMIT_KILLED_BY_DIVERSITY`** | commit L2 lost |

**Coeff:** proxy on seed-2 ge30 with core-only form (`core_majority_hinge_coeff_proxy.json`): raw hinge **0.1936** on **12/12** (core piles fully monopolar). Unit-hinge gate grad ≫ primary → grad-match λ≈**0.09**. Loss-matched 10%≈4.9 rejected (would dominate gate). **Registered λ = 0.25.** Rematch: NO_MOVE→**1.25** once; COMMIT_KILLED→**0.05** once; AXIS_SCRAMBLED → no rematch.

**Make:** `train-v66-fix1-s4-proto-repulsion-scale-l2-core-majority-hinge-stage-a12`

#### RESULT — core-only majority hinge λ=0.25 (ge30, 2026-07-14)

| | |
| --- | --- |
| **Run** | `fix1_s4_stack_core_majority_hinge_stage_a12_cold_v1` |
| **Artifact** | `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/core_majority_hinge_result_ge30.json` |

| Metric (ge30) | Floor | Result |
| ------------- | ----- | ------ |
| frac max-p≥0.60 | ≥15% | **28.7%** PASS |
| τ contrast | ≥0.40 | **1.00** PASS |
| `per_structure_committed_hard_max` | <0.56 | **0.793** FAIL (2Z6H; worse than seed-2 0.745) |
| `committed_hard_share_max` | <0.56 | **0.588** FAIL |
| dehydron partition purity | maj≤0.05, min≥0.90 | **BLURRED on 12/12** |
| `core_majority_conditional` | `CORE_MAJORITY_SPLIT_WIN` | **`AXIS_SCRAMBLED_BY_DIVERSITY`** |

Purity failure mode differs from the agnostic hinge: majority-side dehydron rates are **high** (~0.90–1.0), minority lower — the committed majority is now the **dehydron** pile, not core. Direct dh=1 exclusion held as designed; purity floors correctly caught the scramble (indirect redistribution / partition flip). Local monopole not cleared.

**Verdict: `AXIS_SCRAMBLED_BY_DIVERSITY`.** Per frozen escalation: **do not** rematch at 1.25 / 0.05. Core-only share hinge alone is insufficient; next lever needs a **new pre-reg** (not another λ on this term).

### Architecture check — does routing loss distort trunk embeddings? (2026-07-14)

Feeler / Fix-1+S4 stack uses **`topology_only_gate=True`**. Gate fusion is board-only (`fused_hyp = topo_hyp`); Möbius path on `x_hyp` is not built. Board channels ρ/τ/degree/clustering/ss/SASA come from graph/data; `cone_depth` and optional disc xy/r are trunk-derived but **`.detach()`’d** into the board.

**Routing / share-hinge grads update:** `gate.topo_encoder`, `prototype_bank`, `logit_scale`, `expert_bias` (and curvature on that path). **Not** node encoder / MP / `encoder_h`.

**Implication:** “detach node embeddings before the balance loss” is **already true by construction** on this lineage — that rationale does **not** motivate the next lever. Both hinge failures were **destination relocation** of overflow mass, not trunk embedding distortion. Any Shazeer-style term must still be **retargeted at the core (dh=0) population**, not the dehydron set; stopgrad-on-trunk is a no-op here and should not be sold as the mechanism.

### Cheap check — when does the local monopole form? (idea 3 premise)

Artifact: `monopole_onset_trajectory_core_hinge.json` (core-only λ=0.25 run).

| | |
| --- | --- |
| First nonzero frac max-p≥0.60 | **ge16** |
| First `per_structure_committed_hard_max` ≥0.56 | **ge16** (value **1.0** at onset) |
| ge30 struct_ch | **0.793** (slow decline while frac climbs 2.5%→28.7%) |
| Soft load | stays balanced ~0.25–0.29 throughout |

**Verdict: `LATE_EQUILIBRIUM_NOT_EARLY_LOCKIN`.** Monopole co-emerges with commitment and persists as a late training equilibrium — not an init lock-in by ge5–8. **Idea 3 (early annealed logit noise) is poorly targeted** for this failure; park it unless a future onset check shows early lock-in.

Same pattern on seed-2 stack (onset ~ge14–15 at struct_ch=1.0 → 0.745) and agnostic hinge (onset ~ge15–18).

### Pre-registered gate — core capacity quotas (frozen 2026-07-14, **before** train)

**Hypothesis:** share-hinges can only *penalize* majority share; they cannot choose where overflow lands — both prior runs scrambled purity via destination relocation. A **hard per-expert capacity on the core (dh=0) population** forces overflow to a second-choice expert deterministically (GShard/Switch-style), scoped to the diagnosed over-consolidation side, with an explicit **cross-axis guard** so second-choice cannot be a dehydron-dominant expert.

**Config under test:** same stack lineage (Fix-1+S4+T1a+SASA+repulsion+softplus≈6.612). **No** agnostic/core majority share hinge (`*_share_coeff=0`). **Add** train-time core capacity quota (below). Shazeer `f_i·P_i` on core held as **fallback** only if this fails on commit-L2 / τ (new pre-reg then).

#### Mechanism (pinned)

Per structure, let `C0 = {residues with dehydron==0}` (stopgrad labels). Soft scores `w` as usual.

1. **Capacity:** each expert may hard-receive at most `cap_e = max(⌊τ_cap · |C0|⌋, 1)` core residues, with **τ_cap = 0.40** (forces ≥3 experts if core is fully placed under equal fill; matches “below historical monopole 0.56” with headroom).
2. **Assignment:** sort core residues by max soft score descending. Greedily assign each to its highest-scoring **eligible** expert under remaining core capacity.
3. **Second-choice / cross-axis guard (answers the leak that broke both hinges):**
   - **Frozen dehydron-dominant snapshot (not live):** at **ge0** (ep0 forward, before any optimizer step), for each structure compute soft-argmax dehydron rate per expert. Mark expert **dehydron-dominant** if that rate **> 0.50** among its soft-argmax residues **and** it holds ≥ **5** soft-argmax residues (else undecided → not dominant). If a structure would have **zero** non-dominant experts, instead mark the **two** highest-dehydron-rate experts as dominant (always protect an axis pole). Persist this boolean mask in the run and **never recompute** — quotas must not chase a moving boundary as core redistributes.
   - Eligible second-choice = under-capacity experts that are **not** dehydron-dominant (frozen), ranked by soft score.
4. **Tie-break when every under-cap expert is dehydron-dominant (pinned — collision is expected under τ_cap=0.40):**
   - **Unplaced wins over force.** Never assign a core residue onto a dehydron-dominant expert.
   - That residue keeps its **soft argmax** for the expert mix (no discrete quota reassignment) and increments `quota_unplaced_core`.
   - **Do not** abort mid-run on the first collision — log `quota_collision_unplaced` every scored epoch.
   - At ge30: if corpus-mean `quota_unplaced_core / n_core` **> 0.05** → fail `QUOTA_STARVE`. By construction `QUOTA_FORCED_CROSS_AXIS` count **= 0** (no bypass path).
5. **Training use:** capacity-masked logits / discrete quota assignment for core residues only (declare capacity-masked softmax in impl). Dehydron=1 residues: **unchanged**. Soft `w` still trains the gate.
6. **Eval / metrics:** standing companions use the same hard assignment the mix uses (quota-aware).

#### Eligible purity (standing definition — relative, 2026-07-15)

Eligible structure: `n_committed ≥ 20` **and** `n_minority ≥ 5`.

**Relative floor (default):** let `f` = structure corpus dehydron fraction.
- If `f < 0.5` (core-majority): maj dehydron ≤ **0.05**, min ≥ **0.90**.
- If `f > 0.5` (dehydron-majority): maj dehydron ≥ **0.90**, min ≤ **0.05**.
- If `f = 0.5`: either clean polarity passes.

Fixed `maj_dh ≤ 0.05` is **legacy only** (`floor_mode=fixed_core_majority`) — it assumed core always dominates and false-failed 8/12 Stage A-12 structures. Scoring helper: `score_dehydron_partition_purity` in `prototype_repulsion_epoch.py`.

#### Frozen claims (ge30)

| Claim | Floor |
| ----- | ----- |
| Commit L2 | frac max-p≥0.60 ≥ **15%** |
| Axis | τ contrast ≥ **0.40** |
| Local cleared | `per_structure_committed_hard_max` **< 0.56** + `COMMITTED_DISTRIBUTION_PASS` |
| Purity | eligible structures pass **relative** floor (corpus majority class ↔ committed majority class; see standing definition above) |
| Quota integrity | corpus-mean `quota_unplaced_core / n_core` **≤ 0.05**; `QUOTA_FORCED_CROSS_AXIS` **= 0** by construction (unplaced preferred; never force cross-axis) |

| Verdict | Meaning |
| ------- | ------- |
| **`CORE_QUOTA_WIN`** | all claims above |
| **`CORE_QUOTA_NO_MOVE`** | purity+commit held but local monopole persists |
| **`AXIS_SCRAMBLED_BY_QUOTA`** | purity fail — **no τ_cap ladder inside the run** |
| **`COMMIT_KILLED_BY_QUOTA`** | commit L2 lost |
| **`QUOTA_STARVE`** | unplaced core > 5% or forced cross-axis |

**Escalation (frozen):** one run, τ_cap=0.40. If `CORE_QUOTA_WIN`, stop and claim. If `AXIS_SCRAMBLED_*` or `QUOTA_STARVE`, do **not** soften the cross-axis guard — redesign second-choice under a new pre-reg. If `COMMIT_KILLED_*`, fallback candidate is **core-retargeted Shazeer `f·P`** (new pre-reg), not annealed logit noise. If `CORE_QUOTA_NO_MOVE`, consider lower τ_cap only under a **new** pre-reg.

**Make:** `train-v66-fix1-s4-proto-repulsion-scale-l2-core-quota-stage-a12`

#### RESULT (ge30 cold) — `AXIS_SCRAMBLED_BY_QUOTA`

| | |
| --- | --- |
| **Run** | `fix1_s4_stack_core_quota_stage_a12_cold_v1` |
| **Artifact** | `checkpoints/v66/runs/fix1_s4_stack_core_quota_stage_a12_cold_v1/` (+ `core_quota_dominant_ge0.json`) |
| frac max-p≥0.60 | **0.628** (commit L2 held) |
| best τ contrast | **0.799** (axis held) |
| `per_structure_committed_hard_max` | **0.560** (at floor; not cleared; dist=`COMMITTED_HARD_CONCENTRATED`) |
| purity | **`DEHYDRON_PARTITION_BLURRED`** every scored epoch (incl. ge0 with quotas) |
| mean unplaced/core | **0.017** (≤0.05); forced cross-axis **0** |
| **Verdict** | **`AXIS_SCRAMBLED_BY_QUOTA`** |

Quota integrity passed; commit and τ held. Failure mode is purity (same family as both share hinges), with local monopole only at the 0.56 boundary rather than cleared. Per frozen escalation: **do not** soften the dehydron-dominant guard or ladder τ_cap inside this gate — redesign second-choice under a **new** pre-reg.

#### Leak-direction audit (check before next pre-reg)

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/core_quota_leak_direction_audit.json`.

| Window | What purity fails on | Quota activity |
| ------ | -------------------- | -------------- |
| ge0–25 | **12/12** `min_dh < 0.90` with **maj_dh = 0** (both committed partitions core-only) | `mean_unplaced` flat **≈0.017** |
| ge26–30 | maj_dh rises **0→0.49** (corr with frac max-p ≥0.60 ≈ **0.98**); min_dh stays ≈0 | unplaced still flat (no corr with maj_dh) |

Hard-argmax expert τ already shows a dehydron pole early (e2 ≈0.84–0.94) while committed maj/min stay all-core — soft/hard axis exists; **committed dehydron niche never forms**.

**Verdict on mechanism:** not primarily “dh=1 pulled onto core-majority by quota overflow reshaping.” The one-sided guard held; blur is (1) dehydrons absent from committed minority all run, then (2) late commitment expansion contaminating majority. Symmetric two-pool masking may still be the right next lever (force dh=1 onto dehydron-typical experts), but register it as closing the **unguarded dehydron routing + commitment** hole — not as repairing a quota-induced cross-pull.

#### Commit-lag branch check (gate soft vs quota mix) — 2026-07-15

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/core_vs_dehydron_commit_lag.json`  
Script: `experiments/diagnostics/core_vs_dehydron_commit_lag.py` (top1−top2 logit gap; early ge5→15 / late ge15→30).

| Check | Result |
| ----- | ------ |
| Magnitude gap (core ≫ deh top1−top2) | **Absent — inverted.** @ge15 gap ratio core/deh = **0.24**; @ge30 = **0.54**. Dehydron mean gap 0.55→0.86 vs core 0.13→0.47 |
| Starvation (similar early Δ, deh flat late) | **Absent.** Early: core **+0.062**, deh **+0.130**. Late: core **+0.098**, deh **+0.089** |
| Gate soft frac max-p≥0.60 | **0% both populations** at ge5/15/30 |
| Quota-mix frac max-p≥0.60 @ge30 | core **1.000** (STE one-hots) / dehydron **0.000** (stays soft) |

**Branch: `DEHYDRON_STRONGER_PEAKEDNESS_NEITHER_GAP_NOR_STARVATION`.**  
Neither reverse-hinge (magnitude) nor two-pool-alone (starvation) is justified by gate telemetry. The committed-population “dehydron lag” is largely a **one-sided STE artifact**: quota hardens core to max-p=1 while dehydron soft (~0.51) never enters the ≥0.60 committed set used by purity. Pole e2 is preferred for dehydrons throughout.

**Implication for next pre-reg:** do not draft “accelerate dehydron commit because gate is weak.”

#### Foundation check — was `STACK_WIN_L2` itself gate-soft? (2026-07-15)

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/stack_win_gate_soft_audit.json`.

| Run (ge30) | `use_gumbel` | mix ≡ gate soft | gate-soft frac≥0.60 | Verdict |
| ---------- | ------------ | --------------- | ------------------- | ------- |
| stack seed1 | false | **yes** (L1=0) | **0.806** | **`PURE_GATE_SOFT`** |
| stack seed2 | false | **yes** | **0.807** | **`PURE_GATE_SOFT`** |
| quota, quota off | false | yes | **0.000** | `GATE_SOFT_BUT_NO_COMMIT` |
| quota, quota on | false | **no** | mix 0.49 / gate **0.000** | `HARD_OR_QUOTA_TOUCHED` |

**`STACK_WIN_L2` stands** — original repulsion×scale commitment was pure softmax. The STE artifact is **quota-era specific**. “Symmetric commitment” therefore means **score commit/purity on gate-soft only** (never extend STE one-hots to dehydron). Open problem: quota training left the true router at 0% gate-soft commit — next lever must restore real soft commitment while addressing local monopole without forged companions.

#### Pure-stack purity baseline (recomputed gate-soft, 2026-07-15)

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/stack_gate_soft_purity_baseline.json`.  
Historical stack jsonl lacked purity — recomputed at ge30.

| | seed1 | seed2 |
| --- | --- | --- |
| frac max-p≥0.60 | **0.806** | **0.807** |
| τ contrast | **1.0** | **1.0** |
| `per_structure_committed_hard_max` | **0.714** | **0.727** |
| purity floors (maj≤0.05 ∧ min≥0.90) | **0/12** | **4/12** (1MBN, 1F88, 2Z6H, 1HHP perfect) |
| dominant fail mode | 10/12 BOTH mixed | 8/12 **polarity flip** (maj all-dh, min all-core) |

**Starting point is better than quota-era looked, worse than “axis already solved.”** Commit+τ are real; local monopole is stack-native (~0.71–0.73); purity is partial (soft τ pole exists; committed maj/min only clean on 4/12 seed2). Quota/hinge did not invent the purity gap — they destroyed soft commit while failing to fix monopole. **Park the quota/hinge/STE-output family.** Next lever should act on router decision geometry (same family as scale+repulsion), scored gate-soft-only.

#### Polarity “flips” vs corpus majority (2026-07-15)

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/polarity_flip_vs_corpus_majority.json`.

On seed2, the 8 fixed-floor fails are **exactly** the 8 structures that are dehydron-majority in raw composition (dh frac 0.51–0.69). Committed maj_dh≈1 / min_dh≈0 matches that composition. **True polarity inversions: 0.** Relative floor (corpus majority class ↔ committed majority class): **12/12 pass** on seed2.

| Fixed floor (`maj_dh≤0.05`) | Relative (per-structure) |
| --------------------------- | ------------------------ |
| seed2 **4/12** | seed2 **12/12** |
| seed1 **0/12** | seed1 **0/12** (real dilution, not flips) |

**Floor was miscalibrated**, not the model polarity. Do not design a polarity-symmetry fix for seed2 “flips.”

**Metric correction (standing):** replace fixed `maj_dh ≤ 0.05` with the relative floor as the default purity definition going forward. Scoring against the old fixed floor will keep generating false `AXIS_SCRAMBLED` / `DEHYDRON_PARTITION_BLURRED` verdicts on legitimately dehydron-majority structures.

> **Later correction (2026-07-15):** the absolute local-monopole gate had the **same confound shape**. See § “Absolute monopole gate vs corpus composition” below — under clean bipartition, `committed_hard_share_max ≈ max(frac_core, frac_dh)`. Remaining real routing gap after that reframe: **seed1 committed dilution** (plus a small 1HHP excess residual, watch-only).

#### Pre-draft geometry audits (stack ge30, gate-soft) — 2026-07-15

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/monopole_and_seed_dilution_audit.json`  
Script: `experiments/diagnostics/monopole_and_seed_dilution_audit.py`

**(1) Local monopole ↔ prototype proximity** (analogous question to twin-collinearity)

| | seed1 | seed2 |
| --- | --- | --- |
| n monopole (≥0.56 share) | 6/12 | 8/12 |
| corr(share, frac maj nearest maj-proto) | **−0.22** | **+0.03** |
| corr(share, mean margin on maj) | 0.33 | 0.05 |
| mean d(maj residues → maj proto): mono vs not | **3.79 vs 3.36** | **3.19 vs 2.77** |

Monopolized majors sit **farther** from their majority prototype (mild), not closer. Margins almost identical mono vs non-mono on seed2. **Verdict: `WEAK_OR_NO_PROTOTYPE_PROXIMITY_LINK`.** Within-structure concentration is **not** the twin-style failure mode (residues collapsed onto one too-close attractor). Do not draft “repel maj residues from their prototype” as the monopole fix.

**Working reframe (hypothesis, not yet mechanism):** over-concentration looks more like a **weak-signal fallback** (residues land on a default expert when nothing else claims them more strongly) than strong-signal capture. That points at differentiating the *routing decision* for that population — not further prototype separation.

**(2) Seed1 dilution ↔ repulsion trajectory**

| | seed1 | seed2 |
| --- | --- | --- |
| nearest-pair hyp dist @ge0 | 0.029 | 0.029 |
| nearest-pair @ge30 | **0.541** | **0.319** |
| twin dist @ge30 | 0.595 | 0.350 |
| relative purity | **0/12 blurred** | **12/12 pure** |
| struct committed-hard max | 0.714 | 0.727 |

**Verdict: `NO_SEED1_HAS_STRONGER_NEAREST_PAIR_SEP`.** Seed1 achieved *more* prototype separation than seed2 and still diluted committed partitions. Dilution is **not** “repulsion failed to converge.” Axis-separation quality has real seed variance that is **orthogonal** to nearest-pair distance (seed2: weaker sep, clean axis; seed1: stronger sep, blur).

#### Cheap pre-draft checks — bias ties + bank config (2026-07-15)

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/bias_tie_and_seed_config_audit.json`  
Script: `experiments/diagnostics/bias_tie_and_seed_config_audit.py`

**(A) Equidistance / `expert_bias` tipping near-ties?**

| | seed1 | seed2 |
| --- | --- | --- |
| `expert_bias` range | 0.023 | 0.055 |
| bias argmax | e3 | e3 |
| structures whose maj == bias argmax | **0/12** | **4/12** |
| structures whose maj *changes* if bias removed | **0** | **0** |
| frac maj residues whose argmax flips with bias | **0** | **0** |
| maj expert histogram | e2×10, e1×2 | e0×8, e3×4 |
| frac maj near-tie (d-margin \<0.10), mono vs not | 0.004 vs 0.008 | **0 vs 0** |
| mean d-margin on maj, mono vs not | 0.319 vs 0.314 | 0.229 vs 0.229 |
| corr(share, d-std across experts) | **−0.53** | +0.33 |

**Bias verdict: `BIAS_NOT_DRIVING_MAJORITY`.** Tiny learned bias never flips a majority and does not even match the majority index on seed1. **Do not** draft zero/regularize-`expert_bias` as the monopole fix.

**Equidistance verdict: `NO_CLEAR_EQUIDISTANCE_CONTRAST` (mono cut).** Near-ties are rare; mono vs non-mono margins/tie-fracs are flat. Seed1 alone shows higher share ↔ lower distance-spread (mild continuous anticorrelation) — suggestive of weaker differentiation on heavier monopoles, but **not** “everyone is equidistant and bias breaks the tie.” Weak-signal fallback, if real, is **not** the cheap bias-asymmetry mechanism.

**(B) Seed1 dilution ↔ early bank configuration (beyond nearest-pair)?**

Early ≈ `epoch_001` (no `epoch_000` on disk). Two-seed comparative only.

| Metric | seed1 early | seed2 early | ‖Δ‖ | seed1 ge30 | seed2 ge30 |
| ------ | ----------- | ----------- | --- | ---------- | ---------- |
| nearest-pair d | 0.0424 | 0.0432 | **0.0008** | **0.541** | 0.319 |
| hyp-dist CV | 0.025 | 0.013 | 0.012 | 0.057 | 0.103 |
| angle std (°) | 4.85 | 3.58 | 1.27 | 7.60 | 7.05 |
| gram condition (tangent) | **2.45** | 1.72 | **0.73** | **49.7** | 32.8 |
| gram eig_min | 0.54 | 0.69 | — | **0.032** | 0.046 |
| relative purity @ge30 | — | — | — | **0/12** | **12/12** |

Nearest-pair starts matched; **tangent Gram condition already worse on seed1** and explodes further by ge30 (near-singular eig_min≈0.03). Angles stay ~90–120° on both — not a collapsed pair, but a **worse overall bank conditioning** that distance-to-nearest alone misses.

**Verdict: `SEED1_DILUTION_TRACKS_EARLY_BANK_CONFIG`.** Dilution looks like a **describable init/trajectory failure mode** (ill-conditioned four-prototype arrangement), not “pure RNG that averages out.” A third seed with similar early Gram pathology would be expected to risk the same blur — worth treating as a real mode when drafting, not as seed1 bad luck.

**What Gram condition means (precision over “nearest-pair distance”):**  
Pairwise nearest-distance only checks whether the *closest two* prototypes are far enough apart — it says nothing about whether the *full four-prototype configuration* is well-conditioned as a basis. Gram condition ~50 with `eig_min≈0.03` at ge30 means the four unit-tangent prototypes are nearly linearly dependent (close to a lower-dimensional subspace than nominal). That is exactly what nearest-pair-only repulsion **cannot fix and can worsen**: pushing the closest pair apart does not prevent the other directions from collapsing toward a shared subspace, and if gradient pressure is absorbed by the one pair the hinge targets, the rest of the bank can drift into a degenerate joint configuration. This explains the seed1/seed2 asymmetry: seed2 landed in a better-conditioned four-proto arrangement early; seed1 did not; more nearest-pair separation on seed1 (0.54 vs 0.32) could not repair what was never the optimized quantity.

#### Monopole board-feature distinctiveness (held track, 2026-07-15)

Artifact: `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/monopole_board_distinctiveness_audit.json`  
Script: `experiments/diagnostics/monopole_board_distinctiveness_audit.py`

Within-structure z-scored gate board (`_last_topo_features`[:topo_dim]): maj vs other-committed distance-to-centroid and knn5 uniqueness.

| | seed1 | seed2 |
| --- | --- | --- |
| mean (dist_cent maj / dist_cent oth) | 0.96 | **0.81** |
| mean (knn5 maj / knn5 oth) | 1.24 | **0.88** |
| mean \|Cohen d\| maj vs rest (cont. feats) | 1.62 | 0.78 |
| **Verdict** | `NO_CLEAR_BOARD_DISTINCTIVENESS_DEFICIT` | **`FALLBACK_BOARD_LESS_DISTINCTIVE`** |

On the seed that already has **clean relative purity** (seed2), the monopolized maj pile sits closer to the structure board centroid and is locally denser than the other committed set — consistent with a **board-signal-poor fallback population**, not bank geometry. Seed1’s diluted maj set does **not** show that deficit (dilution mixes the partition). So board poverty is a live lead for the **monopole track only**, and must not be conflated with seed1 Gram dilution.

#### Recommended split (locked 2026-07-15; monopole closed same day)

| Track | Target | Status |
| ----- | ------ | ------ |
| **(1) Dilution / bank conditioning** | Full-bank Gram logdet hinge | **Closed as dilution fix** — bank↑; purity independent / hostile; **no 0.005 rematch** (decision below) |
| **(2) Monopole / fallback** | Absolute within-structure share | **Closed** — measurement artifact under clean relative purity |

Kill list remains: bias zeroing, more nearest-pair-only repulsion, quota/hinge/STE-output, fighting absolute `committed_hard_share_max ≥ 0.56` under clean bipartition, dehydron-persistence board enrichment aimed at core maj piles, **seed1-only Gram λ=0.005 rematch** (chases bank floor, not dilution).

#### Absolute monopole gate vs corpus composition (closed 2026-07-15)

**Resolution class:** same as relative-purity floor — metric confounded with correct behavior on imbalanced structures.

Under stack seed2 ge30 (relative purity **12/12**, within-maj-class committed ≈ **1.0** on all 12):

\[
\texttt{committed\_hard\_share\_max} \approx \max(\texttt{frac\_core}, \texttt{frac\_dh})
\]

A perfectly working router that assigns the corpus majority class to one expert and the minority class to the others will always look like an absolute “monopole” whenever the majority class exceeds 56%. That is success on imbalanced biology, not a routing defect.

| pdb | frac_core | share (seed2) | excess = share − max(f_core, f_dh) | abs ≥0.56? |
| --- | --------- | ------------- | ---------------------------------- | ---------- |
| 1F88 | **0.701** | 0.727 | **+0.025** | yes → **resolved-by-remeasurement** |
| 2Z6H | **0.619** | 0.638 | **+0.018** | yes → **resolved-by-remeasurement** |
| 1HHP | 0.505 | 0.608 | **+0.103** | yes → small residual |
| 1MBN | 0.510 | 0.504 | −0.006 | no (control: same within-class collapse, balanced composition) |

Non-monopoles are not “better-split cores”; they are structures whose majority class is closer to 50%, so the same perfect bipartition stays under 0.56.

**Standing metric correction:** retire absolute `per_structure_committed_hard_max < 0.56` as a fail gate under clean relative purity. If concentration remains a goal, score **excess over corpus majority fraction** (or an explicit within-class multi-expert objective that trades purity — not the default).

**Why the four failed mechanisms looked inert or purity-hostile:** agnostic hinge, core-only hinge, core quota, and Gram-as-monopole-adjacent pressure were aimed at a number that was mostly measuring correct bipartition. Lessons from those runs (destination relocation, STE forged commit, bank↔purity independence) remain durable; the *target* was substantially a mirage.

**Persistence pre-check (killed before draft):** dehydron barcode `local_h1` exists and is non-redundant with ρ/SASA, but the worst absolute-share structures are **core** maj piles (12–17% dehydron touch) — persistence is off-population. `wrap_deficit` correlates |r|≈0.7 with ρ.

##### 1HHP residual — seed1 cross-check (2026-07-15)

| | seed1 stack ge30 | seed2 stack ge30 |
| --- | ---------------- | ---------------- |
| 1HHP share | 0.624 | 0.608 |
| 1HHP excess | **+0.118** | **+0.103** |
| maj class / corpus match | dehydron / **false** | core / **true** |
| within maj-class | 0.907 | 1.000 |
| worst excess on seed | **1UBQ +0.148** (not 1HHP) | **1HHP +0.103** (sole excess\>0.05) |
| seed2 excess z(1HHP vs other-11) | — | **+5.4** |

**Verdict: magnitude reproduces (~+0.10), but do not open a registration.** On seed1 the excess sits inside the dilution / polarity-mismatch cluster (`match=false`); on seed2 it is a single-structure ~10pp over-natural-floor blip. Identity of “worst excess” is seed-dependent (1UBQ vs 1HHP). Flag as **watch-only residual** under the composition-corrected metric — not a fifth mechanism target.

**Track status: `MONOPOLE_CLOSED_REMEASUREMENT`.** Plain resolution for the SSOT: sensitivity and commitment were correctly fixed; the remaining absolute “monopole” problem was largely a measurement artifact, now resolved by reframing against corpus composition — not “five mechanisms failed.”

---

### Pre-registered gate — full-bank Gram conditioning (frozen 2026-07-15, **before** train)

**Status:** **closed as dilution fix** — seed1 bank near-miss + purity still 0/12; seed2 relative purity **12/12→4/12** (`AXIS_SCRAMBLED_BY_GRAM`). **Decision (2026-07-15): do not run λ=0.005 rematch**, including seed1-only. Seed1 dilution needs a **new pre-reg**. Absolute monopole track closed separately by remeasurement.

**Hypothesis:** seed1-style committed dilution is driven by an **ill-conditioned four-prototype bank** (high tangent-Gram condition / tiny `eig_min`), which nearest-pair repulsion cannot see. A full-bank conditioning **hinge** (saturating once registered volume is met) keeps the bank a usable basis so dehydron vs core can occupy distinct committed partitions. Seed2 should stay pure or improve conditioning; seed1 should move toward relative purity. Local monopole may be unchanged — that is **not** a fail of this gate.

**Config under test:** same stack lineage (Fix-1 + S4 + T1a z-norm + `--gate-include-sasa` + softplus init/floor ≈6.612). **Keep** nearest-pair repulsion (`coeff=1.0`, `m=0.25`) as the proven PROTO_SEP term. **Add one term:** saturating Gram logdet hinge (below). **No** encoder_h→gate. **No** new board channel. **No** quota/hinge/STE. Gate-soft scoring only. **Relative** purity floor.

**Runs:** **two** cold Stage A-12 runs — **seed=1** and **seed=2** — same hyperparams. Judge each; success requires the claim on **both** (seed2 must not regress; seed1 must improve purity).

#### Saturation / boundedness (checked before freeze)

| Question | Answer |
| -------- | ------ |
| Does raw `−logdet` push prototypes to the hyperbolic edge? | **Not via magnitude**, if the Gram is formed from **unit-normalized** tangent rows: `G` is a correlation matrix (`diag=1`, `tr=4`), `det(G)≤1`, so `−logdet` is **bounded below** at mutual orthogonality (`G→I`). Radius growth of raw `prototype_tangent` is factored out. |
| Does that mean unbounded maximization is safe? | **No.** Bounded-below ≠ floor-saturating. Raw `−logdet` still keeps gradient toward **full orthogonality** (`eig_min→1`) well past the registered success floors (`eig_min≥0.15` / `cond≤15`) — same class of overshoot risk as unbounded `logit_scale` (opposite direction). |
| Fix | **Saturating hinge** (thread discipline): pressure stops once the registered volume target is met. |

Operating-point logdets (unit Gram, ge30 stack): seed1 **−2.64**, seed2 **−2.28**; floor-edge equal-eig at `λ_min=0.15` ≈ **−1.15**; identity = 0.

#### Loss identity (pinned)

| | |
| --- | --- |
| **Name** | `prototype_gram_logdet_hinge` |
| **Bank vector** | Unit-normalized `prototype_bank.prototype_tangent` → `Â ∈ R^{E×D}`, rows unit |
| **Gram** | `G = Â Âᵀ` (`E×E`, E=4) |
| **Penalty** | `L_g = ReLU(τ_logdet − log det(G + ε I))²` with **ε = 1e-4**, **τ_logdet = −1.15** (floor-edge volume at `λ_min≈0.15` with remaining mass equalized; also implies cond≲9 \< 15) |
| **Saturation** | `L_g = 0` once `logdet ≥ τ_logdet` — no further push toward identity / manifold edge |
| **Why not raw `−logdet`** | Keeps rewarding spread past registered floors (see saturation check) |
| **Why not condition-number alone** | `log(λ_max/λ_min)` ignores mid eigenvalues and is twitchy near crossings; logdet volume is smoother; cond remains a **monitor + success floor** |
| **Why not replace nearest-pair yet** | PROTO_SEP already validated pairwise separation; Gram **supplements** joint conditioning. Replace-only is a separate pre-reg if supplement wins but pairwise becomes redundant |
| **Coeff** | **λ_g = 0.001** (proxy below) |
| **Explicitly out of scope** | Local monopole floors; bias zeroing; raising nearest-pair coeff; STE/quota/share-hinge; new board features; encoder_h→gate |

**Standing monitors (jsonl every scored epoch):** existing stack companions **plus** `gram_eig_min`, `gram_eig_max`, `gram_condition`, `gram_logdet`, `gram_logdet_hinge`, relative `dehydron_partition_purity`. Log at **ge5 / ge15 / ge30** at minimum (catch early overshoot even though hinge should saturate).

#### Frozen floors (judge primarily at ge30; both seeds)

| Claim | Floor |
| ----- | ----- |
| **`GRAM_COMMIT_HELD`** | frac max-p≥0.60 ≥ **15%** (gate-soft) |
| **`GRAM_AXIS_HELD`** | best τ contrast ≥ **0.40** |
| **`GRAM_BANK_CONDITIONED`** | `gram_eig_min` ≥ **0.15** **and** `gram_condition` ≤ **15** **and** `gram_logdet` ≥ **−1.15** |
| **`GRAM_PURITY_HELD`** | relative purity: eligible structures pass (seed1 must reach **≥10/12**; seed2 must hold **12/12**) |

**Not required:** `per_structure_committed_hard_max` \< 0.56. Log it; do not fail the gate on monopole alone.

#### Fail conditions

| Fail | Definition |
| ---- | ---------- |
| **`GRAM_NO_MOVE`** | ge30 `gram_eig_min` still \< **0.08** on the worse seed (seed1 baseline ~0.03) |
| **`COMMIT_KILLED_BY_GRAM`** | frac max-p≥0.60 \< **15%** on either seed |
| **`AXIS_SCRAMBLED_BY_GRAM`** | relative purity regresses on seed2 below **12/12**, or seed1 stays **0/12** with bank floors already cleared |
| **`PAIRWISE_COLLATERAL`** | nearest-pair hyp dist \< **0.10** (undoes PROTO_SEP) |

#### Outcome buckets

| Verdict | Required |
| ------- | -------- |
| **`GRAM_COND_WIN`** | all four claims on **both** seeds |
| **`GRAM_COND_PARTIAL`** | bank floors clear + seed1 purity improves (≥10/12) but seed2 purity slips or commit/axis soft-miss — inspect before rematch |
| **`GRAM_COND_NO_MOVE`** | commit+axis+seed2 purity held but bank / seed1 purity floors missed at registered λ |
| **`COMMIT_KILLED_BY_GRAM`** / **`AXIS_SCRAMBLED_BY_GRAM`** / **`PAIRWISE_COLLATERAL`** | as named |

#### Coefficient policy (proxy-chosen, frozen)

**Proxy artifact:** `checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/gram_logdet_hinge_coeff_proxy.json`  
Script: `experiments/diagnostics/gram_logdet_hinge_coeff_proxy.py`

| Finding | |
| ------- | --- |
| Form | unit-normalized `ReLU(τ − logdet)²`, τ=−1.15 |
| seed1 ge30 | logdet **−2.64**, hinge gap **1.49**, unit-hinge ‖∂proto‖ ≈ **184** |
| seed2 ge30 | logdet **−2.28**, hinge still active (not yet saturated) |
| Nearest-pair repulsion @ge30 | **already saturated** (min dist ≫ 0.25) → repulsion proto grad **0** — match against **primary** proto grad, not repulsion |
| Primary proto grad (seed1) | ≈ **0.211** |
| λ for ≈1× primary proto | **0.00115** |

**Registered λ_g = 0.001.** Rationale: audible on the ill-conditioned bank without dominating primary (~1× proto pressure on seed1). Unit hinge is steep near singularity — do **not** loss-match 10% of total (would imply ≫ primary).

**Pre-authorized one-step rematch:**  
- `GRAM_COND_NO_MOVE` → one rerun at **0.005** (5×). Second NO_MOVE ⇒ new pre-reg.  
- `COMMIT_KILLED_BY_GRAM` → one rerun at **0.0002** (0.2×). Second kill ⇒ new pre-reg.  
- `AXIS_SCRAMBLED_*` or `PAIRWISE_COLLATERAL` → **no** coeff rematch; redesign.

> **Voided 2026-07-15 after seed2:** purity regress → `AXIS_SCRAMBLED_BY_GRAM` ⇒ the pre-auth **0.005** rematch is **not** taken, including as a seed1-only experiment. See RESULT decision below.

**Escalation:** two seeds × one primary λ (+ at most one rematch). If `GRAM_COND_WIN`, stop and claim dilution/conditioning fixed. If bank floors clear but seed1 purity still blurred → `AXIS_SCRAMBLED_BY_GRAM` path (bank was not sufficient cause — new pre-reg). **Actual path taken:** seed2 purity **12/12→4/12** ⇒ new pre-reg; Gram hinge closed as dilution fix (monopole track already closed by remeasurement).

**Make:** `train-v66-fix1-s4-proto-repulsion-scale-l2-gram-cond-stage-a12` (+ `-seed2`).

**Scoring helpers:** `measure_prototype_pairwise` logs Gram eigs/logdet/hinge; `score_gram_cond_ladder` → jsonl `gram_conditional`.

#### RESULT — seed1 ge30 cold (2026-07-15)

| | |
| --- | --- |
| **Run** | `fix1_s4_stack_gram_cond_stage_a12_cold_v1` |
| **Declared** | λ_g=**0.001**, τ=**−1.15**, nearest-pair kept; AMP fix: Gram logdet in float32 |

| Metric (ge30) | Floor | Result |
| ------------- | ----- | ------ |
| frac max-p≥0.60 | ≥15% | **0.531** PASS |
| τ contrast | ≥0.40 | **0.892** PASS |
| `gram_eig_min` | ≥0.15 | **0.129** FAIL (narrow) |
| `gram_condition` | ≤15 | **11.0** PASS |
| `gram_logdet` | ≥−1.15 | **−1.294** FAIL (narrow) |
| relative purity | ≥10/12 | **0/12** BLURRED |
| nearest-pair d | ≥0.10 | **0.367** PASS |
| struct committed-hard max | (monitor) | 0.696 |

Vs stack baseline seed1 ge30: eig_min **0.032→0.129**, cond **49.7→11.0**, logdet **−2.64→−1.29** — hinge moved the bank hard toward the floor but stalled just short; purity unchanged (still full blur).

**Verdict (seed1 alone, contemporaneous): `GRAM_COND_NO_MOVE`.** Commit+axis held; bank improved but floors not cleared; purity not rescued.

**Read before rematch (locked pre-seed2, then superseded):** the three bullets below framed a possible 0.005 rematch. **After seed2, rematch is voided** — see decision section following seed2 RESULT.

1. **Near-miss, not miss.** eig_min 0.129 vs 0.15 / logdet −1.29 vs −1.15 is “lever worked, landed short of floor.”
2. **Purity is the heavier signal.** Bank moved dramatically; relative purity stayed **0/12** — evidence bank conditioning ⊥ dilution target if seed2 purity also fails to improve.
3. **Commit trade.** frac max-p≥0.60 dropped **0.81→0.53** (still ≥15%).

**Decision gate (pre-seed2):** wait for seed2 ge30.

#### RESULT — seed2 ge30 cold (2026-07-15)

| | |
| --- | --- |
| **Run** | `fix1_s4_stack_gram_cond_stage_a12_cold_seed2_v1` |
| **Declared** | λ_g=**0.001**, τ=**−1.15**, nearest-pair kept |

| Metric (ge30) | Floor | Result |
| ------------- | ----- | ------ |
| frac max-p≥0.60 | ≥15% | **0.851** PASS |
| τ contrast | ≥0.40 | **1.0** PASS |
| `gram_eig_min` | ≥0.15 | **0.132** FAIL (narrow, same near-miss band as seed1) |
| `gram_condition` | ≤15 | **11.2** PASS |
| `gram_logdet` | ≥−1.15 | **−1.28** FAIL (narrow) |
| relative purity | hold **12/12** | **4/12** FAIL (8 dehydron-majority structures blurred) |
| nearest-pair d | ≥0.10 | **0.314** PASS |

**Verdict: `AXIS_SCRAMBLED_BY_GRAM`** (purity regress on the clean seed). Automated jsonl bucket still said `GRAM_COND_NO_MOVE` because bank floors also missed — the **decisive** signal is purity **12/12→4/12**, which under frozen escalation forbids coeff rematch.

#### Decision — no λ=0.005 rematch, including seed1-only (locked 2026-07-15)

Question revisited after monopole remeasurement: with absolute share out of scope, is a **seed1-only** rematch still justified to finish clearing the bank near-miss?

**No.** Reasons:

1. **Seed1 already showed scope failure at λ=0.001.** Bank moved hard (eig_min 0.032→0.129, cond 49.7→11.0); relative purity stayed **0/12**. Clearing the last 0.02 on `eig_min` does not address the dilution target.
2. **Seed2 showed purity hostility at the same λ.** The clean seed lost relative purity (**12/12→4/12**). Frozen rule: `AXIS_SCRAMBLED_*` → **no** coeff rematch / redesign.
3. **Seed1-only does not escape that evidence.** Rematch was authorized for coefficient near-miss *when purity held*. Purity did not hold on seed2, and did not move on seed1 — so the rematch would only chase a registered bank monitor, not dilution, while further risking commit (already 0.81→0.53 on seed1).
4. **Monopole closure does not revive Gram.** Removing a confounded co-metric does not make a purity-hostile bank hinge a better dilution fix.

**Standing:** Gram saturating logdet hinge is **closed as a dilution mechanism**. Durable lesson kept: full-bank conditioning is a real quantity nearest-pair misses, and seed1’s early Gram pathology remains a live *description* of the dilution mode — but this particular loss is not the fix. **λ=0.005 rematch is voided** (not “still authorized, deferred”).

**Next pre-reg lead (not drafted yet — park here):** seed1 Gram condition was already worse **early** (2.45 vs seed2 1.72 at early ckpt) before exploding by ge30 — an **initialization-sensitivity** finding, not only late-training pathology. Prefer trying **better-conditioned prototype-bank init** (e.g. explicit orthogonal init, or resample init until four prototypes clear a conditioning threshold **before** training) before another mid-training loss term. Scope: seed1 dilution / relative purity; must not regress seed2’s clean 12/12.

---

### Fresh pre-registration must distinguish (not “commit from nothing”)

| Outcome | Definition |
| ------- | ---------- |
| **`GROWS_REAL_NICHE`** | Committed tail (Hᵢ\<0.5) **grows in count** *and* retains dehydron-axis separation on the inheriting experts — τ contrast comparable to route_v1 committed reference (~0.97 vs ~0.00), not diluted toward 0.5/0.5; large ρ gap preserved (order 5 vs 29). **For the SASA-enrichment run, use the SASA-specific companions** `GROWS_REAL_NICHE_TWIN_COMMIT` / `GROWS_REAL_NICHE_CLEAN_EXPOSURE_SPLIT` above. |
| **`DIFFERENT_AXIS_OR_DIFFUSE`** | Commitment grows in raw count (or mean max-p rises) but axis separation **weakens or disappears** — a way to sharpen routing that abandons the one axis already known to be real. |
| **`AXIS_HELD_SOFT_NO_HARD_GROWTH`** | Soft structure still shows the dehydron axis, but frac Hᵢ\<0.5 does not grow (L1/L2-like). |

---

## Two mechanisms — do not conflate (G1 vs missing v3 shaping)

Flat aleatoric on v6 has **two independent explanations**. The retrain plan must
address **both**, not treat recovering v3's recipe as a substitute for decoupling.

| Mechanism | What it is | Gate | Fix direction |
| --------- | ---------- | ---- | ------------- |
| **Loss-surface coupling** | NIG regularizer `\|y−μ\|·(2ν+α)` entangles ν and α in the objective | **G1** (resolved) | Decorrelation + head split; may need loss reform |
| **Missing aleatoric supervision** | v6 dropped v3's `var_penalty`, `aleatoric_hinge`, and `gaussian_likelihood` ρ residual term | **G4** (required before v3 recipe port) | Explicit shaping **with holdout** — see G4 |

v4 confirms G1 stands alone: same coupled head, pure NIG, no aleatoric shaping —
aleatoric was not engineered to be informative there either. v3's aleatoric
*looked* usable because of **direct supervision on the dehydron mask**, not because
the NIG head magically identified aleatoric under vanilla DER.

**Do not** draft `p4_v3_aleatoric_recovery` until G4 passes and the primary-loss
philosophy is chosen (see `loss_philosophy_options()` in `nig_identifiability.py`).

---

## Pre-retrain gates (cheap, before GPU spend)

| Gate | Question | Status | Resolution |
| ---- | -------- | ------ | ---------- |
| **G1** | Is epi/ale coupling architectural or loss-level? | **Resolved** | **Loss-level.** NIG regularizer couples ν and α. `DecoupledEvidentialHead` alone is **not sufficient**. See `analyze_nig_loss_coupling()`. |
| **G2** | Are uncertainty probes alive (not flat-std)? | Open at retrain | Require P7 + `nu_cv` on epoch health; route_v1: epi alive, ale not informative. |
| **G3** | Is **epistemic** semantics emergent or B-factor/SASA Goodhart? | **Run — re-eval under G4a** | A/B decorr-only vs full epistemic decoupling. `g3_supervision_circularity_report()`. |
| **G4a** | Is **P8** meaningful (not sign-only on flat ale)? | **Required — before G4** | Relative τ lift + informative aleatoric std; see below. |
| **G4** | Is **P8** independent of dehydron-mask aleatoric shaping? | **After G4a** | Held-out residue split for `var_penalty` / `aleatoric_hinge`; P8 on holdout only. |
| **G5** | How much of v6 **P7** is v3-teacher hand-me-down vs v6-native? | **Required before crediting S6** | Provenance audit: distillation path, r(epi,SASA) vs teacher. |

### G3 — B-factor/SASA epistemic circularity gate

**Risk:** `epistemic_decoupling_loss` trains epistemic toward B-factor/SASA. Then P8
(τ aleatoric) and P11 (OOD epistemic on 1PGB) partially measure **supervised
regression fit**, not emergent "model knows what it doesn't know."

**Domain hypothesis (must be stated explicitly):** Epistemic *should* correlate
with crystallographic flexibility / exposure proxies **only if** that is an
independent scientific claim — not because it makes P8/P11 pass.

**Required A/B (two short Phase 4 runs or checkpoint pair):**

| Variant | Config | `epistemic_decoupling_coeff` | `epi_ale_decorrelation_coeff` |
| ------- | ------ | ---------------------------- | ----------------------------- |
| **A** (G3 ablation) | `p4_head_decouple_decorr_only_phase_config` | **0** | 0.75 |
| **B** (full) | `p4_head_decouple_phase_config` | 1.0 | 0.75 |

Compare with `g3_supervision_circularity_report()`:

- **G3 pass:** P8/P11 do **not** pass only in variant B while failing in A.
- **G3 fail:** P8 or P11 passes in B but not A → uncertainty is downstream of
  direct supervision; downgrade claims to "B-factor/SASA regression head," not
  emergent epistemic.

**Reported metrics (not trained on variant A):** `probe_r_epi_sasa`, `probe_r_epi_bf`
(if available), P8 lift, P11 contrast — track in MLflow, do not use as loss on A.

**G3 result (2026-07-08, 12-epoch ablations, pre-G4a P8):** `g3_supervision_circularity_report`
reported `g3_pass=true` with P8 in both A and B — **inconsistent** with aleatoric
std≈0.006 (below informative floor). **Root cause:** legacy P8 used `lift > 0` only
(`TAU_ALE_LIFT_MIN=0`), so direction passed on noise. **Do not treat as clean G3 pass
for P8.** Re-evaluate G3 epistemic circularity on P11/S6 legs; re-run P8 claims only
after **G4a** (below).

---

### G4a — P8 magnitude gate (prerequisite for G4)

**Problem:** P8 can pass while aleatoric is flat — a tiny positive `(ale_τ − ale_non)`
on std(ale)≈0.006 is not biophysical elevation, it's sign noise (same failure family
as the original flat-std incident).

**Fix (implemented in `tau_boundary_aleatoric_elevation`):**

1. Strata: **continuous ρ** (`|ρ − TAU| ≤ band`) — not `tau_flag` (verified in code).
2. **Informative aleatoric:** `std(ale) ≥ 0.05` required before P8 `ok`.
3. **Relative lift:** `(ale_τ − ale_non) / std(ale) ≥ 0.20` (G4a floor; tune with corpus).

Analytical ref: `g4a_p8_magnitude_gate()` in `nig_identifiability.py`.

**G4a pass on route_v1 / G3 checkpoints:** expected **fail** until aleatoric is
actually informative — this resolves the G3 P8 inconsistency.

---

### G4 — Aleatoric shaping holdout gate (dehydron-mask circularity)

**Risk (same pattern as G3, aleatoric side):** v3's `var_penalty` and
`aleatoric_hinge` use `regularity_mask = 1 − target_dehydron` — i.e. the ρ/TAU
label itself. P8 checks aleatoric elevation near ρ≈TAU. If shaping trains aleatoric
low on non-dehydrons and free on dehydrons **on the same residues P8 evaluates**,
P8 passing proves the hinge converged, not that the model discovered biophysical
ambiguity at the rim.

**Holdout granularity (frozen):** default **`protein`** — entire structures excluded
from shaping (`corpus_protein_holdout_ids`, ~20% of Stage A proteins, stable hash).
Alternate `residue_stratified` is within-protein interpolation only (weaker claim;
adjacent residues share context). Constants: `G4_DEFAULT_HOLDOUT_MODE` in
`aleatoric_shaping_holdout.py`.

**G4 pass — ALL required (frozen before `--p4-v3-aleatoric-shaping` runs):**

1. **Holdout P8 (G4a-hardened):** `tau_boundary_aleatoric_elevation(holdout_rows)`
   passes — relative lift `(ale_τ − ale_non) / std(ale) ≥ 0.20` **and**
   `std(ale) ≥ 0.05` on holdout residues only.
2. **No mask memorization:** NOT (`p8_full.ok` AND NOT `p8_holdout.ok`). **Vacuous when
   full P8 fails** — report as `vacuous_full_p8_fail`, not a pass.
3. **Transfer ratio:** `holdout_relative_lift / full_relative_lift ≥ 0.70`
   (`G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN`) **only when** full-corpus P8 is informative
   (`ale_std ≥ 0.05`) **and** `full_relative_lift ≥ 0.20`. Otherwise
   `not_evaluable_sub_threshold` — ratio is noise÷noise and must not count as pass.
4. **ρ coupling report (required in eval, advisory for pass):** compute
   `r(aleatoric, ρ)` on corpus and holdout. If `|r| ≥ 0.85` →
   `aleatoric_rho_reparameterization_risk` — G4 P8 may pass but do **not** cite
   aleatoric as independent biophysical signal (same failure shape as epistemic G5b).

**G4 fail examples:**

| Pattern | Verdict |
| ------- | ------- |
| Full P8 ok, holdout P8 fail | Mask memorization |
| Holdout P8 ok, transfer &lt; 0.70 | Partial memorization |
| Holdout P8 fail (flat ale) | Shaping did not generalize |
| Transfer ratio with flat full P8 | **Not evaluable** — do not count as partial pass |
| Memorization with both P8 fail | **Vacuous** — not a pass |
| G4 pass + \|r(ale,ρ)\| ≥ 0.85 | P8 credit with ρ-reparameterization caveat |
| Eval on `checkpoint_eligible=false` | Diagnostic only — do not certify recipe |

**Pre-recipe-change diagnostics (required on failed shaping runs):**

- **Routing entropy** vs established 1.28–1.33 band — confounds aleatoric read if unstable.
- **Dehydron mask split** (`aleatoric_dehydron_stratification_report`) — if dehydron
  residues are also near-floor, suspect global penalty / mask bug before tuning coeffs.
- **`r(epi,ale)`** only interpretable when `ale_std ≥ 0.05`; near-zero r on collapsed ale
  is mechanical, not decoupling evidence.

Constants: `G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN`, `G4_TRANSFER_RATIO_FULL_REL_LIFT_MIN`, `G4_ALE_RHO_MARGINAL_PROXY` in
`evidential_validation.py`; report: `g4_aleatoric_shaping_holdout_report()`.

**Thin holdout (n=12 corpus, 20% protein holdout):** default seed holds out **one**
protein (e.g. 1F88). Report includes `holdout_corpus_contrast` (fold_id, dehydron
fraction, ρ mean vs corpus z-scores) and `thin_holdout_warning`. Eval-only rotation:
`--holdout-seeds 42,7` — training mask is fixed at seed 42; extra seeds test whether
verdict is holdout-protein-specific. Multi-seed consensus in `multi_seed_consensus`.

**A/B variant (mirrors G3 spirit):**

| Variant | `var_penalty` / `aleatoric_hinge` | P8 eval set |
| ------- | --------------------------------- | ----------- |
| **A** | Off (or decorr-only P4 baseline) | Holdout |
| **B** | On train mask only | Holdout |

**G4 pass (A/B):** B improves holdout P8 vs A under the frozen rule above.

Implementation: `science/training/aleatoric_shaping_holdout.py`; training via
`p4_v3_aleatoric_shaping_phase_config()` + `--p4-v3-aleatoric-shaping`; eval via
`make eval-g4-holdout`.

**Mask audit (2026-07-08, p4_v3 run):** `target_dehydron` is binary `{0,1}` and
matches `data.x[:,1]` (0 mismatches). Symmetric collapse is **not** soft-mask or
field-drift — `var_penalty` on ~42% regular residues updates **shared**
`DecoupledEvidentialHead.ale_trunk` weights that affect all residues.

**Rejected fix:** stop-gradient on dehydron path — var_penalty already excludes
dehydron residues; coupling is through shared **parameters**, not activation flow.

**w_var_penalty coefficient sweep (before architectural change):**

Frozen probe grid: `G4_POPULATION_SEPARATION_WVP_WEIGHTS` = `{2.8, 1.0, 0.3}`.
Short probes only (`G4_WVP_EPOCHS` default 4). Launch: `make g4-var-penalty-sweep`.

Frozen separation pass (pre-registered, not post-hoc):

1. `global_ale_std ≥ 0.05` (same informativeness floor as G4a/P8).
2. `|μ_dehyd − μ_regular| / σ_pooled ≥ 2.0` (`G4_POPULATION_GAP_STD_MULT_MIN`).

Report: `aleatoric_population_separation_report()`; aggregate:
`checkpoints/v6/diagnostics/g4_wvp_sweep_report.json`.

**Sweep results (2026-07-08, resume `slim_moe_route_v1/v6_best.pt`, 4 epochs/probe):**

| `w_var_penalty` | `global_ale_std` | `gap/σ_pooled` | `separation_ok` |
| --------------- | ---------------- | -------------- | --------------- |
| 2.8 | 0.0086 | 1.71 | fail |
| 1.0 | 0.0087 | 1.71 | fail |
| 0.3 | 0.0086 | 1.71 | fail |
| *(route_v1 baseline, no probe)* | 0.0121 | 1.75 | fail |

Verdict: **`partial_gap_at_low_weight`** — gap is not near-zero (~1.7× pooled σ) but
frozen criterion fails on **global σ ≪ 0.05** at every weight; 4-epoch probes barely
move population stats vs baseline. `w=0.3` did cut training `v3_aleatoric_shaping`
loss (~10→~5), confirming coefficient wiring after config fix.

**Single close-out isolation gate (pre-registered):**

- Run exactly one shaping-only isolation training (`make train-v6-p4-g4-shaping-only-isolation`)
  with uncertainty-head-only updates and non-G4 losses zeroed.
- Evaluate with `make eval-g4-holdout G4_CKPT=checkpoints/v6/runs/<run_id>/v6_phase4_12prot.pt`.
- Confirmation criterion (same frozen metric family): `global_ale_std ≥ 0.05` and/or a
  material increase in `gap/σ_pooled` vs route baseline; otherwise park uncertainty.
- Scope cap: isolation run + at most two follow-ups (loss reintroduction probes).

| Sweep verdict | Meaning | Next step |
| ------------- | ------- | --------- |
| `magnitude_sufficient` | ≥1 weight passes frozen separation | Tune weight; no trunk split |
| `partial_gap_at_low_weight` | Gap opens but criterion not met | Extend grid down or mild calibration |
| `structure_likely_required` | Gap ~0 at all weights | Minimal per-population affine on ale output |

**Architectural fallback (only if sweep rules out magnitude):** learned
scale/shift on aleatoric output conditioned on dehydron flag **after** shared
trunk — not duplicate `ale_trunk`, not stop-gradient.

---

### G5 — Epistemic provenance gate (v3 distillation vs v6-native)

**Risk:** v3 epistemic was largely a **SASA proxy** (SASA in `x[:,3]`, coupled head,
`epistemic_temp_scaling=2.8`). v6 distills **depth + epistemic only** from the v3
teacher (`experiments/training/v6/v2_teacher.py`) — not aleatoric. A passing P7 on
route_v1 (epi std ≈ 4.4) may credit **inherited proxy signal**, not v6-learned DER
exposure gap.

**Required before treating P7 / S6 epistemic as a v6/v7 finding:**

1. On Stage A corpus, compute `r(ν_epi, SASA)` and `r(ν_epi, SASA | ρ)` (partial) on
   v6 checkpoint vs frozen v3 teacher on the same graphs.
2. Report **teacher–student epistemic correlation** per structure (distillation alignment).
3. If `r(epi, SASA) ≥ 0.85` and teacher–student `r ≥ 0.80`, flag
   `epistemic_provenance=distilled_proxy` — downgrade "emergent epistemic" claims.
4. **G5 pass:** Either (a) partial r(epi,SASA|ρ) drops materially vs marginal, or
   (b) epistemic on structures **never** in teacher cache diverges from teacher with
   high spread (native learning signal).

**G5 does not block retrain** — it calibrates how much credit S1–S6 give to epistemic
vs routing / disc outcomes.

**Pre-committed credit rule if `distilled_proxy` flags** (marginal r(epi,SASA)≥0.85
AND teacher–student r≥0.80):

| Criterion | Credit |
| --------- | ------ |
| **S1** routing, **S4** hyperbolic MP, **S5** edge telemetry | Full weight |
| **S6** and epistemic-based production claims | **Asterisk only** — wording: *"informative but largely SASA-proxy inherited via v3 teacher distillation — not v6-native DER exposure discovery."* Do **not** count S6 epistemic leg toward production uncertainty gates without native corroboration. |

Checklist: `g5_epistemic_provenance_checklist()` / `g5b_rho_feature_proxy_checklist()` in
`nig_identifiability.py`.

### G5b — ρ feature proxy (extends G5)

ρ is a **direct input** (`x[:,0]`). High `|r(epi,ρ)|` may indicate epistemic is a monotonic
reparameterization of a feature the model already sees — same failure shape as SASA, different
confound.

**Required metrics (folded into `g5_epistemic_provenance_report`):**

1. `|r(epi,ρ)|` marginal on Stage A corpus.
2. Pinned OOD **1PGB**: raw in-corpus vs OOD mean(epi) ratio (P11).
3. Same contrast on epistemic **ρ-residualized** (fit `epi ~ ρ` on in-corpus only).
4. `r(student, teacher | ρ)` — distillation alignment beyond ρ.

**Flag `rho_feature_proxy` if:** `|r(epi,ρ)| ≥ 0.85` **and** raw OOD epistemic elevation
**collapses** after ρ control (`ood_separation_collapsed_after_rho`).

**Teacher–student bootstrap:** resample Stage A proteins with replacement; report CI for
`r(stu,tea)`. If CI straddles 0.80 → `teacher_student_borderline` (not a robust non-trigger).

**G5 route_v1 result (2026-07-08):** SASA-distillation ruled out (|r(epi,SASA)|=0.51,
partial≈0). **Epistemic novelty blocked:** r(epi,ρ)=0.95; pinned OOD 1PGB mean epistemic
**0.90× in-corpus** (inverted, not merely flat); ρ-residual OOD ratio ≈0.98. Teacher–student
r=0.807 with bootstrap CI [0.791, 0.821] straddling 0.80 (n=12 proteins — CI may understate
true spread). **S6 epistemic credit for route_v1:** epistemic is ρ-dominant and shows no OOD
elevation (raw or ρ-residualized) on the pinned OOD test — **do not cite epistemic uncertainty
for novelty/review-flagging claims until this is independently resolved.** G3 ablations break
teacher lock-in (r≈0.59) but do not fix OOD inversion.

**Product consequence (CLOSED 2026-07-16):** Viewer “Investigation” default must not use
evidential `ale×(1−epi)`. See `docs/audit/VIEWER_INVESTIGATION_CORRECTNESS.md` — render
invert fixed; default grounded in ρ/τ physics; evidential kept as experimental overlay.

---

### Primary-loss philosophy (decide before Phase 4 recipe port)

v3 and v6 use **different objectives** for the same outputs. Pick one explicitly;

| Mode | Primary objective | Aleatoric meaning | Epistemic story |
| ---- | ----------------- | ----------------- | --------------- |
| **NIG** (v6 default) | `evidential_regression_loss` | NIG-derived `β/(α−1)` (clamped) | `(1/ν)·temp` — DER exposure |
| **Gaussian** (v3 teacher) | Heteroscedastic NLL on ρ | **Direct variance** fit to ρ residual | Still `(1/ν)·temp` but aleatoric is not NIG-derived |
| **Blended** | NIG + auxiliary `gaussian_likelihood` + shaping (G4 holdout) | Hybrid — document which head output is canonical | Requires G1 + G4 |

**Blended is not "free":** auxiliary gaussian_likelihood makes aleatoric a supervised
heteroscedastic variance — legitimate, but **not** what P9/P11 were written to
validate under pure DER. If blended becomes primary, update P9/P11 interpretation in
`EVIDENTIAL_UNCERTAINTY.md`.

Analytical reference: `loss_philosophy_options()` in `nig_identifiability.py`.

---

## Residue-first aleatoric doctrine (2026-07-08)

**Aleatoric uncertainty is not a corpus-level property.** The NIG head models
genuine molecular ambiguity at each residue / dehydron site — local probability mass
that remains accessible even in the fully observed conformational ensemble. Averaging
or thresholding aleatoric across the corpus erases the spatial signal needed for
druggability and active learning.

From the manifold perspective, informative aleatoric should **concentrate at the rim**
of the Poincaré disc (high `disc_r`, under-wrapped / high-entropy regions), while
stable core residues sit deeper in the cone with low aleatoric. A global
`std(ale) ≥ 0.05` gate smears rim signal across the bulk and systematically
under-flags proteins with one or two critical sites while over-penalizing flexible
constructs.

### Site decisions (local only)

**Investigate** when **all** hold on a residue:

| Signal | Rule | Rationale |
| ------ | ---- | --------- |
| High aleatoric | `ν_ale > t_ale` (default: corpus **P90**) | Local ensemble ambiguity |
| Rim-localized | `disc_r` ≥ corpus P75 | Under-wrapped / rim geometry |
| Low clustering | graph `clustering` ≤ corpus P25 | Not in a dense topological core |
| (optional) Low routing confidence | `expert_routing_max` ≤ P25 | Expert ambiguity |

Implementation: `flag_investigation_sites()` in
`science/training/aleatoric_residue_diagnostics.py`. CLI:
`make aleatoric-corpus-diagnostics`.

**`t_ale` resolution:** default is corpus-relative **P90** (`resolve_t_ale()`), not
the training-health absolute `0.05` floor. Override with `--t-ale` (absolute) or
`T_ALE_PERCENTILE=95` on the Makefile target. Use absolute `0.05` only when head
output is on that scale.

### Global summaries (monitoring / triage only)

| Diagnostic | Purpose | Not for |
| ---------- | ------- | ------- |
| Residue-level aleatoric histogram | Calibration shape — expect **right-skew** (bulk low, long tail) | Site pass/fail |
| `std(ale)` corpus-wide | Head collapse / narrow calibration detector | Druggability certification |
| Per-protein `fraction(ale > t_ale)` | Validate ~0.12–0.20 vs known flexible sites | Global gate |
| Per-protein `max(ale)` ranked | **Active-learning acquisition priority** | Mean-ale ranking |
| Intra-protein `var(ale)` | Conformational heterogeneity (e.g. EGFR L858R) | Corpus mean |
| Dehydron high-ale burden table | Unwrapped-site count per structure | Global threshold |
| Family-level means (when metadata present) | Kinase vs multi-domain separation | Until corpus diverse enough |

**Expected global signatures when well-calibrated on a diverse corpus:** right-skewed
residue distribution; kinase domains lower mean with localized activation-loop spikes;
multi-domain / disordered constructs higher mean and dehydron burden. Current Stage A
corpus is too narrow to show clean family separation — treat absence as a **data**
limit, not proof the head cannot discriminate.

### Aleatoric independence probe (minimal)

**Question:** Does ν_ale vary independently of ρ and expert assignment after controls?

**Not a training gate** — run before crediting site-level aleatoric claims.

| Stat | Meaning |
| ---- | ------- |
| `r(ale, ρ)` marginal | Dehydron-density proxy risk |
| `r(ale, ρ \| expert)` partial | ρ coupling beyond routing |
| `η²(expert)` / `η²(expert \| ρ)` | MoE assignment explains aleatoric? |
| `R²(ale ~ ρ + expert)` | Combined proxy explainability |
| `residual_std_ratio` | `std(resid) / std(ale)` — independent variance left |
| Within-expert `r(ale, ρ)` | ρ monotonicity inside each expert |

**Verdicts** (`aleatoric_independence_probe()`):

| Verdict | Meaning |
| ------- | ------- |
| `not_yet_meaningful` | CV too low — probe cannot decide (route_v1 expected) |
| `rho_proxy` | Aleatoric tracks ρ |
| `expert_proxy` | Aleatoric tracks routing |
| `proxy_fully_explained` | ρ + expert absorb variance |
| `independent_signal_candidate` | Residual spread + weak partials — warrant site follow-up |
| `ambiguous` | Borderline — extend corpus or add geometry |

CLI: `make aleatoric-independence-probe` (optional `INCLUDE_GEOMETRY=1`).

### Impact on frozen gates

| Gate / metric | Old role | New interpretation |
| ------------- | -------- | ------------------ |
| `global_ale_std ≥ 0.05` (G4a, S6, sweep) | Site informativeness certification | **Training collapse monitor** — retain for save health, demote for biology |
| `aleatoric_population_separation_report` | Sweep pass/fail | Shaping coefficient diagnostic — not druggability |
| P8 τ lift on corpus | Aleatoric biology | Residue-stratified holdout P8 on **high-ale sites** + local rim check |
| G4 holdout memorization | Still valid | Independent of global σ floor |

`global_aleatoric_health_monitor()` wraps corpus `std(ale)` with
`not_a_site_gate=True`. Do **not** block structure onboarding or site reports on
corpus mean aleatoric alone.

---

## Success criteria S1–S6 (retrain outcomes)

| ID | Criterion | Pass condition |
| -- | --------- | -------------- |
| S1 | Routing | H ≤ save ceiling; min_routing_fraction ≥ 0.05 (inference) |
| S2 | Disc occupancy | σ₂/σ₁ within corpus band; thickness floor |
| S3 | Topology depth | r(d,τ) ≥ P_DEHYDRON floor (MASTER lineage) |
| S4 | Hyperbolic MP graph | Structural SSOT + hyperbolic edges **during training** |
| S5 | Edge telemetry alive | `track/edge_telemetry_alive = 1` on Stage A corpus |
| **S6** | **Uncertainty calibration** | **Joint — see below** |

### T1 — Trunk hidden-state rank (leading root-cause gate, 2026-07-14)

**Not S2 and not S3.** Official **S2** = 2D disc occupancy; **S3** = topology depth. Keep **T1a** and **T1b** as separate lines — they need different fixes.

| ID | Path | Object |
| -- | ---- | ------ |
| **T1a** | `raw_x → node_emb(=pre_mp) → encoder_h` | Euclidean trunk. **`pre_mp` = `nn.Linear(4→128)` output, not raw features.** |
| **T1b** | `topo_features → topo_tangent` | Topology-only gate hand-board |

#### Cross-lineage trunk collapse (confirmed)

| Checkpoint | encoder_h ER | top-1 |
| ---------- | ------------ | ----- |
| route_v1 | ≈1.02 | ≈0.98 |
| Fix-1 cold_to20 | ≈1.02 | ≈0.98 |
| Fix-1+S4 seed-1 | ≈1.02 | ≈0.98 |

Not Fix-1/S4-specific. Artifact: `checkpoints/v66/diagnostics/trunk_rank_cross_lineage/`.

#### Origin audit (Fix-1+S4; `t1_trunk_origin_fix1_s4/` + `t1a_znorm_forward_fix1_s4/`) — 2026-07-14

| Check | Result |
| ----- | ------ |
| Scope of `pre_mp` | Post-`nn.Linear(4→128)` embed |
| **τ definition** | `tau_flag = 1{ρ < TAU=13}` — **exact match 100%** on Stage A-12; near-duplicate of ρ, not an independent measurement |
| raw_x unnormalized ER | **1.017** (SASA variance share **97.8%**) |
| raw_x z-scored ER | **1.707** — **this is the Linear-rank ceiling after scale fix**, not ~4 |
| z-board with \|ρ−TAU\| instead of τ_flag | ER **2.334** (anti-redundancy headroom) |
| `node_emb.weight` ER | ~3.2 (Linear itself OK) |
| `pre_mp` init → ep20 | **1.015 → 1.014** (**ALREADY_COLLAPSED_AT_INIT**) |
| Fwd-only z-norm (no train) | `pre_mp` **1.014 → 1.774** (`PRE_MP_NEAR_PREDICTED_CEILING`); `encoder_h` **1.552** (`MP_PARTIAL_LOSS` — not crush-back to ~1.0) |
| MP null (T1c) | z-norm relative drop ~**12%** ≤ synth rank-4 ~**24%** → **`MP_LOSS_CONSISTENT_WITH_NULL`** (no separate MP bug / no T1c line). Artifact: `t1c_mp_relative_loss_null_fix1_s4/` |

**Important:** the fwd-pass ranks are on **ep20 weights trained against broken (scale-dominated) inputs**. They show the trunk is not architecturally doomed to ER≈1, but they are **not** the training ceiling. The decisive experiment is a **cold retrain with z-norm from init**.

**T1a expected after scale fix (board / Linear ceiling):** rank ~**1.7–2.0**, not ~4. Hitting that range means the *input* fix worked; richer rank after cold train would mean weights can unlock more than the swapped-inference snapshot.

**T1b (separate — training dynamics):** `topo_features` stays ~1.70→1.62 while `topo_tangent` **degrades 1.78→1.02 over ep1→ep20**. Init-flat for T1a; **collapses during training** for T1b. Do not group as “also thin.” Needs its own loss/feedback investigation.

**Policy:** (T1a) commit input normalization with ceiling ~1.7 documented; optionally drop/replace binary τ. (T1b) dynamics collapse — separate root cause. Tools: `make audit-t1-trunk-origin …`, `make audit-t1a-znorm-forward CHECKPOINT=...`.

#### Pre-registered T1a cold-retrain gates (frozen 2026-07-14 — **before** z-norm Stage A-12 retrain)

**Config under test (controlled first run):** Fix-1 + S4 + **`--input-feature-zscore` only** (no `|ρ−TAU|` yet). Same Stage A-12 corpus / epochs / LR as `fix1_s4_stage_a12_cold_v1`. Optional `|ρ−TAU|` is a **second** lever after z-norm-only is judged.

**Fwd-pass reference (not a training claim):** ep20 weights + z-norm at inference → `pre_mp≈1.77`, `encoder_h≈1.55`.

Judge at **global epoch 20** on Stage A-12 corpus-pooled ranks + train soft loads / H (same sources as Fix-1+S4 usage gates). Log epoch-by-epoch in `t1a_trunk_rank_per_epoch.jsonl` (and `fix1_gates_per_epoch.jsonl`).

| Verdict | Required | Meaning |
| ------- | -------- | ------- |
| **T1A_RANK_MET** | `pre_mp ER ≥ 1.60` **and** `encoder_h ER ≥ 1.50` at ge20 | Cold train at least recovers the forward-pass lift (norm is live under training dynamics) |
| **T1A_RANK_SURPASSED_FWD** | `encoder_h ER ≥ 1.70` at ge20 (implies MET if `pre_mp` also ≥ 1.60) | Training unlocked more than inference-swap on adapted weights — evidence weights were previously locked to degenerate inputs |
| **ROUTING_RESPONDED** | **T1A_RANK_MET** **and** at least one of: `H ≤ 1.32` **or** `max_soft_share ≥ 0.33` **or** `max−min ≥ 0.08` | Rank move co-moved routing off the stuck 1.28–1.37 / near-flat band — minimum bar to keep chasing T1a as the routing root cause |
| **T1A_ROOT_CAUSE_FOR_ROUTING** | **T1A_RANK_MET** **and** existing **USAGE_MOVED** (frozen: max≥0.38 ∧ spread≥0.12 ∧ H≤1.30 ∧ feature hold) | Only verdict that names z-norm as the actual fix for routing collapse / IBU |
| **T1A_RANK_ONLY** | MET or SURPASSED_FWD, but **not** ROUTING_RESPONDED | Norm fixed trunk rank; routing still needs another lever (commitment / T1b / …) — do **not** call T1a “the routing fix” |
| **T1A_FAILED** | `encoder_h ER < 1.50` at ge20 | Cold train did not retain the fwd-pass gain — re-open T1a (fit/install, checkpoint buffers, or train-time collapse) |

Notes:

- Do **not** treat ge20 `encoder_h≈1.55` (fwd-only) as success for the retrain — that number is the *baseline to beat or match under training*, not a destination.
- `|ρ−TAU|` (board ER≈2.33) is optional and **pre-registered as additive**: run only after z-norm-only judgment; success still uses the same routing tables above (rank floors may be revised upward after the first cold run if z-norm alone SURPASSED_FWD).
- T1b remains out of scope for this retrain’s claim set.

#### Cold retrain result — `fix1_s4_t1a_znorm_stage_a12_cold_v1` (2026-07-14)

**Config:** Fix-1 + S4 + `--input-feature-zscore` only (no `|ρ−TAU|`). Stage A-12, 20 ep, cold.

| Epoch | `pre_mp` ER | `encoder_h` ER | H | max / spread |
| ----- | ----------- | -------------- | - | ------------ |
| ep0 | **1.832** | **1.623** | — | — |
| ep1 | 1.832 | 1.591 | 1.386 | 0.250 / 0.001 |
| ep10 | 1.831 | 1.309 | 1.386 | 0.258 / 0.013 |
| ep20 | **1.832** | **1.470** | **1.384** | **0.270 / 0.029** |

**Frozen verdict:** **`T1A_FAILED`** (`encoder_h` 1.470 \< 1.50) — also **not** ROUTING_RESPONDED / USAGE_MOVED (IBU holds: H≈1.38, near-flat loads).

**Read (honest, pre-registered):**

1. **Input scale was the bind for `pre_mp`.** Corpus z-norm locks `pre_mp` at the predicted ~1.83 ceiling for the whole run (vs ~1.01 without). That part of T1a is confirmed under training, not only fwd-swap.
2. **Cold train did not beat the fwd-pass `encoder_h` snapshot** and did not hold `encoder_h` ≥ 1.50 to ge20: ep0 clears MET (~1.62), then **training compresses** toward ~1.30 mid-run and only partially recovers to 1.47. So the promising ep20 fwd-swap (~1.55) is **not** a training destination; something in the train loop still squeezes post-MP trunk rank.
3. **Routing did not move.** Rank lift at init / stable `pre_mp` was insufficient for H/load — do **not** claim T1a z-norm as the routing root-cause fix.
4. **Next (still within T1a/T1 family, not gate geometry):** treat post-MP / loss-driven squeeze as the residual (related to mid-run dip, adjacent to T1b dynamics). Optional `|ρ−TAU|` may lift the board ceiling but is unlikely alone to fix a train-time `encoder_h` squeeze that already starts above MET. Prefer diagnosing **what in the objective compresses `encoder_h` after ep3** before stacking another input transform.

Logs: `checkpoints/v66/runs/fix1_s4_t1a_znorm_stage_a12_cold_v1/{t1a_trunk_rank_per_epoch.jsonl,fix1_gates_per_epoch.jsonl,t1a_input_feature_norm.json}`.

#### Epoch-resolved dip / H coupling (same run — no new train) — 2026-07-14

Full per-epoch logs were already available (`t1a_trunk_rank_per_epoch.jsonl` + `fix1_gates_per_epoch.jsonl`). Joint report: `checkpoints/v66/diagnostics/t1a_dip_recovery_joint_fix1_s4_znorm/`.

| Phase | Epochs | `encoder_h` | H | max / spread |
| ----- | ------ | ----------- | - | ------------ |
| Init | 0 | 1.623 | — | — |
| Early soft decline | 1–3 | 1.591 → 1.556 | **1.3863** (flat) | 0.250→0.252 / ~0 |
| **Steep dip** | **4–7** | **1.468 → trough 1.303 @ ge7** | 1.3863→1.3858 (Δ≪0.001) | 0.253→0.259 |
| Partial recover | 8–19 | 1.314 → **peak 1.480 @ ge19** | 1.3857→1.3848 | crawl to 0.264 |
| End | 20 | 1.470 | 1.3843 | 0.270 / 0.029 |

`pre_mp` stays **1.831–1.833** every epoch (locked at z-norm ceiling).

**Coupling:** over ge1–20, `encoder_h` span **0.288** while H span **0.002**; `r(enc, H) ≈ +0.065`. During the dip, ΣΔenc ≈ −0.29 and ΣΔH ≈ −0.0005; during recovery, ΣΔenc ≈ +0.17 and ΣΔH ≈ **−0.0015** (H drifts *down* slightly while rank climbs). **Verdict: `RANK_H_TEMPORALLY_DECOUPLED`.** Partial rank recovery does not move routing — undercuts “trunk rank is *the* routing bottleneck” for this lineage; they need separately sized levers.

**Feedback-loop hypothesis (early gate preference → gradient concentration → trunk homogenization):** **`WEAKENED`**. Soft loads are already exact uniformity at ge1 (`0.250×4`) and never open a mild preference that then deepens while rank dips (max only crawls 0.250→0.270). No early imbalance to reinforce. Rank compression proceeds under an already-flat gate.

**Loss co-timing (descriptive, not yet causal):** steep `encoder_h` drop (ge4–7) coincides with rapid fall of `cone_consistency` / early `angular_diversity` and a sharp rise in `disc_line_thickness_rms` (geometry settling). Capacity / load-floor terms stay 0. Co-timing ≠ proof — next cheap step is targeted coeff ablations or early diversity aux only if a mechanism candidate is named; do not stack `|ρ−TAU|` or gate-geometry fixes on the strength of this dip alone.

#### Exact-ln(4) pin: dead-grad / balance-term / ep4-schedule audits (2026-07-14)

H on the z-norm cold run sits at **ln(4)−ε** from ge1 (`H−ln4 ≈ −9×10⁻⁷` at ge1; total span 0.002 over 20 ep). Artifact: `checkpoints/v66/diagnostics/t1a_gate_pin_ln4_audit/`.

| Check | Verdict | Evidence |
| ----- | ------- | -------- |
| **(1) Dead gate gradients** | **`RULED_OUT`** | Per-epoch `\|\|Δgate_trainable\|\|` ~0.06–0.20 (topo_encoder moves; comparable order to `convs.0`). Live backward on ep1/4/7/20: `gate_grad_l2` ∈ [0.008, 0.084], nonzero; `freeze_gate=False`. Not a NaN/dead-instrumentation story. |
| **(2) Explicit entropy / balance pin** | **`RULED_OUT`** | `routing_entropy` is **monitor-only** (not in the loss sum). `balance_coeff=0.015` multiplies **asymmetric** `capacity_loss = Σ relu(min_usage−f)²` — starvation-only; **`capacity_loss=0` every logged epoch** at IBU, so the balance term contributes nothing. `routing_load_floor_coeff=0`. In-forward overload penalty only fires for load > `capacity_threshold=0.4` — also inert at ~0.25. No term is *holding* H at ln(4). |
| **(3) Ep4 schedule trigger** | **`RULED_OUT`** | Phase 12 geom-angular-prior: `freeze_radial_epochs=0`, `coeff_ramp_epochs=0`, constant lr. No curriculum boundary at ge4. Geometry-loss co-moves (cone / thickness) are convergence dynamics, not a scheduled handoff. |

**Structural note (stronger than either ruled-out bug):** this lineage runs **`topology_only_gate=True`** (8-D physics board only; `x_tangent` / `encoder_h` **excluded** from the gate by construction — see `apply_master_cold_dehydron_config`). That makes the observed rank↔H decoupling *architecturally expected*, not a mystery: fixing trunk rank cannot move a gate that never reads the trunk. Softmax of near-flat expert logits + tiny `expert_bias` (~10⁻²) still lands at ln(4) while the topo encoder keeps updating — alive gate, undifferentiated routing signal.

**Implication:** separately sized lever is real, but the first lever to name is not “bigger commitment coeff” while capacity is already 0 — it is whether routing is allowed to see a non-collapsed discriminative signal (including whether topology-only remains the right bottleneck), not another trunk-normalization pass.

#### Topology-gate `logit_scale` sweep (forward-only, no retrain) — 2026-07-14

Before choosing "enrich board" vs "feed encoder_h," swept `softplus(logit_scale)` x{1,2,5,10} on frozen `fix1_s4_t1a_znorm_stage_a12_cold_v1` ep20 (`topology_only=True`). Artifact: `checkpoints/v66/diagnostics/topology_gate_logit_scale_sweep_t1a_znorm_ep20/`.

| x softplus | H | max / spread | mean max-p | USAGE letter | MI(hard,tau) | peak soft R2(tau) / R2(depth) |
| ---------- | - | ------------ | ---------- | ------------ | ------------ | ----------------------------- |
| 1 (~1.32) | 1.385 | 0.264 / 0.026 | 0.333 | **IBU_HOLDS** | 0.355 | 0.789 / 0.834 |
| 2 | 1.382 | 0.279 / 0.058 | 0.421 | **IBU_HOLDS** | 0.356 | 0.751 / 0.823 |
| 5 | 1.351 | 0.334 / 0.162 | 0.648 | **AMBIGUOUS** | 0.358 | 0.624 / 0.771 |
| 10 | 1.311 | 0.378 / 0.247 | 0.821 | **AMBIGUOUS** | 0.358 | 0.468 / 0.670 |

**Exact USAGE_MOVED vs x10 (do not round up):** frozen USAGE_MOVED needs `max>=0.38 AND spread>=0.12 AND H<=1.30 AND feature hold`. x10 has max=**0.378** (misses 0.38 by 0.002), H=**1.311** (misses <=1.30 by 0.011), spread=0.247 (pass), feature hold still true. Letter = **`AMBIGUOUS`**, not USAGE_MOVED.

**SCALE_LIMITED still holds** as a *mechanism* claim (board has separable scores; near-unity scale was flattening them). It is **not** yet a verified routing fix.

**Signal vs noise under sharpening (IBU effect-size family):**

| x | MI(hard,tau) vs x1 | peak soft R2(tau) vs x1 | note |
| - | ------------------ | ----------------------- | ---- |
| 2 | +0.001 | -0.037 | mild |
| 5 | +0.003 | **-0.164** | soft R2 drops; MI holds |
| 10 | +0.003 | **-0.321** | soft R2 drops; MI holds |

Hard-assignment MI to tau is rock-stable. Soft continuous R2 dilutes at aggressive scale.

**Soft-R2 null (before trusting R2 as a hard gate):** artifact `topology_gate_scale_soft_r2_null/`.

- Structure-free logits (permute experts / random dists): soft R2 already ~0 at x1 — no saturation drop to observe; mean-max-p still rises with scale (sharpening without structure).
- Clean synthetic informative board: soft R2 **holds or rises** under sharpening while MI stays maxed (2-expert logistic R2 = 1.0 at all scales).
- Noisy informative boards with real-ish x1 R2 (0.55–0.85): partial erosion exists (worst ~−0.07 at x5) but **none** reproduce the real ~−0.16 x5 drop.

**Read:** the real soft-R2 drop is **not** automatic softmax saturation alone — but soft R2 is still a worse primary gate than hard MI (MI stable under both real sharpening and all nulls; R2 moves for reasons that mix geometry and saturation). Demote soft R2 to **secondary diagnostic**; primary SIGNAL_HOLD = hard MI (+ existing USAGE feature-hold floor).

#### L1 result — `fix1_s4_scale_l1_stage_a12_cold_v1` (2026-07-14)

softplus init+floor **2.645**; Fix-1+S4+z-norm; Stage A-12; 20 ep.

| ge | H | max / spread | letter | MI(hard,τ) |
| -- | - | ------------ | ------ | ---------- |
| 0 | 1.386 | 0.251 / 0.002 | IBU_HOLDS | 0.161 (L1 scale); scale≈1 ablation MI=0.161 |
| 20 | **1.385** | **0.263 / 0.023** | **IBU_HOLDS** | **0.400** (≥0.90× ref → SIGNAL_HOLD) |

**Verdict: `SCALE_TRAIN_IBU`.** Inference ×2 did not transfer under training. Ladder → **L2** (softplus ≈6.61).

#### L2 result — `fix1_s4_scale_l2_stage_a12_cold_v1` (2026-07-14)

softplus init+floor **6.613** (×5); same recipe as L1.

| ge | H | max / spread | letter | MI(hard,τ) |
| -- | - | ------------ | ------ | ---------- |
| 0 | 1.386 | 0.252 / ~0 | IBU_HOLDS | 0.064 |
| 20 | **1.386** | **0.260 / 0.015** | **IBU_HOLDS** | **0.644** (SIGNAL_HOLD) |

**Verdict: `SCALE_TRAIN_IBU`.** Ladder **stops** — no ×10. Scale alone does not move usage under training despite inference-time SCALE_LIMITED; next named levers are board enrichment (`|ρ−TAU|`) or gate input policy (what the topology-only gate can see), under a **new** pre-registration.

#### Named finding — two halves of scale (do not conflate)

| Half | Claim | Status |
| ---- | ----- | ------ |
| **A. `SCALE_LIMITED` (frozen)** | On a trained flat checkpoint, raising softplus exposes latent board separation (H/load move at inference). | **True** — verified on ep20 freeze |
| **B. Training re-flattens** | Cold train from the same higher softplus (L1×2, L2×5) returns ge20 to H≈1.386 / max≈0.26 — indistinguishable from scale-naive runs. | **True** — both ladder steps `SCALE_TRAIN_IBU` |

Training is not limited by underscaled softmax; it **does not seek** the inference-time separation. Balance/capacity/floor were already inert — the attractor is elsewhere, but not “scale too low.”

#### L1/L2 pre-softmax logit variance (2026-07-14)

Asked: does training *shrink* gate logit spread (active suppression), or hold/grow while H stays flat? Artifact: `checkpoints/v66/diagnostics/scale_train_logit_variance_l1_l2/`.

Note: logged `routing_entropy` = **H(mean soft load)** — `-(mean_n scores · log mean_n scores)` — **not** mean per-residue entropy.

| Run | ge1 row-var | ge20 row-var | rel | ge20 mean max-p | ge20 H(mean load) |
| --- | ----------- | ------------ | --- | --------------- | ----------------- |
| L1 (×2) | 0.00027 | **0.279** | ×1014 | 0.376 | 1.384 |
| L2 (×5) | 0.0014 | **1.014** | ×734 | **0.602** | 1.382 |

**Verdict: `LOGIT_VAR_GROWS_BUT_H_FLAT`.** Logit spread is **not** being suppressed — it grows by ~10³ while population H stays at ln(4). Per-residue soft also sharpens (mean max-p↑). Flat H/max_share is residue-level specialization **cancelling in the corpus mean load** (classic IBU usage side), not a vanishing-logit / entropy-regularizer attractor.

**Hold — refine before calling it IBU.** Per-residue softmax entropy (not logit row-var) for L2 vs prior IBU: artifact `checkpoints/v66/diagnostics/per_residue_routing_entropy_l1_l2/`.

| Run | H(mean load) | mean max-p | **median Hᵢ** | frac Hᵢ\<0.5 | hard frac |
| --- | ------------ | ---------- | ------------- | ----------- | --------- |
| fix1_s4 ge20 (IBU ref) | 1.384 | 0.361 | **1.300** | **0%** | skewed (~0.03/0.40/0.55/0.02) |
| L1 ge20 | 1.384 | 0.376 | **1.253** | **0%** | skewed |
| L2 ge20 | 1.382 | **0.602** | **0.990** | **0%** | **near-balanced** (~0.22/0.30/0.20/0.28) |
| L2 ge01 | 1.386 | 0.262 | 1.386 | 0% | — |

L2 histogram: **0%** mass below Hᵢ=0.5; **55%** in 0.8–1.0, **45%** in 1.0–1.2 — uniformly moderate, not near-0. Equal-tail math: max-p=0.60 → H≈1.11 if the leftover 0.40 is spread — matches the median. **Rejects** “confident 4-way partition with balanced class sizes” (that needs Hᵢ clustered near 0).

**Shape: `SEMI_COMMITTED_MODERATE_ENTROPY`.** Distinct from early-sprint IBU soft (Hᵢ~ln4) *and* from confident-balanced specialization. Corpus H(mean load) still cannot tell those apart — L2 proves why median Hᵢ / frac Hᵢ\<0.5 must sit beside H in S1/USAGE reads. Do **not** re-score L2 as a sprint positive; USAGE_MOVED still fails. Do **not** collapse it back to “same IBU as feeler cold.”

#### Retrospective Hᵢ — route_v1 / cold_to20 (2026-07-14)

Standing companions to H(mean load): **median Hᵢ** and **frac Hᵢ\<0.5**. Same per-residue histogram on the sprint’s collapse baselines (artifact: `checkpoints/v66/diagnostics/per_residue_entropy_retrospective/`):

| Run | H(mean load) | mean max-p | median Hᵢ | frac Hᵢ\<0.5 | near-ln4 mass | label |
| --- | ------------ | ---------- | --------- | ----------- | ------------- | ----- |
| slim_moe_route_v1 | **1.341** | **0.581** | **1.111** | **6%** | 27% | **MIXED** (heterogeneous: soft near-ln4 *and* a thin committed tail) |
| feeler cold_to20 ge20 | 1.385 | 0.478 | 1.217 | **0%** | 13% | **HIGH_ENTROPY_SOFT** |
| fix1_s4 ge20 | 1.384 | 0.361 | 1.300 | **0%** | **100%** | **NEAR_MAXENT_SOFT** |
| L2 ge20 | 1.382 | 0.602 | 0.990 | **0%** | ~0% | **SEMI_COMMITTED_MODERATE_ENTROPY** |

**Recalibration:** the first collapse diagnosis was not “nothing happening everywhere.” `route_v1` already had mean max-p≈0.58 and median Hᵢ≈1.11 with a small committed tail (6% Hᵢ\<0.5) while corpus H still looked near-maxent — **corpus-mean H alone could not distinguish flat softmax from semi-commitment**. Cold_to20 / fix1_s4 are closer to genuinely soft (high median Hᵢ, 0% low-H mass). “Collapsed” this sprint mostly meant **balance of mean load + no confident commitment**, not identical soft shapes.

#### route_v1 committed-tail characterization (2026-07-14)

Before any fresh scale pre-registration: what *are* the 6% with Hᵢ\<0.5? Artifact: `checkpoints/v66/diagnostics/route_v1_committed_tail/`.

| Check | Result |
| ----- | ------ |
| Size | **204 / 3393 (6.0%)** residues |
| Spread | **11/12** Stage-A proteins (enrichment mild: 4OBE 1.38×, 1TIM 1.35×, 2Z6H 1.25× — not one-structure noise) |
| Expert pattern | **Only e0 + e2** (157 + 47); e1/e3 absent → binary niche |
| Biology | **e0:** τ=0.97, ρ_med=5.0, depth_med=0.24 (underwrapped / dehydron). **e2:** τ=0.00, ρ_med=29.0, depth_med=0.49 (wrapped / non-τ) |
| Corpus τ | committed **0.75** vs background **0.49**; ρ median **6 vs 13** |

**Masking audit (required before calling this “commitment”):** eval forward — dropout and expert-timeout bans are **training-only**. Capacity overload (`raw → adjusted` logits) applies at eval: **0/204** committed residues had argmax flip or \|ΔH\|\>0.05; raw and adjusted committed sets are **identical**. Verdict: **`NOT_CAPACITY_MASK_ARTIFACT`**.

**Read:** the thin tail is a **genuine dehydron-axis specialization signal** that H(mean load) erased. It is closer in *kind* to what L1→L2 later pushed corpus-wide into `SEMI_COMMITTED_MODERATE_ENTROPY` (same underwrap/wrap axis, without recovering Hᵢ\<0.5 mass) than to fix1_s4’s near-maxent soft. Original “collapse” on route_v1 was **MIXED**, not undifferentiated.

**Next:** any fresh scale / board / gate-input pre-registration should treat route_v1’s e0/e2 niches as a **prior hypothesis to expand**, not assume a flat baseline.

#### L1→L2 Hᵢ trajectory — flag only (no L3 under old ladder)

| | L1 ge20 | L2 ge20 | Δ |
| - | ------- | ------- | - |
| median Hᵢ | 1.253 | 0.990 | −0.263 |
| mean max-p | 0.376 | 0.602 | +0.226 |
| frac Hᵢ\<0.5 | 0% | 0% | 0 |

Two trained softplus points cannot separate **deepening toward commitment** from **an asymptote already leveling**. Linear extrapolation of Δ(Hᵢ) would need ~1.9 more L1→L2-sized steps to median Hᵢ=0.5 — but frozen softplus on flat-trained fix1_s4 still has **frac Hᵢ\<0.5 = 0% at ×10** (median Hᵢ→0.55, mean max-p→0.73), so sharpening without commitment mass is easy to over-read.

**Decision:** do **not** reopen L3 under the frozen ladder. Soft-structure audit shows L1→L2 **held** the dehydron axis (L2 twin-splits it) but produced **`AXIS_HELD_SOFT_NO_HARD_GROWTH`** — not `GROWS_REAL_NICHE`. If scale is pursued again, it needs a **fresh pre-registration** that (a) co-registers median Hᵢ + frac Hᵢ\<0.5 **and** dehydron-axis retention (`GROWS_REAL_NICHE` vs `DIFFERENT_AXIS_OR_DIFFUSE`), (b) uses ≥3 trained softplus points or an anneal, (c) treats route_v1’s niche as the prior to expand — see **HEADLINE** at top of this doc.

**Implication for next lever:** prefer board / gate-input enrichment aimed at expanding the existing dehydron niche; metric stack must keep median Hᵢ / frac Hᵢ\<0.5 **and** τ/ρ expert contrast beside H(mean load). Scale-deeper is only a candidate under a `GROWS_REAL_NICHE` pre-reg.

#### Pre-registered gates — trained higher `logit_scale` (frozen 2026-07-14, **before** retrain)

**Config under test (first controlled run = L1):** Fix-1 + S4 Stage A-12 cold; declare z-norm on/off in run README. Init / floor: `softplus(logit_scale) ≈ 2.65` (2× baseline). **No** `|rho-TAU|`, **no** encoder_h→gate.

Judge at **ge20** on locked Stage A-12.

| Verdict | Required |
| ------- | -------- |
| **SCALE_TRAIN_USAGE_MOVED** | Exact frozen **USAGE_MOVED** (max>=0.38 AND spread>=0.12 AND H<=1.30 AND feature hold: peak soft \|r\|(w,tau)>=0.50 **or** MI(hard,tau)>=0.40) |
| **SCALE_TRAIN_SIGNAL_HOLD** | **Primary:** MI(hard,τ) at ge20 ≥ **0.90 ×** a same-run scale≈1 reference — prefer ge0 forward with `softplus` ablated to the prior baseline (~1.32) on the L1 init weights; else a scale=1 sibling. Soft R² is **diagnostic only**. |
| **SCALE_TRAIN_FIX** | **SCALE_TRAIN_USAGE_MOVED** **and** **SCALE_TRAIN_SIGNAL_HOLD** (MI primary) |
| **SCALE_TRAIN_SHARPENED_NOISE** | Usage moves **or** lands **AMBIGUOUS** toward a move, **and** hard-MI SIGNAL_HOLD fails (ge20 MI \< 0.90 × scale≈1 reference). This is the named bucket when training reorganizes the gate around higher scale and **destroys** hard structure — distinct from IBU (flat usage, MI intact) and from the inference-time soft-R² drop. Soft-R² collapse alone does **not** assign this label. |
| **SCALE_TRAIN_IBU** | Still **IBU_HOLDS** at ge20 |
| **SCALE_TRAIN_AMBIGUOUS** | Anything else (including near-miss USAGE numbers) |

**Pre-committed escalation ladder (frozen with this gate — no post-hoc "just try the next scale"):**

| Step | softplus target | If result is… | Next |
| ---- | --------------- | ------------- | ---- |
| **L1** | ~**2.65** (×2) | `SCALE_TRAIN_FIX` | Stop — claim fix |
| L1 | ~2.65 | `SCALE_TRAIN_SHARPENED_NOISE` | **Stop.** Do not escalate. Reopen board enrichment / what-gate-sees. |
| L1 | ~2.65 | `SCALE_TRAIN_IBU` or `SCALE_TRAIN_AMBIGUOUS` | **One** pre-authorized escalation → **L2** |
| **L2** | ~**6.61** (×5) | `SCALE_TRAIN_FIX` | Stop — claim fix |
| L2 | ~6.61 | anything else (IBU / AMBIGUOUS / SHARPENED_NOISE / near-miss) | **Stop.** No L3 at ×10 without a **new** pre-registration. Next named levers: board enrichment (`\|rho-TAU\|`) or gate input policy — not another scale bump. |

Notes: inference x10 AMBIGUOUS is **not** a preview of USAGE_MOVED and is **not** on this ladder. Do not raise H ceiling or lower max floor post-hoc if a run lands at 1.305 / 0.379.

### S6 — Uncertainty (updated after G1)

**Was (conditional on G1):** "depending on G1's diagnosis…"

**Now (G1 resolved):**

1. Retrain config **must** include `p4_head_decouple` (or equivalent) with
   `epi_ale_decorrelation_coeff > 0` — not `decoupled_uncertainty_heads` alone.
2. **S6 passes only if ALL hold jointly on Stage A corpus eval:**
   - `|r(ν_epi, ν_ale)| ≤ 0.70` (decorrelation)
   - **P8 pass (G4a):** τ-boundary **relative** lift ≥ 0.20 **and** `aleatoric_std ≥ 0.05`
   - `nu_cv ≥ 0.02` (scale-invariant exposure spread)
   - `aleatoric_std ≥ 0.05` (informative aleatoric — not decorrelated noise)
3. **Save gate wired:** `require_tau_ale_elevation_save=True` on
   `p4_head_decouple_phase_config` — low r alone cannot promote checkpoint.

Implementation: `uncertainty_s6_joint_pass()`, `uncertainty_save_ineligibility_reasons(require_tau_ale_elevation=True)`.

**G5 must be run before** attributing P7 to v6-native learning. **If
`epistemic_novelty_claim_blocked` (route_v1: yes)** — block production claims that epistemic
flags novel/risky structures for review; routing/disc legs (S1, S4, S5) remain creditable.

---

## Property tests (P7–P11)

| Test | Corpus? | Notes |
| ---- | ------- | ----- |
| P7 | Yes | `epistemic_std` + `nu_cv` — floors on head output, not canonical DER; interpret via **G5** |
| P8 | Yes | **G4a:** relative lift + informative ale; **G4** holdout if shaping; **G3** for epistemic paths |
| P9 | **Yes** (`test_p9_sparsification_monotone_corpus`) | Do not cite triage until passes |
| P10 | After expansion | Stub until 12→25 corpus event |
| P11 | Pinned OOD `1PGB:A` | Independent only if G3 pass; not DER-valid if primary loss is gaussian_likelihood |

### nu_cv floor calibration (empirical)

| Checkpoint | ale_std | nu_cv | Interpretation |
| ---------- | ------- | ----- | -------------- |
| route_v1 (ale flat) | ~0.012 | **~0.16** | nu_cv detects exposure spread; **does not** catch ale flat |
| Synthetic constant ν | — | **~0** | Floor 0.02 rejects |

Floor `0.02` validated against synthetic degenerate + live route_v1 discrimination.

---

## What S6 does **not** authorize

- MoE routing fix
- Production ingest gates on uncertainty
- Trusting edge-flow / same-expert excess without G3 + S6 joint pass

---

## Recommended retrain sequence

1. Confirm G1 (done — document in run params)
2. **G4a** — P8 magnitude gate (done in code; re-eval G3/G3 report through hardened P8)
3. **G5** provenance audit on route_v1 / G3 checkpoints (cheap, no GPU)
4. Re-interpret G3 A/B (epistemic circularity on P11; P8 legs invalid pre-G4a)
5. **G4** holdout wiring — **done** (`--p4-v3-aleatoric-shaping`, `make eval-g4-holdout`)
6. Choose primary-loss philosophy (NIG vs blended) — document in run params
7. If G3 (epistemic) + G4 pass → full Phase 4 + routing retrain with hyperbolic MP graph (S4)
8. Evaluate S1–S6 jointly; apply G5 asterisk rule if `distilled_proxy` flags

**Blocked:** `p4_v3_aleatoric_recovery` until steps 5–6 complete.

---

## Tracked sibling — geometric angular prior (Fix 1, 2026-07-13/14)

**Spec:** `docs/superpowers/specs/2026-07-13-geometric-angular-prior-design.md`  
**Runs:** `feeler_expand_23_geom_angular_prior_v1` (resume ep164, ge165–184); cold `…_cold_v1` → continue `…_cold_to20_v1` (ge20).  
**Gate audit:** `checkpoints/v66/diagnostics/geom_angular_prior_gate_audit/`  
**Routing null + effect sizes:** `checkpoints/v66/diagnostics/geom_prior_cold_to20_routing_null/`

| Layer | Question | Status |
| ----- | -------- | ------ |
| **Fix-1 gates** (probe, 1F88 gap, 4OBE circ-R, corr(r,depth)) | Pre-MoE disc θ grounded without breaking radial story | **PASS** through cold ge20 (`corr(r,depth)~0.99` sustained) |
| **By construction** | Full θ span from dehydron/peptide directions | Expected — do **not** treat HTML “filled disc” as S2 |
| **S1** routing H / load | Differentiated expert *usage* (save ceiling / non-flat load) | **No** — see category below |
| **S2** disc occupancy | 2D disc σ₂/σ₁ / thickness (official S2) | Can look healthy under Fix-1 |
| **T1** trunk HD rank | `pre_mp` / `encoder_h` / topo feed (not disc) | **FAIL** — ER≈1.02 across route_v1, cold_to20, fix1_s4 |

### Named finding: informative-but-undifferentiated (IBU)

**Not** “balanced/healthy specialization.” **Not** “collapsed monopoly” (route_v1 56–72%). **Not** “gate sees only noise.”

| Side | Observation (cold ge20, feeler_expand_23) |
| ---- | ---------------------------------------- |
| **Load** | Soft shares ≈ (0.28, 0.24, 0.24, 0.24); train H≈1.376 ≈ ln(4) |
| **Features** | Soft weights / hard assign track burial–dehydron structure |

**Effect sizes (not just p≈0 vs shuffle):**

| Metric | Value | Practical read |
| ------ | ----- | -------------- |
| MI(hard, depth quartile) | **0.63 nats** (~769× null 0.0008) | Large |
| MI(hard, τ) | **0.69 nats** (~2500× null) | Large — ≈ ln(2), saturates binary-τ information |
| U(expert‖τ) = MI/H(expert) | **≈0.51** | Half of hard-assignment entropy predictable from τ |
| Soft \|r\|(w_e, τ) | **0.76–0.87** → R² **0.58–0.75** | Strong, not tiny-but-real |
| Soft \|r\|(w_e, depth) | **0.60–0.74** → R² **0.36–0.54** | Strong |

**Diagnosis (updated 2026-07-14):** the gate **sees** structure along a near-1-D axis (informative features / strong MI), but expert *usage* stays flat. Logit-spread closed **gate geometry**; trunk occupancy shows the HD encoder / gate feed is still collapsed (ER≈1.03). Commitment levers are secondary to **trunk diversity**. Upstream *disc* geometry alone was the wrong lever.

**Warm vs cold H (do not conflate):** warm resume-ep164 sibling stays H≈1.29; cold stays at ceiling. Different experiments.

**Corpus:** feeler_expand_23 pilot — **not** locked Stage A-12.

### Fix-1 + S4 Stage A-12 — cold seed-1 (`fix1_s4_stage_a12_cold_v1`)

**Status 2026-07-14:** complete (20 ep). Soft loads ge20 ≈ `(0.213, 0.211, 0.297, 0.278)`; `max=0.297`; `max−min=0.086`; `H=1.372`.

**Usage gate:** **AMBIGUOUS** — honest, but *barely*: `max−min=0.086` sits a hair above the frozen IBU threshold `0.08`. On a single seed that is **not** a meaningful exit from IBU; seed noise alone could push it either side. Seed-2 on the same config is required before reading this as “S4 moved usage.”

**Seed-2 (`fix1_s4_stage_a12_cold_seed2_v1`, SEED=2, ge20):** soft ≈ `(0.251, 0.247, 0.253, 0.249)`; `max=0.253`; `max−min=0.006`; `H=1.385` → **IBU_HOLDS**. Seed-1’s hairline AMBIGUOUS was noise; S4 did not move usage off IBU across seeds.

**Fix-1 fidelity cost under S4 (still inside pass bounds):** vs cold Fix-1-alone (~1F88 gap 6.6–7.7°, 4OBE circ-R 0.28–0.30), this run ended ~9° / ~0.35 — both worse, both still passing. Note in the log: S4 bought essentially no usage move while spending a little geometric fidelity. If a future combo pushes Fix-1 outside its gates without resolving usage, that trade stops being free.

**No `v6_best` again** — cumulative sprint fact, not this run alone: routing-entropy save ceiling still blocks every configuration tried.

**Next (load-bearing):** trunk diversity / encoder occupancy (HD ER≈1.03 on seed-1). Seed-2 Fix-1+S4 still useful for usage-boundary noise only: `RUN_ID=fix1_s4_stage_a12_cold_seed2_v1 SEED=2`. Gate geometry **CLOSED**. Commitment knobs deferred until trunk rank moves.

#### Logit-spread diagnostic (seed-1 ge20) — 2026-07-14

Artifact: `checkpoints/v66/diagnostics/fix1_s4_gate_logit_spread_seed1/`.

| Check | Result |
| ----- | ------ |
| Hyp logit row-range mean | **0.93** (AMBIGUOUS_SCORE_SPREAD) |
| `frac(max soft ≥ 0.5)` | **0.0** (mean max soft **0.37**) |
| Origin×rim | **NO_CLEAR_RADIAL_ASYMMETRY** (`corr(row_range, disc_r)≈0.14`) |
| Hyp vs Eucl tangent surrogate | Hyp **wider** than Eucl (`ratio≈2.0`; Eucl range mean **0.47**) |

**Hypothesis reject (this ckpt):** hyperbolic origin-saturation as the unique IBU cause — not supported. Scores have modest inter-expert range but **never** commit past 0.5 at T=1; Euclidean distance on the same fused trunk is flatter, not sharper.

**Gate geometry CLOSED — do not reopen.** Hyp-vs-Euclidean gate choice is ruled out on this evidence; neither readout peels soft loads. Any future flat-routing discussion must not revert to “try a Euclidean gate” on a hunch.

#### Trunk / encoder HD occupancy (**T1**, was sibling “S2/trunk”) — 2026-07-14

Artifact: `checkpoints/v66/diagnostics/trunk_rank_cross_lineage/` (+ per-run dirs).

| Checkpoint | encoder_h ER | top-1 | Origin |
| ---------- | ------------ | ----- | ------ |
| route_v1 | **1.023** | 0.978 | PRE_MP + raw topo low-rank |
| cold Fix-1 to20 | **1.019** | 0.982 | same |
| Fix-1+S4 seed-1 | **1.025** | 0.976 | same |

**Confirmed long-standing bottleneck** — not Fix-1+S4-specific; those runs did not meaningfully worsen trunk rank. Disc can look healthy (σ₂/σ₁≈0.98) while T1 fails.

**Implication:** next work targets embed / early-trunk / gate-feature diversity (T1), not commitment knobs or disc geometry.

### Next tracked sibling (proposed) — tempered expectation

**Fix-1 + hyperbolic graph during training (S4)**, cold, on **locked Stage A-12**.

- **Status 2026-07-14:** launched `checkpoints/v66/runs/fix1_s4_stage_a12_cold_v1` via `make train-v66-fix1-s4-stage-a12` (`--hyperbolic-mp-graph` + geom prior; role edges off).  
- **Tests:** whether better train-time topology + grounded compass moves **load differentiation** (S1 usage).  
- Pre-registered USAGE_MOVED / IBU_HOLDS / AMBIGUOUS apply to final soft loads.  
- Log: `fix1_gates_per_epoch.jsonl` (H, max_share, Fix-1).

#### Pre-registered usage gates (frozen 2026-07-14 — before Fix-1+S4 Stage A-12)

**Baseline (IBU, cold_to20 ge20, feeler_expand_23):** soft loads ≈ `(0.279, 0.241, 0.240, 0.240)`; `max_share=0.279`; `max−min=0.040`; train `H=1.376`.

| Verdict | Required (ALL must hold on eval soft load, Stage A-12 mean) | Else |
| ------- | ------------------------------------------------------------- | ---- |
| **USAGE_MOVED** | `max_soft_share ≥ 0.38` **and** `max−min ≥ 0.12` **and** `H ≤ 1.30` **and** feature hold: peak soft \|r\|(w,τ) ≥ 0.50 **or** MI(hard,τ) ≥ 0.40 | — |
| **IBU_HOLDS** | `max_soft_share < 0.34` **and** `max−min < 0.08` **and** `H > 1.34` | Treat as evidence commitment > upstream geometry |
| **AMBIGUOUS** | Anything else (e.g. max_share 0.34–0.38) | **No claim** — do not read as confirming either side; next action = commitment ablation |

Notes: `0.38` is well above IBU clutter and still below feeler `expert_timeout_max_share=0.45` so a move can register without ban artifacts. Feature floor blocks “usage moved” via destroying structural routing signal. Report effect sizes with any H/load claim. Seed-1 Fix-1+S4 landings near the IBU `max−min` boundary (`0.086` vs `0.08`) count as AMBIGUOUS and require a second seed before any “moved off IBU” claim.

#### Commitment config peek (cheap — cold_to20 vs route_v1)

| Knob | cold geom-prior (phase 12) | Notes |
| ---- | -------------------------- | ----- |
| `balance_coeff` | **0.015** (via capacity path) | Present |
| `capacity_loss` @ ge20 | **0.0** | Soft shares all ≪ `capacity_threshold=0.4` → **no active balance gradient** |
| `expert_timeout_max_share` | **0.45** | Hard anti-monopoly ceiling; irrelevant while max≈0.28 |
| Gate `temperature` | **1.0**; Gumbel **off** | Plain softmax — not an aggressive flatten |
| `expert_dropout_p` | **0.0** (phase 12) | — |
| route_v1 (ref) | H≈1.30; loads more skewed (max soft ~0.37 on best_disc snapshot) | Different attractor |

**Read:** no smoking gun of *active* uniformity pressure at the IBU operating point (`capacity_loss=0`). Commitment story is more **missing specialization pressure once under capacity 0.4** (and a 0.45 timeout wall that never engages) than “balance_coeff is crushing shares to 25%.” Still supports pre-registering commitment ablations if Fix-1+S4 lands in **IBU_HOLDS**.

#### If Fix-1+S4 → IBU_HOLDS **or hairline AMBIGUOUS** — next diagnostic order (frozen)

Do **not** default to lowering `balance_coeff` first (capacity_loss is already 0 at IBU loads — there may be nothing to relax). Prefer this path over another geometry combination when max share stays ~≤0.30.

1. **Pre-softmax gate logits (cheap, required first):** on Stage A-12 eval, histogram / report `logit_std`, per-residue `max(logits)−min(logits)`, and fraction of residues with `max_softmax ≥ 0.5` at T=1. Also compare hyp distance scores vs a Euclidean tangent-distance surrogate on the **same** fused trunk, and check whether score spread correlates with disc/fused radius (origin vs rim). Artifact: `make audit-hyperbolic-gate-logit-spread CHECKPOINT=...`.  
   - **Flat/narrow hyp scores** (low spread → soft≈uniform regardless of coeff): investigate gate distance rescaling / init / metric saturation — **not** balance_coeff.  
   - **Peaked logits but flat loads:** then temperature / aux differentiation loss / capacity-threshold reward structure.
2. Only after (1): commitment knobs (T anneal, differentiation aux, capacity band redesign).  
3. Lowering `balance_coeff` is **low priority** unless (1) shows peaked logits *and* capacity_loss becomes active on a trial that actually tests >0.4 share.
