# |ρ−TAU| Input Swap — Trunk Capacity Registration

**Date locked:** 2026-07-18  
**Status:** **Partial** — Part A Confirmed; Part B passive Partial (PC1 flat); Part B causal knockout **Partial** (methodology regrade; soft Pass withdrawn). Not hist-ρ Win.  
**Part B caveat:** passive PC1 = geometry only; causal = forward knockout telemetry (see ablation Move 2). Clear Pass bar Δ_median ≥ +0.10 not met.  
**Probe defect:** [`../learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md`](../learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md)  
**Path 2 agent pilot:** ORPHANED — [`../learned-flow-influence/PATH2_DIRECTIONALITY.md`](../learned-flow-influence/PATH2_DIRECTIONALITY.md)  
**Filed predicted ceiling:** **2.193**  
**Mechanism:** under z-norm, `||h||` ↔ raw ρ flips while PC1↔betweenness holds.  
**Related:**
- [`../learned-flow-influence/ablation.md`](../learned-flow-influence/ablation.md)
- Probe: `jacobian_flow_influence.py --score-mode pc1_sq`
- Artifacts: `checkpoints/v66/diagnostics/rho_tau_abs_dist_swap/`

---

## 1. Why this, why first

Per the standing T1a diagnosis: τ is a near-duplicate of ρ (binary threshold on the same underlying quantity, historically r≈0.81), contributing almost no independent information to `node_emb`'s input. Replacing it with `|ρ−TAU|` (continuous distance-to-threshold) removes the redundancy without removing information — a strictly richer encoding of the same physics, not a new feature or a new theoretical claim.

This is prioritized **ahead of** depth/reach (containment-style) and loss-lock (Option B / directionality reward) because those act on **how the trunk uses its capacity**. Extending reach or freeing pressure on a trunk already near-saturated (~1.7–1.8 effective rank) tests the wrong stage first. Raising the ceiling first means subsequent depth or loss-competition experiments actually test what they claim, rather than remaining bottlenecked upstream.

---

## 2. What this is NOT

- **Not** a new loss term
- **Not** touching `cone_target_mode=tau_dehydron_rim` or any depth-collision-locked radial target
- **Not** new architecture — `node_dim` unchanged (one column’s *content* changes, not the count)
- **Not** a claim that this alone restores directional flow — this tests **capacity** specifically, separate from loss-competition and reach
- **Not** reusing failed SSE-parent Path B as the vehicle

---

## 3. Prerequisite: re-derive the predicted ceiling on the current lineage

The original ~1.7→~2.33 prediction was computed at T1a-era, before chem edges, z-norm discipline, and other lineage changes. **Do not assume 2.33 still holds.** Before any training:

### Step 1 — correlations on current Stage A-12 / chem-MVP feature mix

Compute on the current Stage A-12 corpus with whatever is active for the chem-MVP arm (`topology_three_vector`: ρ, τ, ss; z-norm if enabled on that recipe; **no** barcode scalars unless that arm actually uses them):

- \(r(\rho,\tau)\)
- \(r(\rho,|\rho-\mathrm{TAU}|)\)

Report both. Forward-pass / corpus stats only.

### Step 2 — analytic ceiling re-estimate

Re-run the T1a-style effective-rank estimate of the **full input feature matrix** with `|ρ−TAU|` substituted for τ. Forward-pass only, no training.

**Deliverable:** an explicit **new predicted ceiling** for chem-MVP. File it; do not train against the legacy 2.33 figure unless Step 2 reproduces it.

### Step 3 — orthogonality / resolution bar

Same discipline as other feature screens in this project (marginal + within-SS-class correlation; redundancy cut in the spirit of `|r|/|ρₛ| ≥ 0.7` where applicable).

This is a **substitution**, not a new independent feature:

- Compare `|ρ−TAU|` against everything **except** τ (which it replaces)
- Expected to correlate with ρ (derived from it) — that is fine
- The question is whether it adds **resolution beyond the binary flag**, not whether it is independent of ρ entirely

**Gate:** if Step 2 shows no meaningful predicted ceiling lift vs current τ encoding, **stop** — do not burn a cold run. Revisit which feature is actually redundant, or whether a genuinely new orthogonal input is required.

---

## 4. Implementation (minimal — only after Steps 1–3 clear)

**Note:** The transform and config flag already exist (`replace_tau_with_abs_dist` / `rewrite_tau_channel`). Matched-arm wiring may still need Makefile targets, feature-set id bump for cache keys, and explicit logging. Do not invent a second code path.

1. Ensure `|ρ−TAU|` is applied as a **replace** of the τ channel (not append) so `node_dim` is unchanged
2. Bump / distinguish feature-set or cache tag so pre-swap graphs cannot silently collide
3. Isolated-seed / gate–prototype identity check: width unchanged ⇒ RNG-shift confound should not apply; still log one-line confirmation that gate/prototype construction matches baseline vs swap under the same seed (cheap insurance)

---

## 5. Matched training

### 5.0 Scale check + z-norm decision (locked before launch)

**Predicted-scale check (Stage A-12, 3393 residues, raw channels):**

| Channel | mean | std | note |
|---------|------|-----|------|
| ρ | 13.0 | **6.96** | |
| τ (binary) | 0.51 | **0.50** | |
| `|ρ−TAU|` | 5.66 | **4.04** | |
| ss | 0.58 | 0.47 | |

`std(|ρ−TAU|)/std(τ) ≈ 8.1` — swapping without z-norm would inject an order-of-magnitude louder second channel (classic T1a / SASA scale-dominance risk). Also: `replace_tau_with_abs_dist=True` already forces z-score in `fit_and_install_input_feature_norm` / forward — enabling the swap alone while leaving baseline z-norm-off would **confound** the matched pair.

**Decision (locked):** both matched arms run with **`input_feature_zscore=True`**. Do **not** reuse `chem_mvp_stage_a12_cold_v1` (z-norm off) as the grade baseline.

| Arm | Recipe |
|-----|--------|
| **Baseline** | Chem-MVP Stage A-12 feeler stack + **z-norm on**; τ unchanged |
| **Swap** | Identical + **z-norm on** + `|ρ−TAU|` substituted for τ (`--replace-tau-abs-dist`) |

Same seed(s), same epoch count, same Stage A-12 corpus — mirror prior matched-arm ablations. Parent stack = chem-MVP feeler (`--chem-edge-mp`); **not** Path B containment-on.

---

## 6. Pre-registered acceptance — two parts (both required for Win)

### Part A — did the ceiling actually move?

Trunk effective rank on `encoder_h` (T1a methodology used in training logs / trunk-rank diagnostics).

**Acceptance track is the z-scored board only** (matches how both arms train). Step-2 primary prediction: baseline input ER **1.520** → swap ceiling **2.193** (Δ+0.673). Raw-board numbers from the prerequisite remain diagnostic telemetry, **not** Part A grade inputs.

| Outcome | Criterion |
|---------|-----------|
| **Confirmed** | Trunk effective rank rises to within a reasonable band of the **Step 2 re-derived z-scored** prediction (**~2.193**) |
| **Partial** | Rank rises, but well short of predicted ceiling |
| **Fail** | No meaningful rank change vs matched **z-norm** baseline |

### Part B — does the freed rank correlate with anything real?

Reuse learned-flow-influence classical-metrics ground truth directly — same tool, same Stage A-12 corpus, same discipline as the standing triangulated conclusion. **No promotion on Part A alone.**

| Outcome | Criterion |
|---------|-----------|
| **Win** | Part A Confirmed **and** (trunk↔classical-betweenness improves beyond baseline ρ≈0.69, **or** a previously weak classical metric — current-flow / ANM / spectral — shows real new trunk signal) |
| **Partial** | Part A Confirmed but classical-metric correlations are flat — capacity exists but is unused |
| **Fail** | Part A fails, **or** rank moves but correlations **degrade** |

A rank increase with flat correlations is filed as **Partial** (informative: capacity was a ceiling, but nothing sought to fill it → strengthens prioritizing Option B / directionality reward next) — **not** a Win.

---

## 7. What each outcome unlocks

| Result | Next registration |
|--------|-------------------|
| **Win** (A+B) | Capacity was real and useful; depth/reach and loss-lock bets become well-motivated on a trunk with confirmed room |
| **Part A Fail** | Revisit ceiling estimate (different redundant feature, or need a new orthogonal input) |
| **Part A Pass / Part B Partial** | Strongest argument to prioritize Option B next — bottleneck was never capacity alone; nothing pressure-filled the room |
| **Fail with degraded correlations** | Stop; do not treat as progress |

---

## 8. Non-goals

- Expanding to sink-set fishing or reopening KRAS 163 as the primary grade for this arm
- Softening Part B via disc-layer pooling
- Claiming directional-flow recovery from rank alone
- Touching depth-collision / rim physics targets
