# |ρ−TAU| Input Swap — Ablation / Gates

**Date:** 2026-07-18  
**Design:** [`design.md`](design.md)  
**Status:** **Overall Partial** — Part A Confirmed; Part B passive Partial; Part B
causal knockout **Partial** (methodology regrade 2026-07-18). Not a design-doc
Win (hist ρ≈0.69 bar uncleared).

**Cleanup (2026-07-18):** Agent Move 2 “Pass” via soft `Δ_holds≥+2` OR was
**over-claimed**. Numbers retained; grade corrected to Partial. Agent Path 2
pilot **orphaned** — do not warm-start from it. See
[`../learned-flow-influence/PATH2_DIRECTIONALITY.md`](../learned-flow-influence/PATH2_DIRECTIONALITY.md).

---

## Part B caveat (read first)

> **Passive PC1 Part B remains geometry-only (Δ ≈ +0.02, flat).** Causal Part B
> is graded separately via **forward knockout** (Jacobian forbidden on z-norm-on
> — [`JACOBIAN_ZNORM_DEFECT.md`](../learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md)).
> Geometry and causal readouts can disagree; do not collapse them.
>
> **Clear causal bar for this registration:** Δ(median) ≥ **+0.10** with swap
> median ≥ 0.30. Soft hold-count OR clauses do **not** clear a Pass under
> project methodology.

Jacobian defect SSOT:  
[`../learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md`](../learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md)

---

## Standing diagnosis

### 1. Mechanism — CONFIRMED

Under z-norm, `||encoder_h||` is almost a perfect anti-proxy for raw ρ:

| Relationship (Stage A-12 median Spearman) | Hist (z-norm off) | Z-norm on |
|-------------------------------------------|-------------------|-----------|
| `\|\|h\|\|` ↔ raw ρ | **+0.997** | **−0.928** |
| `\|\|h\|\|` ↔ betweenness | **+0.49** | **−0.49** |
| PC1 ↔ betweenness (sign-invariant) | **+0.50** | **+0.46** |

Hub **geometry** survives; legacy `s_B=||h||²` tracks a flipped magnitude proxy.
Artifact: `znorm_norm_vs_input_magnitude.json`.

### 2. `pc1_sq` Jacobian — does not restore causal grade

`--score-mode pc1_sq` recovers hist z-norm-off (6/12, +0.51) but **z-norm arms
still fail** (0/12, ≈−0.3). Grad-w.r.t. already-z-scored inputs also stays
negative. Filed as standing probe defect (link above) — not a footnote of this
swap.

### 3. Part B emergency substitution — passive PC1

| Arm | Jacobian `pc1_sq` | Direct PC1↔btw holds | Median PC1↔btw |
|-----|-------------------|----------------------|----------------|
| Hist (no z-norm) | 6/12 | 11/12 | +0.50 |
| Z-norm baseline | 0/12 | **11/12** | **+0.46** |
| Swap (`\|ρ−TAU\|`) | 0/12 | **10/12** | **+0.48** |

Δ(swap − baseline) ≈ **+0.021** (flat).

**Part B passive = Partial:** capacity rose; classical hub **geometry** did not
improve vs matched z-norm baseline.

---

## Move 2 — Causal Part B (forward knockout) — **Partial** (methodology regrade)

### Pre-registration (historical agent prereg — retained)

| Item | Value |
|------|-------|
| Arms | `chem_mvp_znorm_stage_a12_cold_v1` vs `chem_mvp_tau_abs_dist_stage_a12_cold_v1` |
| Corpus | Stage A-12 cache `graphs_38a6993d7a439aa4.pt` |
| Method | Forward input knockout → trunk `encoder_h` displacement (`out_effect`) |
| Primary metric | Stage A-12 **median** Spearman(`out_effect`, classical betweenness) |
| Hold | ρ ≥ 0.30 and bootstrap CI_lo > 0 (hub scaffolding only — **not** a 163 hunt) |

Agent prereg also allowed Pass via `Δ_holds ≥ +2`. **That soft OR is rejected
for Pass under cleanup.** Tooling may still print it; SSOT grade uses the clear
median bar only.

Diagnostic: `experiments/diagnostics/hub_knockout_classical.py`  
Prereg: `checkpoints/v66/diagnostics/rho_tau_abs_dist_swap/part_b_causal_knockout_prereg.json`  
Artifact: `checkpoints/v66/diagnostics/rho_tau_abs_dist_swap/part_b_causal_knockout_stage_a12.json`

### Result (full Stage A-12) — numbers unchanged

| Arm | Median Spearman(out_effect, btw) | Holds (ρ≥0.30, CI_lo>0) |
|-----|---------------------------------:|------------------------:|
| Matched z-norm baseline | **+0.380** | **7 / 12** |
| Swap (`\|ρ−TAU\|`) | **+0.457** | **9 / 12** |
| Δ (swap − baseline) | **+0.077** | **+2** |

**Methodology grade = Partial:** Δ_median **+0.077 < +0.10** clear bar. Hold
gains (+2: 1MBN, 2Z6H) are real telemetry, not a Pass.

**Caveats**

- Passive PC1 Part B remains **Partial / flat** (Δ ≈ +0.02).
- This is **hub-scaffolding** (knockout out-effect ↔ betweenness), **not**
  KRAS 163 / directionality / wet-lab biology.
- Does **not** license “capacity was used → authorize Path 2 training.”

---

## Final grade (two-stage order)

| Gate | Result |
|------|--------|
| Part A (trunk ER vs 2.193) | **Confirmed** — late mean 1.992 → 2.368 |
| Part B (passive PC1 ↔ classical) | **Partial** — flat vs matched baseline (Δ ≈ +0.02) |
| Part B (causal Jacobian) | **Forbidden** — probe defect under z-norm (workaround locked) |
| Part B (causal forward knockout) | **Partial** — median Δ +0.077 below +0.10; holds 7→9 telemetry only |
| Overall | **Partial** — A Confirmed; B geometry + causal both Partial |
| Next training dollar | **Not** agent Path 2. Resume from locked `|ρ−TAU|` / matched z-norm trunk only after a **user-registered** bet |

---

## Move 3 / Path 2 — **orphaned**

Agent Path 2 diam≤9 pilot is **not** an authorized lineage step. See
[`PATH2_DIRECTIONALITY.md`](../learned-flow-influence/PATH2_DIRECTIONALITY.md)
(**ORPHANED**). Do not warm-start from `path2_dir_diam9_swap_warm_v1` (removed).

---

## Status

| Item | Status |
|------|--------|
| Part A | **Confirmed** |
| Part B (passive geometry) | **Partial** |
| Part B (causal knockout) | **Partial** (methodology regrade; soft Pass withdrawn) |
| Jacobian-under-z-norm | **Workaround locked** — Jacobian forbidden on z-norm-on |
| Move 3 / Path 2 | **Orphaned** — not graded; not next |

---

## Outcomes log (abbrev.)

- Ceiling: z-scored ER 1.520→2.193; \(r(\rho,\tau)=-0.814\); \(r(\rho,|\rho-\mathrm{TAU}|)=+0.211\)
- Scale: `std(|ρ−TAU|)/std(τ)≈8.1` → both arms z-norm ON
- Runs: `chem_mvp_znorm_stage_a12_cold_v1`, `chem_mvp_tau_abs_dist_stage_a12_cold_v1`
- Causal knockout (telemetry): baseline median +0.380 (7/12) → swap +0.457 (9/12); grade Partial

---

## Link from flow-influence standing conclusion

Capacity arm: Part A Confirmed; Part B **passive** Partial; Part B **causal
knockout Partial** (hub scaffolding telemetry only). Directionality /
mutation-causal long-range influence remains open — any new bet must be
**user-registered**, graded with knockout (not Jacobian on z-norm-on).
Z-norm-off chem-MVP triangulation remains the causal SSOT for that older lineage.
