# Tokyo Eye EQU — Correct Start (Pure-Hyp Spine)

**Gate ID:** `tokyo_eye_equ_correct_start`  
**Display lineage:** Tokyo Eye EQU  
**Status:** APPROVED & LOCKED (2026-09-15, Ray Swan)  
**Date:** 2026-09-15  
**Experiment (domain charter):** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`  
**Governance:** `tokyo_eye_equ_governance` · Pure-hyp amendment: `tokyo_eye_equ_pure_hyp_v1`  
**Not:** remediation / patch-continue of cold-boot or v5 lesson weights

---

## 1. One-line

Fresh geometry train under the **amended freeze**: post-lift stack is **pure hyperbolic** from day 0 (no tangent QKV / pool / residual as geometry substitute), with sealed hygiene gates from the governance standard. Cold boot and v5 remain **lesson objects**.

---

## 2. Why this card exists

| Sealed lesson | Implication for this card |
|---------------|---------------------------|
| Cold boot hygiene **Fail** (mean probe sat ≈ 0.915; five probes rim-collapsed) | Curriculum / rim must not escalate mass to τ ceiling; volume is a Pass requirement |
| Live `attention.py` / `affinity_head.py` use `exp₀(W·log₀(z))` | **Out of Pass** under pure-hyp; this card builds a **correct** spine path — not a mid-run patch of Fail θ |
| MoE 6/6 alive on cold boot | Keep MoE liveness; alone it does **not** open trunk |
| v5 champion disposition | Forbidden init; lesson-only |

**Operator policy:** amend freeze (done) → correct start (this card) → no remediation theater.

---

## 3. Scope (LOCKED)

### In scope

- Implement / wire a **pure-hyp** attention + pool path compliant with freeze §6.1 / amendment (Einstein/Klein aggregation OK; no tangent Linear as geometry substitute).
- Fresh spine init (Equiformer MPtrj bank pin only for frontend; spine random / fresh under sha guard).
- Geometry Pass train + probe seal under governance §4 gates.
- MLflow run name = `tokyo_eye_equ_correct_start`; gate stamp `data/gates/tokyo_eye_equ_correct_start.json`; artifacts on the run.

### Out of scope

- Affinity / ligand / Pearson promote  
- Biology teleconnections Pass  
- Moving `@champion`  
- Continuing from cold-boot or v5 checkpoints  
- Silent threshold waives  

---

## 4. Architecture contract (LOCKED)

1. **Single Euc→H lift** at projector (`expmap₀` / `project_to_ball`) — once.  
2. **After lift:** on-manifold ops only through hyp graph storage / on-manifold heads.  
3. **Allowed:** Poincaré / hyp distance logits; Einstein / Klein (or equivalent valid) barycenters; gyrovector / Möbius that stay on ball; read-only `log₀` for **diagnostics**.  
4. **Forbidden:** `exp₀(W·log₀(z))` as Q/K/V/output/FFN; tangent mean/sum pool then `exp₀` as graph repr; tangent residual as primary message path; vendor leave-manifold shortcuts.  
5. **Affinity head:** if touched, must obey the same law or stay parked / non-trunk for this card.  
6. **Static / unit gate:** tests fail if `_tangent_linear`-class patterns remain on the EQU trunk forward used by this run.

Implementation detail (algorithms that stay on-manifold for Q/K/V) is filled in the **plan** after this design is APPROVED — design locks the law, not a silent mid-approval code dump.

---

## 5. Train contract (LOCKED) — inherit cold-boot panel unless plan amends with stamp

| Knob | Lock |
|------|------|
| Corpus train | Same 8 PDBs as cold boot: `1MBN,1LYZ,1F88,1HHP,1TEN,1UBQ,1TIM,4OBE` (chains per manifest) |
| Corpus probe | Same 6: `1HNG,1ALC,1A5R,1NAL,2HHB,1GKY` |
| Shock holdout | Still held out: `1BG1,2Z6H,1IVO,2SHP` |
| Frontend bank | MPtrj EquiformerV3 `mptrj_gradient.pt` sha `59c6c235…` (exact pin in plan) |
| Spine init | Fresh — forbid champion / affinity / C1 / cold-boot θ |
| Epochs | Locked N in plan (short geometry card; not whim-extend) |
| Container | `tokyoeye_science` only |
| Curriculum | Must **not** reproduce cold-boot failure mode (mean r locked to τ ceiling by ~epoch 5). Plan must specify τ schedule + any rim regularizer **before** execute |
| Curvature `c` | Prefer learned with contract passthrough; if pinned, document pin |

---

## 6. Sealed advance gates (from governance §4 — pre-registered)

Binary: **any Fail → DISQUALIFIED** (no Pearson waive).

| Gate ID | Pass when | Blocks trunk |
|---------|-----------|--------------|
| `pure_hyp_pass` | `true` (no forbidden post-lift tangent geometry substitutes on the executed forward) | Yes |
| `probe_sat_gate` | mean probe boundary saturation **< 0.50** | Yes |
| `radius_spread_gate` | mean probe radius spread **> 0.15** **and** non-degenerate per-probe volume (not “one PDB carries the mean”) | Yes |
| `finite_h2_gate` | finite H² on all probes + home | Yes |
| `moe_liveness_gate` | normalized routing entropy **≥ 0.60** (plan defines exact estimator; also report per-expert load floors) | Yes |
| `equiv_residual_gate` | Δ_equiv **< 1e-5** on locked E(3) probe protocol in plan | Yes |

**Note:** These thresholds are **stricter / different metrics** than cold-boot’s MoE load-floor seal. That is intentional under governance v1.1. Plan must implement the estimators before execute.

**Hygiene Pass** for this card = all rows above Pass.  
**Biology / affinity** = false / out of scope.

---

## 7. MLflow & naming (LOCKED)

- Experiment: `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine` (existing domain charter — **not** a new experiment)  
- Run name / gate_id: `tokyo_eye_equ_correct_start`  
- Tags: `gate_id`, `display_lineage=Tokyo Eye EQU`, `pure_hyp_pass`, governance refs  
- Log: design + plan + stamp + metrics `gate_*`  
- On seal: update experiment tag `tokyo_eye_equ_next_open` to the **next** card (or `none` if Pass and waiting)  
- Do **not** overwrite `tokyo_eye_equ_ssot_run_id` unless promoting the operator index itself  

---

## 8. Forbidden whim

- Patch cold-boot weights mid-card  
- Waive `pure_hyp_pass` or sat/spread for MoE / loss  
- Rename tangent ops and claim Pass  
- New PDB panel without amending this design + stamp  
- `@champion` move from this card alone  

---

## 9. Exit

| Outcome | Action |
|---------|--------|
| All gates Pass | Stamp Pass; geometry trunk **candidate**; next card pre-registered separately (not affinity by default) |
| Any gate Fail | Seal DISQUALIFIED; lesson object; **no** remediation theater — new card or freeze amend if law was wrong |

---

## 10. Sign-off

**Approver:** _________________ (Ray)  
**Date:** _________________  

On APPROVE: write implementation plan `docs/superpowers/plans/2026-09-15-tokyoeye-equ-correct-start.md`, then execute only under science container with pre-registered estimators.
