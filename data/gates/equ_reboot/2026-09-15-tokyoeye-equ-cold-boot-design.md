# Tokyo Eye EQU — Cold boot (geometry-first restart)

**Status:** APPROVED 2026-09-15 — execute only after implementation plan PDB tables lock  
**Date:** 2026-09-15  
**Display lineage name:** **Tokyo Eye EQU**  
**Depends on:** [`2026-09-15-tokyoeye-equ-champion-disposition-design.md`](2026-09-15-tokyoeye-equ-champion-disposition-design.md); frozen spine design [`2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md`](2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md) (code path `science/tokyo_eye/v8/`); Sprints 1–5 LOCKED/GREEN  
**Amends operator practice:** reboot train theory from sprint docs **exactly**; deviate only where first-pass *last* runs were known mistakes (documented below)  
**Does not open:** affinity Core≥0.40 promote; C1 topology curriculum; pathway/PPI; init from `@champion` v5

---

## 0. What this is

Cold restart of EQU **geometry health** after the Pearson-reprint champion was dispositioned as non-trunk.

Code and graph contracts from Sprints 1–5 stay. **Weights start from Equiformer pretrained bank + random/spine-init hyp stack** — not from v5.

Goal of *this* card: a sealed **geometry-healthy** checkpoint that earns the right to enter the historical Mode / Sprint 8–9 ladder — with meters that would have rejected the Aug smoke path.

---

## 1. Θ (lens)

| Role | Pointer | Use |
|------|---------|-----|
| **Init frontend** | `checkpoints/tokyoeye/pretrained/equiformer_v3_baseline.pt` + weight map (Sprint 5 convention) | Frozen or LR-tiny per boot knobs below |
| **Init spine** | Fresh hyp projector / attention / MoE (no v5 load) | Train |
| **Compare-only** | Mode C S9 local ckpt; `@champion` v5 | Never mixed into means; never init |
| **Forbidden init** | `507d54…`, any `eqf_affinity_*` best, C1 best | Hard fail if load attempted |

---

## 2. Claims boundary

**May say:** under this boot corpus and knobs, train was finite; disc/trunk spread and MoE specialization met or missed pre-registered bars.

**May not say:** production-ready; affinity champion; KRAS understanding; “same as first pass”; Pearson success.

---

## 3. Pure geometry invariant (hard)

After Euclidean→hyperbolic **lift**, stay on the manifold through hyp graph store.

| Allowed | Forbidden |
|---------|-----------|
| Einstein / Klein **barycenters** | Tangent-space linear / mix / pool as geometry substitute (`exp₀(W log₀(·))` layers as MP stand-in) |
| Documented single lift `exp₀` at projector | Vendor shortcuts that leave the ball |

Boot card records whether current `attention.py` `_tangent_linear` remains; any change is a **separate** card (not silent).

---

## 4. Train contract — LOCKED (boot)

| Knob | Lock |
|------|------|
| Corpus | Stage-A / Sprint-documented small panel (same IDs as original boot path in Sprints 7–8 docs — **no** silent KRAS-only shrink). Exact PDB table filled in implementation plan before execute |
| Epochs | Per plan (short boot, not 150 C1). Stop at locked N — no “one more because val moved” |
| Equiformer | Start **frozen** (bank + adapters). Unfreeze only on a later card after geometry Pass |
| Loss | Geometry stack only (dehydron / SDRP / margin / MoE CV+quota). **No** affinity head |
| τ | Curriculum from Sprint engine defaults for boot; document start/end in plan |
| Wrap | Sprint-8 style freeze once set — do not re-median each epoch |
| Curvature | Learned `c` — never hardcoded |
| Device | Science container CUDA |
| Best ckpt | Lowest train loss among epochs with finite `z_hyp` **and** MoE specialization Pass on that epoch’s last step (see §6) |

**Abort:** NaN/Inf; CUDA OOM after one retry skip policy as C1; any load of dispositioned champion bytes.

---

## 5. Fold-diverse probe (mandatory, not optional)

Every K epochs and at end: frozen forward on a **locked** diverse panel (reuse B0 theme coverage idea: ≥1 chain per Ig / lysozyme / grasp / TIM / globin / P-loop — IDs locked in plan).

Probe is **grade-adjacent**: boot cannot Pass if probe shows sat≈1 and MoE monopoly even if train MoE looks flat-healthy.

---

## 6. Hygiene Pass (only Pass on this card)

All must hold at end-of-boot probe τ (documented in plan):

1. **Finite:** probe chains + one home structure load with finite `z_hyp` / learned `c`.  
2. **Volume:** mean probe `boundary_saturation` **< 0.50** (boundary_radius documented; not vacuous vs τ). Trunk radius spread **not** ≈0 (pre-register floor, e.g. spread ≥ 0.05).  
3. **MoE alive:** `moe_load_min` **> ρ_floor** (default 0.05) on ≥4/6 probe themes — **and** loads are **not** required to be uniform 0.25.  
4. **Specialization watch (report + soft fail):** per-expert dehydron or rim enrichment vs chance — report. Soft-fail if monopoly or all experts statistically identical on labels (details in plan). Flat 0.25/0.25/0.25/0.25 alone is **not** Pass evidence.

**Affinity / Pearson:** not scored. Logging optional watch only; cannot Pass or promote.

---

## 7. Deviates from first-pass mistakes (explicit)

| First-pass / retrain mistake | Boot rule |
|------------------------------|-----------|
| Promote on bare Pearson/Spearman scrape | **Forbidden** on this card and until geometry Pass exists |
| `finetune_hyp` promoted after `finetune_all` failed | Affinity ladder is a **later** card; must not skip failed stages without new spec |
| Train MoE health trusted over eval/probe | Probe monopoly **fails** boot even if train MoE looks good |
| Loose-disk memos as truth | MLflow + vault SSOT |

---

## 8. Out of this card

- `@champion` / vault moving-tag retarget  
- Unfreezing Equiformer  
- Sprint 10 affinity  
- C1 / B1 / pathway  
- Tangent-path rewrite (unless separate approved card)  

---

## 9. Artifacts

- Run dir: `checkpoints/tokyoeye/runs/equ_cold_boot_<date>/`  
- Stamp: `data/gates/tokyo_eye_equ_cold_boot.json` — `hygiene_pass`, probe table, Θ sha of frontend bank, **`biology_pass: false`**, **`affinity_pass: false`**  
- MLflow: `tokyoeye/equiformer-v3-moe/geometric/full-stack`, run name `equ_cold_boot_*` (metrics + ckpt artifact; **do not alias**)

---

## 10. Acceptance of *this spec*

Spec done when: init/forbidden init explicit; Pass bars include volume + MoE specialization (not flatness); probe panel locked in follow-on plan; disposition respected.

**Execution** requires: disposition approved + this card approved + implementation plan with PDB tables.
