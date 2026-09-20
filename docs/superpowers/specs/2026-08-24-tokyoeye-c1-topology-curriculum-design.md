# Tokyo Eye C1 — Topology curriculum (25 × 150)

**Status:** APPROVED 2026-08-24 — execute 25×150 from `@champion`; Equiformer frozen; no champion retarget  
**Date:** 2026-08-24  
**Depends on:** [`2026-08-24-tokyoeye-b0-topology-observation-design.md`](2026-08-24-tokyoeye-b0-topology-observation-design.md) (stamped)  
**Amends:** [`docs/specs/tokyo-eye-v8/biology-roadmap.md`](../../specs/tokyo-eye-v8/biology-roadmap.md) — geometry curriculum **between B0 and B1**; this is **not** B1  
**Does not open:** B1 teleconnections, Sprint 10.2, pathway / resistance, MSA/GO losses, chem-MVP, champion retarget

---

## 0. What this is

The first **non-KRAS-specialist** train on the current champion. Twenty-five small/medium chains, six topology themes, 150 epochs. Init from `models:/TokyoEye@champion`. Equiformer stays frozen. Loss is the existing geometry stack (dehydron / SDRP / mechanism margin / MoE balance), not affinity.

B0 (frozen forward) showed the champion **loads** a fold-diverse panel with finite `z_hyp`, but **eval at `tau_end=0.995` parks every residue on the rim**, MoE collapses to **E3**, and theme rhyme is mixed (TIM/globin rhyme; Ig/lysozyme/grasp singleton; P-loop mixed). This card is the first attempt to **teach** those geometric lessons instead of only measuring them.

The curriculum theory (locked 2026-08-24): diversity **inside topology** — different families, shared packing lessons (sheets, barrels, globin helices, P-loop). Conserved patterns are how we **pick** structures, not a new MSA/GO loss. Oncogenic pathway work remains a look-ahead risk, not a milestone here.

---

## 1. Θ (lens)

| Role | Pointer | Use |
|------|---------|-----|
| **Init (SSOT)** | `models:/TokyoEye@champion` via `resolve --alias champion` (registry v5 as of 2026-08-24; sha256 `507d54bd6d8fb6c2…`; Core Pearson 0.404 on `finetune_hyp`) | Starting weights |
| **Graph recipe** | Sprint-8 4OBE wrap retune, **freeze `dehydron_wrap_max=1`** for the whole run | Same physics underwrap as the B0 stamp |
| **Compare-only** | Mode C S9 `checkpoints/tokyoeye/runs/eqf_mode_c_s9_20260823/tokyoeye_best.pt` | Not an init. Not mixed into C1 means |

Do not hardcode a `.pt` as SSOT. Do not init from Mode C. Do not load or train the affinity head on this card.

**Affinity Core ≥ 0.40 is not a C1 Pass.** After the run, one Core eval is a **watch** (catastrophic abort if Core < 0.30). Do not retarget `@champion`. Do not run `finetune_all`.

---

## 2. Claims boundary

**This card may say:** under this 25-chain corpus and this init, train was finite; rim/MoE moved or did not; holdout theme rhyme changed vs B0; 4OBE stayed finite.

**This card may not say:** production-ready geometry; allosteric sites; the model understands KRAS; sheets “do the same thing” in biology; EGFR/SHP2/resistance; B1 conduit vs scramble; a new champion.

**Forbidden (standing rules):** disc-alone hubs ([`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)); evidential Investigation as the trusted signal ([`VIEWER_INVESTIGATION_CORRECTNESS.md`](../../audit/VIEWER_INVESTIGATION_CORRECTNESS.md)); numeric hardcoded curvature; swapping a failed PDB for a different topology.

---

## 3. Why B0 changes the knobs

| B0 fact | C1 lock |
|---------|---------|
| Forward at `tau_end=0.995` → `boundary_saturation=1`, `mean_radius≈0.995` on all 18 | **Curriculum `tau_end=0.90`** (start 0.70). Post-train observe at **0.90**. A diagnostic forward at 0.995 is compare-only, not the grade |
| MoE eval load ≈ `(0,0,0,1)` (E3 monopoly) | Keep Sprint-9 `cv_coeff` + `moe_quota_coeff`. Freeze **entire** Equiformer frontend (bank **and** SE(3)-lite adapters). Only spine trains — same freeze as champion `finetune_hyp`. Unfreezing Equiformer already hurt Core once; this card does not retry it |
| Default wrap τ=19 → ρ=1 everywhere | **`dehydron_wrap_max=1`** (B0 retune). Do not re-median every epoch |
| Training on all 18 B0 chains would make a re-B0 a memorization test | **Hold out one B0 chain per theme** (below). Train 25 = 12 remaining B0 + 13 new |

---

## 4. Split — LOCKED

### 4.1 Hold out of **this train** (post-train observation)

These are the B0 members **not** in the loss. After epoch 150, re-run the B0 observation protocol on this set (plus 4OBE as home, which **is** in train).

| Theme | PDB | Chain | Why held |
|-------|-----|-------|----------|
| Ig-like | `1HNG` | A | Two-domain IgSF; B0 loaded n=175 |
| Lysozyme-like | `1ALC` | A | Same fold, different function (α-lactalbumin) |
| Ubiquitin / β-grasp | `1A5R` | A | SUMO-1 NMR model 1 — keep NMR out of the loss |
| TIM barrel | `1NAL` | `1` | Other 8-barrel family; file uses numeric chain **1** (~291 res). Size stretch + T1000 risk |
| Globin | `2HHB` | B | Hb β; B0 Stage-A holdout list |
| P-loop NTPase | `1GKY` | A | Third P-loop family; tests whether KRAS+ADK in the loss generalizes |

Do not put STAT3 `1BG1`, β-catenin `2Z6H`, EGFR ECD `1IVO`, or SHP2 `2SHP` in train **or** this holdout.

### 4.2 Train corpus — LOCKED (25)

X-ray. One CA-complete polymer. Not three crystals of one gene. Not `4OBE`/`4DSO`/`6OIM`. Size: small ~70–130, medium ~150–250. If a row fails to load, log `n_skip`, **do not swap topology**, continue if that theme still has ≥2 train chains.

**From B0 (12):** `1TEN` A, `1FNA` A, `1LYZ` A, `1GHL` A (first CA-complete polymer), `1UBQ` A, `1PGB` A (`grasp_cousin`), `1TIM` A, `1HTI` A, `1MBN` A, `1ASH` A, `4OBE` A (KRAS **home — stays in the loss**), `1AKE` A.

**New (13):**

| Theme | PDB | Chain | Why it is in the lesson | n_res band |
|-------|-----|-------|-------------------------|------------|
| Ig-like | `1WIT` | A | Twitchin fnIII (not tenascin / not 10Fn3) | small |
| Ig-like | `2RHE` | A | VL domain — same sandwich, antibody lineage | small |
| Lysozyme-like | `1LZ1` | A | Human C-type (organism replicate vs hen/avian already in) | small |
| Lysozyme-like | `153L` | A | Goose g-type — same packing field, not a third hen crystal | medium |
| Ubiquitin / β-grasp | `1NDD` | A | NEDD8 (X-ray) | small |
| Ubiquitin / β-grasp | `1WM3` | A | SUMO-2 X-ray (SUMO lesson without NMR in the loss) | small |
| TIM barrel | `1YPI` | A | Yeast TIM | medium |
| TIM barrel | `1BTM` | A | Bacterial TIM | medium |
| TIM barrel | `1MXS` | A | KDPG aldolase (*P. putida*) — other 8-barrel family (`1NAL` is holdout) | medium (~225) |
| Globin | `1HHO` | A | Human Hb **α** (β is holdout) | medium |
| Globin | `1ECA` | A | *Chironomus* Hb | small/medium |
| P-loop NTPase | `1TEV` | A | Human UMP/CMP kinase — not RAS, not ADK | medium (~196) |
| P-loop NTPase | `1WE2` | A | Mtb shikimate kinase — fourth P-loop family | medium (~176) |

**Theme counts in the loss:** Ig 4, lysozyme 4, grasp 4, TIM 5, globin 4, P-loop 4. TIM gets the extra chain because it was the only B0 sandwich/barrel **rhyme**.

**Do not use:** `1UD7` (designed ubiquitin mutant), `1TIT` (NMR Ig), three RAS crystals, `1F88` (7TM), Stage A shock chains listed above.

---

## 5. Train contract

| Knob | Lock |
|------|------|
| Epochs | **150**. Stop at 150. No “one more continue because val moved.” |
| Device | Science container CUDA (T1000). Keep Ollama off the GPU |
| Init | `resolve_alias_checkpoint(alias="champion")` → `load_state_dict(..., strict=False)`. Affinity-head keys ignored |
| Freeze | `system.frontend` **all** `requires_grad=False`. Spine (projector, hyp attn, MoE, SDRP / mechanism / evidential) trains |
| LR | Spine `lr_hyperbolic` from weight map (`3e-4`). No backbone LR |
| Loss | Existing v8 step: dehydron/mechanism margin + SDRP (`sdrp_coeff=0.1`) + MoE CV + quota. **No** affinity, MSA, GO, chem |
| τ radius | `CurriculumRadiusController(tau_start=0.70, tau_end=0.90, epochs=150)` |
| Gumbel | Weight-map exponential cool-down, stretched across 150 epochs (not left at Mode C’s 24-epoch α) |
| Wrap | `set_dehydron_wrap_max(1)` once at start |
| Curvature | Learned `c` from loaded spine via `require_learned_curvature`. Never hardcode |
| Batch | One structure per step; cycle the 25. Seed 0 |
| Best ckpt | Lowest train loss among epochs with finite `z_hyp` and `moe_load_min ≥ 0.02` on that epoch’s last step |
| Graph cache | Allowed; hash includes wrap_max=1 |

**Abort (stop, write stamp, do not promote):**

- NaN/Inf `z_hyp` or loss
- CUDA OOM on a chain after one retry at empty cache — skip that chain (`n_skip`), do not replace it; abort the **run** only if a theme drops below 2 train chains
- Core Pearson watch **< 0.30** if the optional end-of-run affinity eval is executed
- Any alias / vault / `@experimental` move attempted by the operator script (script must not expose those flags)

**Holdout probe during train:** every 25 epochs, frozen forward on the 6 holdouts at `tau=0.90`. Log H2–H5. Do not backprop on holdouts.

---

## 6. What to record (grade of the **run**, not of biology)

Reuse B0 IDs. Per holdout and per `4OBE`, at end-of-run `tau=0.90`:

| ID | Record |
|----|--------|
| H1 | Load OK, `n_res`, helix/sheet/coil counts |
| H2 | Finite `z_hyp`, finite learned `c` |
| H3 | Trunk radius + `boundary_saturation`, paired with mean disc `r` (no disc hub rank) |
| H4 | `moe_load_e*` — collapse vs spread is health |
| H5 | Mean ρ (and wrap τ) vs helix/sheet/coil |
| H6 | Dehydron AUPRC if labels exist; else `auprc_na` |

**C1 hygiene Pass (the only Pass on this card):** all 6 holdouts + `4OBE` finite (H2); mean holdout `boundary_saturation` **< 0.50** at τ=0.90 (B0 was 1.0 at 0.995 — this is a different operating point, stated as such); `moe_load_min` **> 0** on at least 4 of 6 holdouts.

**Theme rhyme:** same narrative labels as B0 (rhyme / singleton / collapse / mixed / insufficient), computed on the **6 holdouts** (one per theme — so “rhyme” here means holdout class-ρ sits with the **train-theme band from a frozen snapshot of train chains**, not three-holdout rhyme). If that comparison is too thin, report holdout H5 vs B0 H5 on the same PDB and stop. **Do not** turn rhyme into a Pass.

**KRAS home:** `4OBE` H2–H5 vs holdout `1GKY`. If only KRAS looks sane, say so. Not a pathway claim.

**Affinity watch (optional, end only):** Core Pearson on the sealed split. Record. Do not Pass/Fail C1 on 0.40.

---

## 7. Out of this card

- Champion / vault / `@experimental` moves
- Unfreezing Equiformer or a second `finetune_all`
- 10.2, B1 AlleleSens / conduit vs scramble
- STAT3 / EGFR ECD / SHP2 / β-catenin
- MSA, conservation, or GO as inputs or losses
- Parked chem track
- Raising `tau_end` back to 0.995 as a “fix” without a new spec

---

## 8. Artifacts

- Manifest: `manifests/v8_c1_topology_curriculum_v1.json` (`enabled`, `theme`, `role=train|holdout`, B0 flags as needed)
- Run dir: `checkpoints/tokyoeye/runs/eqf_c1_topology_curriculum_<date>/`
- Stamp: `data/gates/tokyo_eye_v8_c1_topology_curriculum.json` — hygiene Pass/Fail, per-PDB metrics, theme narratives, Θ sha256, wrap_max, **`biology_pass: false`**
- MLflow: experiment `tokyoeye/equiformer-v3-moe/geometric/full-stack`, run name `c1_topology_curriculum_champion_init` (metrics + ckpt artifact; **do not alias**)

---

## 9. Acceptance of *this spec* (not of the model)

The spec is done when: 25 train IDs + 6 holdouts match the tables; freeze/τ/wrap/init are explicit; B0 facts drove the knobs; 10.2 and B1 stay out; no champion retarget.

C1 **execution** is a follow-on plan after this file is approved.
