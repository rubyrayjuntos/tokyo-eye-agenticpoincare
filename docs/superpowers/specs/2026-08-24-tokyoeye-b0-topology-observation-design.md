# Tokyo Eye B0 — Topology observation on champion

**Status:** DRAFT for review — not an implementation license until this spec is approved  
**Date:** 2026-08-24  
**Amends:** [`docs/specs/tokyo-eye-v8/biology-roadmap.md`](../../specs/tokyo-eye-v8/biology-roadmap.md) B0 (Θ lock + forward smoke)  
**Does not open:** B1 teleconnections, Sprint 10.2, pathway / resistance inference, 25×150 train

---

## 0. What this is

Observation of how the **current champion** behaves when shown a **small/medium, fold-diverse** panel. Weights stay frozen. No affinity retune. No biology Pass/Fail.

The curriculum theory (locked in chat, 2026-08-24): diversity **inside topology** — different families, shared geometric lessons (sheets, barrels, globin packing, P-loop). Conserved patterns are how we **pick** structures, not a new loss. Oncogenic pathway work is a look-ahead risk of KRAS-only training, not a milestone of this card.

---

## 1. Θ (lens)

| Role | Pointer | Use |
|------|---------|-----|
| **Primary** | `models:/TokyoEye@champion` (registry v5 as of 2026-08-24; Release `tokyoeye-eqf-507d54bd6d8fb6c2`; Core Pearson 0.404 on `finetune_hyp`) | All B0 numbers |
| **Compare-only** | `checkpoints/tokyoeye/runs/eqf_mode_c_s9_20260823/tokyoeye_best.pt` | Same panel, same metrics, labeled `mode_c_s9_compare`. Never mixed into a champion mean |

Resolve champion via governance (`resolve --alias champion`). Do not hardcode a local `.pt` as SSOT. Mode C S9 may be a filesystem cache; if missing, skip compare and stamp `mode_c_s9_absent`.

**Affinity Core ≥ 0.40 is not a B0 Pass.** It only explains which file is champion.

---

## 2. Claims boundary

**This card may say:** under these structures and this checkpoint, `z_hyp` was finite; rim/trunk looked like X; MoE loads were Y; sheet residues on the three Ig-like chains did or did not rhyme.

**This card may not say:** this residue is an allosteric site; the model understands KRAS; sheets “do the same thing” in biology; EGFR/SHP2/resistance; production-ready geometry; B1 conduit vs scramble.

**Forbidden (existing standing rules):** disc-alone hubs or pairwise claims ([`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)); evidential Investigation as the trusted signal ([`VIEWER_INVESTIGATION_CORRECTNESS.md`](../../audit/VIEWER_INVESTIGATION_CORRECTNESS.md)); numeric hardcoded curvature.

---

## 3. Panel — LOCKED (6 themes × 3 proteins)

X-ray unless noted. One CA-complete chain. **Not** three crystals of one gene. **Not** `4OBE`/`4DSO`/`6OIM`.

If a row fails to load (no CA-complete, NMR unusable, multi-model chaos), log `n_skip` for that PDB, do not silently swap a different topology, and still grade the theme if ≥2 of 3 loaded.

| Theme | PDB | Chain | Why it is in the lesson | Notes |
|-------|-----|-------|-------------------------|--------|
| Ig-like sandwich | `1TEN` | A | Tenascin fnIII (Stage A) | Home example for this theme |
| Ig-like sandwich | `1FNA` | A | Fibronectin 10th type III | Same sandwich, different protein |
| Ig-like sandwich | `1HNG` | A | CD2 IgSF (X-ray) | Two Ig domains in one chain (~176 res). Replaces NMR `1TIT`. If load fails: `1CD2` A, same theme only |
| Lysozyme-like | `1LYZ` | A | Hen C-type (Stage A) | |
| Lysozyme-like | `1ALC` | A | Baboon α-lactalbumin | Same fold, different function |
| Lysozyme-like | `1GHL` | A | Avian C-type (pheasant/guinea-fowl entry) | Pick first CA-complete polymer; still C-type, not a third hen crystal |
| Ubiquitin / β-grasp | `1UBQ` | A | Ubiquitin (Stage A) | |
| Ubiquitin / β-grasp | `1PGB` | A | GB1 | Neighboring CATH `3.10.20.10`; **flag `grasp_cousin`** |
| Ubiquitin / β-grasp | `1A5R` | A | SUMO-1 | NMR, **model 1 only**. If loader rejects NMR: skip and keep 2/3 |
| TIM barrel | `1TIM` | A | Trypanosome TIM (Stage A) | |
| TIM barrel | `1HTI` | A | Human TIM | Same enzyme, different organism (within-theme replicate, not a new family) |
| TIM barrel | `1NAL` | A | N-acetylneuraminate lyase | Other 8-barrel family; tetramer in file — **chain A only** (~291 res, top of medium) |
| Globin | `1MBN` | A | Sperm-whale myoglobin (Stage A) | |
| Globin | `2HHB` | B | Human Hb β (Stage A holdout list) | |
| Globin | `1ASH` | A | *Ascaris* Hb domain I | Invertebrate globin, not lupin; still globin topology |
| P-loop NTPase | `4OBE` | A | KRAS G-domain (home) | |
| P-loop NTPase | `1AKE` | A | Adenylate kinase (Stage A `ADK`) | Different family, same P-loop topology |
| P-loop NTPase | `1GKY` | A | Yeast guanylate kinase | Third family; not a RAS paralog |

**Held out of this card (too large / shock):** STAT3 `1BG1`, β-catenin `2Z6H`, EGFR ECD `1IVO`, SHP2 `2SHP`.

**Size intent:** small ~70–130; medium ~150–250; `1NAL` A is the one stretch. No STAT3-class chains.

---

## 4. What to record (hygiene + theme rhyme)

Per structure, both Θ copies (champion required; Mode C if present):

| ID | Record | Notes |
|----|--------|--------|
| H1 | Load OK, `n_res`, DSSP helix/sheet/coil counts | Skip line if no CA-complete |
| H2 | Finite `z_hyp`, finite learned `c` from checkpoint | Fail hygiene if NaN/Inf |
| H3 | Trunk radius stats + `boundary_saturation` | Pair with disc `r`; **do not rank hubs on disc** |
| H4 | MoE `moe_load_e*` | Collapse vs spread is health, not “expert = kinase” |
| H5 | Physics underwrap: ρ and τ vs DSSP class | Mean ρ on sheet vs helix vs coil |
| H6 | Dehydron AUPRC only if labels exist for that graph | Missing → `auprc_na`, not 0.0 |

**Theme rhyme (observation, not Pass):** for each theme, compare H5 sheet (or helix, for globin) distributions **across the three chains**. Narrative:

- **Rhyme:** the three Ig-like chains put sheet residues in a similar ρ/τ band, and that band is **not** the same as globin helix residues.
- **Singleton:** only `1TEN` looks like that.
- **Collapse:** every residue class looks the same (no geometric differentiation).

Do not threshold rhyme into a gate on this card.

**KRAS home check (observation):** `4OBE` H2–H5 vs the other two P-loop chains. If KRAS is the only P-loop that looks sane, say so. That is not a pathway claim.

---

## 5. Out of this card

- Training (including 25 proteins × 150 epochs). That is a **later** spec, init from this champion, after B0 is stamped.
- B1 AlleleSens / conduit vs scramble (needs its own v8 prereg).
- MSA/conservation or GO as inputs or losses.
- Side-chain chemistry train (parked chem track).
- Champion retarget, vault, or `@experimental` moves.

---

## 6. Artifacts

- Manifest: `manifests/v8_b0_topology_observation_v1.json` (this panel; `enabled` + `theme` + `grasp_cousin` / `nmr_model1` flags).
- Stamp: `data/gates/tokyo_eye_v8_b0_topology_observation.json` — hygiene, per-PDB metrics, theme narratives, Θ sha256, **no Pass on biology**.
- MLflow: experiment `tokyoeye/equiformer-v3-moe/geometric/full-stack`, run name `b0_topology_observation_champion` (metrics only; do not alias).

---

## 7. Acceptance of *this spec* (not of the model)

The spec is done when: panel IDs/chains match the table; claims boundary is explicit; Θ is champion; Mode C is compare-only; 10.2 and B1 are out of scope.

B0 **execution** is a follow-on plan after this file is approved.
