# Tokyo Eye EQU — Current champion disposition (v5 / sha 507d54…)

**Status:** APPROVED 2026-09-15 — record-only; no GPU train  
**Date:** 2026-09-15  
**Display lineage name:** **Tokyo Eye EQU** (do not promote new work under version-number marketing)  
**Depends on:** MLflow archaeology `equ_archaeology_first_vs_retrain_20260915` (`21ff1994be0d4013ab8d611992fda95b`); B0 stamp; C1 stamp  
**Does not open:** C1.1 from this champion; affinity rematch; pathway / PPI; alias retarget to a new `@champion` without a later boot+ladder Pass

---

## 0. What this is

Operator disposition of `models:/TokyoEye@champion` **v5** (sha256 `507d54bd6d8fb6c2…`, vault `tokyoeye-eqf-champion` / `tokyoeye-eqf-507d54bd6d8fb6c2`).

This weight is the **Aug 2024–2025 retrain seal** that reprinted Core Pearson ≈ 0.404 via **`finetune_hyp`**, after first-pass weights were lost. It is **not** accepted as the EQU geometry trunk.

---

## 1. Evidence summary (sealed)

| Fact | Pointer |
|------|---------|
| First-pass seal Core R ≈ 0.407 via **finetune_all** ladder | MLflow `2bc5e1f7…` + `data/gates/tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json` |
| First-pass **weights missing** | `checkpoints/v8/runs/tokyo_eye_v8_affinity_s1011_finetune_all` absent |
| Current champion Core R ≈ 0.404 via **finetune_hyp** | MLflow `d6c2f4af…` |
| Aug `finetune_all` rematches **failed** 0.40 gate | local `eqf_affinity_s1011_finetune_all_*` summaries (Core ≈ 0.35–0.39) |
| Fold-diverse forward: rim park + MoE monopoly | B0 stamp `tokyo_eye_v8_b0_topology_observation.json` |
| Topology curriculum cannot rehabilitate | C1 stamp: hygiene **Fail**; train MoE ≠ holdout MoE |

---

## 2. Claims boundary

**This card may say:** current `@champion` v5 is a lesson object / affinity reprint; do not continue EQU geometry curriculum from it; evidence must be retained.

**This card may not say:** first-pass weights were proven geometrically healthier (bytes gone); biology understanding; destroy vault releases; silent alias move.

---

## 3. Disposition — LOCKED once approved

| Action | Rule |
|--------|------|
| **Continue training from v5** | **Forbidden** for EQU reboot / C1.x / B0-style curriculum |
| **Destroy evidence** | **Forbidden** — keep MLflow runs, gate stamps, GitHub Release vault tags, local ckpt cache |
| **`@champion` alias** | Remain pointing at v5 until a **new** boot+ladder Pass + explicit human promote. Treat as **non-trunk for EQU geometry** in operator practice (compare-only / affinity archaeology) |
| **Init for next train** | Cold / pretrained Equiformer bank only — never `507d54…` / v5 best |
| **Analysis SSOT** | MLflow (+ vault). No loose job-hunt analysis JSON as record of truth |

---

## 4. Forbidden (standing)

- Pearson / Spearman alone as promote criterion  
- Substituting `finetune_hyp` for a failed `finetune_all` without a new card  
- Tangent-space shortcuts post-lift (barycenters OK; log/exp linear as geometry substitute **veto**)  
- Mixing v6 / v7 / Fix-1 / Mode-C-as-champion contamination into EQU trunk weights  

---

## 5. Artifacts

- This spec  
- MLflow archaeology run `21ff1994be0d4013ab8d611992fda95b`  
- Existing B0 / C1 stamps (unchanged)  

## 6. Acceptance of *this spec*

Approved when: disposition table is accepted; next work is the EQU cold-boot card, not C1.1 from v5.
