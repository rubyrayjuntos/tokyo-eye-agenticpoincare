# Tokyo Eye EQU — Dehydron wrap threshold AMEND (§5)

**Status:** **SIGNED** — Option A (Ray Swan, 2026-09-17)  
**Display lineage:** Tokyo Eye EQU  
**Gate stamp:** [`data/gates/tokyo_eye_equ_wrap_threshold.json`](../../../data/gates/tokyo_eye_equ_wrap_threshold.json)  
**Evidence:** [`checkpoints/v8/runs/freeze_recon_wrap_threshold_amend/corpus_median_descend_search.json`](../../../checkpoints/v8/runs/freeze_recon_wrap_threshold_amend/corpus_median_descend_search.json)  
**Does not open:** router epsilon-greedy; soft MoE; cone-geometry redesign  
**Amends:** freeze §5 wrap gate; addendum §1.3 row #11 (historical REVERT → superseded by this AMEND)

---

## 0. Disposition

**Chosen: A — AMEND `wrap_max`.**

| Rejected | Why |
|----------|-----|
| **C — fix wrap pipeline** | Double-cone (45° / 6.5Å / H→O) is deliberate Sprint-8 design, not an undercount bug. Classical Fernández ~19–26 was sphere-calibrated; lower cone counts are the expected mathematical consequence. Cone shape may be a future biophysics card; it must not gate label repair now. |
| **B — keep 19, separate label rule** | `dehydron_labels_from_edges` *is* R2 incidence. A second undocumented threshold would hide which number supervises training (same failure shape as dual-meaning eval metrics / multiple MLflow names). |

**Forbidden still:** silent runtime `set_dehydron_wrap_max` retune; adopting the offline 1UBQ thr=2 diagnostic as production without corpus search.

---

## 1. Binding procedure (corpus median-then-descend)

Same search the Sprint-8 curated-loader suite encoded for one structure, run once across the full Stage-A-12 enabled manifest (`manifests/v6_corpus_stage_a_small_v1.json` / `v8_stage_a_small_v1.json`):

1. Build R0–R5 graphs for all **12** enabled structures; collect undirected H-bond wrap counts.  
2. Pool wraps; start `τ = floor(pooled median)`.  
3. Descend `τ` while `max(per-structure dehydron_frac) ≥ 0.60`.  
4. Stop at first `τ` with max frac `< 0.60` (or `τ = 0`).  
5. **Require** the per-structure table at the chosen `τ` (not only the aggregate) — flag any structure still ≥ 0.60 or < 0.05.

---

## 2. Sealed corpus result (2026-09-17)

**Pooled wrap distribution (n=3651 H-bonds):**

| Stat | Value |
|------|-------|
| min / max | 0 / 17 |
| mean | ≈ 3.58 |
| p10 / p50 / p90 / p95 | 0 / **3** / 8 / 9 |

**Search:** start `τ=3` → `τ=2` still max frac 0.736 → **chosen `τ=1`**.

**Why not the offline 1UBQ thr=2:** the search starts at pooled median (p50=3) and descends past it until *every* structure clears `<0.60`; at τ=2 seven of twelve still sat ≥ ceiling. 1UBQ alone balances near thr=2; denser/larger Stage-A chains pull the corpus stop one step lower. Procedure agreement, not contradiction.

**Per-structure `dehydron_frac` at `wrap_max=1`:**

| Structure | frac | n_r1 | n_r2 |
|-----------|------|------|------|
| 1MBN:A | 0.392 | 126 | 41 |
| 1LYZ:A | 0.581 | 93 | 56 |
| 1BG1:A | 0.486 | 406 | 188 |
| 1F88:A | 0.260 | 310 | 59 |
| 2Z6H:A | 0.306 | 479 | 105 |
| 1HHP:A | 0.424 | 77 | 28 |
| 1TEN:A | 0.528 | 58 | 32 |
| 1UBQ:A | 0.329 | 64 | 16 |
| 1TIM:A | 0.385 | 197 | 62 |
| 4OBE:A | 0.467 | 127 | 55 |
| 1IVO:A | 0.487 | 375 | 175 |
| 2SHP:A | 0.503 | 349 | 173 |

- max = 0.581, min = 0.260, mean ≈ 0.429  
- **0** structures ≥ 0.60; **0** structures < 0.05  
- `n_r1 > 0` on every structure  

**Chosen `DEHYDRON_WRAP_MAX`:** **1** (rule unchanged: `wrap_count ≤ wrap_max → R2`).

**Retroactive caveat:** freeze_recon REVERT-era wrap=19 dehydron scores — [`tokyo_eye_v8_wrap19_label_saturation_caveat_v1`](../../../checkpoints/v8/runs/freeze_recon_wrap_threshold_amend/WRAP19_LABEL_SATURATION_CAVEAT.json). Theme biology/restore AUPRC already used wrap=1 — out of scope.

---

## 3. Acceptance checklist

1. ✅ Corpus histogram + search procedure recorded  
2. ✅ Chosen wrap yields dehydron_frac ∈ (0.05, 0.60) on all 12 Stage-A structures  
3. ✅ `n_r1 > 0` on all 12  
4. ✅ Gate stamp + freeze addendum §2.7 + code constant updated together  
5. Graph caches keyed by wrap hash — wrap=19 caches auto-miss; rebuild on next load  

---

## 4. Sign-off

**Approver:** Ray Swan  
**Signed:** Option A AMEND `DEHYDRON_WRAP_MAX` 19 → 1 from Stage-A-12 median-then-descend (Ray Swan 9/17/26)
