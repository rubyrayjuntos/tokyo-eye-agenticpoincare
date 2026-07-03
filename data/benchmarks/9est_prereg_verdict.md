# 9EST/1FLE Cryptic-Interface Pre-Registration — Verdict Record

**Date (UTC):** 2026-07-02  
**Status:** **REFUTE** (locked decision rule §8)  
**Pre-registration:** `experiments/diagnostics/PREREG_9EST_1FLE_cryptic_interface.md`  
**Harness:** `experiments/diagnostics/evaluate_9est_prereg.py`  
**Results artifact:** `data/benchmarks/9est_phase34_results.json`

---

## Summary

On the hand-picked maximally favorable single case (static 9EST elastase vs 1FLE elafin interface), **Tokyo Eye S1 does not recover the cryptic interface** under frozen confirmatory criteria.

| Endpoint | PeSTo | S1 | S2 | SASA |
|---|---|---|---|---|
| recovery@3 | 0 | 0 | 0 | 0 |
| ROC AUC | 0.404 | 0.560 | 0.546 | 0.603 |
| SASA margin | — | −0.043 | −0.058 | — |

**Verdict:** `recovery@3=0` AND `ROC_AUC(S1)=0.560 < 0.65` → **REFUTE**

---

## What this means

1. **Thesis kill on this discriminator** — static dehydron + disc-radial shell signal does not carry the elafin binding interface on unbound elastase where PeSTo also fails at k=3.
2. **Exposure confound** — S1 underperforms freesasa heavy-atom SASA (margin negative). Any weak ranking signal is not separable from solvent exposure on this structure.
3. **No benchmark license** — PPDB5/MaSIF full benchmark is **not** authorized by this result (CONFIRM gate not met).
4. **Negative result is informative** — n=1 REFUTE on a PeSTo-published static failure is the intended falsification outcome; record and move on.

---

## Procedure integrity

| Phase | Gate | Result |
|---|---|---|
| 0 Freeze | harness + prereg committed | ✅ |
| 1 Label lock | \|I_gold\|=26, \|U\|=229 | ✅ |
| 2 PeSTo precondition | recovery@3=0 | ✅ PASS |
| 3 Tokyo Eye channels | lever_a on 9EST | ✅ |
| 4 Blind evaluate | `decide()` | ✅ REFUTE |
| 5 Record | §12 filled | ✅ |

S1 recovery@k sweep k=1..10: all **0**.

---

## Next steps (not confirmatory)

- Select another PeSTo static-failure from the 20-complex MD benchmark set.
- Do **not** run PPDB5/MaSIF on the strength of this case.
- Any scoring or threshold changes require §11 amendment and are exploratory only.
