# Platform concordance policy — closeout (hard Fail)

**Date locked:** 2026-07-21  
**Decision:** Retain smoke **FAIL**; retire generic knockout↔\(C_B\) concordance as a primary Pass claim. Do **not** lower the 0.50 bar.  
**Machine stamp:** [`data/gates/platform_concordance_policy_hard_fail.json`](../../../data/gates/platform_concordance_policy_hard_fail.json)  
**Protocol:** [`platform-concordance.md`](platform-concordance.md)  
**Grade artifact:** `checkpoints/v66/diagnostics/routing_sparsity/general_hub_alignment_smoke.json`

---

## What was asked

On the sparsity champion, does forward-knockout `out_effect` rank-order with classical Cα betweenness \(C_B\) on a non-KRAS `TRAINING_TARGETS` panel (`3PP0`, `2SHP`, `2HHB`), with Spearman > **0.50** and cutoff stability?

## What was measured

| Structure | Spearman(out_effect, \(C_B\)) | Pass (>0.50)? |
|-----------|-------------------------------|---------------|
| `3PP0` | ≈0.42 | No |
| `2SHP` | ≈0.41 | No |
| `2HHB` | ≈0.54 | Yes |
| **Panel** | median ≈0.42; **1/3** | **FAIL** |

Stability at 7.5 / 8.5 Å held. Soft modular ρ is real, not a cutoff artifact.

## Decision

**Hard Fail. Closed.**

- Do **not** re-scope to a softer bar (e.g. 0.40) to manufacture Pass.  
- Do **not** claim generic CB concordance for the platform / champion.  
- Smoke artifact remains the historical Fail record; further runs of the same bar are **report-only** unless a *new* pre-reg is stamped first.

## Related questions this raised (not answered here)

Soft concordance on modular folds (SRC/SHP2) reopened whether **different gradient / influence magnitudes are interpreted differently** by the trunk (Jacobian vs forward knockout; z-norm gradient-sign pathology).

**We cannot answer that with current tooling and governance:**

- Jacobian flow-influence rankings are **forbidden** as Pass on z-norm-on trunks ([`learned-flow-influence/ablation.md`](../learned-flow-influence/ablation.md); sign-flip defect).  
- Forward knockout is the allowed hub metric — it does **not** resolve whether distinct gradient channels mean distinct biology.  
- Disc vs trunk proxies remain non-interchangeable ([`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)).

Those threads stay **open as research questions**. They do **not** reopen or soften this concordance Fail.

## Forbidden claims

- “Platform CB concordance Pass” / generic hub-alignment Pass on `TRAINING_TARGETS`  
- Lowering Ledger A Spearman bar to explain modular ρ ≈ 0.42  
- Treating Jacobian / gradient-channel interpretation as settled by this smoke
