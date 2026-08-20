# Ledger B — Interface Policy Closeout (Option 1)

**Date locked:** 2026-07-21  
**Decision:** **Option 1** — retain FAIL; retire latch/tunnel \(I\) as a primary Pass claim; append discovery explanation.  
**Machine stamp:** [`data/gates/ledger_b_interface_policy_option1.json`](../../../data/gates/ledger_b_interface_policy_option1.json)  
**Pre-reg (unchanged historical lock):** [`ledger-b-interface-prereg.md`](ledger-b-interface-prereg.md)  
**Grade artifact (FAIL retained):** `checkpoints/v66/diagnostics/routing_sparsity/ledger_b_interface_alignment.json`  
**Discovery explanation:** [`ledger-b-phase4b-hub-literature-map.md`](ledger-b-phase4b-hub-literature-map.md)  
  Machine: `checkpoints/v66/diagnostics/routing_sparsity/ledger_b_phase4b_hub_map.json`

---

## Decision

Tokyo Eye graded the **locked** question: do champion knockout hubs recover the pre-registered latch/tunnel (SHP2) and ATP-spine / A-loop (SRC) sets at recall@top-10% ≥ 0.25?

**Answer: No.** Panel FAIL stands. We do **not** expand \(I\) post hoc, do **not** adopt Phase 4b hubs as a silent Pass set, and do **not** re-stamp interface concordance.

## What the pre-reg asked vs what the model weighted

| Structure | Locked \(I\) (primary) | Observed hub skew (Phase 4b) | recall@top-10% |
|-----------|------------------------|------------------------------|----------------|
| `2SHP` | Latch + tunnel / N-SH2–PTP band (SHP099-class) | PTP catalytic neighborhood + N-SH2 bulk (e.g. Arg32); thin \(H \cap I\) | ≈0.07 |
| `3PP0` | ATP spine + activation loop | C-lobe / αC-flank packing more than spine \(I\) | ≈0.07 |

Phase 4b is **interpretative only** — a clear record of discovery candidates, not a new Pass form.

## Governance consequences

1. **`ledger_b_interface_recall` remains FAIL** in the biology phase status ledger.
2. Latch/tunnel (and SRC spine) \(I\) is **retired as a primary Pass claim** — historical pre-reg + Fail grade remain the record; future grades of the same \(I\) are report-only unless a **new** pre-reg is stamped *before* a new grade.
3. **Re-prereg of discovery hubs** (PTP/N-SH2; Src C-lobe/αC) is a *separate* science claim — allowed later only with a fresh stamp before scoring.
4. **Still allowed:** Ledger B focal sensitivity Pass; SHP2 inactive→open OOD; monopoly Gini; KRAS classical rematch. These do not imply interface concordance with locked \(I\).

## Forbidden claims

- “Ledger B interface Pass” / latch–tunnel concordance on the champion  
- Treating Phase 4b hubs as if they were the original pre-reg \(I\)  
- Lowering Ledger A Spearman bar to “explain” modular ρ ≈ 0.42 via this Fail
