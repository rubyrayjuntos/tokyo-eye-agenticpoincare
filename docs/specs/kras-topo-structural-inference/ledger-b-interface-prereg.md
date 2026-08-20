# Ledger B — Interface Alignment Pre-Registration (SRC / SHP2)

**Status:** HISTORICAL LOCK — graded **FAIL**; primary Pass claim **retired** (Option 1, 2026-07-21)  
**Policy closeout:** [`ledger-b-interface-policy-closeout.md`](ledger-b-interface-policy-closeout.md) · stamp `data/gates/ledger_b_interface_policy_option1.json`  
**Date locked (pre-reg):** 2026-07-20  
**Checkpoint graded:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Machine stamp (pre-reg):** `data/gates/ledger_b_interface_prereg_src_shp2.json`  
**Grade artifact:** `checkpoints/v66/diagnostics/routing_sparsity/ledger_b_interface_alignment.json` (FAIL retained)  
**Discovery explanation (not a Pass):** [`ledger-b-phase4b-hub-literature-map.md`](ledger-b-phase4b-hub-literature-map.md)

> Do **not** re-run this \(I\) expecting a Pass, and do **not** expand \(I\) from Phase 4b hubs. A new functional claim requires a fresh pre-reg stamp before grading.

---

## Dual-ledger reminder

| Ledger | Metric | Bar | Role |
|--------|--------|-----|------|
| **A — Geometric** | Spearman(out_effect, \(C_B\)) | **> 0.50** + cutoff stability | Primary geometric platform grade |
| **B — Functional** | Flow enrichment / recall on **this** interface set | See Pass form below | Co-equal for modular proteins |

Do **not** lower Ledger A to 0.40. Ledger B explains modular \(\rho \approx 0.42\) only if interfaces were locked first.

---

## Pass form (Ledger B) — locked before run

For each structure (`3PP0`, `2SHP`):

1. Compute forward-knockout `out_effect` on the champion (same as concordance smoke).
2. Let \(H\) = top **10%** residues by `out_effect` (model hubs; **not** \(C_B\)).
3. Let \(I\) = pre-registered interface residue set below ∩ deposited residues present in the graph.
4. **Primary Pass:** recall@top-10% = \(|H \cap I| / |I|\) **≥ 0.25**  
   (at least one-quarter of documented interface sites land in model top-decile flow).
5. **Secondary (report):** enrichment = mean(`out_effect` on \(I\)) / mean(`out_effect` on complement) **> 1.25**.
6. **Panel Pass:** both structures meet primary Pass.

KRAS landmarks (81/114/156) stay audit-only and are **not** in these sets.

---

## SHP2 — `2SHP` (autoinhibited; chain A)

**Literature basis:** tunnel / latch allosteric sites at N-SH2–C-SH2–PTP interfaces (SHP099-class); N-SH2 backside latch; PTP catalytic cysteine.  
Refs: Fodor et al. 2018; Janes/Nichols SHP099 structural series; Roskoski / domain maps; Tier-2 packet tunnel-2 100–110.

### Set B1 — Tunnel allosteric core (primary)

| Resseqs | Rationale |
|---------|-----------|
| 111, 114 | C-SH2 tunnel contacts (R111, H114) |
| 249, 250, 253, 254, 257 | PTP tunnel wall (E249, E250, T253, L254, Q257) |
| 491, 492, 495 | PTP tunnel / covalent-mapped (P491, K492, Q495) |

### Set B2 — Latch / N-SH2–PTP interface (primary)

| Resseqs | Rationale |
|---------|-----------|
| 79, 80, 84 | N-SH2 latch (Q79, Y80, H84) |
| 265, 269, 281 | PTP latch (R265, Q269, N281) |
| 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110 | Tunnel-2 / N-SH2–PTP interface band (Tier-2 pre-reg) |

### Set B3 — Catalytic PTP (secondary; report enrichment)

| Resseqs | Rationale |
|---------|-----------|
| 459 | Catalytic Cys (PTP active site) |

**Union used for Pass:** \(I_{\mathrm{SHP2}} = B1 \cup B2\) (B3 report-only).

---

## SRC — `3PP0` (kinase domain construct; chain A)

**Literature basis:** canonical Src KD regulatory / catalytic elements (chicken/human Src numbering as used in Roskoski / Xu et al. KD literature).  
**Construct caveat:** `3PP0` is a **kinase-domain** deposit — SH2/SH3 intramolecular sites may be absent from the graph. Only residues present in the deposited chain enter \(I\).

### Set S1 — Catalytic spine / ATP site (primary)

| Resseqs | Rationale |
|---------|-----------|
| 273, 274, 275, 276, 277, 278, 279, 280, 281 | Gly-rich P-loop |
| 295 | Catalytic Lys (Lys295) |
| 310 | αC Glu (Glu310; Lys–Glu salt bridge) |
| 339, 340, 341, 342, 343, 344, 345 | Hinge |
| 384, 385, 386 | HRD catalytic loop |
| 404, 405, 406 | DFG |

### Set S2 — Activation loop / A-loop helix (primary)

| Resseqs | Rationale |
|---------|-----------|
| 404–432 | Activation loop window (inclusive) |
| 416 | Autophosphorylation Tyr416 |

### Set S3 — Regulatory domain contacts (conditional)

| Resseqs | Rationale |
|---------|-----------|
| 81–142 | SH3 (include **only if** present in `3PP0` graph) |
| 148–245 | SH2 (include **only if** present) |
| 527 | C-terminal regulatory Tyr (include **only if** present) |

**Union used for Pass:** \(I_{\mathrm{SRC}} = (S1 \cup S2)\) ∩ residues in graph; S3 appended only when those resseqs exist in the deposit.

---

## Mapping rule (mandatory before score)

1. Load PDB; build `auth_seq` → graph index map (same `parse_resseq` as topo matrix).
2. If `literature_to_auth_offset` is set on the structure stamp, map  
   `auth_seq = literature_resseq + offset` **before** intersection with the deposit.  
   For `3PP0`, offset **+433** (literature Lys295 → deposit `A:728:`; P-loop 273 → `A:706:`).  
   This is a **numbering map**, not a change to which functional sites were pre-registered.
3. Drop any mapped resseq missing from the graph (document `missing_from_deposit`).
4. Never add residues because they appeared in smoke top-k flow lists.

### Mapping amendment (2026-07-20, post first grade attempt)

First formal grade found `|I|=0` on `3PP0` because the stamp stored Roskoski/human KD numbers while the graph uses chicken-style auth_seq **706–993**. Functional set unchanged; offset stamped as `literature_to_auth_offset: 433`. Re-grade required for a valid Ledger B panel score.

---

## Explicit non-goals

- Do not use \(C_B\) top-k as the interface set.
- Do not change Ledger A bar.
- Do not grade MEK1/ERK2 in this stamp (separate Tier-2 historical packet).
- Do not run hub extraction in the same commit that only adds this pre-reg (ordering: stamp → then grade).

---

## Grade command

```bash
make grade-v66-fix1-ledger-b-interface-alignment
```
