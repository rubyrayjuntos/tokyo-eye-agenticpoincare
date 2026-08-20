# KRAS nucleotide-only basin transition — pre-registration

**Status:** LOCKED for grading — written **before** nucleotide-only grade runs  
**Date locked:** 2026-07-21  
**Workstream:** `kras_basin_routing_automaton` child **2**  
**Depends on:** Child 1 `kras_basin_observation_classify` FAIL closeout (nucleotide axis Pass; allele OFF Fail — bars held)  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Make (to wire):** `make grade-v66-kras-nucleotide-basin-transition`  
**Stamp:** `data/gates/kras_nucleotide_basin_transition_prereg.json`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_nucleotide_basin_transition.json`

---

## Motivation

Child 1 showed **channel split**:

| Channel | Result |
|---------|--------|
| Nucleotide macro-basin (OFF GDP vs ON GppNHp) | Pass — same_nuc 0.941 > cross_nuc 0.911 |
| Allele micro-state on OFF | Fail — \(N_{12}\) \|Δ\| absorbed into Switch / backbone |
| Allele on ON | Pass alone — not enough for joint Child 1 Pass |

This child **promotes the validated macro-state sensor** into the state-machine transition rules and **explicitly parks** allele gating (`BLOCKED_ON_ALLELE_SENSOR`). No bar softening on Child 1.

Architecture (unchanged):

| Role | Actor |
|------|--------|
| Sensor | GNN hyp depth on \(R_\star\) (Switch I∪II ∪ arm \(N_{12}\)) |
| Referee | Deposit nucleotide class only |
| Memory | Discrete basins `{OFF, ON}` (allele ignored) |

## Roster

Same four-quadrant matrix; allele is **label-only / report-only**, not Pass:

| PDB | Nucleotide basin | Allele (report) |
|-----|------------------|-----------------|
| **4LPK** | OFF | G12 |
| **5US4** | OFF | G12D |
| **6GOD** | ON | G12 |
| **6GOF** | ON | G12D |

## Observation (sensor — locked)

Identical to Child 1 nucleotide axis:

- Forward champion once per structure.  
- \(F =\) `cone_depth` / `dist0(x_hyp)` on \(R_\star\).  
- \(R_\star =\) Switch I (25–40) ∪ Switch II (57–75) ∪ arm \(N_{12}\) (OFF arm uses Child-2 \(N_{12}\) from `5US4`/`4LPK`; ON from `6GOF`/`6GOD`).  
- Pairwise Spearman on shared \(R_\star\) resseqs.

## Physics guards (referee — locked)

1. Nucleotide class parse: `GDP` → OFF, `GppNHp|GNP|GTP…` → ON (Fail if unparseable).  
2. Parsed class must match roster table for all four.  
3. **Allele / G12 identity is not a Pass guard** (report-only audit).

## Transition rules (memory — locked smoke)

Static classify + legal-transition table (no MD, no multi-step dynamics engine):

| From → To | Legal? | Guard |
|-----------|--------|-------|
| OFF → ON | yes | nucleotide deposit is ON-class |
| ON → OFF | yes | nucleotide deposit is OFF-class |
| OFF → OFF | stay | — |
| ON → ON | stay | — |

**Grade procedure:**

1. Guards parse; assign each structure basin \(B \in \{\mathrm{OFF},\mathrm{ON}\}\) from deposit.  
2. Compute same_nuc / cross_nuc as Child 1.  
3. **Nearest-centroid assign** (leave-one-out): for each structure, mean \(F\) of the other same-roster structures in each basin (using shared \(R_\star\) alignment to the held-out map); assign predicted basin by higher mean Spearman to basin members (or Euclidean distance in aligned \(F\) — locked below).  
   **Locked assigner:** predicted basin = argmax over \(\{\mathrm{OFF},\mathrm{ON}\}\) of mean Spearman of held-out \(F\) vs other structures labeled that basin.  
4. Transition smoke: for each structure, legal self-stay and cross-basin flip are consistent with deposit guard (i.e. predicted basin equals deposit basin).

## Pass form

All required:

1. **Guards:** all four nucleotide classes match roster.  
2. **Separation:** `same_nuc > cross_nuc` on \(R_\star\) hyp-depth (replicate Child 1 nucleotide gate).  
3. **Classify:** leave-one-out predicted basin == deposit basin for **all four** structures.  

**Report-only:** allele labels; pairwise Spearmans; Child 1 allele OFF/ON deltas; disc views.

## Interpretation

- **Pass:** macro-state sensor + nucleotide referee support OFF↔ON basin memory for the automaton — allele channel remains parked.  
- **Fail:** cannot replicate nucleotide separation and/or LOO classify errors — do not fall back to allele; do not soften.

## Explicit non-goals

- Allele / G12D Pass gates (parked)  
- Softening Child 1 allele bars  
- MD / coordinate integration  
- Hyperbolic MP trunk  
- Dynamic multi-step simulation beyond LOO classify + legal table smoke  
- Hop-2 / current-flow allele pierce (future pre-reg under parked allele sensor)
