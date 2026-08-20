# KRAS basin routing automaton — observation classify (pre-registration)

**Status:** CLOSED — Fail (see [`kras-basin-observation-classify-closeout.md`](kras-basin-observation-classify-closeout.md); bars held)  
**Date locked:** 2026-07-21  
**Workstream:** `kras_basin_routing_automaton` child **1**  
**Depends on:** `gnn_perturbation_boundary` children 1–4 PASS (perturbation map sealed)  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Make:** `make grade-v66-kras-basin-observation-classify`  
**Stamp:** `data/gates/kras_basin_observation_classify_prereg.json`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_basin_observation_classify.json`

---

## Motivation

Children 1–4 mapped the perturbation boundary: seeds nudge, conduits must flex, post-lift `x_hyp` injection is site-true but thin under backbone inertia (WT↔mut hyp-depth ρ≈0.975). The next claim is **not** better continuous grafting — it is whether GNN **observation** fields plus deposit **physics guards** can place each structure in the correct **discrete basin** on the inhibitor-free four-quadrant roster.

Architecture (locked):

| Role | Actor |
|------|--------|
| Sensor | GNN hyperbolic inference fields (post-lift) |
| Referee | Deposit-derived chemical physics guards |
| Memory | Discrete basin labels (state machine) |

This child is the **static classify smoke** for that automaton — no MD, no coordinate integration, no hyperbolic MP rewrite.

## Roster (ground-truth basins)

| Basin | Nucleotide | Allele | PDB | Chain |
|-------|------------|--------|-----|-------|
| `OFF_WT` | GDP | Gly12 | **4LPK** | A |
| `OFF_MUT` | GDP | Asp12 (G12D) | **5US4** | A |
| `ON_WT` | GppNHp / GNP | Gly12 | **6GOD** | A |
| `ON_MUT` | GppNHp / GNP | Asp12 (G12D) | **6GOF** | A |

Historical `4OBE` / `4DSO` / `5VQ2` are **out of scope** (wrong allele / superseded).

## Observation field (sensor — locked)

Per structure, forward champion once. Primary observation vector:

**F = hyperbolic depth `dist0(x_hyp)` / `cone_depth` on residues in**

\[
R_\star = (\text{Switch I } 25\text{–}40) \cup (\text{Switch II } 57\text{–}75) \cup N_{12}^{\text{union}}
\]

where \(N_{12}^{\text{union}}\) = union of Child-2/3 \(N_{12}\) definitions recomputed per structure’s mut-matched recipe on each mut (`5US4`, `6GOF`) then intersected with that structure’s present resseqs (WT uses the mut \(N_{12}\) of its nucleotide arm).

Aligned Spearman uses shared resseqs across the pair under comparison.

**Report-only (not Pass):** full-chain depth Spearman; routing entropy scalar; disc/Klein views; `out_effect` hub lists.

## Physics guards (referee — locked)

Parsed from deposit / sequence **before** looking at GNN scores:

1. **Nucleotide class:** `GDP` vs `GppNHp|GNP|GTP` (Fail structure if unparseable).  
2. **Allele at 12:** Gly vs Asp (Fail if missing / mismatch to roster table).  
3. **Chain:** A only.

Guards do **not** use GNN outputs. They label expected basin axes and validate roster integrity.

## Protocol

1. Load four PDBs; parse guards; abort Fail if any guard fails.  
2. Forward each with champion (`topology_three_vector`).  
3. Build \(R_\star\) depth vectors; compute all six pairwise Spearmans on shared residues in \(R_\star\).  
4. **Nucleotide axis:**  
   - same_nuc = mean( ρ(4LPK,5US4), ρ(6GOD,6GOF) )  
   - cross_nuc = mean( ρ(4LPK,6GOD), ρ(4LPK,6GOF), ρ(5US4,6GOD), ρ(5US4,6GOF) )  
5. **Allele axis (both arms):** For OFF (`4LPK`/`5US4`) and ON (`6GOD`/`6GOF`):  
   - δ = |F_mut − F_wt| on shared \(R_\star\) residues  
   - mean_δ_N12 = mean δ on \(N_{12}\) ∩ shared  
   - mean_δ_scramble = mean δ on same-cardinality sites drawn from sorted \((R_\star \cap \text{shared}) \setminus N_{12}\)  
     (within-observation null; Child-2 distal donors lie outside \(R_\star\) by construction and cannot serve this control)  
6. Optional report: assign each structure nearest nucleotide centroid in F-space (leave-one-out) — not Pass.

## Pass form

All required:

1. **Guards parse:** all four structures nucleotide + G12 allele match roster.  
2. **Nucleotide separation:** `same_nuc > cross_nuc` on observation F.  
3. **Allele site-specificity:** on **both** OFF and ON arms, `mean_δ_N12 > mean_δ_scramble`.

## Interpretation

- **Pass:** hyp transport observation separates nucleotide basins; allele stress concentrates on \(N_{12}\) vs distal scramble under physics-correct labels — supports sensor+referee basin memory without MD.  
- **Fail:** observation cannot separate OFF/ON and/or allele delta is non-specific — automaton stays blueprint; do not invent softer bars post-hoc.

## Explicit non-goals

- Continuous coordinate / MD integration  
- Hyperbolic message-passing trunk  
- Disc-alone Pass  
- Jacobian / gradient hub rankings  
- Dynamic multi-step transition simulation (future child; needs this smoke first)  
- Hop-2 Euclidean grafts or G12V/C fan-out (separate pre-regs under perturbation workstream)
