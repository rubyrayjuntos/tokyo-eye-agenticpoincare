# TP53 DNA-Binding Domain: Structure Inventory for DTIE Analysis

**Date:** 2026-05-12
**Purpose:** Identify optimal PDB structure pair for DTIE dehydron/conductance analysis of TP53 cancer mutations.

## Selected Pair: 2XWR (WT) vs 4IBS (R273H)

| Property | 2XWR (WT) | 4IBS (R273H) |
|----------|-----------|--------------|
| Resolution | 1.68Å | 1.78Å |
| State | Apo (no DNA/ligand) | Apo (no DNA/ligand) |
| Construct | Residues 89–293 (205aa) | Residues 94–293 (200aa) |
| R248 | R (WT) | R (WT) |
| R273 | R (WT) | H (MUTANT) |
| Rescue mutations | None | None |
| Polymer entities | 1 (protein only) | 1 (protein only) |
| Deposited | 2010 (Joerger et al) | 2013 (Cañadillas et al) |

**Common core:** Residues 94–293 (200 residues).
**Resolution match:** Excellent (0.10Å difference).
**Confounds:** None — both apo, single chain, no ligands.

## Why R273H (Not R248W or R175H)

### R248W — No clean structure exists
All R248W crystal structures in PDB carry 4–5 thermostabilizing background mutations (M133L, V203A, N239Y, N268D + R248W). R248W is a "structural" mutant that destabilizes the p53 fold — it cannot crystallize without rescue mutations. Using a rescue-mutant background would confound DTIE analysis (the stabilizing mutations themselves alter the dehydron network).

### R175H — Same problem
R175H (the most common TP53 hotspot) is also a structural mutant. No clean single-mutation crystal structure exists.

### R273H — Ideal for DTIE
R273H is a "contact" mutant: it maintains the overall DBD fold but loses the direct guanidinium–DNA phosphate contact. This means:
1. The protein is structurally stable → crystallizes without rescue mutations
2. Any dehydron/conductance changes detected by DTIE reflect genuine allosteric rewiring, not global unfolding
3. The question becomes: does loss of a DNA-contact residue propagate through the dehydron network, or is it purely local?

## Predictions (Falsifiable)

| Outcome | λ₂ Direction | Interpretation |
|---------|-------------|----------------|
| Archetype I | +increase | Contact loss propagates allosterically (like KRAS/SPOP) |
| Archetype III | −decrease | Regulatory decoupling (like KEAP1) |
| New (Archetype IV) | ~unchanged (<5%) | Pure contact loss — no network rewiring |

The third outcome would define a new archetype: mutations that abolish function through direct contact loss without allosteric propagation. This would be biologically distinct from the structural mutants (R248W, R175H) which likely destroy the network entirely.

## Alternative Pairs Considered

| Pair | WT | Mutant | Issue |
|------|-----|--------|-------|
| 2XWR / 4IBS | WT apo | R273H apo | **SELECTED** — optimal |
| 2OCJ / 4IBS | WT apo (219aa) | R273H (200aa) | Lower resolution WT (2.05Å) |
| 7B4H / 7B4A | WT+DNA+MQ | R273H+DNA | Drug (MQ) confound in WT |
| 2XWR / 4IJT | WT apo | R273H form II | 4IJT equivalent to 4IBS (same mutation, different crystal form) |
| 2XWR / 4IBQ | WT apo | R273C | R273C less common clinically than R273H |

## Execution Plan

```bash
# Download structures
uv run python -c "
from data_science.sub_agents.pdb.client import download_structure
download_structure('2XWR')  # WT apo
download_structure('4IBS')  # R273H apo
"

# Run DTIE pipeline
# GDP = WT (reference state)
# GTP = R273H (perturbed state)
# This follows the same convention as KRAS (GDP=WT, GTP=mutant)
```

## Residue Numbering Reference

For the DTIE pipeline, residue numbering follows PDB auth_seq_id:
- R273 is residue 273 in both structures (standard p53 numbering)
- The 5 extra N-terminal residues in 2XWR (89–93) are handled by normalization alignment
- Key functional residues to watch for doorways:
  - R248 (loop L3, DNA minor groove contact)
  - R273 (helix H2, DNA backbone contact) — the mutation site
  - R280 (helix H2, DNA major groove contact)
  - K120 (loop L1, DNA contact + acetylation site)
  - C176, H179, C238, C242 (zinc coordination, structural)
  - S249 (adjacent to R248, hepatocellular carcinoma hotspot)

## Context in Framework

This analysis extends the allosteric archetype framework (`2026-05-11-allosteric-archetype-framework.md`) to a tumor suppressor. All previous analyses (SPOP, DDX3X, KEAP1, KRAS) were on oncogenes or regulatory proteins. TP53 R273H tests whether the λ₂ directionality discriminator applies to loss-of-function mutations in tumor suppressors.
