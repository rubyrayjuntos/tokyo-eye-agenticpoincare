# KRAS basin observation classify — closeout (Child 1)

**Status:** FAIL (bars held)  
**Date:** 2026-07-21  
**Machine stamp:** [`data/gates/kras_basin_observation_classify_closeout.json`](../../../data/gates/kras_basin_observation_classify_closeout.json)  
**Pre-reg:** [`kras-basin-observation-classify-prereg.md`](kras-basin-observation-classify-prereg.md)  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_basin_observation_classify.json`  
**Make:** `make grade-v66-kras-basin-observation-classify`  
**Workstream:** `kras_basin_routing_automaton` child **1**  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`

---

## Verdict

**Fail.** Nucleotide basin separation and deposit guards pass; allele site-specificity fails on the OFF arm (ON arm alone would pass). Bars not softened.

| Gate | Result |
|------|--------|
| Guards (GDP/GppNHp + G12/G12D) | ✓ all four |
| same_nuc (0.941) > cross_nuc (0.911) | ✓ |
| OFF mean\|Δ\| N12 (0.032) > scramble (0.036) | ✗ |
| ON mean\|Δ\| N12 (0.038) > scramble (0.021) | ✓ |

### Pairwise R★ Spearmans

| Pair | ρ | Axis |
|------|---|------|
| 4LPK–5US4 | 0.922 | same-nuc OFF |
| 6GOD–6GOF | 0.960 | same-nuc ON |
| 4LPK–6GOD | 0.913 | cross |
| 4LPK–6GOF | 0.906 | cross |
| 5US4–6GOD | 0.911 | cross |
| 5US4–6GOF | 0.914 | cross |

## Interpretation

- **Sensor sees nucleotide basins:** same-state pairs are more concordant on Switch∪\(N_{12}\) hyp-depth than cross-state pairs — conformational axis is readable without MD.  
- **Allele stress is not OFF-site-true under R★ null:** on GDP scaffolds, mean \|Δdepth\| on \(N_{12}\) does **not** exceed elsewhere in \(R_\star\) (Switch-heavy pool). ON (GppNHp) does concentrate allele Δ on \(N_{12}\).  
- Consistent with sealed perturbation map: backbone / switch geometry dominate; allele is a weak local overlay, stronger when the active scaffold already flexes conduits.

## Architecture honesty

Automaton remains **blueprint** for allele-gated transitions. Do not claim full sensor+referee memory Pass. Dynamic multi-step child stays deferred. No bar changes.

## Next (requires new pre-reg)

Options (pick explicitly — do not invent Pass):

1. Allele observation on a different locked field (`out_effect` / conduit knockout) with new bars  
2. Report-only allele axis; promote nucleotide-only classify as a narrower child  
3. Hold automaton until stronger allele sensor exists
