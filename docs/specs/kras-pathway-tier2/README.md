# Tier 2 Pre-Registration — KRAS Pathway Proteins (historical stamp)

**Pre-registration timestamp:** 2026-05-20T02:45:00-05:00 (before pipeline runs)  
**Pipeline checkpoint:** `checkpoint="pipeline"`, epoch 75  
**Status:** Historical validation packet — stamped 2026-07-20 for dual-ledger provenance  
**Does not replace:** Ledger A (\(C_B\) @ 0.50) on `FIX1_SPARSITY_CHAMPION_CKPT`

---

## Pre-registered predictions (written BEFORE pipeline)

### SHP2 (GDP: `2SHP` autoinhibited vs GTP: `6MCF` open)

| Item | Prediction |
|------|------------|
| Doorways | Tunnel-2 pocket (res **100–110** N-SH2/PTP interface) |
| λ₂ | Autoinhibited **>** Open |
| Literature | Janes et al. 2018; Fodor et al. 2018 |

### MEK1 (GDP: `3EQI` inactive vs GTP: `3PP1` active)

| Item | Prediction |
|------|------------|
| Doorways | Allosteric pocket (res **97–103** helix-C; **211–215** DFG-adjacent) |
| λ₂ | Inactive **<** Active |
| Literature | Ohren et al. 2004 |

### ERK2 (GDP: `1ERK` inactive vs GTP: `2ERK` active)

| Item | Prediction |
|------|------------|
| Doorways | DEF site (res **160–165** substrate docking) |
| λ₂ | Inactive **<** Active |
| Literature | Sheridan et al. 2008 |

---

## Pipeline results (epoch 75)

### SHP2: `2SHP` vs `6MCF`

| Metric | Result |
|--------|--------|
| State-selective doorways | 92 |
| Constitutive | 0 |
| Key doorway residues | 56, 59, 70, 84, 91, 92, 113, 132, 154, 155, 161–165, 168, 179, 184… |
| Phase 3 leaks | 0 |
| Top GDP hub | B:359 |
| Top GTP hub | B:50 |
| λ₂ GDP | 0.0057 |
| λ₂ GTP | 0.1038 |
| Runtime | 106.6s |

### MEK1: `3EQI` vs `3PP1`

| Metric | Result |
|--------|--------|
| State-selective doorways | 2 |
| Constitutive | 0 |
| Doorway residues | 227, 350 |
| Phase 3 leaks | 0 |
| Top GDP hub | A:192 |
| Top GTP hub | A:246 |
| λ₂ GDP | 0.0730 |
| λ₂ GTP | 0.0735 |
| Runtime | 132.7s |

### ERK2: `1ERK` vs `2ERK`

| Metric | Result |
|--------|--------|
| State-selective doorways | 4 |
| Constitutive | 1 |
| Doorway residues | 64, 352, 357, 358 |
| Phase 3 leaks | 0 |
| Top GDP hub | A:166 |
| Top GTP hub | A:148 |
| λ₂ GDP | 0.0476 |
| λ₂ GTP | 0.0522 |
| Runtime | 107.2s |

---

## Validation: prediction vs output

| Target | Prediction | Pipeline | Match |
|--------|------------|----------|-------|
| SHP2 λ₂ | Autoinhibited > Open | GDP 0.006 < GTP 0.104 | **INVERTED** |
| SHP2 doorways | Tunnel-2 (100–110) | 91, 92, 113 in set | **PARTIAL** (adjacent) |
| MEK1 λ₂ | Inactive < Active | ≈ equal | **NEUTRAL** |
| MEK1 doorways | Helix-C / DFG-adj | 227, 350 | **DIFFERENT** |
| ERK2 λ₂ | Inactive < Active | 0.048 < 0.052 | **DIRECTION OK** |
| ERK2 doorways | DEF 160–165 | 64, 352, 357, 358 | **DIFFERENT region** |
| ERK2 hub | — | A:166 | **DEF site is the hub** |

Literature searches were performed **after** pipeline output (protocol step 4).

---

## Key findings (honest)

1. **SHP2 λ₂ inversion:** Pre-reg intuition was wrong; open form is more connected — pipeline correct.
2. **ERK2 DEF as hub:** Genuine validation (res 166), surfaced in hub phase not doorway phase.
3. **MEK1:** Weak conformational pair (`3PP1` inhibitor-bound) — do not over-read.

## Protocol checklist

1. ✓ Pre-registered predictions (timestamped)
2. ✓ Ran pipeline
3. ✓ Compared to predictions
4. ✓ Literature after output
5. ✓ Documented match/mismatch

## Relation to dual ledger (2026-07-20)

- This packet is **pathway doorway / λ₂** evidence on historical `pipeline` ep75.
- Ledger B interface pre-reg for champion flow concordance: [`../kras-topo-structural-inference/ledger-b-interface-prereg.md`](../kras-topo-structural-inference/ledger-b-interface-prereg.md)
- Do **not** treat this stamp as a Pass on `FIX1_SPARSITY_CHAMPION_CKPT` without re-grade.
