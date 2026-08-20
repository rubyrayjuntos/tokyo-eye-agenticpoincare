# Section 13 Amendment — G12D·GppNHp Active Observational Reference

**Status:** Intent locked; **not executable** until PDB is named and countersigned  
**Parent:** [`design.md`](design.md) v2.0  
**Effect on primary gates:** **None** — audit / observational only

---

## Purpose

Add an allele-matched **G12D · GppNHp (or GTP analogue)** deposit as a secondary Active Observational structure so allele-variance critiques of `5VQ2` (G12V·GppNHp) can be audited without changing Pass/Fail triangulation.

---

## Constraints (must initial)

| # | Constraint | Sign-off |
|---|------------|----------|
| 1 | Primary calibration remains **`5VQ2`**. Pass rules \(\rho(4DSO,5VQ2)>\rho(4OBE,5VQ2)\) and \(\|\Delta E(4DSO,5VQ2)\|<\|\Delta E(4OBE,5VQ2)\|\) are unchanged. | ☐ |
| 2 | Named G12D·GppNHp structure is **report-only** — no Pass bar, no champion veto, no sparsity hyperparameter change. | ☐ |
| 3 | Deposit screened for **ligand-distortion artifacts** (prefer natural active ensemble geometry over heavily constrained co-crystal poses). | ☐ |
| 4 | Residue index map matches locked canonical set (12 / 81 / 114 / 156 / 163). | ☐ |
| 5 | Uncertainty telemetry remains **monitor-only**. | ☐ |

---

## PDB selection (fill before executable)

| Field | Value |
|-------|--------|
| PDB ID | *[TBD]* |
| Chain | *[TBD]* |
| Ligand / nucleotide | *[TBD — must be GTP or GppNHp / GNP]* |
| Allele at 12 | **ASP (G12D)** required |
| Resolution (Å) | *[TBD]* |
| Artifact screen notes | *[TBD]* |
| Proposed by | *[name / date]* |
| Structural biologist | ☐ signed _____________ date ______ |
| GNN academic | ☐ signed _____________ date ______ |

---

## Execution after countersignature

```bash
# Observational only — does not replace primary matrix Make target
PYTHONPATH=. python -m experiments.diagnostics.kras_topo_structural_matrix \
  --checkpoint checkpoints/v66/runs/fix1_s4_sparsity_confirm_continue_v2/v66_sparsity_champion.pt \
  --active-obs <PDB> \
  --observational-only
```

Stamp result under `checkpoints/v66/diagnostics/routing_sparsity/kras_topo_matrix_observational_<pdb>.json`.
