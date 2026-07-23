# TokyoEye-v8 Sprint 10.1.1 — Production MOL2/SDF Ligand Upgrade

**Date:** 2026-07-22  
**Status:** APPROVED 2026-07-22 — implement per plan  
**Depends on:** Sprint 10.1.0 HETATM bootstrap — Core Pearson \(R \approx 0.366\) (clears 0.30 rematch band; short of 0.40 gate)  
**Parent:** [`2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface-design.md`](2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface-design.md)  
**Splits SSOT (unchanged):** `manifests/v8_pdbbind_refined_cluster30_v1.json`  
**Plan:** [`docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint10-1-1-production-ligand.md`](../plans/2026-07-22-tokyo-eye-v8-sprint10-1-1-production-ligand.md)

---

## Problem

Sprint 10.1.0 proved R6 + asymmetric cross-attn unlocks ligand-conditioned affinity (Core \(R: 0.007 \rightarrow 0.366\)). The residual gap to the provisional **0.40** gate is attributed to **Option A HETATM information loss**: missing formal charges, bond/aromatic typing, and clean connectivity. Per frozen policy: **do not** inflate via LR/MoE knobs — upgrade the ligand input source.

---

## Non-goals

- Touching `JointPocketAffinityHead` attention math (only `lig_in` input dim if feature width grows)  
- Mutating R0–R5, `v8_biophys_s8` cache, MoE guilds, or leak-wall splits  
- Adding RDKit / Open Babel as a hard dependency for 10.1.1 (native MOL2/SDF text parse preferred)  
- Softening the 0.40 gate or claiming docking ΔG equivalence  

---

## Answers locked from exploration

| Question | Lock |
|----------|------|
| Write production spec under `docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-1-1-production-ligand.md` | **Yes** (this file) |
| Prefer local ligand assets under `data/pdbbind/` before HETATM fallback | **Yes** (Option C hybrid, now enforced) |

---

## Frozen operational contracts

| Contract | Lock |
|----------|------|
| Resolution order | `data/pdbbind/ligands/{pdb_id}.mol2` → `{pdb_id}.sdf` → filtered HETATM (10.1.0) |
| Staging root | `data/pdbbind/ligands/` (create if missing; empty OK until dump staged) |
| R6 geometry | Unchanged: \(d(C_\beta/C_\alpha,\mathrm{lig})\le 4.5\,\text{Å}\), on-the-fly, off protein cache |
| Head / MoE / cache | Frozen layout; only ligand feature tensor richness changes |
| Splits | Untouched cluster30 manifest |

### Path resolver (pseudocode)

```python
def resolve_ligand_path(pdb_id: str, root=Path("data/pdbbind/ligands")) -> Path | None:
    pid = pdb_id.lower()
    for ext in (".mol2", ".sdf", ".MOL2", ".SDF"):
        p = root / f"{pid}{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None  # → HETATM fallback
```

Telemetry: log `ligand_source ∈ {mol2, sdf, hetatm}` per complex; summary rates in `run_summary.json`.

---

## Feature tensor upgrade

### Sprint 10.1.0 (frozen archaeology)

`LIGAND_FEAT_DIM = 10`: elements C/N/O/S/P/Halogen/Other + charge bins (HETATM default neutral).

### Sprint 10.1.1 production width

**Expand to `LIGAND_FEAT_DIM = 12`** (backward-compatible constant bump; `JointPocketAffinityHead.lig_in` reads `LIGAND_FEAT_DIM`):

| Ch | Meaning |
|----|---------|
| 0–6 | Element one-hot: C, N, O, S, P, Halogen(F/Cl/Br/I), Other *(unchanged)* |
| 7–9 | Formal charge bins: Negative / Neutral / Positive — **from MOL2 charge column or SDF property; never silent zero when present** |
| 10 | **Aromatic flag** (1 if mol2 atom type marks aromatic / `@` / `ar`, else 0) |
| 11 | **Heavy-atom bonded degree bin** (normalized \( \min(\mathrm{deg},4)/4 \) from MOL2 bond block; SDF connectivity; HETATM fallback → 0) |

HETATM fallback (still Option A): channels 7–9 default Neutral; channels 10–11 → 0 (explicit deficit telemetry `hetatm_feature_pad=1`).

### Parser scope (lightweight, no RDKit)

1. **MOL2:** `@<TRIPOS>ATOM` + `@<TRIPOS>BOND` — element, charge, atom_type → aromatic heuristic; bond endpoints → degree.  
2. **SDF:** atom block (coords + atomic number) + bond block; charge from atom charge column / `M  CHG`; aromatic from bond type 4 if present.  
3. **Coordinate frame:** ligand coords must match the complex PDB frame used for protein (PDBBind refined ligands are typically already aligned). If a file is missing coords or empty → fall through to HETATM.

---

## Architecture (unchanged topology)

```
data/pdbbind/ligands/{pdb}.mol2|.sdf  ──┐
                                        ├──> LigandAtoms + feats[N,12]
pdb_cache/{pdb}.pdb HETATM (fallback) ──┘
                                        │
                         R6 on-the-fly (unchanged)
                                        │
              JointPocketAffinityHead (lig_in: 12 → d)
```

---

## Train progression

Mirror 10.1 discipline:

1. Unit tests: MOL2/SDF parse fixtures; charge/aromatic/degree; resolver order; HETATM fallback pads ch 10–11.  
2. `head_only` smoke with `--joint-head` (expect `ligand_source=mol2` rate > 0 once dump staged).  
3. `finetune_hyp` rematch from Sprint 9 spine (same LRs / λ_aux=0.10).  
4. Gate: Core Pearson \(R \ge 0.40\); if \(0.30 \le R < 0.40\) after full MOL2 coverage → new pre-reg (not LR churn).

**Prerequisite:** stage ligand files for train∪val∪core IDs under `data/pdbbind/ligands/`. Until staged, runs remain HETATM-equivalent (no false gate claim).

---

## Files to add/touch

| Path | Role |
|------|------|
| `docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-1-1-production-ligand.md` | This design |
| `science/tokyo_eye/v8/ligand_interface.py` | MOL2/SDF parse, 12-ch feats, resolver |
| `science/tokyo_eye/v8/affinity_head.py` | `lig_in` uses `LIGAND_FEAT_DIM` (already) — verify dim=12 |
| `experiments/training/v8/run_affinity_s10.py` | Log `ligand_source` rates |
| `tests/v8/test_ligand_mol2_sprint1011.py` | Parser + resolver unit tests |
| Plan (after approval) | `docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint10-1-1-production-ligand.md` |

**Do not modify:** `r0_r5_graph.py`, biophysics cache version, MoE architecture, cluster30 manifest.

---

## Open points (approve to lock)

1. **Resolver root** `data/pdbbind/ligands/` + mol2→sdf→HETATM — **LOCKED**.  
2. **Feature width 12** (aromatic + degree) — **LOCKED**.  
3. **No RDKit for 10.1.1** — native text parsers — **LOCKED**.  
4. **Ligand dump staging** via `experiments/training/v8/stage_ligand_assets.py` — **LOCKED**.  
5. **Telemetry** `ligand_source` + `run_summary` rates — **LOCKED**.

---

## Docs after approval

- Implementation plan under `docs/superpowers/plans/`  
- Then: parsers + tests → stage ligands → `head_only` smoke → `finetune_hyp` Core gate  
