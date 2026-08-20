# B1 — Teleconnections / functional allostery (prereg)

**Status:** FROZEN  
**Date locked:** 2026-07-21  
**Parent:** [`biology-roadmap.md`](biology-roadmap.md)  
**Θ SSOT:** `HEALTHY_V7_CKPT` = `checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt`  
**Stamp:** [`data/gates/tokyo_eye_v7_b1_teleconnections_prereg.json`](../../../data/gates/tokyo_eye_v7_b1_teleconnections_prereg.json)  
**Defs:** [`investigation-allele-epistasis-metrics.md`](investigation-allele-epistasis-metrics.md)  
**Conduit \(N_{12}\):** same as Child-2/3 — [`kras-g12-neighborhood-graft-prereg.md`](../kras-topo-structural-inference/kras-g12-neighborhood-graft-prereg.md) / `neighborhood_n12()`  
**Draft trail:** [`b1-teleconnections-prereg-draft.md`](b1-teleconnections-prereg-draft.md)

---

## Claim

Under sealed Hyp MP Θ, a local G12 allele perturbation produces a **conduit-distributed** post-lift response (pathway neighborhood), stronger on the ON arm than OFF, and stronger than an off-pathway scramble — measured by \(\mathrm{AlleleSens}\) on \(N_{12}\).

**Investigation (not Pass):** whether that response looks **hub–hub** vs **hub–pocket** (or other destination) in latent space. Working prior is hub–hub; numbers may prefer pocket — report both, do not Fail on the winner.

---

## Panel — LOCKED

| Arm | WT | Mut | Chain |
|-----|----|-----|-------|
| OFF | `4LPK` | `5US4` | A |
| ON | `6GOD` | `6GOF` | A |

Shared deposited resseqs only. New bars — **no** inheritance of Fix-1 Pass claims.

---

## Conduit \(N_{12}\) — LOCKED

On each arm (mut graph contacts, intersect WT∩mut):

1. Seed resseq **12**  
2. ∪ Cα neighbors of 12 on **mut** @ **10 Å**  
3. ∪ Switch lock partners **32, 60, 61** if present  

**Scramble:** Child-2 scheme — same cardinality, mut residues outside {12} ∪ Switch-I(25–40) ∪ Switch-II(57–75), sorted map onto sorted \(N_{12}\).

---

## Metrics

### Pass-relevant

| ID | Metric |
|----|--------|
| M1 | \(\mathrm{AlleleSens}(S_{\mathrm{wt}}, S_{\mathrm{mut}}; N_{12})\) per arm |
| M2 | \(\mathrm{AlleleSens}(S_{\mathrm{wt}}, S_{\mathrm{mut}}; N_{\mathrm{scramble}})\) per arm |
| M3 | Finite learned curvature; no NaN in `x_hyp` / distances |

### Investigation-only (never Pass/Fail)

| ID | Metric |
|----|--------|
| I1 | Absolute AlleleSens values (soft monitor — log only) |
| I2 | Singleton \(d_{\mathbb{B}}\) WT↔mut at candidate **hub** residue(s) (report prior hub-migration indices where alignable) |
| I3 | Singleton \(d_{\mathbb{B}}\) WT↔mut at candidate **pocket** / catalytic residues (prereg list in grade script from literature lock) |
| I4 | Hub vs pocket: which destination moved more (narrative only) |
| I5 | Per-residue conduit displacement profile (pathway ordering) |
| I6 | Fix-1 G12D hub-migration rank / related — **soft monitor** |
| I7 | Euc Cα distance of same pairs — report-only |

---

## Pass form — LOCKED (relative + scramble; B + soft absolute)

**Primary (all required on sealed Θ):**

1. **ON vs OFF (conduit):**  
   \(\mathrm{AlleleSens}_{\mathrm{ON}}(N_{12}) > \mathrm{AlleleSens}_{\mathrm{OFF}}(N_{12})\)  
   (propagation stronger than absorption on the same conduit definition).

2. **Conduit vs scramble (ON arm):**  
   \(\mathrm{AlleleSens}_{\mathrm{ON}}(N_{12}) > \mathrm{AlleleSens}_{\mathrm{ON}}(N_{\mathrm{scramble}})\).

3. **Conduit vs scramble (OFF arm):**  
   \(\mathrm{AlleleSens}_{\mathrm{OFF}}(N_{12}) > \mathrm{AlleleSens}_{\mathrm{OFF}}(N_{\mathrm{scramble}})\)  
   *or* both OFF values below a tiny numerical noise floor (report; if OFF is near-zero absorption, scramble need not “win” — lock in grade: if \(\mathrm{AlleleSens}_{\mathrm{OFF}}(N_{12}) < 10^{-4}\), bar 3 is **waived** with note).

4. **Hygiene:** finite curvature; all AlleleSens finite; disc/views not used as Pass.

**Secondary (soft absolute monitor — not Pass):**

- Log \(\mathrm{AlleleSens}\) magnitudes on ON/OFF/scramble.  
- No Fail if absolute values are small (Child-2 lesson).

**Explicitly not Pass:** hub vs pocket destination winner; Fix-1 hub-migration rank; Euc distances; NIG channels; champion parity vs Fix-1.

---

## Destination policy — LOCKED

- Report **both** hub-oriented and pocket-oriented singleton displacements (I2–I4).  
- Prior belief: **hub–hub** teleconnection.  
- If pocket displaces more, that is an **investigation finding**, not a Fail.  
- Do not require a single destination for Pass.

---

## Non-claims

Hidden-driver discovery; ligands; phylogeny; NIG Pass; biology champion vs Fix-1; free energy; disc-rim pathway Pass; destination-type Pass.

---

## Implementation notes

- Forward on sealed Θ only; Jacobian forbidden (consistent with graft probes).  
- Align residues by deposited resseq ∩ both structures of the arm.  
- Curvature from ckpt — never hardcode \(c\).  
- Artifact dir: `checkpoints/v7/diagnostics/b1_teleconnections/`  
- Make: `make grade-v7-b1-teleconnections`

## Exit

Closeout stamp `data/gates/tokyo_eye_v7_b1_teleconnections_closeout.json` → open B2 epistasis prereg if Pass or after explicit waiver.
