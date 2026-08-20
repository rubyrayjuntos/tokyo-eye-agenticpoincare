# Tokyo Eye v7 — biology roadmap (hub / teleconnection track)

**Status:** OPEN (discussion-locked order)  
**Date:** 2026-07-21  
**Θ SSOT:** `checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt` (`HEALTHY_V7_CKPT`)  
**Machine stamp:** [`data/gates/tokyo_eye_v7_biology_roadmap.json`](../../../data/gates/tokyo_eye_v7_biology_roadmap.json)  
**Geometry:** [`EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md) · disc ≠ pathway ([`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md))  
**Investigation defs:** [`investigation-allele-epistasis-metrics.md`](investigation-allele-epistasis-metrics.md)  
**Evidential uncertainty:** PARKED — [`bprime-uncertainty-unpark.md`](bprime-uncertainty-unpark.md) (not a blocker for frozen-Θ probes)

---

## Locked priority (agreed)

| Order | ID | Theme | Status |
|-------|-----|--------|--------|
| **0** | **B0** | Prep (Θ lock, metric code, probe inventory, forward smoke) | **PASS** |
| **1** | **B1** | Functional allostery / long-range teleconnections | **FAIL** (first grade) — [`b1-teleconnections-prereg.md`](b1-teleconnections-prereg.md) · closeout [`tokyo_eye_v7_b1_teleconnections_closeout.json`](../../../data/gates/tokyo_eye_v7_b1_teleconnections_closeout.json) |
| **2** | **B2** | Epistasis cascades (residue-level; not multimeric theater) | After B1 grade |
| **3** | **B3′** | Directional topography (transport fields — **not** free energy) | After B2 |
| — | **B4** | Evolutionary / ancestral trajectories | Deferred (origin ≠ phylogeny) |
| — | **B5** | Zero-shot ligand / substrate | Deferred (chem-MVP parked) |

**Do not** run B1→B5 as “emergent discovery” without prereg bars. Each open item gets a prereg + gate stamp before grade.

---

## B0 — Prep (do before B1)

Why: sealed v7 is **disc-health Pass**, not a graded biology champion. Prep makes B1 honest and cheap.

| # | Task | Done when |
|---|------|-----------|
| B0.1 | Lock Θ: only `HEALTHY_V7_CKPT` for biology probes (rematch unc weights = compare-only) | Documented here + stamp |
| B0.2 | Naming: `AlleleSens` / `Epistasis` ≠ NIG aleatoric/epistemic | Defs doc + stamp |
| B0.3 | Implement pure metric helpers + unit tests (synthetic ball points) | Code + `make test` slice |
| B0.4 | Inventory Fix-1 / KRAS probes reusable as **compare-only** baselines | Table below |
| B0.5 | Forward smoke: sealed Θ → finite `x_hyp` + curvature on a small KRAS/Stage-A panel | Smoke note / gate |
| B0.6 | Draft B1 prereg skeleton (panels, \(N\), bars) — discuss before freeze | Open for review |

**Not required for B0:** evidential unpark; chem-MVP; promote; Pass vs Fix-1 champion pack (that can be a parallel later gate).

### Reusable compare-only inventory (v66 archaeology)

| Probe / closeout | Spec | Use for v7 |
|------------------|------|------------|
| G12D hub migration | `grade-v66-fix1-sparsity-g12d-hub-migration` | External baseline for hub rank / migration |
| KRAS topo matrix / four-quadrant ΔE | `kras-topo-structural-inference/` | Classical / edge controls — not hyp teleconnection Pass |
| G12 residue / neighborhood / latent grafts | `kras-g12-*-graft-*` | Perturbation-boundary patterns to **re-prereg** under Hyp MP |
| Ledger B / platform concordance | policy Fail retained | Do not reopen as Pass theater |
| AlleleSens / Epistasis defs | `investigation-allele-epistasis-metrics.md` | B1/B2 primary metrics |

---

## B1 — Teleconnections (functional allostery)

**Claim class:** Distal sites that share a **hyperbolic geodesic chain** with a transport hub show coordinated `x_hyp` response under a local allele perturbation (AlleleSens on prereg’d \(N\)), even when far in Euclidean 3D.

**Mechanism (disciplined):** Hyp MP + ball geodesics route hierarchical signals; measure post-lift, not disc-alone.

**Prep inputs:** B0 metrics + sealed Θ + KRAS-class WT/mut panel (reuse graft PDBs where possible).

**Out of scope:** “Hidden driver discovery” marketing; ligand pockets; phylogeny.

**Exit:** B1 closeout stamped (`make grade-v7-b1-teleconnections`). First grade **FAIL** — ON>OFF conduit held; conduit vs scramble failed both arms (scramble larger). Destination investigation: pocket > hub (not Pass). Next: human waiver vs redesign before B2.

---

## B2 — Epistasis cascades

**Claim class:** Double-mutant latent change is not the tangent sum of singles (`Epistasis` residual); cancel vs amplify on a prereg’d suppressor/driver panel.

**Mechanism:** `logmap₀` residual on frozen Θ ([defs](investigation-allele-epistasis-metrics.md)).

**Out of scope:** Giant multimeric assembly clustering at the rim (later, if ever).

**Exit:** B2 closeout; then optional B3′.

---

## B3′ — Directional topography (rewritten)

**Not** thermodynamic free energy / MD metastability from static density gradients.

**Claim class:** Directed transport / sink–source fields on `x_hyp` (and any directionality objectives already in-tree) are stable under B1/B2 perturbations and correlate with hub↔pocket coupling.

**Forbidden wording until ensemble prereg:** ΔG, saddle, folding pathway, thermodynamic landscape.

---

## B4 — Evolutionary trajectories (deferred)

Ball radius encodes **within-structure** hierarchy/burial, not phylogenetic time. Homologs near the origin ≠ ancestral consensus without a phylo-controlled prereg. Do not schedule before B1–B2.

---

## B5 — Ligand / substrate zero-shot (deferred)

Chem-MVP **PARKED**. No ligand channel on sealed trunk. Revisit only via [`chem-mvp-reengage`](../chem-mvp-reengage/README.md).

---

## Parallel (optional, not blocking)

| Item | Note |
|------|------|
| Formal biology pack vs Fix-1 champion | New prereg; compare-only champion under `checkpoints/v66/` |
| Evidential NIG unpark | Separate; correlation audit with AlleleSens/Epistasis only after informative NIG |

---

## Non-claims (standing)

- Sealed health = biology champion  
- Disc rim = allosteric pathway  
- NIG `ale`/`epi` = AlleleSens / Epistasis  
- Energy landscape from embedding density  
- Zero-shot ligand affinity  

---

## Next action

1. Implement **`make grade-v7-b1-teleconnections`** per frozen prereg.  
2. Grade → closeout stamp → open B2 if Pass (or waiver).

### B0 checklist

- [x] B0.1–B0.6 complete  
- [x] B1 prereg frozen ([`b1-teleconnections-prereg.md`](b1-teleconnections-prereg.md))  

