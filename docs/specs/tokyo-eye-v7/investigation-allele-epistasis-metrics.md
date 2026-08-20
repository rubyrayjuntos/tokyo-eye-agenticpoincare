# Investigation metrics on invariant Θ — allele sensitivity & epistatic coupling

**Status:** STARTING DEFINITIONS (parked with uncertainty work, 2026-07-21)  
**Applies to:** Frozen Tokyo Eye v7 (`Θ` fixed) · output `x_hyp` · hyperbolic inference layer  
**Unpark / evidential track:** [`bprime-uncertainty-unpark.md`](bprime-uncertainty-unpark.md)  
**Geometry SSOT:** [`docs/audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md)

These are **deterministic metric extraction functions** on top of an invariant model. They do **not** require a calibrated NIG head, and they do **not** replace evidential training when that track unparks.

---

## Naming lock (non-negotiable)

| Abbreviation in this doc | Full name | Symbol | **Not** the same as |
|--------------------------|-----------|--------|---------------------|
| **allele / AlleleSens** | Allele sensitivity | \(\mathrm{AlleleSens}\) | NIG **aleatoric** (`aleatoric_std` / training `ale`) |
| **epistasis / EpistasisCoupling** | Epistatic coupling residual | \(\mathrm{Epistasis}\) | NIG **epistemic** (`epistemic_std` / training `epi`) |

Historical training/code still uses `ale`/`epi` for evidential channels. Investigation metrics **must** use the long names (or `AlleleSens` / `Epistasis`) in code, gates, and MLflow keys until a rename sweep is prereg’d. Do not overload `ale`/`epi` in new probes.

---

## Premise

Invariant \(\Theta\) (e.g. `HEALTHY_V7_CKPT` / sealed health) is a fixed lens. Mutational / epistatic **investigation** is then:

\[
\text{metric} = f\big(x_{\mathrm{hyp}}(S; \Theta),\, \ldots\big)
\]

with \(\Theta\) never updated by the metric. That is compatible with “model is healthy” and still gives programmable handles for ON/OFF absorption, driver vs background, and non-additive rewiring.

---

## 1. Allele sensitivity (\(\mathrm{AlleleSens}\))

Local mutational disruption in hyperbolic space for a focus site (e.g. KRAS site 12) and neighborhood \(N\) (e.g. \(N_{12}\)):

\[
\mathrm{AlleleSens}(S_{\mathrm{wt}}, S_{\mathrm{mut}}; N)
=
\frac{1}{|N|}
\sum_{i \in N}
d_{\mathbb{B}}\Big(
  x_{\mathrm{hyp},i}(S_{\mathrm{wt}}),\;
  x_{\mathrm{hyp},i}(S_{\mathrm{mut}})
\Big)
\]

where \(d_{\mathbb{B}}\) is the Poincaré-ball (curvature-aware) geodesic distance on the model’s ball, and residue indices are aligned by the structure’s residue map.

**Intended read:** How hard a single substitution moves the local latent geometry after invariant GNN lift. Low ⇒ local absorption / damping; high ⇒ sharp neighborhood response (candidate driver / state-sensitive site).

**Starting-point caveats (acceptable for park; fix on unpark):**

- \(N\) must be prereg’d (1-hop Cα, conduit set, graft neighborhood, …).
- Cross-structure alignment (same chain/index convention) is load-bearing.
- Use learned curvature \(c\) from the ckpt — never a hardcoded \(c\).

---

## 2. Epistatic coupling (\(\mathrm{Epistasis}\))

Non-additive interaction of two mutations \(A\), \(B\) vs double mutant \(AB\), relative to WT. Work in the **origin tangent** via \(\mathrm{logmap}_0\) so displacements add:

\[
\Delta\mathbf{x}_A
=
\mathrm{logmap}_0\big(x_{\mathrm{hyp}}(S_A)\big)
-
\mathrm{logmap}_0\big(x_{\mathrm{hyp}}(S_{\mathrm{wt}})\big)
\]

(and likewise for \(B\), \(AB\)), then

\[
\mathrm{Epistasis}(S_A, S_B, S_{AB}; S_{\mathrm{wt}})
=
\big\|
  \Delta\mathbf{x}_{AB} - \big(\Delta\mathbf{x}_A + \Delta\mathbf{x}_B\big)
\big\|
\]

Norm and reduction (per-residue mean over a focus set, or global) must be stated in the probe prereg.

**Intended read:** Zero ⇒ additive drift in the tangent chart; large ⇒ cooperative / compensatory rewiring beyond the sum of singles.

**Starting-point caveats:**

- \(\mathrm{logmap}_0\) additivity is a **chart** approximation; large radii may later need parallel transport or geodesic parallelograms — good enough to park, not final geometry dogma.
- Requires four matched structures (WT, A, B, AB) with aligned residues.
- Separates **additive structural drift** from **non-linear coupling**; it is not “model hasn’t seen this” (that remains evidential / OOD epistemic when that track returns).

---

## Relation to evidential (NIG) channels — statistical filter vs biological translation

The intersection of an **NIG head** and the geometric metrics touches a crucial bridge: **statistical uncertainty vs biological state change.** They live on different layers; abbreviations must stay distinct (see naming lock).

### Layer A — NIG head (statistical filter)

An NIG head replaces point predictions with conjugate prior parameters \((\gamma, \nu, \alpha, \beta)\) for a Gaussian likelihood. From one forward pass:

| Channel | Typical readout | Biological intuition (statistical) |
|---------|-----------------|--------------------------------------|
| **Aleatoric** \(\mathbb{E}[\sigma^2]\) | Inherent ambiguity | Flexible loops, unresolved density, ensemble blur — noise *in the data* |
| **Epistemic** \(\mathrm{Var}[\mu]\) | Model ignorance | Motif / mutant outside training coverage — “haven’t seen this” |

Code / training keys: `aleatoric_*`, `epistemic_*` (legacy short `ale`/`epi` **only** here).

### Layer B — Geometric investigation metrics (biological translation)

| Metric | Formula (short) | What it measures |
|--------|-----------------|------------------|
| \(\mathrm{AlleleSens}\) | mean \(d_{\mathbb{B}}\) WT↔mut on \(N\) | Site-local **state change** on the invariant manifold |
| \(\mathrm{Epistasis}\) | \(\|\Delta\mathbf{x}_{AB}-(\Delta\mathbf{x}_A+\Delta\mathbf{x}_B)\|\) | **Non-additive** rewiring vs sum of singles |

These need frozen \(\Theta\) + matched structures; they do **not** require a calibrated NIG head to be *defined* or *computed*.

### Proposed cross-layer relationships (hypotheses — not yet measured)

These are the conceptual bridge to test when both layers are live. **Do not treat as Pass criteria until prereg’d.**

1. **AlleleSens ↔ aleatoric (hypothesis)**  
   OFF-state absorption of an allele (e.g. G12D) → low \(\mathrm{AlleleSens}\) (small geodesic move) *and*, if NIG is informative, elevated **aleatoric** (structural variability / damping without a clean directional shift).  
   ON-state tension propagation → high \(\mathrm{AlleleSens}\) with a more **deterministic** latent move (not “just noise”).

2. **Epistasis ↔ epistemic (hypothesis)**  
   Cooperative / suppressor epistasis often lands in rare intermediate regimes. Large \(\mathrm{Epistasis}\) residual is expected to **correlate** with an **epistemic** spike: the joint perturbation is off the linear sum of singles and may sit outside single-mutant coverage.

```text
NIG head     →  statistical filter  (aleatoric noise vs epistemic blindness)
AlleleSens /
Epistasis    →  biological translation (displacement + non-linear rewiring on x_hyp)
```

| Track | Object | Needs calibrated NIG? | Park status |
|-------|--------|----------------------|-------------|
| Evidential aleatoric / epistemic | NIG head | Yes — [`bprime-uncertainty-unpark.md`](bprime-uncertainty-unpark.md) | PARKED |
| AlleleSens / Epistasis | Metrics on \(x_{\mathrm{hyp}}\) | No | STARTING DEFS |
| Cross-layer correlation audit | Same residues, both layers | Yes (informative NIG) | Deferred until evidential unpark *or* explicit waiver |

Sealed health \(\Theta\) is enough to compute Layer B today; it does **not** fix collapsed NIG stds. Related surface: agent `compare_wt_mutant` — \(\mathrm{AlleleSens}\) is a local-neighborhood specialization of that idea.


---

## Acceptance as starting point

**Agreed for park:** These two equations are concrete, programmable, and aligned with “Θ invariant + metrics on top.” They are good enough to:

1. Name the investigation handles without vague “emergent readouts”
2. Guide future probe code / preregs (KRAS G12 neighborhood, double-mutant panels)
3. Keep evidential unpark from being confused with mutational epistasis

**Not claimed:** Bars, corpus panels, or Pass/Fail for biology. Those need a separate prereg when probes are implemented.

---

## When implementing (later)

- [ ] Code keys: `allele_sens_*`, `epistasis_coupling_*` (never bare `ale`/`epi`)
- [ ] Curvature passthrough from ckpt
- [ ] Prereg \(N\), reduction, and structure panels before grading
- [ ] Optional: side-by-side with NIG channels on the same residues for correlation audit (compare-only)
