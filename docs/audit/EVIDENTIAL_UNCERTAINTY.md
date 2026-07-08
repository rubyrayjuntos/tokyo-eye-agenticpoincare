# Evidential uncertainty — framework and validation

**Scope:** Diagnostic infrastructure **alongside** unresolved routing collapse (H≈1.28–1.33).
Passing P7 does **not** mean GNNv6 is healthier — see
`docs/audit/GNNV6_DIAGNOSTIC_SYNTHESIS.md`.

**Framework:** Deep Evidential Regression (DER, Amini et al.) — **not** MC dropout or
deep ensembles. Single deterministic forward pass; NIG evidence `(γ, ν, α, β)` per node.

Implementation: `science/dtie/v5/gnn/model.py` (`EvidentialHead`, NIG loss),
`science/dtie/v6/gnn/evidential.py` (decoupled trunks + canonical audit formulas).

## Conceptual split (domain)

| Kind | Meaning | Should respond to |
| ---- | ------- | ----------------- |
| **Aleatoric** | Irreducible ambiguity — ρ genuinely near TAU | Biophysics only; **not** corpus size |
| **Epistemic** | Exposure gap — sparse structure-space coverage | Corpus expansion; OOD proteins |

**Master property test (P10):** expand corpus → epistemic on shared residues shrinks;
aleatoric stable. If both move equally, the split is one label wearing two hats.

## Head output vs canonical DER (units / floors)

**Reported by live head:**

```
epistemic  = (1/ν) · epistemic_temp_scaling   # default temp=1.0, constructor hyperparam
aleatoric  = exp(clamp(log(β/(α−1))))
```

**Canonical DER (audit only — `der_uncertainty_from_evidence`):**

```
aleatoric  = β / (α − 1)
epistemic  = β / (ν · (α − 1))   # = aleatoric / ν
```

### Floor calibration

| Floor | Applies to | Notes |
| ----- | ---------- | ----- |
| `1e-4` alive | Head-reported epi/ale | Technical non-zero |
| `1e-3` corpus epi | Head-reported epistemic std | Order-of-magnitude vs flat-std≈0.0041 incident — **not** inherited from canonical DER |
| `0.05` informative ale | Head-reported aleatoric std | route_v1 fails (~0.011) |
| `0.02` nu_cv | std(ν)/mean(ν) on evidence | **Scale-invariant** — temp cannot fake this |

**P7 pass on std(epistemic) alone is insufficient** if `nu_cv` fails: raising
`epistemic_temp_scaling` inflates reported epistemic without informative ν.

**P9 sparsification** ranks by reported epistemic — scale-invariant (ranking unchanged under temp).

## Loss identifiability (open before Phase 4 retrain)

NIG loss = NLL + `coeff · |y−μ| · (2ν + α)`.

The regularizer **structurally couples ν and α** in the loss surface. Known DER
critique (Meinert & Lavin; Bengs et al.): epistemic/aleatoric may not be identifiable
from data + vanilla NIG alone.

**`DecoupledEvidentialHead` is not sufficient:**

- Splits MLP trunks (ν from epi trunk; μ,α,β from ale trunk).
- **Same** `evidential_regression_loss` underneath — all four evidence params in NLL.
- r(epi,ale)≈0.977 on route_v1 may be **loss-level**, not shared-weight architecture.

**What Phase 4 can add (training signal, not just parameters):**

| Signal | Module | Preset |
| ------ | ------ | ------ |
| r²(epi, ale) penalty | `epi_ale_decorrelation_loss` | `p4_head_decouple_phase_config` (0.75) |
| B-factor / SASA on epistemic | `epistemic_decoupling_loss` | `p4_epistemic_decoupling_phase_config` |
| Save gate r(epi,ale) | `max_probe_r_epi_ale_save` | ≤0.70 head-decouple |

Analytical module: `science/training/nig_identifiability.py`

**Cheap pre-retrain check:** confirm Phase 4 enables decorrelation and/or epistemic
decoupling coeffs — not `decoupled_uncertainty_heads=True` alone.

## Phase 4 / retrain gates (G1, G3–G5, S6)

Full criteria: **`docs/audit/GNNV7_SUCCESS_CRITERIA.md`**

- **G1 (resolved):** Loss-level coupling — head split insufficient.
- **G3 (required):** A/B epistemic B-factor/SASA — `g3_supervision_circularity_report`.
- **G4a (required before G4):** P8 relative lift `(ale_τ−ale_non)/std(ale)` ≥ 0.20 + informative ale — not sign-only.
- **G4 (after G4a):** Holdout split for v3 `var_penalty` / hinge; P8 on holdout only.
- **G5 (provenance):** v3 teacher vs v6-native epistemic — before crediting P7 to v6.
- **S6 (joint):** `|r| ≤ 0.70` **and** P8 τ-ale lift **and** informative aleatoric — wired in save gate.

**Two mechanisms:** G1 (NIG coupling) and missing v3 shaping are independent — see v7 doc § Two mechanisms.

**Loss philosophy:** `loss_philosophy_options()` in `nig_identifiability.py` — decide before `p4_v3_aleatoric_recovery`.

## Validation checks

| # | Check | Function | Test |
| - | ----- | -------- | ---- |
| P7 | Epistemic + ν_cv non-degenerate | `epistemic_var_non_degenerate`, `exposure_non_degenerate` | unit + corpus |
| P8 | Aleatoric elevated at ρ≈TAU | `tau_boundary_aleatoric_elevation` | unit + corpus; **G4a:** relative lift + informative std |
| P9 | Sparsification monotonicity | `sparsification_curve` | synthetic + **corpus** |
| P10 | Corpus-expansion sensitivity | `corpus_expansion_sensitivity` | stub until expansion |
| P11 | OOD epistemic > aleatoric ratio | `out_of_corpus_epistemic_contrast` | synthetic + **pinned OOD** |

Module: `science/training/evidential_validation.py`

### P11 OOD policy (pinned)

OOD must be **structurally distant from Stage A training**, not merely "not trained this epoch."

| Tier | Definition |
| ---- | ---------- |
| **In-corpus** | `enabled: true` in `manifests/v6_corpus_stage_a_small_v1.json` |
| **Pinned OOD** | `1PGB:A` — `enabled: false` in manifest (explicit drop), CATH `3.10.20.10` β-grasp, GB1 domain unlike GTPase-heavy Stage A |
| **Distance basis** | Manifest disposition + CATH fold not represented in enabled training folds — not sequence identity alone |

Constants: `OOD_PINNED_STRUCTURES`, `OOD_DISTANCE_BASIS` in `evidential_validation.py`.

**1MBN is in-corpus** — useful for residue-level audit, **not** for P11 OOD.

### Running diagnostics

```bash
TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/residue_uncertainty_audit.py \
  --checkpoint checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \
  --structures 1MBN:A \
  --validate-decomposition \
  --pdb-local
```

Corpus property tests: `tests/test_evidential_validation.py` (set `TRAINING_LOAD_FROM_PDB=1`).

## Edges

Edge uncertainty = endpoint mean of same node evidential head. One framework, one
flat-std guard, P7 logic at edge granularity (`edge_epistemic_var_std`, etc.).

## route_v1 honest read (1MBN + Stage A)

| Check | Result | Authorizes |
| ----- | ------ | ---------- |
| P7 epistemic std | Pass (~4.4) | Track epistemic magnitude |
| P7 nu_cv | **Verify on checkpoint** | Informative exposure |
| Aleatoric informative | **Fail** (~0.011) | Nothing on ale |
| r(epi, ale) | **Fail** (~0.977) | No split semantics |
| P8 τ ale lift | **Fail** | No τ-boundary claims |
| P9 corpus | **Run test** — may fail | Triage only if passes |

## MLflow

`track/epistemic_std`, `track/aleatoric_std`, `track/epi_ale_corr`,
`track/uncertainty_informative_ale`, `track/epistemic_non_degenerate` — see
`docs/TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md` §3.2.
