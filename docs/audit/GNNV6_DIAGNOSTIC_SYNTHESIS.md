# GNNv6 diagnostic synthesis — three threads, one scope boundary

This document keeps **disc geometry**, **MoE routing**, and **evidential uncertainty**
separate so partial passes are not read as platform health.

> **Standing SSOT (2026-07-16):** Routing track closed under **decision (a)** —
> sensitivity + commitment banked; purity/Gram named limitation (not open);
> uncertainty parked (product symptoms patched). See
> `docs/audit/GNNV7_SUCCESS_CRITERIA.md` § WHERE WE ARE. The 2026-07-07 table
> below is the historical three-thread snapshot that started this separation —
> do not treat its “MoE routing unresolved” row as current status.

## Thread status (2026-07-07) — historical snapshot

| Thread | Instrument | Current read | Fixes routing? |
| ------ | ---------- | ------------ | -------------- |
| **Disc geometry** | σ₂/σ₁, disc_thick, structural SSOT | Occupancy measurable; compose uses checkpoint c | No |
| **MoE routing** | H, min_routing_fraction, expert collapse | **Unresolved** — H≈1.28–1.33, dominant expert | Needs training with hyperbolic MP graph |
| **Uncertainty (DER)** | P7–P11, edge telemetry, MLflow `track/*` | Epistemic spread alive; **aleatoric not informative**; r(epi,ale)≈0.977 | No |

**Net (as of 2026-07-07):** GNNv6 is **not healthier** because epistemic passes P7. You have better
instruments to tell when it becomes healthy.

## What each thread authorizes

### Disc geometry
- Trust σ₂/σ₁ trends when corpus context matches Stage A baseline caveats.
- Do **not** treat structural SSOT frozen as curvature pinned.

### Routing
- Do **not** promote checkpoints on routing entropy alone when H > save ceiling.
- Hyperbolic graph at inference only did not fix collapse — retrain with MP edges active.

### Uncertainty (this sprint’s deliverable)
- **Authorize:** epistemic ranking experiments, sparsification plots, MLflow tracking.
- **Do not authorize:** τ-boundary ale claims, corpus-expansion ale/epi split, aleatoric gates, edge-flow interpretation — until Phase 4 **loss + head** calibration passes P8/P10 and r(epi,ale) drops.

## Phase 4 gate (before retrain spend)

Cheap analytical checks (no GPU):

1. `analyze_nig_loss_coupling()` — regularizer `(2ν+α)` couples evidence params.
2. `decoupled_head_changes()` — architecture split ≠ loss change; need `epi_ale_decorrelation_coeff` and/or `epistemic_decoupling_coeff`.
3. Floor units: `head_output_quantity()` — floors on `(1/ν)·temp`, parallel `nu_cv` guard.

Recommended preset: `p4_head_decouple_phase_config` (decorrelation + save gate `max_probe_r_epi_ale_save=0.70`), not `decoupled_uncertainty_heads` alone.

## Artifacts map

| Concern | Doc / module |
| ------- | ------------ |
| **GNNv7 retrain gates (G1–G3, S6)** | `docs/audit/GNNV7_SUCCESS_CRITERIA.md` |
| Edge telemetry | `docs/audit/EDGE_TELEMETRY.md` |
| Evidential validation | `docs/audit/EVIDENTIAL_UNCERTAINTY.md` |
| NIG identifiability | `science/training/nig_identifiability.py` |
| MLflow track metrics | `docs/TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md` §3.2 |
| PyG attach mutation | `docs/audit/PYG_ATTACH_MUTATION.md` |
| Curvature consumers | `docs/audit/CURVATURE_CONSUMERS.md` |

## Regression discipline

Passing P7 (epistemic non-degenerate) **does not** imply:
- MoE specialization
- Trustworthy aleatoric at ρ≈TAU
- Valid edge-flow / same-expert excess interpretation
- Ready for production ingest gates on uncertainty

Each thread has its own mandatory metrics and gates in MLflow governance core vs `track/*`.
