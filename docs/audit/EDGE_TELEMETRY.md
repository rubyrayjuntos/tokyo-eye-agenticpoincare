# Edge telemetry MVP

Read-only edge-level instrumentation for v6 GNN. **Not a gate criterion** —
MLflow tags: `telemetry/track`, `not_p_entry_gate`.

## Purpose

Answer “can we see edge-level structure that already exists?” before adding
capacity or loss terms. Same discipline as the dead-gradient telemetry incident:
prove the probe is alive before reading any signal.

## Four fields (per structure, per forward)

| Field | Source |
|-------|--------|
| `edge_embed_resistance_corr` | Pearson(radial MLP norm on MP edges, physics R_eff) |
| `edge_epistemic_var` / `edge_aleatoric_var` | Endpoint mean of node evidential head |
| `same_expert_rate` | Fraction of MP edges with same argmax expert |
| `same_expert_null_rate` | Independent-assignment baseline Σ p_e² |
| `edge_flow_score` | conductance × mean(betweenness) per edge |

Implementation: `science/training/edge_telemetry.py`

Physics graph: Cα conductance from cone depth (Phase 4 stack). MP edges from
`resolve_message_passing_edges()` (hyperbolic when attached).

## Pre-registered signatures

**Telemetry-alive (run first):**
- `edge_epistemic_var_std` and `edge_aleatoric_var_std` both > floor (not flat)
- All fields finite across Stage A corpus

Residue-level diagnostic (verify head before edge aggregation). Full DER validation:
``docs/audit/EVIDENTIAL_UNCERTAINTY.md``.

```bash
TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/residue_uncertainty_audit.py \\
  --checkpoint checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \\
  --structures 1MBN:A \\
  --validate-decomposition \\
  --json-out checkpoints/v6/diagnostics/residue_uncertainty_1mbn_route.json \\
  --csv-out checkpoints/v6/diagnostics/residue_uncertainty_1mbn_route.csv \\
  --pdb-local
```

## Flow stratification (spec §3 healthy signature)

Per structure, edges split at median `edge_flow_score`:
- `flow_stratification.high_flow.same_expert_excess` vs `low_flow`
- `same_expert_excess_high_minus_low` > 0 ⇒ experts align with flow (healthy)

Baseline JSON includes `per_edge` arrays when `--per-edge` (default on).

**Collapsed routing (expected today, H≈1.28–1.33):**
- Compare `same_expert_rate` vs `same_expert_null_rate` (Σ p_e²) — excess quantifies
  clustering beyond independent assignment
- Pre-registered near-zero `edge_embed_resistance_corr` — **MVP baseline did not match**
  (observed ~−0.32 stable negative on hyperbolic MP edges)

## MVP baseline results (Stage A small v1)

| Checkpoint | telemetry_alive | same | null (Σp²) | excess | corr |
|------------|-----------------|------|------------|--------|------|
| cold_v1 | 100% | 0.836 | 0.298 | +0.54 | −0.31 |
| route_v1 | 100% | 0.784 | 0.322 | +0.46 | −0.32 |

Artifact: `checkpoints/v6/runs/edge_telemetry_mvp_baseline.json`

Probe is **alive**. Same-expert **exceeds** null by ~0.5 — not fully explained by soft
load alone (argmax + graph locality). Resistance correlation is **anti-aligned**, not absent.

**Healthy (future, after retrain — not MVP claim):**
- `same_expert_excess` > 0 on high-`edge_flow_score` edges vs low-flow
- Stable non-zero `edge_embed_resistance_corr` across seeds/checkpoints
- Elevated `edge_aleatoric_var` on ρ≈TAU boundary edges

## Commands

```bash
TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/edge_telemetry_baseline.py \
  --corpus manifests/v6_corpus_stage_a_small_v1.json \
  --compare-checkpoints \
    checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/v6_best.pt \
    checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \
  --json-out checkpoints/v6/runs/edge_telemetry_mvp_baseline.json \
  --pdb-local

make test  # includes tests/test_edge_telemetry_mvp.py
```

## Curvature note

Structural compose uses checkpoint `model.curvature`; learned c reaching its own
equilibrium is acceptable — telemetry does not require pinning c for MVP baseline.
