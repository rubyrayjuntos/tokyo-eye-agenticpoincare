# GNNv7 success criteria — pre-retrain gates and Phase 4 uncertainty

**Status:** Pre-retrain planning (2026-07-07)  
**Scope boundary:** See `docs/audit/GNNV6_DIAGNOSTIC_SYNTHESIS.md` — uncertainty
diagnostics sit **next to** unresolved routing collapse (H≈1.28–1.33). Passing
P7 does not mean GNNv6 is healthier.

Related: `docs/audit/EVIDENTIAL_UNCERTAINTY.md`, `science/training/nig_identifiability.py`

---

## Pre-retrain gates (cheap, before GPU spend)

| Gate | Question | Status | Resolution |
| ---- | -------- | ------ | ---------- |
| **G1** | Is epi/ale coupling architectural or loss-level? | **Resolved** | **Loss-level.** NIG regularizer `\|y−μ\|·(2ν+α)` couples ν and α. `DecoupledEvidentialHead` alone is **not sufficient**. See `analyze_nig_loss_coupling()`. |
| **G2** | Are uncertainty probes alive (not flat-std)? | Open at retrain | Require P7 + `nu_cv` on epoch health; route_v1: epi alive, ale not informative. |
| **G3** | Is epistemic semantics emergent or supervised Goodhart? | **New — required** | Run **A/B** before trusting P8/P11: decorr-only vs full B-factor/SASA supervision. See below. |

### G3 — B-factor/SASA circularity gate

**Risk:** `epistemic_decoupling_loss` trains epistemic toward B-factor/SASA. Then P8
(τ aleatoric) and P11 (OOD epistemic on 1PGB) partially measure **supervised
regression fit**, not emergent "model knows what it doesn't know."

**Domain hypothesis (must be stated explicitly):** Epistemic *should* correlate
with crystallographic flexibility / exposure proxies **only if** that is an
independent scientific claim — not because it makes P8/P11 pass.

**Required A/B (two short Phase 4 runs or checkpoint pair):**

| Variant | Config | `epistemic_decoupling_coeff` | `epi_ale_decorrelation_coeff` |
| ------- | ------ | ---------------------------- | ----------------------------- |
| **A** (G3 ablation) | `p4_head_decouple_decorr_only_phase_config` | **0** | 0.75 |
| **B** (full) | `p4_head_decouple_phase_config` | 1.0 | 0.75 |

Compare with `g3_supervision_circularity_report()`:

- **G3 pass:** P8/P11 do **not** pass only in variant B while failing in A.
- **G3 fail:** P8 or P11 passes in B but not A → uncertainty is downstream of
  direct supervision; downgrade claims to "B-factor/SASA regression head," not
  emergent epistemic.

**Reported metrics (not trained on variant A):** `probe_r_epi_sasa`, `probe_r_epi_bf`
(if available), P8 lift, P11 contrast — track in MLflow, do not use as loss on A.

---

## Success criteria S1–S6 (retrain outcomes)

| ID | Criterion | Pass condition |
| -- | --------- | -------------- |
| S1 | Routing | H ≤ save ceiling; min_routing_fraction ≥ 0.05 (inference) |
| S2 | Disc occupancy | σ₂/σ₁ within corpus band; thickness floor |
| S3 | Topology depth | r(d,τ) ≥ P_DEHYDRON floor (MASTER lineage) |
| S4 | Hyperbolic MP graph | Structural SSOT + hyperbolic edges **during training** |
| S5 | Edge telemetry alive | `track/edge_telemetry_alive = 1` on Stage A corpus |
| **S6** | **Uncertainty calibration** | **Joint — see below** |

### S6 — Uncertainty (updated after G1)

**Was (conditional on G1):** "depending on G1's diagnosis…"

**Now (G1 resolved):**

1. Retrain config **must** include `p4_head_decouple` (or equivalent) with
   `epi_ale_decorrelation_coeff > 0` — not `decoupled_uncertainty_heads` alone.
2. **S6 passes only if ALL hold jointly on Stage A corpus eval:**
   - `|r(ν_epi, ν_ale)| ≤ 0.70` (decorrelation)
   - **P8 pass:** τ-boundary aleatoric elevation (`aleatoric_tau_lift > 0`)
   - `nu_cv ≥ 0.02` (scale-invariant exposure spread)
   - `aleatoric_std ≥ 0.05` (informative aleatoric — not decorrelated noise)
3. **Save gate wired:** `require_tau_ale_elevation_save=True` on
   `p4_head_decouple_phase_config` — low r alone cannot promote checkpoint.

Implementation: `uncertainty_s6_joint_pass()`, `uncertainty_save_ineligibility_reasons(require_tau_ale_elevation=True)`.

**G3 must pass before S6 counts for production claims** on OOD / τ-boundary.

---

## Property tests (P7–P11)

| Test | Corpus? | Notes |
| ---- | ------- | ----- |
| P7 | Yes | `epistemic_std` + `nu_cv` — floors on head output, not canonical DER |
| P8 | Yes | Independent only if G3 pass |
| P9 | **Yes** (`test_p9_sparsification_monotone_corpus`) | Do not cite triage until passes |
| P10 | After expansion | Stub until 12→25 corpus event |
| P11 | Pinned OOD `1PGB:A` | Independent only if G3 pass |

### nu_cv floor calibration (empirical)

| Checkpoint | ale_std | nu_cv | Interpretation |
| ---------- | ------- | ----- | -------------- |
| route_v1 (ale flat) | ~0.012 | **~0.16** | nu_cv detects exposure spread; **does not** catch ale flat |
| Synthetic constant ν | — | **~0** | Floor 0.02 rejects |

Floor `0.02` validated against synthetic degenerate + live route_v1 discrimination.

---

## What S6 does **not** authorize

- MoE routing fix
- Production ingest gates on uncertainty
- Trusting edge-flow / same-expert excess without G3 + S6 joint pass

---

## Recommended retrain sequence

1. Confirm G1 (done — document in run params)
2. Run G3 A/B Phase 4 ablations (short epochs acceptable for gate)
3. If G3 pass → full Phase 4 + routing retrain with hyperbolic MP graph (S4)
4. Evaluate S1–S6 jointly; do not promote on S6 partial (r only without P8)
