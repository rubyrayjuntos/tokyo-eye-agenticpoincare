# GNNv7 success criteria — pre-retrain gates and Phase 4 uncertainty

**Status:** Pre-retrain planning (2026-07-08)  
**Scope boundary:** See `docs/audit/GNNV6_DIAGNOSTIC_SYNTHESIS.md` — uncertainty
diagnostics sit **next to** unresolved routing collapse (H≈1.28–1.33). Passing
P7 does not mean GNNv6 is healthier.

Related: `docs/audit/EVIDENTIAL_UNCERTAINTY.md`, `science/training/nig_identifiability.py`

---

## Two mechanisms — do not conflate (G1 vs missing v3 shaping)

Flat aleatoric on v6 has **two independent explanations**. The retrain plan must
address **both**, not treat recovering v3's recipe as a substitute for decoupling.

| Mechanism | What it is | Gate | Fix direction |
| --------- | ---------- | ---- | ------------- |
| **Loss-surface coupling** | NIG regularizer `\|y−μ\|·(2ν+α)` entangles ν and α in the objective | **G1** (resolved) | Decorrelation + head split; may need loss reform |
| **Missing aleatoric supervision** | v6 dropped v3's `var_penalty`, `aleatoric_hinge`, and `gaussian_likelihood` ρ residual term | **G4** (required before v3 recipe port) | Explicit shaping **with holdout** — see G4 |

v4 confirms G1 stands alone: same coupled head, pure NIG, no aleatoric shaping —
aleatoric was not engineered to be informative there either. v3's aleatoric
*looked* usable because of **direct supervision on the dehydron mask**, not because
the NIG head magically identified aleatoric under vanilla DER.

**Do not** draft `p4_v3_aleatoric_recovery` until G4 passes and the primary-loss
philosophy is chosen (see `loss_philosophy_options()` in `nig_identifiability.py`).

---

## Pre-retrain gates (cheap, before GPU spend)

| Gate | Question | Status | Resolution |
| ---- | -------- | ------ | ---------- |
| **G1** | Is epi/ale coupling architectural or loss-level? | **Resolved** | **Loss-level.** NIG regularizer couples ν and α. `DecoupledEvidentialHead` alone is **not sufficient**. See `analyze_nig_loss_coupling()`. |
| **G2** | Are uncertainty probes alive (not flat-std)? | Open at retrain | Require P7 + `nu_cv` on epoch health; route_v1: epi alive, ale not informative. |
| **G3** | Is **epistemic** semantics emergent or B-factor/SASA Goodhart? | **Run — re-eval under G4a** | A/B decorr-only vs full epistemic decoupling. `g3_supervision_circularity_report()`. |
| **G4a** | Is **P8** meaningful (not sign-only on flat ale)? | **Required — before G4** | Relative τ lift + informative aleatoric std; see below. |
| **G4** | Is **P8** independent of dehydron-mask aleatoric shaping? | **After G4a** | Held-out residue split for `var_penalty` / `aleatoric_hinge`; P8 on holdout only. |
| **G5** | How much of v6 **P7** is v3-teacher hand-me-down vs v6-native? | **Required before crediting S6** | Provenance audit: distillation path, r(epi,SASA) vs teacher. |

### G3 — B-factor/SASA epistemic circularity gate

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

**G3 result (2026-07-08, 12-epoch ablations, pre-G4a P8):** `g3_supervision_circularity_report`
reported `g3_pass=true` with P8 in both A and B — **inconsistent** with aleatoric
std≈0.006 (below informative floor). **Root cause:** legacy P8 used `lift > 0` only
(`TAU_ALE_LIFT_MIN=0`), so direction passed on noise. **Do not treat as clean G3 pass
for P8.** Re-evaluate G3 epistemic circularity on P11/S6 legs; re-run P8 claims only
after **G4a** (below).

---

### G4a — P8 magnitude gate (prerequisite for G4)

**Problem:** P8 can pass while aleatoric is flat — a tiny positive `(ale_τ − ale_non)`
on std(ale)≈0.006 is not biophysical elevation, it's sign noise (same failure family
as the original flat-std incident).

**Fix (implemented in `tau_boundary_aleatoric_elevation`):**

1. Strata: **continuous ρ** (`|ρ − TAU| ≤ band`) — not `tau_flag` (verified in code).
2. **Informative aleatoric:** `std(ale) ≥ 0.05` required before P8 `ok`.
3. **Relative lift:** `(ale_τ − ale_non) / std(ale) ≥ 0.20` (G4a floor; tune with corpus).

Analytical ref: `g4a_p8_magnitude_gate()` in `nig_identifiability.py`.

**G4a pass on route_v1 / G3 checkpoints:** expected **fail** until aleatoric is
actually informative — this resolves the G3 P8 inconsistency.

---

### G4 — Aleatoric shaping holdout gate (dehydron-mask circularity)

**Risk (same pattern as G3, aleatoric side):** v3's `var_penalty` and
`aleatoric_hinge` use `regularity_mask = 1 − target_dehydron` — i.e. the ρ/TAU
label itself. P8 checks aleatoric elevation near ρ≈TAU. If shaping trains aleatoric
low on non-dehydrons and free on dehydrons **on the same residues P8 evaluates**,
P8 passing proves the hinge converged, not that the model discovered biophysical
ambiguity at the rim.

**Required before `p4_v3_aleatoric_recovery`:**

1. **Holdout split:** Per structure (or corpus-level), hold out fraction `h` of
   residues from `var_penalty` and `aleatoric_hinge` entirely. Holdout must include
   both τ-near and non-τ residues; stratify by `target_dehydron` where possible.
2. **Train** with shaping on the **train mask only**; never backprop shaping on holdout.
3. **Evaluate P8 only on holdout residues** (`tau_boundary_aleatoric_elevation(holdout_rows)`).
4. **G4 pass:** P8 holdout lift > 0 **and** aleatoric_std on holdout ≥ informative floor.
5. **G4 fail:** P8 passes on full corpus but fails on holdout → shaping memorized the mask.

**A/B variant (mirrors G3 spirit):**

| Variant | `var_penalty` / `aleatoric_hinge` | P8 eval set |
| ------- | --------------------------------- | ----------- |
| **A** | Off (or decorr-only P4 baseline) | Holdout |
| **B** | On train mask only | Holdout |

**G4 pass:** B improves holdout P8 vs A without full-corpus-only inflation.

Implementation sketch: `aleatoric_shaping_holdout_masks()` in
`science/training/nig_identifiability.py`; wire into Phase 4 when recipe is ported.

---

### G5 — Epistemic provenance gate (v3 distillation vs v6-native)

**Risk:** v3 epistemic was largely a **SASA proxy** (SASA in `x[:,3]`, coupled head,
`epistemic_temp_scaling=2.8`). v6 distills **depth + epistemic only** from the v3
teacher (`experiments/training/v6/v2_teacher.py`) — not aleatoric. A passing P7 on
route_v1 (epi std ≈ 4.4) may credit **inherited proxy signal**, not v6-learned DER
exposure gap.

**Required before treating P7 / S6 epistemic as a v6/v7 finding:**

1. On Stage A corpus, compute `r(ν_epi, SASA)` and `r(ν_epi, SASA | ρ)` (partial) on
   v6 checkpoint vs frozen v3 teacher on the same graphs.
2. Report **teacher–student epistemic correlation** per structure (distillation alignment).
3. If `r(epi, SASA) ≥ 0.85` and teacher–student `r ≥ 0.80`, flag
   `epistemic_provenance=distilled_proxy` — downgrade "emergent epistemic" claims.
4. **G5 pass:** Either (a) partial r(epi,SASA|ρ) drops materially vs marginal, or
   (b) epistemic on structures **never** in teacher cache diverges from teacher with
   high spread (native learning signal).

**G5 does not block retrain** — it calibrates how much credit S1–S6 give to epistemic
vs routing / disc outcomes.

**Pre-committed credit rule if `distilled_proxy` flags** (marginal r(epi,SASA)≥0.85
AND teacher–student r≥0.80):

| Criterion | Credit |
| --------- | ------ |
| **S1** routing, **S4** hyperbolic MP, **S5** edge telemetry | Full weight |
| **S6** and epistemic-based production claims | **Asterisk only** — wording: *"informative but largely SASA-proxy inherited via v3 teacher distillation — not v6-native DER exposure discovery."* Do **not** count S6 epistemic leg toward production uncertainty gates without native corroboration. |

Checklist: `g5_epistemic_provenance_checklist()` in `nig_identifiability.py`.

---

### Primary-loss philosophy (decide before Phase 4 recipe port)

v3 and v6 use **different objectives** for the same outputs. Pick one explicitly;

| Mode | Primary objective | Aleatoric meaning | Epistemic story |
| ---- | ----------------- | ----------------- | --------------- |
| **NIG** (v6 default) | `evidential_regression_loss` | NIG-derived `β/(α−1)` (clamped) | `(1/ν)·temp` — DER exposure |
| **Gaussian** (v3 teacher) | Heteroscedastic NLL on ρ | **Direct variance** fit to ρ residual | Still `(1/ν)·temp` but aleatoric is not NIG-derived |
| **Blended** | NIG + auxiliary `gaussian_likelihood` + shaping (G4 holdout) | Hybrid — document which head output is canonical | Requires G1 + G4 |

**Blended is not "free":** auxiliary gaussian_likelihood makes aleatoric a supervised
heteroscedastic variance — legitimate, but **not** what P9/P11 were written to
validate under pure DER. If blended becomes primary, update P9/P11 interpretation in
`EVIDENTIAL_UNCERTAINTY.md`.

Analytical reference: `loss_philosophy_options()` in `nig_identifiability.py`.

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
   - **P8 pass (G4a):** τ-boundary **relative** lift ≥ 0.20 **and** `aleatoric_std ≥ 0.05`
   - `nu_cv ≥ 0.02` (scale-invariant exposure spread)
   - `aleatoric_std ≥ 0.05` (informative aleatoric — not decorrelated noise)
3. **Save gate wired:** `require_tau_ale_elevation_save=True` on
   `p4_head_decouple_phase_config` — low r alone cannot promote checkpoint.

Implementation: `uncertainty_s6_joint_pass()`, `uncertainty_save_ineligibility_reasons(require_tau_ale_elevation=True)`.

**G3 must pass before S6 counts for production claims** on OOD / τ-boundary **for
epistemic-supervised paths**. **G4 must pass before** v3 aleatoric shaping or P8
claims on aleatoric recovery. **G5 must be run before** attributing P7 to v6-native learning.

---

## Property tests (P7–P11)

| Test | Corpus? | Notes |
| ---- | ------- | ----- |
| P7 | Yes | `epistemic_std` + `nu_cv` — floors on head output, not canonical DER; interpret via **G5** |
| P8 | Yes | **G4a:** relative lift + informative ale; **G4** holdout if shaping; **G3** for epistemic paths |
| P9 | **Yes** (`test_p9_sparsification_monotone_corpus`) | Do not cite triage until passes |
| P10 | After expansion | Stub until 12→25 corpus event |
| P11 | Pinned OOD `1PGB:A` | Independent only if G3 pass; not DER-valid if primary loss is gaussian_likelihood |

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
2. **G4a** — P8 magnitude gate (done in code; re-eval G3/G3 report through hardened P8)
3. **G5** provenance audit on route_v1 / G3 checkpoints (cheap, no GPU)
4. Re-interpret G3 A/B (epistemic circularity on P11; P8 legs invalid pre-G4a)
5. If pursuing v3 aleatoric recipe: **G4** holdout wiring + A/B — only after G4a
6. Choose primary-loss philosophy (NIG vs blended) — document in run params
7. If G3 (epistemic) + G4 pass → full Phase 4 + routing retrain with hyperbolic MP graph (S4)
8. Evaluate S1–S6 jointly; apply G5 asterisk rule if `distilled_proxy` flags

**Blocked:** `p4_v3_aleatoric_recovery` until steps 5–6 complete.
