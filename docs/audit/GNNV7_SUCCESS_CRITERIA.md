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

**Holdout granularity (frozen):** default **`protein`** — entire structures excluded
from shaping (`corpus_protein_holdout_ids`, ~20% of Stage A proteins, stable hash).
Alternate `residue_stratified` is within-protein interpolation only (weaker claim;
adjacent residues share context). Constants: `G4_DEFAULT_HOLDOUT_MODE` in
`aleatoric_shaping_holdout.py`.

**G4 pass — ALL required (frozen before `--p4-v3-aleatoric-shaping` runs):**

1. **Holdout P8 (G4a-hardened):** `tau_boundary_aleatoric_elevation(holdout_rows)`
   passes — relative lift `(ale_τ − ale_non) / std(ale) ≥ 0.20` **and**
   `std(ale) ≥ 0.05` on holdout residues only.
2. **No mask memorization:** NOT (`p8_full.ok` AND NOT `p8_holdout.ok`). **Vacuous when
   full P8 fails** — report as `vacuous_full_p8_fail`, not a pass.
3. **Transfer ratio:** `holdout_relative_lift / full_relative_lift ≥ 0.70`
   (`G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN`) **only when** full-corpus P8 is informative
   (`ale_std ≥ 0.05`) **and** `full_relative_lift ≥ 0.20`. Otherwise
   `not_evaluable_sub_threshold` — ratio is noise÷noise and must not count as pass.
4. **ρ coupling report (required in eval, advisory for pass):** compute
   `r(aleatoric, ρ)` on corpus and holdout. If `|r| ≥ 0.85` →
   `aleatoric_rho_reparameterization_risk` — G4 P8 may pass but do **not** cite
   aleatoric as independent biophysical signal (same failure shape as epistemic G5b).

**G4 fail examples:**

| Pattern | Verdict |
| ------- | ------- |
| Full P8 ok, holdout P8 fail | Mask memorization |
| Holdout P8 ok, transfer &lt; 0.70 | Partial memorization |
| Holdout P8 fail (flat ale) | Shaping did not generalize |
| Transfer ratio with flat full P8 | **Not evaluable** — do not count as partial pass |
| Memorization with both P8 fail | **Vacuous** — not a pass |
| G4 pass + \|r(ale,ρ)\| ≥ 0.85 | P8 credit with ρ-reparameterization caveat |
| Eval on `checkpoint_eligible=false` | Diagnostic only — do not certify recipe |

**Pre-recipe-change diagnostics (required on failed shaping runs):**

- **Routing entropy** vs established 1.28–1.33 band — confounds aleatoric read if unstable.
- **Dehydron mask split** (`aleatoric_dehydron_stratification_report`) — if dehydron
  residues are also near-floor, suspect global penalty / mask bug before tuning coeffs.
- **`r(epi,ale)`** only interpretable when `ale_std ≥ 0.05`; near-zero r on collapsed ale
  is mechanical, not decoupling evidence.

Constants: `G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN`, `G4_TRANSFER_RATIO_FULL_REL_LIFT_MIN`, `G4_ALE_RHO_MARGINAL_PROXY` in
`evidential_validation.py`; report: `g4_aleatoric_shaping_holdout_report()`.

**Thin holdout (n=12 corpus, 20% protein holdout):** default seed holds out **one**
protein (e.g. 1F88). Report includes `holdout_corpus_contrast` (fold_id, dehydron
fraction, ρ mean vs corpus z-scores) and `thin_holdout_warning`. Eval-only rotation:
`--holdout-seeds 42,7` — training mask is fixed at seed 42; extra seeds test whether
verdict is holdout-protein-specific. Multi-seed consensus in `multi_seed_consensus`.

**A/B variant (mirrors G3 spirit):**

| Variant | `var_penalty` / `aleatoric_hinge` | P8 eval set |
| ------- | --------------------------------- | ----------- |
| **A** | Off (or decorr-only P4 baseline) | Holdout |
| **B** | On train mask only | Holdout |

**G4 pass (A/B):** B improves holdout P8 vs A under the frozen rule above.

Implementation: `science/training/aleatoric_shaping_holdout.py`; training via
`p4_v3_aleatoric_shaping_phase_config()` + `--p4-v3-aleatoric-shaping`; eval via
`make eval-g4-holdout`.

**Mask audit (2026-07-08, p4_v3 run):** `target_dehydron` is binary `{0,1}` and
matches `data.x[:,1]` (0 mismatches). Symmetric collapse is **not** soft-mask or
field-drift — `var_penalty` on ~42% regular residues updates **shared**
`DecoupledEvidentialHead.ale_trunk` weights that affect all residues.

**Rejected fix:** stop-gradient on dehydron path — var_penalty already excludes
dehydron residues; coupling is through shared **parameters**, not activation flow.

**w_var_penalty coefficient sweep (before architectural change):**

Frozen probe grid: `G4_POPULATION_SEPARATION_WVP_WEIGHTS` = `{2.8, 1.0, 0.3}`.
Short probes only (`G4_WVP_EPOCHS` default 4). Launch: `make g4-var-penalty-sweep`.

Frozen separation pass (pre-registered, not post-hoc):

1. `global_ale_std ≥ 0.05` (same informativeness floor as G4a/P8).
2. `|μ_dehyd − μ_regular| / σ_pooled ≥ 2.0` (`G4_POPULATION_GAP_STD_MULT_MIN`).

Report: `aleatoric_population_separation_report()`; aggregate:
`checkpoints/v6/diagnostics/g4_wvp_sweep_report.json`.

**Sweep results (2026-07-08, resume `slim_moe_route_v1/v6_best.pt`, 4 epochs/probe):**

| `w_var_penalty` | `global_ale_std` | `gap/σ_pooled` | `separation_ok` |
| --------------- | ---------------- | -------------- | --------------- |
| 2.8 | 0.0086 | 1.71 | fail |
| 1.0 | 0.0087 | 1.71 | fail |
| 0.3 | 0.0086 | 1.71 | fail |
| *(route_v1 baseline, no probe)* | 0.0121 | 1.75 | fail |

Verdict: **`partial_gap_at_low_weight`** — gap is not near-zero (~1.7× pooled σ) but
frozen criterion fails on **global σ ≪ 0.05** at every weight; 4-epoch probes barely
move population stats vs baseline. `w=0.3` did cut training `v3_aleatoric_shaping`
loss (~10→~5), confirming coefficient wiring after config fix.

**Single close-out isolation gate (pre-registered):**

- Run exactly one shaping-only isolation training (`make train-v6-p4-g4-shaping-only-isolation`)
  with uncertainty-head-only updates and non-G4 losses zeroed.
- Evaluate with `make eval-g4-holdout G4_CKPT=checkpoints/v6/runs/<run_id>/v6_phase4_12prot.pt`.
- Confirmation criterion (same frozen metric family): `global_ale_std ≥ 0.05` and/or a
  material increase in `gap/σ_pooled` vs route baseline; otherwise park uncertainty.
- Scope cap: isolation run + at most two follow-ups (loss reintroduction probes).

| Sweep verdict | Meaning | Next step |
| ------------- | ------- | --------- |
| `magnitude_sufficient` | ≥1 weight passes frozen separation | Tune weight; no trunk split |
| `partial_gap_at_low_weight` | Gap opens but criterion not met | Extend grid down or mild calibration |
| `structure_likely_required` | Gap ~0 at all weights | Minimal per-population affine on ale output |

**Architectural fallback (only if sweep rules out magnitude):** learned
scale/shift on aleatoric output conditioned on dehydron flag **after** shared
trunk — not duplicate `ale_trunk`, not stop-gradient.

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

Checklist: `g5_epistemic_provenance_checklist()` / `g5b_rho_feature_proxy_checklist()` in
`nig_identifiability.py`.

### G5b — ρ feature proxy (extends G5)

ρ is a **direct input** (`x[:,0]`). High `|r(epi,ρ)|` may indicate epistemic is a monotonic
reparameterization of a feature the model already sees — same failure shape as SASA, different
confound.

**Required metrics (folded into `g5_epistemic_provenance_report`):**

1. `|r(epi,ρ)|` marginal on Stage A corpus.
2. Pinned OOD **1PGB**: raw in-corpus vs OOD mean(epi) ratio (P11).
3. Same contrast on epistemic **ρ-residualized** (fit `epi ~ ρ` on in-corpus only).
4. `r(student, teacher | ρ)` — distillation alignment beyond ρ.

**Flag `rho_feature_proxy` if:** `|r(epi,ρ)| ≥ 0.85` **and** raw OOD epistemic elevation
**collapses** after ρ control (`ood_separation_collapsed_after_rho`).

**Teacher–student bootstrap:** resample Stage A proteins with replacement; report CI for
`r(stu,tea)`. If CI straddles 0.80 → `teacher_student_borderline` (not a robust non-trigger).

**G5 route_v1 result (2026-07-08):** SASA-distillation ruled out (|r(epi,SASA)|=0.51,
partial≈0). **Epistemic novelty blocked:** r(epi,ρ)=0.95; pinned OOD 1PGB mean epistemic
**0.90× in-corpus** (inverted, not merely flat); ρ-residual OOD ratio ≈0.98. Teacher–student
r=0.807 with bootstrap CI [0.791, 0.821] straddling 0.80 (n=12 proteins — CI may understate
true spread). **S6 epistemic credit for route_v1:** epistemic is ρ-dominant and shows no OOD
elevation (raw or ρ-residualized) on the pinned OOD test — **do not cite epistemic uncertainty
for novelty/review-flagging claims until this is independently resolved.** G3 ablations break
teacher lock-in (r≈0.59) but do not fix OOD inversion.

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

## Residue-first aleatoric doctrine (2026-07-08)

**Aleatoric uncertainty is not a corpus-level property.** The NIG head models
genuine molecular ambiguity at each residue / dehydron site — local probability mass
that remains accessible even in the fully observed conformational ensemble. Averaging
or thresholding aleatoric across the corpus erases the spatial signal needed for
druggability and active learning.

From the manifold perspective, informative aleatoric should **concentrate at the rim**
of the Poincaré disc (high `disc_r`, under-wrapped / high-entropy regions), while
stable core residues sit deeper in the cone with low aleatoric. A global
`std(ale) ≥ 0.05` gate smears rim signal across the bulk and systematically
under-flags proteins with one or two critical sites while over-penalizing flexible
constructs.

### Site decisions (local only)

**Investigate** when **all** hold on a residue:

| Signal | Rule | Rationale |
| ------ | ---- | --------- |
| High aleatoric | `ν_ale > t_ale` (default: corpus **P90**) | Local ensemble ambiguity |
| Rim-localized | `disc_r` ≥ corpus P75 | Under-wrapped / rim geometry |
| Low clustering | graph `clustering` ≤ corpus P25 | Not in a dense topological core |
| (optional) Low routing confidence | `expert_routing_max` ≤ P25 | Expert ambiguity |

Implementation: `flag_investigation_sites()` in
`science/training/aleatoric_residue_diagnostics.py`. CLI:
`make aleatoric-corpus-diagnostics`.

**`t_ale` resolution:** default is corpus-relative **P90** (`resolve_t_ale()`), not
the training-health absolute `0.05` floor. Override with `--t-ale` (absolute) or
`T_ALE_PERCENTILE=95` on the Makefile target. Use absolute `0.05` only when head
output is on that scale.

### Global summaries (monitoring / triage only)

| Diagnostic | Purpose | Not for |
| ---------- | ------- | ------- |
| Residue-level aleatoric histogram | Calibration shape — expect **right-skew** (bulk low, long tail) | Site pass/fail |
| `std(ale)` corpus-wide | Head collapse / narrow calibration detector | Druggability certification |
| Per-protein `fraction(ale > t_ale)` | Validate ~0.12–0.20 vs known flexible sites | Global gate |
| Per-protein `max(ale)` ranked | **Active-learning acquisition priority** | Mean-ale ranking |
| Intra-protein `var(ale)` | Conformational heterogeneity (e.g. EGFR L858R) | Corpus mean |
| Dehydron high-ale burden table | Unwrapped-site count per structure | Global threshold |
| Family-level means (when metadata present) | Kinase vs multi-domain separation | Until corpus diverse enough |

**Expected global signatures when well-calibrated on a diverse corpus:** right-skewed
residue distribution; kinase domains lower mean with localized activation-loop spikes;
multi-domain / disordered constructs higher mean and dehydron burden. Current Stage A
corpus is too narrow to show clean family separation — treat absence as a **data**
limit, not proof the head cannot discriminate.

### Aleatoric independence probe (minimal)

**Question:** Does ν_ale vary independently of ρ and expert assignment after controls?

**Not a training gate** — run before crediting site-level aleatoric claims.

| Stat | Meaning |
| ---- | ------- |
| `r(ale, ρ)` marginal | Dehydron-density proxy risk |
| `r(ale, ρ \| expert)` partial | ρ coupling beyond routing |
| `η²(expert)` / `η²(expert \| ρ)` | MoE assignment explains aleatoric? |
| `R²(ale ~ ρ + expert)` | Combined proxy explainability |
| `residual_std_ratio` | `std(resid) / std(ale)` — independent variance left |
| Within-expert `r(ale, ρ)` | ρ monotonicity inside each expert |

**Verdicts** (`aleatoric_independence_probe()`):

| Verdict | Meaning |
| ------- | ------- |
| `not_yet_meaningful` | CV too low — probe cannot decide (route_v1 expected) |
| `rho_proxy` | Aleatoric tracks ρ |
| `expert_proxy` | Aleatoric tracks routing |
| `proxy_fully_explained` | ρ + expert absorb variance |
| `independent_signal_candidate` | Residual spread + weak partials — warrant site follow-up |
| `ambiguous` | Borderline — extend corpus or add geometry |

CLI: `make aleatoric-independence-probe` (optional `INCLUDE_GEOMETRY=1`).

### Impact on frozen gates

| Gate / metric | Old role | New interpretation |
| ------------- | -------- | ------------------ |
| `global_ale_std ≥ 0.05` (G4a, S6, sweep) | Site informativeness certification | **Training collapse monitor** — retain for save health, demote for biology |
| `aleatoric_population_separation_report` | Sweep pass/fail | Shaping coefficient diagnostic — not druggability |
| P8 τ lift on corpus | Aleatoric biology | Residue-stratified holdout P8 on **high-ale sites** + local rim check |
| G4 holdout memorization | Still valid | Independent of global σ floor |

`global_aleatoric_health_monitor()` wraps corpus `std(ale)` with
`not_a_site_gate=True`. Do **not** block structure onboarding or site reports on
corpus mean aleatoric alone.

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

**G5 must be run before** attributing P7 to v6-native learning. **If
`epistemic_novelty_claim_blocked` (route_v1: yes)** — block production claims that epistemic
flags novel/risky structures for review; routing/disc legs (S1, S4, S5) remain creditable.

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
5. **G4** holdout wiring — **done** (`--p4-v3-aleatoric-shaping`, `make eval-g4-holdout`)
6. Choose primary-loss philosophy (NIG vs blended) — document in run params
7. If G3 (epistemic) + G4 pass → full Phase 4 + routing retrain with hyperbolic MP graph (S4)
8. Evaluate S1–S6 jointly; apply G5 asterisk rule if `distilled_proxy` flags

**Blocked:** `p4_v3_aleatoric_recovery` until steps 5–6 complete.
