# Tokyo Eye v8 — Freeze Reconciliation Addendum

**Status:** SIGNED (Ray Swan, 2026-09-16)  
**Date:** 2026-09-16  
**Amends:** [`2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md`](2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md) (FROZEN, signed 2026-07-22)  
**Does not replace:** [`2026-09-15-tokyoeye-equ-pure-hyp-freeze-amendment.md`](2026-09-15-tokyoeye-equ-pure-hyp-freeze-amendment.md) (`tokyo_eye_equ_pure_hyp_v1`) — this addendum **implements** that veto as a trunk-wide check and dispositions the rest of the freeze against the 2026-09-16 audit  
**Trigger:** Independent audit (2026-09-16) found the live hyperbolic-spine experiment (MLflow exp id 14) diverged from the 7/22 freeze on 14 of 22 scored clauses. `pure_hyp_pass` — the freeze’s veto gate — was computed by grepping two filenames rather than checking the trunk, and is a false positive on `geoopt_restore` (Euclidean `nn.Linear` confirmed on ball points in MoE experts and on `log₀` in SDRP/affinity heads).

**Decision:** Rebuild from a **cold checkpoint** against the 7/22 freeze’s clauses, corrected by this section’s dispositions. This is a **reconciliation, not a re-architecture**. Two months of failure-knowledge that produced the freeze remain authoritative unless explicitly amended below with a stated rationale. Nothing warm-starts from `geoopt_restore` or any prior checkpoint; none of them clear the veto gate as specified.

**Goal of the rebuild:** the 7/22 freeze **plus** the corrections this week’s diagnostics independently earned (§2). Not a byte-for-byte museum of 7/22, and not a license to relitigate the architecture.

---

## 0. How to read dispositions

| Token | Meaning |
|-------|---------|
| **KEEP** | Audit found the live path already matches the freeze. Do not “improve” it. |
| **REVERT** | Live code is unjustified drift. Restore the frozen clause. Most rows. |
| **AMEND** | The frozen clause itself changes, with rationale, and becomes binding. |
| **SIGNED** | Operator confirmed REVERT vs AMEND on §1.1 (2026-09-16). Rebuild may start at §3 step 1. |

**DEFER is not used in this addendum.** §9.3 marks TransparencyEngine / rim↔core / per-R fractions **mandatory**. Parking them would also require striking the §14 Ontology sign-off bar — that is not recommended and is not done here.

---

## 1. Disposition of the 22-clause audit

### 1.1 Operator decisions — SIGNED 2026-09-16

Evidence lookups retained below. Dispositions are no longer proposals.

#### §6 lift — **REVERT** to Option B (signed)

**Git.** `git log --follow -- science/tokyo_eye/v8/projector.py` has **one** commit: `2512cd2` (2026-08-20, “Save remaining local workbench…”). `HEAD` is still Option B (`v_lifted = α · â` then `exp₀`). The geoopt projector is **uncommitted working-tree only**. There is no commit message and no PR description for a switch to `geoopt.PoincareBall.expmap0`.

**Design note (not a git commit), quoted verbatim, not strengthened:**

- Plan `docs/superpowers/plans/2026-09-16-tokyoeye-equ-geoopt-restore.md`: “Kill SE(3)-lite + hand-rolled lift as the sealed spine path.”
- Contract `docs/superpowers/specs/2026-09-16-tokyoeye-equ-geoopt-assembly-contract.md` §3.B: “Lift via **`geoopt.manifolds.PoincareBall`** (`expmap0` + `proj`) wrapping projector output as `ManifoldTensor` where needed.”

Those sentences name geoopt as a restore-card primitive vs a hand-rolled `exp_map_zero`. They do **not** say numerical stability, NaN avoidance, or library maturity. They also never landed `projector.py` on `master`.

**Signed:** REVERT. Freeze §6 Option B remains law. (Ray 9/16/26) *I agree to revert — without a reason for the past change then I cannot support it — that is why REVERT is the correct action.*

#### §16 `lr_hyperbolic` — **REVERT** to `1e-3` (signed)

**MLflow (localhost:5000, all 15 experiments, 2026-09-16).**

| `lr_hyperbolic` | n runs (param present) | What it shows |
|-----------------|------------------------|---------------|
| `0.0003` | 22 | EQU reboot including `equ_cold_boot_geometry_first` `f217ade3` (`sat=100%`, `mean_r=0.895`, `tau_ceiling=0.895`). 100% rim sat happened **at 3e-4**, not 1e-3. |
| `0.001` | 1 | `tokyo-eyes-v8` / `tokyo_eye_v8_live` `eb1a09a2815247eaa7d04b04c029eca2` (2026-07-22). `equiformer_mode=stub_missing_ckpt`. Wall clock **3 seconds**. `diag_boundary_saturation_pct=0`, `loss_total=2.785` finite, status FINISHED. Not a trained rejection of 1e-3. |

No run at `1e-3` shows NaN, 100% sat, FAILED, or ABORTED attributable to that LR. `3e-4` was already in `equiformer_v3_weight_map.json` / Sprint 5; it was not logged as a response to a 1e-3 blow-up.

**Signed:** REVERT to freeze §16 `lr_hyperbolic=1e-3`. Rebuild includes a real 1e-3 train on the restored spine; if that run NaNs or saturates, AMEND then with that run id. (Ray 9/16/26) *REVERT confirmed for the reasons stated.*

#### Canonical MLflow name — **AMEND** to `full-stack` (signed)

Canonical experiment going forward: `tokyoeye/equiformer-v3-moe/geometric/full-stack`.

- Sprint 5 harness + `tests/v8/test_sprint5_harness.py` already lock that string.
- Freeze `tokyo-eyes-v8` was created 2026-07-22 and used for early v8 smokes; it is **not** the Sprint 5 contract name.
- `…/hyperbolic-spine` (id 14) is the 2026-09-16 EQU reboot experiment; no Sprint 5 test coverage.

**Historical aliases (do not train new work here):** `tokyo-eyes-v8`, `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`.

**Signed:** AMEND freeze §3/§12 MLflow field to `tokyoeye/equiformer-v3-moe/geometric/full-stack`. Aliases above are historical only. (Ray 9/16/26) AMEND confirmed for the reasons stated.

### 1.2 KEEP — already on freeze

| # | Clause | Audit | Disposition | Note |
|---|--------|-------|-------------|------|
| 4 | §5 Dual-layer R0 coexistence + Layer-B exclusivity + bidirectional emit | Compliant | **KEEP** | Loader already emits R0 and chemistry on the same pair. Do not “simplify.” |
| 5 | §7 Sparse segmented softmax; no dense N×N | Compliant | **KEEP** | PyG `softmax` over `E`. |
| 6 | §8 Train Gumbel `hard=True` STE | Compliant | **KEEP** | Forward path is hard. Logging is not — see #12. |
| 7 | §8 Eval argmax one-hot (forward) | Compliant | **KEEP** | Eval *forward* is argmax. Eval *metrics* were not — §2.1. |
| 8 | §6 Curriculum class `τ` 0.70 → 0.995 | Compliant (controller) | **KEEP** | Last-step `scheduled_tau≈0.916` is schedule math, not a missing controller. |
| 9 | §8 Four experts E0–E3 | Compliant (count) | **KEEP** | Niches and gate features are #11, not this row. |
| 10 | Lineage isolation `science/tokyo_eye/v8/` | Compliant | **KEEP** | No v7/v66 imports in the spine. |

### 1.3 REVERT — unjustified drift

| # | Clause | Audit | Disposition | Rationale |
|---|--------|-------|-------------|-----------|
| 11 | §5 wrap ≤ 19; R5 demoted | `dehydron_wrap_max=1` on live probes; `edge_frac_r5≈0.61`, `r0≈0.165`; R1–R4 unlogged | **REVERT** → **superseded by §2.7 AMEND** | Historical REVERT restored freeze text wrap ≤ 19 before evidence existed. §2.7 (2026-09-17) AMENDs the number to **1** from Stage-A-12 median-then-descend; cone geometry KEEP. |
| 12 | §8 Gate features ρ, τ, SS, degree, SASA | Live: `[radius, −d_H density, degree, h_proj]` | **REVERT** | Missing chemistry is a plausible independent cause of eval 2-expert collapse. Restore spec’d features before re-evaluating routing health. |
| 13 | §8 Anti-collapse: load-balance + majority hinge + load floor | Only one term shipped; computed on a masked (soft / Gumbel) quantity | **REVERT + fix** | All three mechanisms are named in §8; only one shipped, and it did not work. Load-balance on **sharp clean-logit** probabilities (validated this week); add majority hinge; add eval-mode load floor as a **hard gate** (§2.1). STE `hard=True` forward tensor **is** the tensor logged for load metrics. |
| 14 | §7 / `tokyo_eye_equ_pure_hyp_v1` no tangent Linear post-lift | Attention QKV gyro (ok); MoE experts `nn.Linear` on ball; SDRP/affinity `Linear(log₀)` | **REVERT** | Veto-gate violation. Rewrite MoE experts and hyp heads as gyro / Einstein-midpoint ops, or the trunk does not clear `pure_hyp_pass` regardless of sat / Pearson. |
| 15 | §9 TransparencyEngine; rim↔core flow; per-R message fractions | Missing; only radius/sat/entropy | **REVERT** | §9.3 is **mandatory**. It is the only way to score the §14 Ontology bar (rim/core vs R2/R1). Not deferred. Parking would require an AMEND that strikes or rewrites that bar — not done. |
| 16 | §6 Learned `c` (pin only if documented) | `c=1.0` float pin; not `nn.Parameter` | **REVERT** | Freeze prefers learned `c` with pin-and-document as fallback. Pinning was never documented. Unpin **or** document the pin in a signed AMEND. Until that sentence exists, treat as revert-to-learned. |
| 17 | §16 Checkpoint path convention | Live `checkpoints/tokyoeye/pretrained/…` vs freeze `checkpoints/v8/pretrained/…` | **REVERT** | Cosmetic but lineage-bearing. Restore the convention path (symlink ok). |
| 18 | §7 `L ∈ {2,3}`; softmax index | Ctor allows `L=1` and `>3`; tests use `L=1`; softmax on `edge_index[0]` not §7.1’s `[1]` | **REVERT** | Default and harness `L=2`. Reject `L∉{2,3}` at ctor. Softmax/scatter index: freeze §7.1 says destination `edge_index[1]`. Bidirectional emit masks the bug; still restore the contract index. |
| 19 | §10 Radius-aware evidential | Head exists on Euclidean skip; no radius vacuity curriculum | **REVERT** (in-lineage head) | Keep the evidential head **in** v8. Restore radius-aware vacuity (high `r` → epistemic pressure). Do not park uncertainty again. |
| 20 | `pure_hyp_pass` computation | Filename grep for `_tangent_linear` in `attention.py` + `affinity_head.py` | **REVERT** | Do not restore the grep. Replace with §2.4’s trunk-wide scanner. All pre-2.4 `pure_hyp_pass` values are retired (§2.5). |

### 1.4 AMEND — freeze text was behind live (keep the stricter fact)

| # | Clause | Audit | Disposition | Rationale |
|---|--------|-------|-------------|-----------|
| 21 | §5 / Sprint 2 note: DSSP `E ≤ −0.5` deferred | Live Kabsch–Sander cutoff is −0.5 | **AMEND (upgrade, keep)** | Already stricter than the Sprint 2 deferral. Formalize as baseline. No code action beyond deleting the “deferred” sentence from freeze §12. |
| 22 | Sprint unit tests s1–s6 | Byte-identical to Downloads / 7/22 | **AMEND (extend, don’t rewrite)** | Tests are not the source of drift. They still encode the original unit contract, including Sprint 5’s pre-existing `lr_hyperbolic=3e-4` / MLflow-name conflict. **Extend** them for: real `pure_hyp_pass`, restored gate features, three anti-collapse mechanisms, R1–R4 logging, TransparencyEngine. Do not silently rewrite s1–s6 to match drifted code. |

---

## 2. Additions — corrections earned this week, registered as spec, not drift

These did **not** exist in the 7/22 freeze. They were found through this week’s diagnostics and are **binding additions**, so a future audit does not flag them as new deviations.

### 2.1 Eval-mode routing gate

`moe_liveness_gate` runs on **eval-mode hard assignments**, not train-time probabilities.

Train-time `moe_load_e*` (Gumbel-noised **or** softmax at any temperature ≳ 0.5) is **retired as evidence of deployed routing**. It structurally cannot detect eval-time argmax collapse (`geoopt_restore` train ~0.25 uniform vs eval `[0, ~0.66, ~0.34, 0]`).

The routing-entropy / liveness gate must be computed on eval-mode argmax (or the validated sharp `softmax(logits / T)`, `T ≪ 1`, proxy). It **hard-blocks** under freeze cascade semantics, not watch-only.

The STE `hard=True` forward tensor is the **same tensor** logged for load metrics. No separate soft-probability logging path that can be mistaken for deployed load.

### 2.2 Dual geometry seals (pre-MoE and post-MoE)

`sat`, `spread`, and `Δ_equiv` must each record which representation they were computed from (`z_lift` / `z_attn` vs `z_moe`).

A pre-MoE PASS does not imply the post-MoE (deployed) embedding is healthy, and vice versa. `geoopt_restore` QUALIFIED spread came almost entirely from two MoE-parked radii (pre-MoE `std(‖z‖) ≈ 0.002`, post-MoE ≈ 0.093), not from genuine hyperbolic volume utilization by the lift/attention stack.

### 2.3 Frontend pretraining claim retracted; equivariant architecture retained

Stage-0 LOOCV+L2 residue-separability (frozen MPtrj vs random-init-same-architecture vs raw CA vs GearNet-Edge) showed MPtrj-pretrained weights statistically indistinguishable from random init at the same architecture; both clear raw CA. **The SE(3)-equivariant decomposition carries the signal; MPtrj pretraining does not.**

Going forward: no claim that the frontend is biomolecularly- or physics-informed via its pretraining. **Cold random-init is the default frontend state** pending a named, shape-compatible, dehydron/H-bond-relevant protein checkpoint clearing a CA-sized margin under the same probe. That checkpoint has not been found (GearNet-Edge did not clear it under held-out evaluation).

### 2.4 `pure_hyp_pass` is a trunk-wide structural scan

The check must walk every module reachable from the model’s forward pass for:

1. `nn.Linear` / `nn.Sequential` of Linears / Euclidean matmul applied **directly to ball-point tensors** after the single Euc→H lift  
2. Any `log₀(·) → [Euclidean op] → exp₀(·)` sandwich, regardless of filename  

It ships as `science/tokyo_eye/v8/pure_hyp_pass.py` with a property test:

- **Known-bad fixture:** tangent-Linear expert on ball points (matches live MoE) — **must fail**  
- **Known-good fixture:** gyro / Einstein-midpoint only — **must pass**  

The known-bad fixture must be shown to fail **before** this check gates any train / promote stamp. Until that test is green, do not attach `pure_hyp_pass` to MLflow.

Grep of `_tangent_linear` in two files is **forbidden** as a substitute.

### 2.5 Retired-metric list (must never again be cited as eval/deployed evidence)

- Train-step `moe_load_e*` (any temperature, any noise regime)  
- Train-step `h_norm` as a proxy for routing diversity (it measures embedding-norm health only)  
- Any `pure_hyp_pass` value computed prior to §2.4  
- Any geometry seal (`sat` / `spread` / `Δ_equiv`) not labeled pre-MoE or post-MoE per §2.2  

Prior citations already known to rest on one of these (checkpoint-selection C1, Sprint 9 `moe_load_e*≥0.05`, “rim-volume healthy MoE H_norm=0.876”, `geoopt_restore` original QUALIFIED) are **superseded**. Do not cite them as supporting evidence in external-facing documents (SBIR, patent, GOSP) going forward.

### 2.6 SIGNED AMEND — §8 train-time epsilon-greedy hard exploration (with decay)

**Status:** **SIGNED** (Ray Swan, 2026-09-17). Binding. Implement per this clause.

**Evidence package (2026-09-17):** `checkpoints/v8/runs/freeze_recon_router_data_gumbel_audit/`

| Check | Verdict | Why it matters |
|-------|---------|----------------|
| Gumbel runway | `RUNWAY_NOT_THE_BOTTLENECK` | Quota×140 3-step still had `T=1.00→0.82`; monopoly + rising-and-stall while exploration was hot. 30ep monopoly present at ep0 with `T=1.0`. |
| 4OBE real chemistry | `REAL_CHEMISTRY_DOES_NOT_RESCUE` (caveat: `dehydron_frac=1.0`, `n_r1=0`) | Varied SS/degree/SASA; task→gate ~1e-4; argmax monopoly. |
| 1UBQ production wrap=19 | Same monopoly / task~1e-4 | Degenerate labels are **systemic** under wrap=19 (`wrap_count≤19→R2`), not 4OBE-specific. |
| 1UBQ offline balanced thr=2 | `TASK_ABSENT_RISING_STALL_MONOPOLY` | `dehydron_frac=0.671`, `ρ` varies (`rho_std≈0.08`), `n_r1=42`/`n_r2=38`; still task→gate ~2e-4 and argmax `76/0/0/0`. **Decisive** — closes “data artifact vs mechanism.” |

Also prior: `Σ ReLU(floor−load)²` rising-and-stalling under equalized quota×140; T=0.5 negative control; STOP coeff search on quota.

**Separate card (CLOSED by §2.7):** production wrap threshold — [`tokyo_eye_equ_wrap_threshold`](2026-09-17-tokyoeye-equ-wrap-threshold-decision.md) (`SIGNED_AMEND` → `DEHYDRON_WRAP_MAX=1`). Independent of this AMEND.

**Signed clause (amends freeze §8 routing, not hardness):**

During **training**, after Gumbel-Softmax `hard=True` produces a per-node one-hot assignment, apply a **per-node independent Bernoulli(`ε(t)`)** draw. On heads, **replace** that node’s one-hot with a freshly sampled uniform random one-hot over E0–E3 (discrete swap — never a convex combination of two one-hots, never a batch-level single coin flip for all nodes). The tensor that leaves this step **must** lie in `{0,1}^{N×E}` with exactly one `1` per row **before** it multiplies or mixes any expert output. Any continuous mixture / confidence scale / soft residual is forbidden and will fail `pure_hyp_pass` / `euclidean_mix_on_ball` the same way the historical confidence-scale bug did. Eval forward remains pure argmax (ε ignored). `ε(t)` **must** decay on an explicit schedule — a fixed forever-ε is underspecified and forbidden by this AMEND.

**Epsilon schedule (binding):** mirror the existing Gumbel cool-down infrastructure:

| Symbol | Default | Meaning |
|--------|---------|---------|
| `eps_start` | `0.20` | Per-node Bernoulli probability of force-random override at epoch 0 |
| `eps_end` | `0.00` | Floor — exploration ends; gate owns selection |
| `eps_half_epochs` | same as `gumbel_exp_half_epochs` (12) | Exponential half-life, shared with Gumbel so one curriculum clock |
| `ε(t)` | `max(eps_end, eps_start · exp(−α_ε · t))` with `α_ε = log(eps_start/eps_end_eff)/eps_half_epochs` | Same functional form as `GumbelTemperatureSchedule` |

When `eps_end=0`, treat `eps_end_eff` as a small positive for α only (e.g. `1e-3`) so ε reaches numerical zero by ~`2·half_epochs`, then clamp to 0. Log `eps_t` every epoch next to `gumbel_temperature`.

**Verification plan (binding — same mutation-test standard as loader oracle / `pure_hyp_pass`):**

1. **Known-bad / inert fixture — `eps=0`:** with exploration forced off, routing tensors and eval-mode loads must be **byte-identical** (or exact equal within floating STE path) to the pre-AMEND baseline on a fixed seed + batch. Proves the override path is genuinely inert when off — not a silent always-on branch.
2. **Known-good / forced fixture — `eps=1.0`:** every node’s assignment must be uniform-random over E0–E3 (empirical load ≈ `1/E` within sampling tolerance; not gate-argmax). Proves the override path actually executes and is not a no-op. Only after (1) and (2) pass may a real run use `eps_start=0.20`.
3. **Decay-floor hold (the interesting failure mode):** after `ε(t)` first reaches `eps_end` (0), continue training and log **eval-mode** `moe_eval_utilization` / load / `n_alive` on the epochs **immediately after** the floor — not only during the exploration window. If eval collapses back to a single-expert monopoly once the training wheels are off, epsilon-greedy bought temporary appearance of health rather than a durable fix; that outcome **fails** this AMEND’s acceptance even if mid-schedule loads looked healthy.

**Compliance reading (include both sides):**

- *Supporting:* Freeze §8 forbids soft relaxation in the forward path (“Soft MoE mush”). Epsilon-greedy never emits a soft / blended expert combination — the forward tensor remains a hard one-hot. Gumbel noise at high temperature **already** randomly overrides what would otherwise be the clean top-1 choice, and nobody calls that soft mush because the STE forward stays hard. Epsilon-greedy is the same category — **noisy hard selection** — decoupled from temperature so it can persist on a named schedule after Gumbel has cooled. It succeeds for the **same reason** Gumbel exploration is already legal, not despite the anti-mush intent.
- *Counterargument (state explicitly):* A reviewer can argue that randomly overriding argmax on a fraction of steps recreates the *practical* failure Soft MoE mush named — unreliable expert assignment — through a different mechanism. That concern is about **assignment noise**, not literal softness. The reply is: Gumbel already injects that class of noise under the frozen contract; the AMEND only adds a second, schedule-bounded knob when temperature alone proved insufficient (check #1). If permanent noise is the fear, `eps_end=0` and the shared half-life are the control — not rejecting the mechanism. The decay-floor hold check above is how we detect “temporary appearance of health.”

**Out of scope / still forbidden without a separate AMEND:** soft forward (`hard=False`), annealed hardness that blends experts, Einstein-midpoint / top-k mixture of expert outputs (manifold-legal ≠ §8-legal).

**Signed:** AMEND §8 exploration policy under hard commitment. Hardness KEEP. (Ray Swan 9/17/26)

### 2.7 SIGNED AMEND — §5 dehydron wrap threshold (Option A)

**Status:** **SIGNED** (Ray Swan, 2026-09-17). Binding. Implements Option A from [`tokyo_eye_equ_wrap_threshold`](2026-09-17-tokyoeye-equ-wrap-threshold-decision.md).

**Why not C:** Double-cone wrap (45° half-angle, H→O axis, 6.5Å) is intentional Sprint-8 geometry. Classical Fernández ~19–26 was calibrated on a full-sphere midpoint count; cone restriction structurally yields lower counts — expected, not a measurement bug. Cone redesign is a separate future biophysics card.

**Why not B:** Training dehydron labels are R2 incidence (`dehydron_labels_from_edges`). Keeping wrap=19 for “edge physics” while inventing a second label threshold would leave R2 uninformative and hide the real supervision number.

**Why not silent thr=2:** That value was a one-structure diagnostic. Production constant must come from the corpus search below.

**Procedure (binding evidence standard — same as lift / `lr_hyperbolic`):**

1. Pool undirected H-bond wrap counts across all **12** Stage-A enabled structures.  
2. Start `τ = floor(pooled median)`; descend while `max(per-structure dehydron_frac) ≥ 0.60`.  
3. Record pooled distribution + full per-structure table at the chosen `τ` before signing.

**Sealed result:** artifact `checkpoints/v8/runs/freeze_recon_wrap_threshold_amend/corpus_median_descend_search.json`

| Item | Value |
|------|-------|
| Pooled wraps | n=3651; min=0; max=17; mean≈3.58; p50=**3**; p90=8 |
| Search path | τ=3 → 2 → **1** |
| Chosen `DEHYDRON_WRAP_MAX` | **1** |
| Per-struct dehydron_frac at τ=1 | min=0.260 (1F88); max=0.581 (1LYZ); mean≈0.429 |
| Outliers ≥0.60 or <0.05 | **none** of 12 |

**Why τ=1, not the offline 1UBQ thr=2:** p50=3 sits *above* the chosen threshold — the search correctly kept descending past the median until `max(per-struct frac) < 0.60`. At τ=2 that max was still 0.736 (7/12 structures ≥ ceiling); only τ=1 cleared all twelve. The earlier 1UBQ-only diagnostic balanced near thr=2 because that one small chain is easier to bring under 0.60; denser/larger Stage-A members (e.g. 1BG1, 1IVO, 2SHP, 1F88) keep max-frac above the ceiling until one more step down — that is why the corpus AMEND lands a step below the single-structure number, not a disagreement about procedure.

**Signed clause:** Freeze §5 wrap gate is `wrap_count ≤ 1 → R2 (dehydron)`, else R1. Rule form unchanged; integer AMENDed. Runtime retune remains forbidden. Graph caches keyed by wrap hash auto-invalidate wrap=19 bytes.

**Retroactive caveat (named, same discipline as pool epoch-0):** freeze_recon probes that ran under the temporary REVERT wrap=19 — [`tokyo_eye_v8_wrap19_label_saturation_caveat_v1`](../../checkpoints/v8/runs/freeze_recon_wrap_threshold_amend/WRAP19_LABEL_SATURATION_CAVEAT.json). Those dehydron losses / saturated-label scores are not biology. **Out of scope:** `theme_biology` / `theme_restore` AUPRC (already `dehydron_wrap_max=1` with fracs ~0.29–0.56).

**Signed:** AMEND freeze §5 `DEHYDRON_WRAP_MAX` 19 → 1 from Stage-A-12 median-then-descend. (Ray Swan 9/17/26)

---

## 3. Rebuild order (cold init throughout)

Reuses freeze §12 Sprint **2 → 1 → 3 → 4 → 5**. Sequencing was correct; implementations drifted.

1. **R0–R5 loader** — wrap ≤ **1** (addendum §2.7), R1–R4 populated **and logged**, R5 demoted; exclusivity/priority **property-tested** against freeze §5.2 (not “does the file run”).  
2. **Sparse attention** — §7 gyro ops; §2.4 `pure_hyp_pass` **passing on this module in isolation** before proceeding.  
3. **Hard-commit MoE** — restored gate features; load-balance (sharp clean logits) + majority hinge + eval-mode load floor as **distinct** mechanisms; STE hard tensor = logged load tensor.  
4. **Spine integration** — `pure_hyp_pass` re-run **trunk-wide**.  
5. **Equiformer bind** — frontend per §2.3 (cold random-init default).  

No “QUALIFIED” language for any checkpoint until it clears freeze promotion rules **and** §2.4.

**Do not write model-rewrite code except in §3 order.** §1.1 is signed. First rewrite is R0–R5 loader (step 1). The §2.4 scanner + property test already shipped.

---

## 4. Sign-off

**Approver:** Ray Swan  
**Date:** 2026-09-16  
**Addendum ID:** `tokyo_eye_v8_freeze_reconciliation_v1`

This addendum locks: dispositions in §1.1–§1.4; additions in §2.1–§2.6 as binding spec (once signed), not drift; rebuild order in §3.

**Signed sentences:**

1. **§6:** REVERT. Freeze Option B remains law. Geoopt is uncommitted; `git log --follow` on `projector.py` never records the switch. (Ray 9/16/26) I agree to revert — without a reason for the past change then I cannot support it — that is why REVERT is the correct action.
2. **§16:** REVERT to `lr_hyperbolic=1e-3`. No trained 1e-3 run failed; cold-boot 100% sat was at `3e-4`. (Ray 9/16/26) REVERT confirmed for the reasons stated.
3. **MLflow:** AMEND canonical name to `tokyoeye/equiformer-v3-moe/geometric/full-stack`; aliases `tokyo-eyes-v8` and `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`. (Ray 9/16/26) AMEND confirmed for the reasons stated.
4. **§2.6:** AMEND §8 train-time epsilon-greedy hard exploration with explicit decay (`eps_start=0.20→eps_end=0.00`, shared half-life with Gumbel); per-node Bernoulli discrete `{0,1}` swap; mutation fixtures + decay-floor eval hold required. Hardness KEEP. (Ray Swan 9/17/26) Signed.
5. **§2.7 / §5 wrap:** AMEND `DEHYDRON_WRAP_MAX` 19 → **1** from Stage-A-12 pooled median-then-descend (max per-struct frac < 0.60; all 12 in 0.26–0.58). Cone KEEP; Options B/C rejected. (Ray Swan 9/17/26) Signed.

**TransparencyEngine is REVERT (restore), not DEFER.** Changing that requires an AMEND to freeze §9 plus a corresponding change to the §14 Ontology bar.
