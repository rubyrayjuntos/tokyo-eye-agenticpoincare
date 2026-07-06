# MLflow Metrics Guide — V6 GNN Training

**Audience:** Anyone monitoring v6 training in MLflow (especially first-time ML developers)  
**Companion to:** `[HYPERBOLIC_FRAMEWORK_EVAL.md](HYPERBOLIC_FRAMEWORK_EVAL.md)`, `[../DEVELOPER_ONBOARDING.md](../DEVELOPER_ONBOARDING.md)`

This guide explains what each chart in the `tokyo-eyes-v6` MLflow experiment measures, what “good” looks like, and whether you want the line to go **up** or **down** over epochs.

---

## How MLflow charts work in this project


| Concept            | Meaning                                                                   |
| ------------------ | ------------------------------------------------------------------------- |
| **Step (X-axis)**  | Global training epoch number (continues across resume)                    |
| **Value (Y-axis)** | End-of-epoch average over the proteins in that run                        |
| **One run**        | One curriculum phase (e.g. `mlflow_theory_baseline`, `full_hyp_moe_test`) |
| **Compare runs**   | Overlay charts from different runs in the MLflow UI                       |


Every epoch, `StageRunner` logs a flat dict of **loss terms** (what the optimizer minimizes) and **health probes** (what we measure on held-style eval proteins). See `experiments/training/v6/stage_runner.py`.

### Two kinds of numbers (easy to confuse)


| Prefix         | When computed                                      | Use for                                             |
| -------------- | -------------------------------------------------- | --------------------------------------------------- |
| `probe_r_*`    | After training step, eval mode, capped protein set | **Save gates**, console `shell r(d,s)=…`, promotion |
| `shell_r_*`    | During loss forward pass on training batch         | Training feedback; should track `probe_r_*`         |
| `shell_corr_*` | Loss = penalty when correlations are weak          | **Lower is better** (optimizer target)              |


**Loss vs probe:** A low `shell_corr_depth_sasa` loss is good. A high `probe_r_depth_sasa` correlation is good. They measure related things but opposite sign conventions.

---



## The dashboard — six metrics to pin first

If you only watch six charts, use these:


| Chart                   | Good direction   | Healthy band (theory / P2)               | Why                                   |
| ----------------------- | ---------------- | ---------------------------------------- | ------------------------------------- |
| **score**               | ↑                | Beat prior best (~2.2–2.3+)              | Ranks epochs for `v6_best.pt`         |
| **checkpoint_eligible** | → **1** on saves | 1 = all gates passed                     | Binary “would we save this epoch?”    |
| **routing_entropy**     | ↓ (not too fast) | ≤ **routing_save_max** (ramps 1.32→1.18) | MoE specialization                    |
| **probe_r_depth_sasa**  | ↑                | **≥ 0.65** save, **≥ 0.80** strong       | Core shell: depth vs solvent exposure |
| **probe_r_proj_depth**  | ↑                | **≥ 0.95**                               | Disc radius tracks burial depth       |
| **proj_frac_mean**      | → **0**          | **< 0.05**                               | Boundary collapse detector            |


Everything else supports *why* those moved.

---

## What needs work vs what really matters

Training now logs **focus metrics** every epoch so MLflow answers: *“Which numbers still need training, and which of those actually block quality?”*

### MLflow charts to pin (focus)

| Chart | Good direction | Meaning |
| ----- | -------------- | ------- |
| **focus_critical_count** | → **0** | Metrics that block saves or threaten shell/MoE |
| **focus_matters_needs_work_count** | → **0** | Subset that **really matter** for ingest quality |
| **focus_healthy_count** | ↑ | Saturated / production-grade on tracked probes |
| **focus_flag_routing_entropy** | → **0** | 1 = routing above save ceiling this epoch |
| **focus_flag_probe_r_depth_sasa** | → **0** | 1 = core shell below save floor |
| **focus_flag_proj_frac_mean** | → **0** | 1 = boundary collapse |

At run end, open **Artifacts → focus/focus_summary.json** for the full breakdown, or read **Tags**:

| Tag | Example | Purpose |
| --- | ------- | ------- |
| **focus_primary** | `routing_entropy` | Top metric(s) to train next |
| **focus_recommendation** | Human sentence | What to do this run |
| **focus_critical_count** | `1` | Quick scan across runs |
| **focus_matters_needs_work_count** | `1` | High-leverage gaps only |

### Priority matrix (your ep85 / ep100 situation)

Logic lives in `science/training/metric_focus.py`. Summary for mature P2 / theory-test runs:

| Metric | Matters for ingest? | Your runs (typical) | Train more? |
| ------ | ------------------- | ------------------- | ----------- |
| **routing_entropy** | **Yes** — MoE specialization | ep85 **1.24** ✓ save; ep100 **1.31** ✗ vs 1.18 ceiling | **Yes** — primary P2 focus until ≤ 1.18 (save) and ideally ≤ 1.20 (stretch) |
| **probe_r_depth_sasa** | **Yes** — core shell signal | **0.80+** excellent | **No** — protect during MoE tuning |
| **probe_r_proj_depth** | **Yes** — disc ↔ burial | **0.95+** saturated | **No** |
| **proj_frac_mean** | **Yes** — geometry collapse | **0** ideal | **No** |
| **expert_starvation_count** | **Yes** — expert dropout | **0** in winners | Only if ≥ 1 |
| **probe_r_epi_sasa** | Secondary | **0.7–0.9** strong | **No** — monitor only |
| **cone_range_mean** | Moderate | **0.8+** stable | **No** unless &lt; 0.04 |
| **disc_r_std_mean** | Low once shell strong | Often OK | **Low leverage** after P1d |
| **expert_load_spread** | Yes when routing high | Small if route_H &gt; 1.2 | Tune balance_coeff / dropout if experts stay at ~0.25 each |
| **total** (loss) | **No** alone | ↓ slowly | Do not chase — use tier-1 probes |

**Plain English for your curriculum:** v6 clearly improved shell and disc geometry vs earlier phases. The remaining gap is **MoE routing specialization** (`routing_entropy`) catching up to the **tightening save ceiling** (`routing_save_max`). That is why ep100 can show r(d,s)=0.80 and still skip: geometry is production-grade while routing is “too uniform for this epoch’s gate.”

### How focus status is assigned

| Status | Meaning |
| ------ | ------- |
| **needs_work** | Below save gate or critical threshold — train this |
| **watch** | Passes gates but below stretch target — optional polish |
| **healthy** | Saturated — do not spend epochs here |

| Priority | Meaning |
| -------- | ------- |
| **critical** | Blocks `v6_best` save or inverted/collapsed geometry |
| **important** | Affects score/ranking or MoE quality |
| **monitor** | Diagnostic; low leverage once shell is strong |
| **healthy** | No action |

### Example: ep100 focus output

```
primary_focus: routing_entropy
recommendation: Train more on: routing_entropy. These block saves or threaten shell/MoE quality.
healthy: probe_r_depth_sasa, probe_r_proj_depth, proj_frac_mean
needs_work: routing_entropy (1.306 > 1.18 save ceiling)
```

Compare to ep85 winner: `focus_primary=none` or routing in **watch**, `checkpoint_eligible=1`, recommendation mentions eligible epoch.

---



## Tier 1 — Decision metrics (saves & promotion)

These drive whether an epoch becomes `v6_best.pt`. Logic lives in `science/training/checkpoint_score.py`.

### score

**What:** Composite rank — higher is better among eligible epochs.

**Formula (simplified):** Rewards wide cone depth range, low boundary clipping, specialized routing, and strong shell probes.

**Trend:** ↑ over training within a phase; compare across runs to pick a champion.

**Example:** 2.29 = strong eligible epoch; 2.15 with `checkpoint_eligible=0` still useful for diagnostics but won’t save.

### checkpoint_eligible

**What:** `1.0` if all gates pass, else `0.0`.

**Gates include:** `proj_frac`, `routing_H` vs ceiling, expert starvation, minimum cone range, shell probe floors, optional `probe_r_depth_sasa` save floor (0.65 in theory test).

**Trend:** You want spikes to **1** when geometry + MoE align; many epochs will be **0** during P2 — that is normal.

### routing_entropy (route_H)

**What:** Shannon entropy of expert routing weights, averaged over residues.


| Value             | Meaning                                                         |
| ----------------- | --------------------------------------------------------------- |
| **ln(4) ≈ 1.386** | Uniform routing (4 experts, equal weight) — “no specialization” |
| **~1.2**          | Moderate specialization (promotion target)                      |
| **~1.0**          | Strong specialization                                           |


**Trend:** ↓ gently during P2 bridge / hypmix / theory test. Too fast → unstable shell; too slow → MoE not learning roles.

**Related:** `routing_save_max` — dynamic ceiling for *this* epoch (not a model output). If `routing_entropy` > `routing_save_max`, epoch is ineligible even if score is high.

### probe_r_depth_sasa

**What:** Pearson correlation between **cone depth** (burial) and **SASA** (solvent exposure). Positive = buried residues are deeper.

**Trend:** ↑ during P1/P1d; **hold ≥ 0.65** in P2 for saves; **≥ 0.80** is excellent.

**Abort guard:** Two consecutive epochs **< 0.60** aborts theory-test training (shell erosion).

### probe_r_epi_sasa

**What:** Correlation between epistemic uncertainty and SASA.

**Trend:** ↑; **≥ 0.20** = signal alive; **0.7–0.9** = strong (surface residues more uncertain).

### probe_r_proj_depth

**What:** Correlation between **disc radius** |p| on the Poincaré disc and cone depth.

**Trend:** ↑; **≥ 0.30** minimum; **≥ 0.95** = disc geometry tracks burial (what you want).

### probe_r_disc_sasa

**What:** Correlation between disc radius and SASA (exposed → different disc position).

**Trend:** ↑; **0.7+** is strong.

### proj_frac_mean

**What:** Fraction of residues whose hyperbolic embedding hit the **ball boundary** (clamped).

**Trend:** → **0**. **> 0.95** = collapsed / useless geometry. Your runs at **0** are ideal.

### cone_range_mean

**What:** Mean over proteins of (max cone depth − min cone depth) — spread of burial signal.

**Trend:** ↑ in early training; stable **0.8–1.0** in mature P2 runs. Gate: **≥ 0.04** (trivial once trained).

### expert_starvation_count

**What:** Count of proteins where at least one expert received ~zero mass.

**Trend:** → **0**. **≥ 2** with phase ≥ 2 → ineligible.

### elapsed_s

**What:** Wall time for one epoch (seconds).

**Trend:** Informational only; drops after epoch 1 (CUDA warmup).

---



## Tier 2 — Mixture-of-Experts (MoE)



### expert_load_0 … expert_load_3

**What:** Mean routing probability per expert (sums ≈ 1.0 across experts).

**Trend:** Lines **diverge** from uniform 0.25 each; no expert should flatline at 0 for many epochs.

**Read:** e.g. 0.21 / 0.33 / 0.18 / 0.29 = expert 1 handles ~⅓ of residues on average.

### capacity_loss

**What:** Penalty when expected expert load exceeds capacity threshold.

**Trend:** ↓ toward **0**.

### expert_{0,1,2}_depth_mean

**What:** Average cone depth for residues whose **dominant** expert is *e* (eval pass).

**Trend:** Lines **separate** — different experts carve different burial regimes.

### expert_{e}_tau_mean / expert_{e}_sasa_mean / expert_{e}_rho_mean / expert_{e}_ss_coil_frac

**What:** Per-expert **biology role card** from dominant routing: mean dehydron flag τ, SASA, wrap count ρ, and fraction of coil residues (ss_type ≥ 0.75).

**Trend:** Lines **diverge** when experts specialize by regime (rim/dehydron vs shell vs coil).

### expert_{e}_disc_r_mean

**What:** Average Poincaré disc radius for residues routed to expert *e*.

**Trend:** Separation (e.g. one expert with higher disc_r = more “surface/disc-expanded” role).

### expert_{e}_r_depth_sasa

**What:** Shell correlation **within** expert *e* only.

**Trend:** Higher = that expert’s residues still respect depth↔SASA; surface experts may run lower than core experts.

---



## Tier 3 — Shell training losses (`shell_corr_*`)

These are **loss terms** (minimize). Approximate form: `loss ≈ 1 − correlation` for correlation targets.


| Metric                     | What it penalizes                       | Good trend     |
| -------------------------- | --------------------------------------- | -------------- |
| **shell_correlation**      | Weighted total shell alignment loss     | ↓              |
| **shell_corr_depth_sasa**  | depth not aligned with SASA             | ↓ (→ 0 as r→1) |
| **shell_corr_epi_sasa**    | uncertainty not aligned with SASA       | ↓              |
| **shell_corr_proj_depth**  | disc radius not aligned with depth      | ↓              |
| **shell_corr_disc_spread** | disc radii too similar (collapsed disc) | ↓              |
| **shell_corr_disc_sasa**   | disc radius not aligned with SASA       | ↓              |


**Companion probes (maximize):** `shell_r_depth_sasa`, `shell_r_epi_sasa`, `shell_r_proj_depth`, `shell_r_disc_sasa` — same correlations as `probe_r_`* but from the training batch; should track the probe lines.

---



## Tier 4 — Disc & hyperbolic layout


| Metric                   | What                                          | Good trend                         |
| ------------------------ | --------------------------------------------- | ---------------------------------- |
| **disc_r_mean**          | Mean Poincaré disc radius |p|                 | Stable **0.2–0.4** in P2 (not → 0) |
| **disc_r_std_mean**      | Spread of disc radii across residues          | ↑ during disc expansion phases     |
| **disc_r_std**           | Batch-level disc std (loss path)              | Same idea as above                 |
| **disc_depth_scale**     | Loss pushing disc scale to match depth target | ↓                                  |
| **radial_std_mean**      | Spread of radial head outputs                 | Non-zero; stable                   |
| **cone_depth_mean_mean** | Average normalized depth                      | Context-dependent                  |
| **cone_depth_std_mean**  | Depth variability across residues             | ↑ vs collapsed; stable in P2       |


---



## Tier 5 — Other loss components


| Metric                         | What                                     | Good trend                      |
| ------------------------------ | ---------------------------------------- | ------------------------------- |
| **total**                      | Full multi-term training loss            | ↓ slowly                        |
| **evidential**                 | Uncertainty head (NLL + regularizer)     | ↓; often largest term early     |
| **cone_consistency**           | Radial vs angular head agreement         | ↓                               |
| **cone_depth_anticollapse**    | Penalty if depth std too flat            | → **0**                         |
| **angular_diversity**          | Angular head collapse penalty            | ↓ / stable                      |
| **neighborhood_consistency**   | Graph neighbor smoothness                | ↓                               |
| **domain_separation_2d / _3d** | Push domain centroids apart              | Often **0** if coeff ramped off |
| **capacity_loss**              | MoE overload (see Tier 2)                | ↓                               |
| **v2_teacher_depth**           | Distill match to frozen v3 teacher depth | ↓                               |
| **v2_teacher_epistemic**       | Distill match to teacher uncertainty     | ↓                               |
| **v2_teacher_total**           | Combined teacher distill loss            | ↓                               |


---



## Tier 6 — Ignore or rare


| Metric                                               | Notes                                                  |
| ---------------------------------------------------- | ------------------------------------------------------ |
| **grad_radial**, **grad_angular**, **grad_backbone** | Often **NaN** — gradient norms not logged on this path |
| **routing_save_max**                                 | Curriculum reference line, not model quality           |


---



## Reading a typical epoch (example)

```
Ep 100 | loss=8.28 route_H=1.306 | score=2.15 [skip]
       | shell r(d,s)=0.80 r(e,s)=0.86 r(|p|,d)=0.99
       | route_max=1.18
       | ineligible: routing_H=1.306
```


| Signal               | Reading                                           |
| -------------------- | ------------------------------------------------- |
| r(d,s)=0.80          | Shell excellent                                   |
| r(                   | p                                                 |
| route_H=1.306 > 1.18 | Too uniform for **this** epoch’s save gate → skip |
| score=2.15           | Below locked-in best 2.29 anyway                  |


**Takeaway:** Geometry can be production-grade while MoE routing is still “catching up” to the tightening ceiling — compare `routing_entropy` to `routing_save_max` on the same step.

---



## MLflow tags (not charts, but important)

Set at run start / on best save. View under **Tags** in the run page.


| Tag                             | Example                       | Purpose                                            |
| ------------------------------- | ----------------------------- | -------------------------------------------------- |
| **phase_preset**                | `full_hyp_moe_test`           | Which Makefile preset                              |
| **resume_from**                 | Path to warm-start checkpoint | Lineage                                            |
| **parent_run_id**               | Prior MLflow run              | Auto-linked when resuming from a tagged checkpoint |
| **checkpoint_path**             | Path to best `.pt`            | Used by `promote-v6-from-run`                      |
| **checkpoint_sha256**           | Hash                          | Verify promotion                                   |
| **best_score** / **best_epoch** | On eligible save              | Quick audit                                        |
| **focus_primary**               | `routing_entropy`             | Top metric to train next (end of run)              |
| **focus_recommendation**        | Sentence                      | Human-readable next step                           |
| **focus_critical_count**        | `0` / `1`                     | Count of blocking metrics                          |
| **focus_matters_needs_work_count** | `0`                        | High-leverage gaps only                            |


---



## Workflow cheat sheet

```bash
# Train with tracking
make train-v6-theory-test-mlflow RUN_ID=my_run DEVICE=cuda \
  RESUME=checkpoints/v6/runs/mlflow_theory_baseline/v6_best.pt

# Compare charts
make mlflow-ui    # http://localhost:5000 → tokyo-eyes-v6

# Promote winner
make promote-v6-from-run MLFLOW_RUN_ID=<run_id> STATUS=production CHECKPOINT_ID=tokyo_eyes_v6
make verify-v6-gnn
docker compose restart science
```

---



## Further reading


| Topic                       | Location                                                            |
| --------------------------- | ------------------------------------------------------------------- |
| **Training focus logic**    | `science/training/metric_focus.py`                                  |
| Checkpoint scoring gates    | `science/training/checkpoint_score.py`                              |
| Shell probe definitions     | `science/dtie/v6/loss.py` → `shell_correlation_loss`                |
| Epoch health measurement    | `experiments/training/v6/train_loop.py` → `measure_geometry_health` |
| Curriculum presets          | `science/training/config.py`                                        |
| Shell diagnostics (offline) | `experiments/diagnostics/shell_signal_diagnostics.py`               |


---



## Glossary (quick)


| Term                    | Plain English                                                                 |
| ----------------------- | ----------------------------------------------------------------------------- |
| **SASA**                | Solvent accessible surface area — how exposed a residue is to water           |
| **Cone depth**          | Model’s burial-depth signal per residue                                       |
| **Disc / |p|**          | 2D Poincaré projection radius — where the residue sits on the hyperbolic disc |
| **Routing entropy**     | How “spread out” expert choices are — lower = more specialized                |
| **Eligible**            | Epoch passed all gates and could save `v6_best.pt`                            |
| **Warm-start / resume** | Continue training from a previous `.pt` instead of random init                |


