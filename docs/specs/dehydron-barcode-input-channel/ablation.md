# Dehydron Barcode Ablation — Runbook

**Related:** [`design.md`](design.md) · [`tasks.md`](tasks.md) · [`../gnn-topology-input/tasks.md`](../gnn-topology-input/tasks.md)

**Purpose (P1):** Matched-epoch **representation** ablation — baseline `[ρ, τ, ss]` vs **orthogonal dehydron scalars** (+ missing) — on a **plain learned master-cold** topology-three-vector parent. Loss / MoE / geometry-freeze / routing-stack levers are **non-goals** (design §2).

> **2026-07-09 course correction:** Slim MoE + structural SSOT freezes `node_emb`/`convs` and bypasses MP→geometry, so barcode in `data.x` cannot affect inference. Never combine barcode with `--slim-moe-structural-ssot` unless `--allow-dead-feature-channel`.

> **2026-07-16 parameter lock:** Corpus z-norm at cache-build; smaller H1 scalar subset after orthogonality; empirical long-lived threshold; min-dehydron → missing-mask; cold arms require `isolated_init`; investigation audits use **`physics_investigation`**, not evidential. See design §5, §8.2, §11.

> **2026-07-17 parent lock (this is the load-bearing one):** P1 resumes from a **plain** topology-three-vector master-cold parent under the **v6.6** entrypoint/tag, **not** from the Fix-1+S4+SASA-gate+repulsion+scale-L2 routing stack (`fix1_s4_stack_initseed_controlled_3d_seed*`), and **not** from v66 feeler `p3_geom` / edge-barcode recipes. Those are real experiments; they answer a different question (“does barcode help *on top of* the routing stack / feeler geometry?”). P1 asks whether scalars improve representation quality in isolation.

> **Drift log 2026-07-17:** Earlier drafts pointed at abandoned v65 master-cold, then at v66 feeler edges / controlled 3-D routing seeds. Corrected to plain three-vector master-cold. Same class of silent drift as env-forwarding / stale colormap / hardcoded `v1` cache paths.

> **2026-07-17 labeling vs architecture (decision = relaunch):** An earlier plain three-vector cold (`checkpoints/v6/runs/master_cold_topology_three_vector_v1/phase_2.pt`, ge200, `node_emb` width 3, end `route_H≈1.30` / `eff_rank≈1.80`) is **architecturally** the right baseline content. It was launched via `experiments.training.v6.launch_training` / `tokyo-eyes-v6` / `checkpoints/v6/` by mistake. **Do not** silently move that file into `checkpoints/v66/` (quiet reclassification). Chosen path: **relaunch** with `make train-v66-master-cold-topology-three-vector` and treat the new `phase_2.pt` as the P1 parent. The v6-path artifact stays as a documented reference only.

---

## Locked parent (verify before any resume)

**Canonical (after relaunch completes):**

```text
checkpoints/v66/runs/master_cold_topology_three_vector_v1/phase_2.pt
```

| Check | Expected |
|-------|----------|
| Path / lineage | `checkpoints/v66/…`, `--gnn-lineage v6.6`, MLflow `tokyo-eyes-v66` |
| `node_emb.weight.shape[1]` | **3** |
| `global_epoch` | **200** (P1→P2 master-cold) |
| Recipe | `make train-v66-master-cold-topology-three-vector RUN_ID=master_cold_topology_three_vector_v1 SEED=1` |
| Routing-stack / feeler flags | **Absent** (`--v66-feeler-lineage`, `prototype_repulsion_*`, elevated `gate_logit_softplus_*`, `gate_include_sasa`, feeler rim/geom **not** passed) |

### Verification record — 2026-07-17

#### Parent identity (formal adoption)

Direct inspection of the landed `phase_2.pt`, its epoch-200 snapshot, resolved
master-cold phase configuration, and the live MLflow record established:

| Check | Observed |
|-------|----------|
| `node_emb.weight.shape` | `(128, 3)` |
| Epoch / phase | `global_epoch=200`, phase 2 |
| MLflow | Experiment `tokyo-eyes-v66` (id 3), finished run `c54c89ce590a4236a4548cf84f7a6f06`, run name `master_cold_topology_three_vector_v1`, `gnn_lineage=v6.6` |
| Learned geometry | `structural_disc_frozen=False`, `disc_layout_source=gnn_learned` |
| Gate recipe | `hyperbolic_gate=True`, `topology_only_gate=True`, `gate_include_sasa=False`, `v66_feeler_lineage=False` |
| Resolved shell–SASA weights | `shell_corr_depth_sasa_weight=0.0`, `shell_corr_epi_sasa_weight=0.0`, and `shell_corr_disc_sasa_weight=0.0` in every resolved phase |

**Corrected tensor expectation:** this plain master-cold recipe intentionally has
no `gate.mobius*` tensors. `topology_only_gate=True` removes the 128-D Möbius
trunk from gate logits; the checkpoint retains
`gate.prototype_bank.prototype_tangent`. Both the v6 reference and the v66
relaunch have this same gate shape. Requiring four `gate.mobius*` tensors would
silently change the experiment into a different routing recipe, so their
absence is an identity check for this parent, not a failed launch.

On this evidence, the canonical `phase_2.pt` above is the adopted P1 baseline
parent.

#### Origin-collapse correction: two defects, not one

The collapsed controlled-seed viewer was independently reconstructed using
`fix1_s4_stack_initseed_controlled_3d_seed2_v1`. Its
`v66_best_disc.pt` is epoch 7 of a 30-epoch run. Epoch 7 was genuinely
near-origin during training (`disc_r_mean=0.0026`); by epoch 30 the training
metric was `disc_r_mean=0.3136`. Separately, the old bare-checkpoint render path
dropped the T1a input z-normalization buffers and could collapse a mature
checkpoint.

| Forward pass on 4OBE | Checkpoint | T1a z-norm | `r_mean` | Interpretation |
|----------------------|------------|------------|----------|----------------|
| A | `v66_best_disc.pt`, epoch 7 | Present | 0.0026 | Genuine early, undertrained model state |
| B | `v66_best_disc.pt`, epoch 7 | Dropped | 0.0045 | Still collapsed; wrong checkpoint dominates |
| C | `phase_12.pt`, epoch 30 | Present | 0.3332 | Correct mature render |
| D | `phase_12.pt`, epoch 30 | Dropped | 0.0073 | Independent z-norm-drop collapse |

Therefore the screenshot was not fabricated geometry: the viewer selected a
real but inappropriate early checkpoint. Checkpoint selection and z-norm
restoration were separate bugs with separate blast radii.

#### Rim-push does not depower the locked corpus-12 criterion

The adopted v66 parent is the worst-case tested parent for this concern
(`disc_r_mean≈0.78`). The official physics-only audit scored all 12 exported
structures and passed rim enrichment on **12/12** (required: ≥11/12).

- Per-structure `depth_hi - depth_lo`: minimum `+2.1212`, median `+2.1648`.
- Corpus `cone_depth`: standard deviation `1.052`, range `5.572..8.000`.
- 4OBE: `depth_hi=7.9261`, `depth_lo=5.7220`, gap `+2.2041`.

This refutes the proposed loss-of-power mechanism for the parent: learned disc
radius may be rim-heavy, but the locked criterion partitions residues by
`physics_investigation` and compares `cone_depth`, whose variation remains
large. The high cohort is nevertheless close to the `cone_depth=8.0` ceiling
(`depth_hi≈7.93`). Treat that as a possible attenuation of a future
Scalars/Full **delta**; it does not invalidate the parent pass.

The v6-path reference viewers expose only legacy evidential `investigation`,
not `physics_investigation`, and therefore refusal-fail the locked audit. They
cannot substitute for the adopted v66 parent.

**Reference only (mislabeled launch — do not graft into v66 dirs):**

```text
checkpoints/v6/runs/master_cold_topology_three_vector_v1/phase_2.pt
```

Verified 2026-07-17: width 3, ge200, plain master-cold recipe via **v6** entrypoint; useful for comparing relaunch numbers (`eff_rank≈1.80`, `route_H≈1.30`), **not** the SSOT parent once the v66 relaunch lands.

**Do not use as P1 parent:**

| Candidate | Why wrong for P1 |
|-----------|------------------|
| `fix1_s4_stack_initseed_controlled_3d_seed{1,2}_v1` | Full Fix-1+S4+SASA-gate+repulsion+scale-L2 **routing** stack — seed-dependent commit, purity fail; confounds “did scalars help?” |
| `fix1_s4_stack_initseed_controlled_4d_seed{1,2}_v1` | Controlled **4-D** arm of the same routing ablation; closed as `NO_CONSISTENT_FOUR_D_WIN` |
| `feeler_expand_23_p3_geom_*` / `train-v66-feeler-p3-geom-edges` | Feeler geometry / edge-barcode recipe; prior disc collapse on node-global scalars; not the P1 baseline |
| v65 `dbh_*_master_cold_v1_2_seed1` archive | Wrong lineage for this continue; κ≈0.647 ring+cluster screenshot came from there |
| Quietly relocated copy of the v6-path `phase_2.pt` under `checkpoints/v66/` | Labeling lie — same confusion class as the original mislaunch |

---

## Prerequisites

1. **Phase 1 diagnostics green** (see [`tasks.md`](tasks.md)) — bar-length, orthogonality subset, min-dehydron guard, corpus z-score. Do **not** train on July 11-dim / `long_lived=2.0` scaffolding defaults. Diagnostic SSOTs may live under `checkpoints/v65/diagnostics/` (feature locks only).

2. **Gate stamp (Stage A-12 / small corpus)**
   ```bash
   make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json
   ```

3. **Barcode sidecars** for the Stage A-12 corpus under `dehydron_barcode_v1_2`:
   ```bash
   make precompute-dehydron-barcodes \
     CORPUS=v6_corpus_stage_a_small_v1.json \
     OUT_DIR=checkpoints/v6/dehydron_barcode_v1 \
     MAX_PROTEINS=12
   ```

4. **Environment** — `GNN_INPUT_MODE=topology_three_vector`. `--master-cold-lineage` via **`experiments.training.v66.launch_training`**. Do **not** pass `--slim-moe-structural-ssot`, and do **not** pass Fix-1 routing-stack flags (`--prototype-repulsion-*`, elevated `--gate-logit-softplus-*`, `--gate-include-sasa`, `--input-feature-zscore`, `--v66-feeler-*` rim/geom stack) on P1 arms.

---

## Ablation arms (P1)

| Arm | Node inputs | Role | Status |
|-----|-------------|------|--------|
| **Parent (cold)** | `[ρ, τ, ss]` only | Correctly tagged v66 plain master-cold (200 ep) | **Adopted** — identity verified; physics audit 12/12 |
| **Baseline** | `[ρ, τ, ss]` only | Control continue off locked v66 parent | **Complete** — ge215; eligible |
| **Scalars** | + orthogonal dehydron scalars + missing (`node_dim=7`) | “Do barcodes help at all?” | **Complete** — ge215; liveness LIVE (MLflow + offline stamp); awaiting physics audit |
| **Full / binned** | + scalars + bins | — | **Deferred** until Scalars verdict |
| Routing-stack + barcode | scalars on Fix-1/S4/… stack | Different question | **Out of P1 scope** |
| Feeler edge barcode | `--dehydron-edge-barcode` on feeler | Separate post-P1 / feeler track | **Out of P1 scope** |

Shared knobs: `RESUME` (default = locked v66 `phase_2.pt` above), `EPOCHS` (short continue, e.g. 15), `RUN_ID`, `DEVICE`, `SEED` (required for any cold / width-changing arm).

**Warm-start note:** Scalars widen `node_dim` 3→7. On `--resume`, `node_emb` is resized (overlap copied, new cols zero-filled) and remains trainable under master-cold — no fresh RNG draw for the bank.

**Cold-start note:** any arm that widens `node_emb` **must** pass `--seed` → `init_seed` → `isolated_torch_seed` / `derived_seed`. Continues off the locked 3-input parent are safe for gate/prototype init.

---

## Exact commands

```bash
make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json
make precompute-dehydron-barcodes \
  CORPUS=v6_corpus_stage_a_small_v1.json \
  OUT_DIR=checkpoints/v6/dehydron_barcode_v1 \
  MAX_PROTEINS=12

# (1) Relaunch plain parent under correct lineage — do this before matched continues
make train-v66-master-cold-topology-three-vector \
  RUN_ID=master_cold_topology_three_vector_v1 \
  SEED=1

PARENT=checkpoints/v66/runs/master_cold_topology_three_vector_v1/phase_2.pt
# Verify: Training node_dim=3; GNN lineage v6.6; no feeler/routing-stack flags in the launch log

# (2) Matched Phase-2 continues — parent is ge200 Phase 2; P3_ENTRY_GATE correctly
# blocks Phase 3 (routing H not stable for 5 consecutive epochs). Phase 2 keeps
# gate/backbone trainable — right surface for a representation feature ablation.
make train-v66-dbh-baseline RUN_ID=dbh_ablation_baseline_v1 SEED=1 EPOCHS=15
make train-v66-dbh-scalars  RUN_ID=dbh_ablation_scalars_v1  SEED=1 EPOCHS=15
# Defaults: RESUME=$PARENT, DBH_DIR=checkpoints/v65/dehydron_barcode_v1 (Stage A-12 v1_2)
# Note: --master-cold-lineage + --resume is recipe-preserving continue (not a cold-start claim).
```

The v66 parent `phase_2.pt` is verified and continue targets are wired to it. **Do not** launch `train-v66-feeler-*` or `train-v65-dbh-*` and call it P1. **Do not** resume barcode arms from the v6-path reference artifact (option 2 was rejected in favor of relaunch).

Outputs for P1: under `checkpoints/v66/runs/<RUN_ID>/` · MLflow `tokyo-eyes-v66`.

---

## Evaluation checklist

### 0. Liveness (required)

`liveness_barcode_alive=1` (and `liveness_mp_alive=1`). If barcode delta stays ~0, **do not** interpret investigation audits.

> **2026-07-17 visibility note:** On the completed P1 Scalars continue
> (`dbh_ablation_scalars_v1`), these keys were written to **MLflow** every epoch
> (`liveness_barcode_alive=1` for all 15 steps) but were **not** persisted into
> `metrics.json` (entry builder only kept health/losses). Direct tensor checks
> confirmed the channel was live (`node_emb` width 7; barcode `data.x` cols
> nonzero + structure-varying; offline probe `alive=True`). Treat missing
> `metrics.json` liveness as a logging gap until the stage_runner persistence
> fix lands — do **not** confuse it with a dead feature. Prefer MLflow or an
> offline `probe_barcode_liveness` stamp before refusing audits.

> **2026-07-17 hardcoded-slice audit (the `[:, 4:]` bug class).** After fixing the
> liveness probe (`[:, 4:]` → `gnn_input_dim()`), grepped for other literal
> column-offset slices that assume a fixed base width. **It was not fully
> isolated**, but the live P1 train/score path is clean:
> - **Legitimate (not the bug):** `data.x[:, 0/1/2]` reads of ρ/τ/ss and
>   `[:, 3]` SASA reads under explicit `legacy_four_vector` — these are fixed base
>   positions by definition.
> - **Latent, same class, NOT on P1 path:**
>   `science/dtie/common/residue_features.py:159` `residue_sasa_from_data` returns
>   `data.x[:, 3:4]` as "SASA" whenever width>3 and no `data.sasa` side-channel —
>   on a 3-vector+barcode graph this silently returns the **first barcode scalar**.
>   Guarded (prefers `data.sasa`, master-cold zeroes SASA weights) so it did not
>   fire here, but it is fragile. Same pattern in diagnostics
>   `t1a_znorm_forward_probe.py:139` and `t1c_mp_relative_loss_null.py:73`
>   (`if x.size(1) > 4: cat([x4, x[:, 4:]])`) — 4-vector-only diagnostics, not the
>   barcode scoring path.
> - **Fixed:** the liveness probe (the only instance on the live P1 path).
> Conclusion: the class is a repeated pattern (3 further latent sites), but none
> corrupted the P1 baseline/scalars run. Left as tracked debt, not a P1 blocker.

### 1. 4OBE investigation motifs — **physics path only**

Spot-check Switch I/II, α3 ~105–107, C-term pivot using **`physics_investigation`** — not evidential. Judge against the **same plain v66 parent** (`checkpoints/v66/.../phase_2.pt` / matched baseline arm), not against routing-stack, feeler, or v65 screenshots.

### 2. Corpus-12 rim enrichment — **physics path only**

```bash
python -m experiments.diagnostics.investigation_audit_corpus12 \
  --run-dir checkpoints/v66/runs/<RUN_ID>
```

### 3. Cone / τ probes

Stable vs baseline arm; prefer per-structure distributions over loss-wiring near-1.0.

### 4. MoE routing health

No new starvation / eligibility collapse **attributable to barcode**. Do not treat purity/Gram as a P1 promotion gate (closed as not-solved-at-this-scale on the routing track).

### 5. Curvature provenance

Read κ from the checkpoint under test. Do not compare across unrelated lineages.

---

### P1 result — 2026-07-17 (baseline vs scalars, matched off locked v66 phase_2)

Liveness gate open (see visibility note above). Both arms scored on `physics_investigation`.

| Criterion | Baseline | Scalars | Read |
|-----------|----------|---------|------|
| Corpus-12 rim enrichment | **12/12** pass (≥11 req) | **12/12** pass | Both clear the floor — non-discriminating |
| Per-structure depth gap (mean `depth_hi−depth_lo`) | 2.5085 | 2.5830 | Scalars **+0.0746**, positive on **12/12** structures |
| 4OBE physics motifs (ranks) | Switch I/II, α3, C-term present | **identical** | `physics_investigation` is ρ/τ-input-derived → cannot discriminate arms by construction |
| Cone/τ probe | 0.9923 | 0.9918 | Flat (within noise) |
| MoE health | best score 2.440, final eligible | best score 2.381, final **ineligible on route_H** | Scalars marginally **worse** |

**Verdict — mixed / marginal, not a clean win.**

- The barcode channel is genuinely live and produces a *consistent* effect: the
  cone_depth separation between physics-hi and physics-lo cohorts widens on
  **12/12** structures (uniform sign; sign-test p≈2⁻¹² ≈ 2e-4 — real, not noise).
  Magnitude is small (~+0.075 on a ~2.5 gap, ~3% relative).
- It does **not** improve any named pass/fail promotion criterion beyond what the
  plain baseline already achieves (both 12/12 rim, motifs retained-not-improved,
  cone/τ flat), and it is marginally **worse** on MoE routing health (lower best
  score, final checkpoint route_H-ineligible).
- The 4OBE motif spot-check cannot discriminate the arms: it ranks residues by
  `physics_investigation`, an input-side ρ/τ quantity identical across arms. Only
  the learned-cone_depth rim-gap is arm-discriminating in this audit.
- Ceiling caveat holds: `depth_hi≈7.93` sits near the `cone_depth=8.0` ceiling, so
  the observed delta is a floor on the true representation effect, not a ceiling —
  a real benefit could be partly masked by saturation.

Decision per the table below: this lands between "checklist pass vs plain
baseline" (true, but baseline also passes) and no improvement on the promotion
axes plus a slight routing-health cost. **Do not auto-promote scalars as default.**
The live, consistent depth-gap sharpening is a legitimate positive signal worth
carrying into Full/edge experiments, but on the named P1 criteria the scalars-only
channel is inconclusive-to-marginal, not promotion-worthy on its own.

### OOD KRAS G12D coupled-lock probe — 2026-07-17

A forward-only hypothesis probe compared the matched Baseline and Scalars
checkpoints on two G12D structures not enabled in the Stage A-12 training set:
1AGP (the pure-G12D structure named in the suppressor figure) and 4DSO.

The OOD barcode sidecars were featurized per structure but normalized with the
**locked Stage A-12** statistics. The single-PDB path refuses missing or
wrong-version stats; it never recomputes μ/σ on the target structure.

Before inference, both PDBs passed a residue-numbering/identity refusal gate:
chain A model rows resolve exactly to ASP12, TYR32, GLY60, and GLN61
(`A:12:`, `A:32:`, `A:60:`, `A:61:`).

| Structure | Baseline lock mean | Scalars lock mean | Raw Δ | Global-normalized Δ |
|-----------|-------------------:|------------------:|------:|--------------------:|
| 1AGP | 0.2748 | 0.2965 | **+0.0217** | **+0.0979** |
| 4DSO | 0.2655 | 0.3061 | **+0.0406** | **+0.1440** |

Positive Δ means Scalars places the three Asp12 lock partners **farther** away
in learned Poincaré distance. Pair-level agreement:

- 1AGP: 12–32 `+0.0466`, 12–60 `+0.0274`, 12–61 `−0.0088`
  (2/3 farther).
- 4DSO: 12–32 `+0.0502`, 12–60 `+0.0233`, 12–61 `+0.0484`
  (3/3 farther).

**Layer pin (required follow-up):** the first-pass numbers above are Poincaré
distances on the **final disc** (`hyp_projections_2d`), not trunk hidden state.
Re-running the same lock check on post-MP `encoder_h` (Euclidean L2, same hook
path as T1a trunk rank) yields:

| Structure | Disc normalized Δ | Trunk normalized Δ | Layer verdict |
|-----------|------------------:|-------------------:|---------------|
| 1AGP | **+0.0979** | **+0.0007** | `projection_only` |
| 4DSO | **+0.1440** | **−0.0013** | `projection_only` |

Trunk raw lock means do move slightly (`+0.11` / `+0.13`), but the all-residue
median moves in lockstep, so **normalized trunk compactness is flat**. The
disc-level separation therefore does **not** originate as a trunk relational
reorganization; it is specific to the projection step. Given this layer's known
fragility (rim-push vs filled-disc recipe effects, τ-coupled radial target,
recent z-norm/render bugs), treat the disc-only Δ as closer to projection
artifact / global-feature side-effect than as evidence of pairwise coupling.

**Payload-scope note (expected negative):** two of the three locked scalars
(`total_persistence_h1`, `fraction_long_lived_h1`) are structure-broadcast
constants among touching residues; only `n_dehydrons_touching` varies
per-residue. Per `design.md` §5.1, structure-level witness persistence does not
associate bars back to individual midpoints, so this payload structurally cannot
encode a specific shared-bar identity between residues 12 and 32. The negative
result constrains the **current scalar payload**, not the deeper
"shared persistent feature should pull residues together" hypothesis.

**Correct next experiment:** typed edges for residue pairs sharing a persistent
dehydron bar (`design.md` §9 deferred item 3 / local-association pass), as its
own registration — not folded into P1. Any such probe must record **trunk and
disc distances separately from the start**.

**Standing rule (escalated):** the trunk↔disc disagreement seen here recurred in
Chem-MVP (sign disagreement) and again in the flow-influence pilot (sign-*flipped*
betweenness correlations). Disc alone is forbidden for relational claims — see
[`docs/audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md).

Reproducible artifact:
`checkpoints/v66/diagnostics/kras_coupled_lock_ood/report.json` (schema v2).

## Promotion decision

| Outcome | Action |
|---------|--------|
| Scalars + liveness green + checklist pass vs plain baseline (physics scoring) | Promote scalars as default **representation** feature candidate |
| Scalars fail | Keep barcode off; only then consider Full / edge / routing-stack-on-top as separate experiments |
| Liveness dead | Fix recipe (not barcode math) — refuse slim+SSOT |
| Result only on Fix-1 / feeler parent | **Not a P1 answer** — re-run on locked plain parent |

See [`design.md`](design.md) §2 / §8 / §11 and [`docs/audit/LEARNED_GNN_VS_SLIM_SSOT.md`](../../audit/LEARNED_GNN_VS_SLIM_SSOT.md).
