# Standing probe defect: Jacobian flow-influence under input z-norm

**Date filed:** 2026-07-18  
**Status:** Workaround locked (2026-07-18) — Jacobian L2 flow-influence remains
**forbidden** on any `input_feature_zscore=True` lineage; trusted causal
substitute = forward knockout; trusted geometry readout = direct PC1  
**SSOT home:** this file; indexed from [`ablation.md`](ablation.md) §Standing probe defects  
**Discovered during:** [`../rho-tau-abs-dist-swap/`](../rho-tau-abs-dist-swap/) matched chem-MVP arms (not a property of `|ρ−TAU|` itself)

---

## Claim

On trunks **trained** with T1a input z-norm, the Jacobian flow-influence probe’s
flow-centrality **anti-correlates** with classical betweenness (Stage A-12 median
≈−0.3, 0/12 holds), while the **same** trunk’s static PC1 geometry **positively**
correlates with betweenness (median ≈+0.46, 11/12 holds).

This is **not** fixed by changing the probe score from `s_B=||h||²` to
`s_B=(h·û)²` (`--score-mode pc1_sq`). Direct PC1 is fine; **backward-pass
influence** relative to that axis is not.

**Scope of contamination:** Standing triangulated flow-influence negatives
(containment asymmetry, 163 hub migration, knockout) used **z-norm-off**
chem-MVP — those results are **not** directly invalidated by this defect.
**Any future Jacobian experiment on a z-norm-on lineage is untrustworthy**
under the locked workaround below. Z-norm is the locked default for new
T1a-era work, so most future causal grades must use knockout (or wait for a
real probe redesign).

---

## Workaround locked (Move 1 — 2026-07-18)

### Recipe (exact)

| Tool | `input_feature_zscore=False` | `input_feature_zscore=True` |
|------|------------------------------|-----------------------------|
| Jacobian `jacobian_flow_influence` (`raw_x` / `post_zscore` / `post_node_emb`) | **Allowed** (hist chem-MVP SSOT) | **Forbidden** for causal / Part-B grades |
| Direct PC1 ↔ classical | Allowed (geometry only) | **Allowed** (geometry only — not causal) |
| Forward knockout / feature ablation | Allowed | **Trusted causal substitute** |

CLI diagnostic only (does **not** recover polarity on z-norm-on):

```bash
GNN_INPUT_MODE=topology_three_vector PYTHONPATH=. python -m \
  experiments.diagnostics.jacobian_flow_influence \
  --checkpoint <ckpt> --pdb-id 4OBE --score-mode pc1_sq \
  --grad-site {raw_x|post_zscore|post_node_emb} --layers encoder_h
```

### What remains forbidden

1. Do **not** promote, refute, or Part-B-grade causal claims from Jacobian
   flow-influence on `input_feature_zscore=True` checkpoints — including with
   `--grad-site post_zscore` or `post_node_emb`.
2. Do **not** treat `--grad-site` as a fix; it only relocates the same
   anti-betweenness ranking.
3. Passive-norm-on passive geometry (PC1, etc.) must stay labeled **not causal**.

### Acceptance numbers (4OBE, `pc1_sq`, trunk `encoder_h`)

Artifact:
`checkpoints/v66/diagnostics/learned_flow_influence/jacobian_znorm_grad_site_sweep/`

| Arm | `grad_site` | Spearman(flow, betweenness) | vs ANM MSF | Holds? |
|-----|-------------|----------------------------:|-----------:|:------:|
| z-norm-off (hist) | `raw_x` | **+0.690** | −0.699 | yes |
| z-norm-off | `post_zscore` | **+0.690** | −0.699 | yes |
| z-norm-off | `post_node_emb` | **+0.688** | −0.698 | yes |
| z-norm-on | `raw_x` | **−0.505** | +0.493 | no |
| z-norm-on | `post_zscore` | **−0.497** | +0.497 | no |
| z-norm-on | `post_node_emb` | **−0.472** | +0.472 | no |

- Acceptance (1) hist z-norm-off stays healthy (~+0.69): **PASS**
- Acceptance (2) z-norm-on Jacobian polarity agrees with direct PC1 (~+0.54):
  **FAIL** at all three sites → substitute locked (knockout + PC1 geometry)

### Mechanism read (locked after grad-site sweep)

Layer-tap smoke ruled out mid-trunk LayerNorm flips and showed B-local raw-x
act·grad wash-out (`≈+0.05`, pool `attenuation_compressed_near_zero`). The
grad-site sweep shows that relocating `‖∂s_B/∂feat_A‖` to post-zscore or
post-`node_emb` **does not** restore betweenness polarity — so the defect is
**not** “wrong leaf tensor / raw-x scale alone.”

Working hypothesis (still open for a future redesign, not blocking the
workaround): L2 aggregation of per-source feature grads under z-norm-trained
trunks **ranks an anti-hub / flexibility-aligned axis** (note ANM Spearman
flips +0.49 vs healthy −0.70) rather than flipping a coherent hub map.
Near-noise manufacture at the input remains compatible; it is not required to
explain the failed post-site recovery.

---

## Evidence (locked — pre-workaround)

| Readout on z-norm chem-MVP Stage A-12 | Result |
|---------------------------------------|--------|
| Direct sign-invariant PC1 ↔ betweenness | **+0.46** median, **11/12** holds |
| Jacobian `norm_sq` flow-centrality ↔ betweenness | **−0.32** median, **0/12** |
| Jacobian `pc1_sq` flow-centrality ↔ betweenness | **−0.31** median, **0/12** |
| Same `pc1_sq` on hist **z-norm-off** chem-MVP | **+0.51** median, **6/12** (healthy) |
| 4OBE: grad w.r.t. **already z-scored** `x` (`pc1_sq`) | still **−0.50** (transform bypass does not restore) |
| 4OBE: clean `--grad-site post_zscore` / `post_node_emb` | still **≈−0.47…−0.50** (Move 1 sweep) |

Artifacts:
- `checkpoints/v66/diagnostics/rho_tau_abs_dist_swap/znorm_isolation_*.json`
- `.../znorm_norm_vs_input_magnitude.json`
- `.../part_b_*_pc1_sq_stage_a12.json`
- `.../part_b_pc1_regrade.json`
- `checkpoints/v66/diagnostics/learned_flow_influence/jacobian_znorm_grad_site_sweep/`

Related magnitude mechanism (separate, confirmed): under z-norm,
`||h||` ↔ raw ρ flips (+0.997 → −0.928). That explains why **`norm_sq`** fails;
it does **not** explain why **`pc1_sq` Jacobian** still fails while direct PC1
succeeds.

---

## Minimal affine repro (rules out naive std-sign bug)

Synthetic check in `tests/test_jacobian_znorm_gradient_repro.py`:

- Affine z-score `(x−μ)/σ` has **positive** channel scales `1/σ`.
- For linear `s = (W z)[b]²`, `∂s/∂x_raw` and `∂s/∂z` agree up to positive
  `1/σ` — **no sign flip** from standardization alone.

Therefore the standing defect is **not** “autograd through mean/std is
sign-wrong in isolation.” It is: **on z-norm-trained GNN trunks, Jacobian
flow-centrality polarity disagrees with the representation’s own geometry↔GT
correlation.**

---

## Rules (superseded detail — see Workaround locked)

1. Do **not** promote, refute, or Part-B-grade causal claims from Jacobian
   flow-influence on `input_feature_zscore=True` checkpoints.
2. Passive-norm-off Jacobian results (standing chem-MVP triangulation) remain the
   causal SSOT for that lineage.
3. Passive-norm-on work may use **passive** geometry readouts (PC1, etc.) only with
   the caveat that they are **not** causal influence.
4. Knockout / forward interventions remain valid causal tools (no gradient).

---

## Fix candidates (status after Move 1)

- ~~Compare influence w.r.t. post-`node_emb` / post-zscore vs raw input~~ —
  **Done** — does not recover; not a fix
- ~~Priority: why z-norm training attenuates raw-x act·grad at B~~ — still
  true as a local polarity fact; **not** sufficient to explain aggregate
  anti-betweenness (post-sites also fail)
- If redesigning the probe later: test whether a **non-L2** influence
  definition (or directed path score) can recover hub polarity without
  tracking ANM; until then keep Jacobian quarantined on z-norm-on
- Forward knockout / feature ablation — **locked interim causal grade** on
  z-norm-on arms

---

## Ordered candidate ops: `data.x` → probe `encoder_h` (2026-07-18)

**Critical scope note:** the Jacobian probe hooks `encoder_h` as the **input to
`radial_head`** — i.e. the Euclidean trunk **after** MP, **before** hyperbolic
lift. On chem-MVP z-norm (`EquivariantConvMultiRel` ×6 + `LayerNorm` ×6):

| # | Op | Division / norm / hyp? | Notes |
|---|-----|------------------------|-------|
| 0 | `data.x` (raw ρ,τ,ss) | — | Default probe autograd target (`--grad-site raw_x`) |
| 1 | `transform_node_features` → `(x−μ)/σ` | **DIV** | Affine only; minimal repro already rules out sign bug *in isolation*; `--grad-site post_zscore` = `node_emb` input |
| 2 | `node_emb` `Linear` | — | `--grad-site post_node_emb` |
| 3a…6a | `EquivariantConvMultiRel` | **NORM** | `o3.spherical_harmonics(..., normalize=True)` — unit direction of edge `rel_pos`; e3nn TP + radial MLP (SiLU), no LayerNorm inside |
| 3b…6b | `nn.LayerNorm(hidden)` | **DIV** | Per-node feature std; **demoted** by layer-tap smoke (no local flip) |
| 3c…6c | `F.silu` | — | Smooth; derivative always ≥0 for typical regimes |
| 3d…6d | residual `x + x_res` | — | |
| ★ | **`encoder_h` (= `x` into `radial_head`)** | — | **Probe capture / score site** |

**Off this path (do not prioritize for this defect):** `radial_head` /
`angular_head` → `rescale_tangent_before_expmap` → `expmap0` → `dist0` /
`logmap0` / disc project. Those are **after** `encoder_h`. Saturating
hyperbolic / tanh-tail hypotheses are **out of scope** for
`∂s/∂x` when `s` is a function of probe `encoder_h` alone.

**Suspect ranking after layer-tap + grad-site sweep:**
1. **L2 influence aggregation under z-norm training** — ranks anti-hub /
   ANM-aligned structure at `raw_x`, `post_zscore`, and `post_node_emb`
2. Raw-x act·grad wash-out — real local fact; **not** the sole cause
3. ~~6× `LayerNorm` local polarity flip~~ — **ruled out** on 4OBE matched B
4. SH normalize — near-zero post-conv cos on **both** arms; not z-norm-specific

**Instruments:**
- `experiments/diagnostics/jacobian_znorm_layer_grad_sign.py`
- `experiments/diagnostics/jacobian_flow_influence.py` (`--grad-site`)

```bash
GNN_INPUT_MODE=topology_three_vector PYTHONPATH=. python -m \
  experiments.diagnostics.jacobian_znorm_layer_grad_sign \
  --checkpoint checkpoints/v66/runs/chem_mvp_znorm_stage_a12_cold_v1/v66_best.pt \
  --pdb-id 4OBE --target-b 92
```

Artifact:
`checkpoints/v66/diagnostics/learned_flow_influence/jacobian_znorm_layer_grad_sign/4OBE_layer_grad_sign.json`

---

## Layer-tap smoke (4OBE, matched `target_b=92`, 2026-07-18)

### What this rules in / out (be precise)

| Claim | Status |
|-------|--------|
| Flip happens inside the Euclidean trunk (LN / MP) | **Ruled out** — 0 DIV sign-flips; B-local `encoder_h` act·grad ≈+0.58 |
| B-local raw_x cos ≈+0.05 is a **sign flip** | **Ruled out** — weakly positive, not negative |
| B-local raw_x cos ≈+0.05 is **wash-out / near-orthogonal** | **Confirmed** |
| Pool at raw_x under z-norm is **wide ± incoherence** | **Ruled out** (see distribution) |
| Pool at raw_x under z-norm is **compressed near zero** (attenuation) | **Confirmed** |
| Relocating grad site to post-zscore / post-`node_emb` restores hub polarity | **Ruled out** (Move 1 sweep) |

**Do not call this closed as “the input flipped.”** A near-zero positive is
closer to “input-level directional gradient signal washed out” than “sign
inverted.” Aggregate Jacobian↔betweenness ρ≈−0.3 could still be manufactured
later by ranking/aggregating near-noise `‖∂s_B/∂x_A‖` — and Move 1 shows the
same anti-sign persists even when differentiating in healthier post-zscore /
post-emb spaces.

### B-local tap table

| Tap | z-norm-on cos(act, ∂s) | z-norm-off control |
|-----|------------------------|--------------------|
| `00_raw_x` | **≈+0.05** (wash-out) | **≈+0.98** |
| `01_after_input_zscore` | ≈+0.58 | (n/a) |
| `02_after_node_emb` | ≈+0.35 | ≈+0.97 |
| `L0_b` LayerNorm | ≈+0.43 | ≈+0.02 |
| `99_encoder_h` | ≈+0.58 | ≈+0.63 |
| sign-flips at DIV (LayerNorm) | **0** | 0 (control flips are residual restores) |

### Pool distribution of per-residue cos(x[i], ∂s_B/∂x[i]) at `00_raw_x`

Same single backward of `s_B` (`pc1_sq`); N≈169 residues. Instrument:
`pool_act_grad_cos_distributions` in
`experiments/diagnostics/jacobian_znorm_layer_grad_sign.py`.

| Arm | mean | std | q10 | q90 | frac \|c\|<0.1 | frac \|c\|>0.5 | `shape_read` |
|-----|------|-----|-----|-----|----------------|----------------|--------------|
| z-norm-on | **+0.06** | **0.20** | −0.09 | +0.25 | **0.57** | **0.05** | **`attenuation_compressed_near_zero`** |
| z-norm-off | +0.04 | **0.69** | −0.89 | +0.87 | 0.08 | **0.70** | `incoherence_wide_scatter` |

**How to read the control pool:** wide ± scatter across A with mean≈0 is
**expected** for A≠B — there is no reason residue A’s features should align with
`∂s_B/∂x_A`. The healthy control signal is **localized at B** (B-local cos≈+0.98).
Do not treat control pool mean≈0 as “control is also washed out.”

**How to read the z-norm pool:** mass compressed into roughly [−0.1, +0.25],
almost no strong ± — including at B itself (+0.05). That is **attenuation /
directional wash-out at the input**, not bipolar incoherence.

Also note: post-zscore pool on the z-norm arm **does** go wide
(`incoherence_wide_scatter`) — directional structure reappears in z-scored
feature space for A≠B, while raw-x space stays compressed. Combined with the
grad-site sweep (post-zscore influence still ≈−0.50), healthy local polarity
at a tap is **not** sufficient for hub-ranking via L2 aggregation.

---

## Do not conflate with

- `|ρ−TAU|` Part A (rank) — still Confirmed  
- `|ρ−TAU|` Part B **passive** Partial — see that ablation’s explicit caveat  
- `|ρ−TAU|` Part B **causal knockout Partial** (Move 2 methodology regrade) —
  `../rho-tau-abs-dist-swap/ablation.md` +
  `checkpoints/v66/diagnostics/rho_tau_abs_dist_swap/part_b_causal_knockout_stage_a12.json`
  (hub scaffolding telemetry; soft agent Pass withdrawn — not Jacobian)  
- T1a z-norm itself (still correct for scale; Fix-1+S4 lineage verified)
- Agent Path 2 pilot — **ORPHANED** ([`PATH2_DIRECTIONALITY.md`](PATH2_DIRECTIONALITY.md))
