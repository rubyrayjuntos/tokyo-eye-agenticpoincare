# Routing entropy sparsity — design

**Date locked:** 2026-07-20  
**Status:** Implemented — sparsity save gates registered; confirm-continue pending  
**Parent:** Fix-1 SSOT restore + expand ladder (`docs/specs/fix1-s4-restore/`)  
**Evidence:** hub knockout sealed ρ(4OBE)=0.596 vs P2/P3 champions ≤0.29; Phase 12 routing H≈1.33–1.38 (near-uniform)  
**Make (init):** `make train-v66-fix1-sparsity-sealed-continue` (default `RUN_ID=fix1_s4_sparsity_sealed_continue_v1`)  
**Make (confirm):** `make train-v66-fix1-sparsity-confirm-continue` (resume `epoch_045.pt`, default 5 ep)  
**Resume (init):** `v66_healthy_sealed.pt` only  
**λ peak / warmup:** `0.0075` / `8` (confirm continue: warmup=0, λ stays at peak)

### Save / monitor gates (registered 2026-07-20 post Task 6)

| Gate | Bound | Role |
|------|-------|------|
| `routing_entropy_mean_residue` (final) | ∈ **[0.50, 0.90]** | Hard save eligibility |
| `routing_entropy_mean_residue` (final-3 mean) | ∈ **[0.50, 0.90]** | Sparsity Pass bar |
| `max_share` | **&lt; 0.45** | Hard save eligibility |
| `H(f̄)` | warn **&lt; 1.00**; abort **≤ 0.80** | Diversity monitor / circuit breaker |
| Legacy `H(f̄) ≤ 1.21` | **removed** as save blocker | Was fighting healthy global balance |
| 4OBE hub ρ | **≥ 0.45** | Biology hold (vs sealed 0.596) |

---

## 1. Problem

Fix-1 continue training (esp. Phase 2 contested routing) drove the MoE gate into a **uniform equilibrium**:

- Batch routing entropy `H(f̄)` ≈ ln(4) ≈ 1.386
- Save ceiling `H ≤ 1.21` never met → no `v6_best` eligibility
- Capacity / timeout stack only fights **starvation** and **monopoly**, not uniformity
- Existing `routing_entropy` is **monitor-only** (not in `total` loss)

Consequence: experts stay interchangeable; trunk hub scaffolding vs Cα betweenness collapses (sealed 0.596 → P2/P3 ~0.24–0.29). Without partitioned routing, directional biology claims (e.g. KRAS G12D hub migration) cannot be graded as expert-lens phenomena.

---

## 2. Locked recipe

| Item | Choice |
|------|--------|
| Penalty | \(L_{\mathrm{sparse}} = \frac{1}{N}\sum_i H(p_i)\) over residues (nats) |
| Scale | \(\lambda_{\mathrm{sparse}} \in [0.5, 1.0]\times\lambda_{\mathrm{cap}}\) |
| Warmup | Linear over first **8** epochs of resume (default; allow 5–10) |
| Init | Resume **only** from `v66_healthy_sealed.pt` |
| Guards | Keep `capacity_loss` + 45% expert timeout |
| Deferred | Gini on routing weights (only if soft two-expert mush) |

Feeler Phase 12 reference: `balance_coeff` (\(\lambda_{\mathrm{cap}}\)) = **0.015** → \(\lambda_{\mathrm{sparse}}^{\mathrm{peak}} \in \{0.0075, 0.015\}\).

Default registration: **\(\lambda_{\mathrm{sparse}}^{\mathrm{peak}} = 0.5\times\lambda_{\mathrm{cap}} = 0.0075\)** for first sealed continue; escalate to \(1.0\times\) only if mean residue-H stalls above 0.9 after warmup.

---

## 3. Critical distinction: two entropies

The model today exposes:

```text
routing_entropy = H( mean_i p_i )     # H(f̄) — batch load entropy (MONITOR)
```

The sparsity loss **must** use a different quantity:

```text
routing_entropy_mean_residue = mean_i H(p_i)   # L_sparse (LOSS + TELEMETRY)
```

| Metric | Low when… | High when… |
|--------|-----------|------------|
| `mean_i H(p_i)` | Each residue commits to few experts | Each residue is near-uniform |
| `H(f̄)` | Corpus load collapses to one expert | Corpus uses all experts evenly |

**Desired regime:** low `mean_i H(p_i)` **and** moderate `H(f̄)` / balanced load — local commitment, global diversity. That is the adversarial geometry vs capacity.

Do **not** put `H(f̄)` into the sparsity loss (that would fight capacity and encourage monopoly).

---

## 4. Loss wiring

```text
total += λ_sparse(t) * mean_i H(p_i)
λ_sparse(t) = λ_peak * min(1, t / T_warmup)    # t = 1..epochs in this continue phase
```

- Compute `H(p_i)` on soft expert weights **before** timeout masking for the loss value used in telemetry; apply the same `scores` tensor the model routes with for the gradient path (training-time bans already reshape logits — do not double-mask).
- Log both raw `routing_entropy_mean_residue` and weighted `sparse_loss = λ_sparse * L_sparse`.
- Keep existing `routing_entropy` (= `H(f̄)`) as governance monitor + save ceiling input.

### Warmup schedule (locked default)

| Epoch in continue (1-indexed) | \(\lambda_{\mathrm{sparse}} / \lambda_{\mathrm{peak}}\) |
|-------------------------------|--------------------------------------------------------|
| 1 | 0.125 |
| 2 | 0.250 |
| … | linear |
| 8 | 1.000 |
| ≥9 | 1.000 |

Abort / downscale rule: if any epoch in warmup produces a **new** expert timeout ban **and** `max_soft_share ≥ 0.45`, freeze \(\lambda\) at the previous epoch’s value for 2 epochs (hold), then resume ramp. This is a soft governor, not a hard train abort.

---

## 5. Collision telemetry (answer to warmup monitor question)

**Question:** At which epoch does rising \(\lambda_{\mathrm{sparse}}\) begin to actively collide with `capacity_loss` / approach the 45% timeout?

**Primary collision detector — log every epoch:**

| Signal | Role |
|--------|------|
| `routing_entropy_mean_residue` | Sparsity objective; should fall toward [0.5, 0.9] |
| `capacity_loss` (raw, pre-λ) | Capacity objective; near-zero when no starvation |
| `λ_sparse * L_sparse` vs `λ_cap * capacity_loss` | Weighted contributions to `total` |
| `usage_max_soft_share` / eval `max_routing_fraction` | Monopoly approach to **0.45** timeout |
| `min_routing_fraction` / soft min load | Approach to `min_usage` (typically 0.05) starvation floor |
| `expert_timeout_bans` (ids + count this epoch) | Hard failsafe trips |

**Define “collision epoch” \(t^\*\) as the first continue epoch where all hold:**

1. `routing_entropy_mean_residue` decreased vs previous epoch (sparsity still biting), **and**
2. `capacity_loss ≥ max(1e-4, 2 × capacity_loss at ep1)`, **or** `usage_max_soft_share ≥ 0.40` (warning band below timeout)

**Timeout-imminent band:** `usage_max_soft_share ∈ [0.40, 0.45)` → reduce \(\lambda\) ramp slope by half for remaining warmup.  
**Timeout trip:** any ban under 45% rule → hold \(\lambda\) (see §4).

**Secondary ratio (dashboard / MLflow):**

```text
sparse_vs_cap_ratio = (λ_sparse * L_sparse) / (λ_cap * capacity_loss + ε)
```

Rising ratio with flat `capacity_loss` = sparsity winning cleanly.  
Rising ratio **with** rising `capacity_loss` = adversarial collision (expected; healthy if max_share stays &lt; 0.40).  
Spike in max_share toward 0.45 = sparsity overpowering capacity → intervene.

Log to MLflow under experiment `tokyo-eyes-v66-fix1-expand` (new child of lineage root; tag `sparsity_bet=entropy_mean_residue`).

---

## 6. Pass / Fail (pre-registered)

Run: Fix-1 stack continue from sealed, Stage A-12 or feeler-expand corpus, 15–20 epochs, \(\lambda_{\mathrm{peak}}=0.0075\), \(T_{\mathrm{warmup}}=8\).

| Gate | Pass |
|------|------|
| Mean residue-H (final 3-ep mean) | ∈ [0.5, 0.9] |
| Mean residue-H (final epoch) | ∈ [0.5, 0.9] (save eligibility) |
| Batch `H(f̄)` | **monitor**: warn &lt; 1.00; abort ≤ 0.80 (diversity floor — **not** ≤ 1.21 save ceiling) |
| Max soft share | &lt; 0.45 every epoch after warmup (no sustained timeout loop) |
| Capacity | `capacity_loss` finite; min load ≥ `min_usage − ε` |
| Hub scaffolding | 4OBE knockout vs betweenness ρ ≥ 0.45 (vs sealed 0.596; allow mild regression) |
| Disc | `probe_r_proj_depth` ≥ 0.95; σ₂/σ₁ ≥ 0.80 (expand bars) |

**Fail:** mean residue-H still ≥ 1.2 after warmup+5 ep; **or** monopoly timeout thrash (≥3 bans in 5 ep); **or** 4OBE hub ρ &lt; 0.35.

**Partial:** H band met but hub ρ ∈ [0.35, 0.45) — record; do not promote as biology-ready.

---

## 7. Standing rules

1. Never overwrite sealed run dir / `v66_healthy_sealed.pt`
2. Do not resume from P2/P3 expand champions for this bet
3. Jacobian hub rankings remain forbidden under z-norm; grade hubs via forward knockout
4. Gini deferred
5. Chem-MVP / Path2 / HA remain PARKED

---

## 8. Files (implementation map)

| Area | Touch | Status |
|------|--------|--------|
| Loss | `science/dtie/v66/loss.py` — weighted mean residue-H | Done |
| Model output | `science/dtie/v66/gnn/model.py` — emit `routing_entropy_mean_residue` | Done |
| Coeffs | `science/training/config.py` — sparsity coeff + warmup | Done |
| Train loop / stage | `experiments/training/v66/stage_runner.py` — schedule λ(t); collision jsonl | Done |
| Make | `train-v66-fix1-sparsity-sealed-continue` | Done |
| Tests | mean-H, warmup, governor, collision detector | Done |
| GPU verify | Task 6 — hub knockout + \(t^\*\) | Pending |

---

## 9. Biological imperative (non-negotiable framing)

Uniform routing → flat static analysis. Partitioned MoE → distinct analytical lenses over residue sub-graphs. Mapping how KRAS G12D rewires inactive→active-like conductance requires that partitioning as a **prerequisite**, not a post-hoc visualization choice.
