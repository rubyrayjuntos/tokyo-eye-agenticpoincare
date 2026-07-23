# TokyoEye-v8 Sprint 10 — Affinity Regression Implementation Plan

> **For agentic workers:** Spec SSOT:
> [`docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-affinity-regression-design.md`](../specs/2026-07-22-tokyo-eye-v8-sprint10-affinity-regression-design.md)

**Status:** APPROVED design locked 2026-07-22 — execute in order below.  
**Goal:** Protein-only \(-\log K\) from pocket-gated tangent pool; Refined@30% train/val; Core external test; `head_only` from Sprint 9 ckpt.

**Frozen:** R0–R5 / DSSP / cone / cache / MoE structure. Ligand encoder = Sprint **10.1**.

---

## Operational safety contracts (non-negotiable)

### 1. Strictly isolated gradient boundaries in `head_only`

All backbone / projector / hyp-attn / MoE parameter tensors must be decoupled from the optimizer graph before any affinity step:

```python
# Force absolute isolation of the geometry banks (v8 names)
for param in system.frontend.parameters():       # backbone
    param.requires_grad = False
for param in system.spine.projector.parameters():
    param.requires_grad = False
for param in system.spine.attn_layers.parameters():  # hyp_layers
    param.requires_grad = False
for param in system.spine.moe.parameters():
    param.requires_grad = False
```

Helper: `freeze_geometry_banks()` in `experiments/training/v8/run_affinity_s10.py`.  
Also freeze euc_skip / SDRP / evidential / mechanism heads. Optimizer may contain **only** `PocketGatedAffinityHead` weights. Forward spine under `torch.no_grad()` + detached `z_hyp` / `mechanism_score`.

### 2. Softmax over variable graph node count

In `science/tokyo_eye/v8/affinity_head.py`:

```python
# scores: [N, 1]
w = F.softmax(scores, dim=0)
t_pool = torch.sum(w * log_map_zero(z_hyp), dim=0)  # [d]
```

Never softmax over feature dim. Unit tests cover N ∈ {7, 31, 64}.

### 3. Absolute Core leak purge

`build_pdbbind_splits.py` must hard-fail if any train/val PDB ID (or exact Core sequence / mmseqs hit / k-mer wall hit) intersects CASF-2016 Core. No partial leakage, no exception paths that re-admit purged clusters.

```python
assert_no_core_leak(train_ids, val_ids, core_ids)  # raises AssertionError
```

---

## File map

| File | Role |
|------|------|
| `science/tokyo_eye/v8/seq_cluster.py` | mmseqs / exact / k-mer Core wall |
| `science/tokyo_eye/v8/pdbbind_loader.py` | LP-PDBBind CSV → refined/core frames |
| `science/tokyo_eye/v8/affinity_head.py` | PocketGate + log₀ pool + FFN |
| `experiments/training/v8/build_pdbbind_splits.py` | Emit `manifests/v8_pdbbind_refined_cluster30_v1.json` |
| `experiments/training/v8/run_affinity_s10.py` | `head_only` train + Core Pearson/Spearman/RMSE |
| `tests/v8/test_affinity_sprint10.py` | Pool + leak-wall unit tests |

---

## Execution sequence

### Task 1 — Compile splits artifact

```bash
PYTHONPATH=. python experiments/training/v8/build_pdbbind_splits.py \
  --csv data/pdbbind/LP_PDBBind_refined_core.csv \
  --out manifests/v8_pdbbind_refined_cluster30_v1.json
```

Assert: `core_test` ∩ (`train` ∪ `val`) = ∅. Freeze the JSON.

### Task 2 — Pocket readout head

`PocketGatedAffinityHead`: features \(h_{\mathrm{inv}}, r_i, \sigma(\mathrm{mech}_i), \mathrm{dehyd}_i\) → softmax gate → tangent midpoint → FFN → scalar \(-\log K\).

### Task 3 — External Core baseline (`head_only`)

```bash
PYTHONPATH=. python experiments/training/v8/run_affinity_s10.py \
  --splits manifests/v8_pdbbind_refined_cluster30_v1.json \
  --init-ckpt checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt \
  --run-name tokyo_eye_v8_affinity_s10_head_only
```

Smoke:

```bash
PYTHONPATH=. python experiments/training/v8/run_affinity_s10.py --smoke
```

Report Core **Pearson \(R\)**, **Spearman \(\rho\)**, **RMSE**. Gate: provisional pass if Core \(R \ge 0.40\).

### Task 4 — Unit tests

```bash
pytest tests/v8/test_affinity_sprint10.py -q
```

---

## Acceptance checklist

- [x] Manifest written; `NO_LEAK` assert green (`n_train=2727`, `n_val=698`, `n_core_test=285`, `n_purged=1625`)
- [x] `affinity_head.py` softmax `dim=0`; pool finite
- [x] `head_only` freezes geometry banks; only AffinityFFN+gate train (`freeze_geometry_banks` → 265 tensors)
- [x] Subset Core metrics logged (`tokyo_eye_v8_affinity_s10_head_only_subset`); full Refined/Core PDB cache still required for gate claim
- [x] `pytest tests/v8/test_affinity_sprint10.py` green (9 passed)

### Subset baseline (2026-07-22, 64/16/32 prefetch, 8 epochs, Sprint 9 init)

| Split | n | Pearson R | Spearman ρ | RMSE |
|-------|---|-----------|------------|------|
| Core (external) | 27 | **0.054** | 0.068 | 3.65 |
| Val (best epoch) | 16 | −0.208 | −0.209 | 3.40 |

`CORE_PEARSON ≥ 0.40` **not met**. Spec rematch trigger: Core \(R < 0.30\) → **`finetune_hyp`**.

### Rematch protocol — `finetune_hyp` (activated)

```bash
PYTHONPATH=. python experiments/training/v8/run_affinity_s10.py \
  --mode finetune_hyp \
  --device cuda \
  --epochs 12 \
  --lr 1e-3 \
  --lr-hyperbolic 3e-4 \
  --aux-coeff 0.10 \
  --ckpt-baseline 0.054 \
  --init-ckpt checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt \
  --run-name tokyo_eye_v8_affinity_s10_finetune_hyp
```

| Knob | Value |
|------|-------|
| Frontend / Equiformer bank | **frozen** |
| projector + `attn_layers` + MoE + `mechanism_head` | **train** @ `lr_hyperbolic=3e-4` |
| `PocketGatedAffinityHead` | **train** @ `lr=1e-3` |
| λ_aux (dehydron BCE) | **0.10** |
| Checkpoint | `v8_affinity_best.pt` on any val Pearson ↑; `improve_ckpts/v8_affinity_improve_e*_r*.pt` when val Pearson **> 0.054** |

### Rematch result (2026-07-22, full split, 12 epochs, ~50 min)

| Split | n | Pearson R | Spearman ρ | RMSE |
|-------|---|-----------|------------|------|
| Core (external) | 257 | **0.007** | −0.019 | 2.17 |
| Val (best) | — | 0.045 | — | — |

`CORE_PEARSON ≥ 0.40` **fail**; Core \(R < 0.30\) after rematch → **capacity / representation diagnosis** (new pre-reg) before Sprint 10.1 ligand encoder. No improve-ckpts (best val 0.045 never cleared 0.054 baseline). Artifacts: `checkpoints/v8/runs/tokyo_eye_v8_affinity_s10_finetune_hyp/`.

### Pivot — Sprint 10.1 (activated)

Capacity diagnosis confirmed: protein-only graphs cannot resolve ligand-conditioned \(-\log K\).  
**Design draft:** [`docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface-design.md`](../specs/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface-design.md)

| Lock | Value |
|------|-------|
| Production ligand source | Option **C** (hybrid mol2/SDF → HETATM fallback) |
| Sprint **10.1.0** bootstrap | Option **A** filtered HETATM from `pdb_cache/` |
| R6 | \(d(C_\beta/C_\alpha,\ \mathrm{lig})\le 4.5\) Å, on-the-fly, **off** protein graph cache |
| Readout | Asymmetric cross-attn \(Q=z_{\mathrm{hyp}}\), \(K,V=\) ligand one-hots along R6 |
