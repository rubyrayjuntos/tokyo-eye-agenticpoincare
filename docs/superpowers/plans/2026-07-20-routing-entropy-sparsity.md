# Routing Entropy Sparsity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire per-residue routing entropy minimization into Fix-1 training, resume from `v66_healthy_sealed.pt`, with warmup and collision telemetry so sparsity does not thrash the 45% monopoly timeout.

**Architecture:** Add `routing_entropy_mean_residue = mean_i H(p_i)` from soft expert weights; include `λ_sparse(t) * L_sparse` in `total` loss. Keep existing `routing_entropy = H(f̄)` as monitor/save ceiling only. Capacity loss + 45% expert timeout remain unchanged.

**Tech Stack:** PyTorch, v66 `GOSPConeMapperV66`, feeler Phase 12 Fix-1 stack, MLflow `tokyo-eyes-v66-fix1-expand`

**Spec:** [`docs/specs/routing-entropy-sparsity/design.md`](../../specs/routing-entropy-sparsity/design.md)

## Global Constraints

- Init checkpoint: `checkpoints/v66/runs/fix1_s4_stack_initseed_controlled_3d_seed2_v1/v66_healthy_sealed.pt` only
- Do not resume from P2/P3 expand champions for this bet
- Default `λ_peak = 0.5 × balance_coeff` (feeler: `0.5 × 0.015 = 0.0075`); `T_warmup = 8`
- Target mean residue-H ∈ [0.5, 0.9]; batch H(f̄) ≤ 1.21; max soft share &lt; 0.45
- Jacobian hub rankings forbidden under z-norm; hub grade = forward knockout
- Gini deferred

---

## File map

| File | Responsibility |
|------|----------------|
| `science/dtie/v66/gnn/model.py` | Emit `routing_entropy_mean_residue` from per-row `scores` |
| `science/dtie/v66/loss.py` | Add `λ_sparse * L_sparse` into `total`; return telemetry fields |
| `science/training/config.py` | Coeffs + warmup epochs on `LossCoeffs` / TrainingConfig |
| `experiments/training/v66/stage_runner.py` | Schedule λ(t); log collision fields; optional λ hold on timeout |
| `experiments/training/v66/launch_training.py` | CLI flags for sparsity coeff / warmup |
| `Makefile` | `train-v66-fix1-sparsity-sealed-continue` |
| `tests/test_routing_entropy_sparsity.py` | Unit tests for mean-H, warmup, collision detector |
| `science/training/routing_sparsity.py` (new, small) | Pure helpers: mean residue entropy, warmup λ, collision epoch |

---

### Task 1: Pure helpers + failing tests

**Files:**
- Create: `science/training/routing_sparsity.py`
- Create: `tests/test_routing_entropy_sparsity.py`

- [ ] **Step 1: Write failing tests**

```python
import math
import torch
from science.training.routing_sparsity import (
    mean_residue_routing_entropy,
    sparse_coeff_at_epoch,
    detect_sparse_capacity_collision,
)

def test_mean_residue_entropy_uniform_four():
    scores = torch.full((10, 4), 0.25)
    h = mean_residue_routing_entropy(scores)
    assert abs(h.item() - math.log(4)) < 1e-5

def test_mean_residue_entropy_one_hot():
    scores = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    h = mean_residue_routing_entropy(scores)
    assert h.item() < 1e-5

def test_warmup_linear():
    assert sparse_coeff_at_epoch(1, peak=0.01, warmup=8) == 0.01 * (1 / 8)
    assert sparse_coeff_at_epoch(8, peak=0.01, warmup=8) == 0.01
    assert sparse_coeff_at_epoch(20, peak=0.01, warmup=8) == 0.01

def test_collision_detector_triggers_on_capacity_rise():
    series = [
        {"routing_entropy_mean_residue": 1.3, "capacity_loss": 1e-6, "usage_max_soft_share": 0.28},
        {"routing_entropy_mean_residue": 1.1, "capacity_loss": 5e-4, "usage_max_soft_share": 0.30},
    ]
    assert detect_sparse_capacity_collision(series) == 2  # 1-indexed epoch
```

- [ ] **Step 2: Run tests — expect fail**

```bash
PYTHONPATH=. python -m pytest tests/test_routing_entropy_sparsity.py -q
```

- [ ] **Step 3: Implement helpers**

```python
# science/training/routing_sparsity.py
def mean_residue_routing_entropy(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """scores: [N, E] soft weights; returns scalar mean_i H(p_i)."""
    p = scores.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    h = -(p * p.log()).sum(dim=-1)
    return h.mean()

def sparse_coeff_at_epoch(epoch_1indexed: int, *, peak: float, warmup: int) -> float:
    if warmup <= 0:
        return float(peak)
    t = max(1, int(epoch_1indexed))
    return float(peak) * min(1.0, t / float(warmup))

def detect_sparse_capacity_collision(epoch_rows: list[dict], *, cap_floor: float = 1e-4) -> int | None:
    """First 1-indexed epoch where mean-H fell and capacity rose or max_share≥0.40."""
    if not epoch_rows:
        return None
    cap0 = float(epoch_rows[0].get("capacity_loss") or 0.0)
    for i in range(1, len(epoch_rows)):
        prev, cur = epoch_rows[i - 1], epoch_rows[i]
        h_prev = float(prev["routing_entropy_mean_residue"])
        h_cur = float(cur["routing_entropy_mean_residue"])
        cap = float(cur.get("capacity_loss") or 0.0)
        mx = float(cur.get("usage_max_soft_share") or 0.0)
        if h_cur < h_prev and (cap >= max(cap_floor, 2.0 * cap0) or mx >= 0.40):
            return i + 1
    return None
```

- [ ] **Step 4: Run tests — expect pass**

```bash
PYTHONPATH=. python -m pytest tests/test_routing_entropy_sparsity.py -q
```

- [ ] **Step 5: Commit**

```bash
git add science/training/routing_sparsity.py tests/test_routing_entropy_sparsity.py
git commit -m "add routing entropy sparsity helpers and unit tests"
```

---

### Task 2: Model emits mean residue entropy

**Files:**
- Modify: `science/dtie/v66/gnn/model.py`
- Modify: `tests/test_routing_entropy_sparsity.py` (optional forward smoke if fixture exists)

- [ ] **Step 1: After soft `scores` are computed, add**

```python
from science.training.routing_sparsity import mean_residue_routing_entropy
routing_entropy_mean_residue = mean_residue_routing_entropy(scores)
```

Keep existing:

```python
routing_entropy = -(scores.mean(dim=0) * torch.log(scores.mean(dim=0) + 1e-8)).sum()
```

- [ ] **Step 2: Include `routing_entropy_mean_residue` in forward output dict** (same place as `routing_entropy`)

- [ ] **Step 3: Smoke**

```bash
PYTHONPATH=. python -c "from science.training.routing_sparsity import mean_residue_routing_entropy; import torch; print(mean_residue_routing_entropy(torch.rand(5,4).softmax(-1)))"
```

- [ ] **Step 4: Commit**

```bash
git add science/dtie/v66/gnn/model.py
git commit -m "emit per-residue mean routing entropy from MoE gate scores"
```

---

### Task 3: Loss wiring + config coeffs

**Files:**
- Modify: `science/training/config.py` (`LossCoeffs`: `routing_entropy_sparsity_coeff`, `routing_entropy_sparsity_warmup_epochs`)
- Modify: `science/dtie/v66/loss.py`
- Modify: `experiments/training/v66/launch_training.py` (CLI)
- Test: extend `tests/test_routing_entropy_sparsity.py` with a tiny loss mock if feasible

- [ ] **Step 1: Add to `LossCoeffs`**

```python
routing_entropy_sparsity_coeff: float = 0.0
routing_entropy_sparsity_warmup_epochs: int = 8
```

Default peak for Fix-1 make target will set coeff to `0.0075` (0.5 × feeler `balance_coeff=0.015`).

- [ ] **Step 2: In `combined_loss_v6` / v66 combined loss**

```python
L_sparse = output.get("routing_entropy_mean_residue")
# if missing, compute from expert_weights
sparse_coeff = float(routing_entropy_sparsity_coeff)  # already scheduled by caller OR pass scheduled value
sparse_term = sparse_coeff * L_sparse if L_sparse is not None and sparse_coeff > 0 else 0
total = total + sparse_term
```

Prefer: **stage_runner passes the scheduled λ** into coeffs each epoch (clearer than hiding schedule inside loss).

- [ ] **Step 3: Return keys**

```python
"routing_entropy_mean_residue": L_sparse,
"routing_entropy_sparsity_loss": sparse_term,
"routing_entropy_sparsity_coeff": sparse_coeff,
```

- [ ] **Step 4: CLI**

```text
--routing-entropy-sparsity-coeff 0.0075
--routing-entropy-sparsity-warmup-epochs 8
```

- [ ] **Step 5: Commit**

```bash
git add science/training/config.py science/dtie/v66/loss.py experiments/training/v66/launch_training.py tests/test_routing_entropy_sparsity.py
git commit -m "wire mean-residue routing entropy sparsity into v66 loss"
```

---

### Task 4: Stage runner schedule + collision logging

**Files:**
- Modify: `experiments/training/v66/stage_runner.py`
- Optional: `science/training/routing_sparsity.py` (already has detector)

- [ ] **Step 1: Each epoch before train**

```python
peak = self.config.routing_entropy_sparsity_coeff  # peak
warmup = self.config.routing_entropy_sparsity_warmup_epochs
lam = sparse_coeff_at_epoch(epoch_in_phase, peak=peak, warmup=warmup)
# if timeout hold active: lam = held_lam
phase_cfg.coeffs.routing_entropy_sparsity_coeff = lam
```

- [ ] **Step 2: After epoch metrics, append collision row**

```python
row = {
  "routing_entropy_mean_residue": health/losses field,
  "capacity_loss": ...,
  "usage_max_soft_share": ...,
  "min_routing_fraction": ...,
  "routing_entropy": ...,  # H(f̄)
  "lambda_sparse": lam,
  "sparse_vs_cap_ratio": (lam * L_sparse) / (balance_coeff * capacity + 1e-12),
  "expert_timeout_bans": list(...),
}
# write jsonl: routing_sparsity_per_epoch.jsonl
```

- [ ] **Step 3: Timeout hold**

If new ban and `usage_max_soft_share >= 0.45`: set `_sparse_hold_lam = previous_lam`, `_sparse_hold_epochs = 2`.

If `0.40 <= max_share < 0.45`: half remaining ramp slope (document in log).

- [ ] **Step 4: Commit**

```bash
git add experiments/training/v66/stage_runner.py
git commit -m "schedule sparsity λ warmup and log capacity collision telemetry"
```

---

### Task 5: Make target + sealed continue recipe

**Files:**
- Modify: `Makefile`
- Modify: `docs/specs/fix1-s4-restore/corpus-expansion.md` (one-line pointer) or design status → implementing

- [ ] **Step 1: Add target**

```make
train-v66-fix1-sparsity-sealed-continue: ## Fix-1 sealed continue + mean-residue entropy sparsity
	# resume HEALTHY_FIX1_CKPT / sealed
	# --routing-entropy-sparsity-coeff 0.0075
	# --routing-entropy-sparsity-warmup-epochs 8
	# corpus: Stage A-12 small (prefer for hub grade) OR feeler_expand
	# mlflow experiment tokyo-eyes-v66-fix1-expand
	# Fix-1 stack flags unchanged
```

- [ ] **Step 2: Document RUN_ID default** e.g. `fix1_s4_sparsity_sealed_continue_v1`

- [ ] **Step 3: Commit**

```bash
git add Makefile docs/specs/routing-entropy-sparsity/design.md
git commit -m "add Fix-1 sealed sparsity continue Make recipe"
```

---

### Task 6: Verification run (agent or human GPU)

- [ ] **Step 1: Launch**

```bash
make train-v66-fix1-sparsity-sealed-continue EPOCHS=15
```

- [ ] **Step 2: After run, grade**

```bash
PYTHONPATH=. python -c "from science.training.routing_sparsity import detect_sparse_capacity_collision; import json; ..."
# Inspect routing_sparsity_per_epoch.jsonl for t*
make grade via hub knockout on 4OBE if Pass band on H
```

- [ ] **Step 3: Record Pass/Fail in** `checkpoints/v66/runs/<RUN_ID>/sparsity_gate.json`

---

## Collision telemetry (SSOT answer)

Use this set every epoch to find when \(\lambda_{\mathrm{sparse}}\) collides with capacity / timeout:

1. **`routing_entropy_mean_residue`** — sparsity pressure (should fall)
2. **`capacity_loss`** (raw) — starvation pressure (rises when sparsity starves an expert)
3. **`sparse_vs_cap_ratio`** = \((\lambda_{\mathrm{sparse}} L_{\mathrm{sparse}}) / (\lambda_{\mathrm{cap}} \cdot \mathrm{capacity\_loss} + \varepsilon)\)
4. **`usage_max_soft_share`** — monopoly approach (**warn ≥ 0.40**, **timeout ≥ 0.45**)
5. **`expert_timeout_bans`** — hard failsafe events

**Collision epoch \(t^\*\)** = first epoch where mean residue-H decreased **and** (`capacity_loss ≥ max(1e-4, 2× ep1)` **or** `max_share ≥ 0.40`).

That is the monitor for the warmup schedule — not loss alone, and not batch `H(f̄)` alone.

---

## Out of scope

- Gini penalty
- Resuming from P2/P3 expand champions
- Directionality / Path2 loss
- Chem-MVP / HA edges
- Changing save ceiling formula
