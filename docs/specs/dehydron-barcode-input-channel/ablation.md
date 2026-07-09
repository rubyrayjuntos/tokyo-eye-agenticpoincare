# Dehydron Barcode Ablation — Runbook

**Related:** [`design.md`](design.md)  
**Purpose:** Matched-epoch representation ablation (baseline vs scalars vs full binned) off a fixed slim MoE lineage.

---

## Prerequisites

1. **Gate stamp** — training halts without it:
   ```bash
   make gate-p-feature-01
   ```
   Expect `data/gates/p_feature_01_passed.json`.

2. **Barcode sidecars** — one `.pt` per corpus structure under `checkpoints/v65/dehydron_barcode_v1/`:
   ```bash
   make precompute-dehydron-barcodes
   ```
   Scalars-only sidecars suffice for the **scalars** arm. The **full** arm needs binned tensors in each sidecar:
   ```bash
   make precompute-dehydron-barcodes BINNED=1
   ```
   Re-running with `BINNED=1` overwrites sidecars in place (scalars + `binned` field).

3. **Resume checkpoint** — default slim MoE P3e best (override with `RESUME=`):
   ```text
   checkpoints/v65/runs/cold_start_v8_p3e/v65_best.pt
   ```
   Alternate: `checkpoints/v65/runs/cold_start_v8_p2/v65_best.pt`.

4. **Environment** — `GNN_INPUT_MODE=topology_three_vector` (set by Makefile targets). Slim MoE structural SSOT recipe unchanged; do not alter cone/τ loss coeffs or unfreeze structural disc in this experiment.

---

## Ablation arms

| Arm | Node inputs | Makefile target | Default `RUN_ID` |
|-----|-------------|-----------------|------------------|
| **Baseline** | `[ρ, τ, ss]` only | `train-v65-dbh-baseline` | `dbh_ablation_baseline` |
| **Scalars** | + dehydron scalars + missing mask | `train-v65-dbh-scalars` | `dbh_ablation_scalars` |
| **Full** | + scalars + binned vector | `train-v65-dbh-full` | `dbh_ablation_full` |

All arms: **phase 3** continue, **15 epochs** default (use `EPOCHS=20` for upper bound), **12 proteins**, same `--resume` lineage.

Shared knobs: `RESUME`, `EPOCHS` (default 15), `RUN_ID`, `DEVICE`, `MAX_PROTEINS` (default 12).

**Warm-start note:** Scalars and full arms widen `node_dim`. On `--resume`, the launcher automatically resizes `node_emb`: overlapping input columns copied from the checkpoint, new barcode columns zero-filled. Check launch logs for the resize message.

---

## Exact commands

```bash
# Once per environment
make gate-p-feature-01
make precompute-dehydron-barcodes              # scalars arm
make precompute-dehydron-barcodes BINNED=1     # full arm (includes scalars)

# Matched continues (run all three for comparable arms)
make train-v65-dbh-baseline
make train-v65-dbh-scalars
make train-v65-dbh-full

# Optional overrides
make train-v65-dbh-scalars EPOCHS=20 RESUME=checkpoints/v65/runs/cold_start_v8_p2/v65_best.pt RUN_ID=dbh_scalars_e20
```

Outputs land under `checkpoints/v65/runs/<RUN_ID>/` with MLflow experiment `tokyo-eyes-v65`.

---

## Evaluation checklist

After each arm completes, compare against baseline (`cold_start_v8_p3e` or the baseline ablation run).

### 1. 4OBE investigation motifs

Spot-check **4OBE:A** for retained or improved high-investigation signal at literature motifs:

- Switch I / Switch II
- α3 loop (residues ~105–107)
- C-terminal pivot region

Use the interactive viewer or residue-level investigation export from the run checkpoint.

### 2. Corpus-12 rim enrichment

Run the corpus-12 investigation audit workflow (same approach as the p3e reference run). Write results beside the run:

```text
checkpoints/v65/runs/<RUN_ID>/investigation_audit_corpus12.json
```

Reference artifact from the fixed lineage:

```text
checkpoints/v65/runs/cold_start_v8_p3e/investigation_audit_corpus12.json
```

**Pass:** high-investigation residues remain rim-enriched (`depth_hi > depth_lo`) on **≥11/12** Stage A structures.

### 3. Cone / τ probes

`make audit-dehydron-topology` loads 3-wide topology graphs **without** barcode sidecars. It is valid only for the **baseline** arm (node_dim 3). Scalars/full checkpoints (node_dim 15/55) will fail on forward if run through the current audit script.

**Baseline arm** — run the offline audit:

```bash
make audit-dehydron-topology CHECKPOINT=checkpoints/v65/runs/dbh_ablation_baseline/v65_best.pt
```

**Scalars / full arms** — evaluate cone/τ from training artifacts already written under the run directory (or the same metrics in MLflow experiment `tokyo-eyes-v65`):

| Source | Path | Cone / τ fields |
|--------|------|-----------------|
| Epoch history | `checkpoints/v65/runs/<RUN_ID>/metrics.json` | Last phase-3 epoch: `health.probe_r_depth_tau`, `health.cone_range_mean`, `health.probe_r_depth_rho` |
| End-of-run focus | `checkpoints/v65/runs/<RUN_ID>/focus_summary.json` (if present) | `items[]` entries for `probe_r_depth_tau` and `cone_range_mean` |
| MLflow | Run UI → Metrics | Same keys logged each epoch by `StageRunner` |

Quick read of the final phase-3 epoch:

```bash
python3 - <<'PY'
import json, sys
run = sys.argv[1]
rows = [r for r in json.load(open(f"checkpoints/v65/runs/{run}/metrics.json")) if r.get("phase") == 3]
h = rows[-1]["health"]
print("probe_r_depth_tau", h.get("probe_r_depth_tau"))
print("cone_range_mean", h.get("cone_range_mean"))
PY dbh_ablation_scalars
```

**Pass:** `probe_r_depth_tau` ≥ **0.15** (P_DEHYDRON_CONE_01 floor) and `cone_range_mean` stable; neither regresses vs baseline ablation run or `cold_start_v8_p3e`.

> **Follow-up:** Wire `--use-dehydron-barcode` / `--dehydron-barcode-dir` into `experiments/training/v6/dehydron_cone_alignment_audit.py` so `make audit-dehydron-topology` works on barcode checkpoints.

### 4. MoE routing health

Inspect per-structure routing audits and MLflow expert-utilization metrics. **Pass:** no new starvation, eligibility collapse, or timeout storms attributable to widened node inputs.

---

## Promotion decision

| Outcome | Action |
|---------|--------|
| Scalars arm meets all four checks | Promote scalars to default training feature; consider ingest (post-P1) |
| Scalars fail, full passes | Revisit binned payload / projector only |
| Both fail | Keep `use_dehydron_barcode` off; retain cache pipeline for viewer-only use |

See [`design.md`](design.md) §8.2 for full success/fail criteria.
