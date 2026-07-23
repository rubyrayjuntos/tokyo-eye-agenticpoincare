# Tokyo Eye v7 — Hyp MP lineage

> **DEAD / ARCHAEOLOGY (2026-07-23).** v7 is a different architecture and was **never** the production trunk.  
> **Do not open this tree for new work.** Active trunk: [`docs/specs/tokyo-eye-v8/README.md`](../tokyo-eye-v8/README.md).  
> Biology TODOs ported to [`../tokyo-eye-v8/biology-roadmap.md`](../tokyo-eye-v8/biology-roadmap.md).

**Status:** CLOSED  
**Date:** 2026-07-21 (opened) · **closed:** 2026-07-23  
**Module (archaeology):** [`science/tokyo_eye/TokyoEye.py`](../../../science/tokyo_eye/TokyoEye.py)  
**Checkpoints:** `checkpoints/v7/`  
**Training:** `experiments/training/v7/`  
**Gate:** [`data/gates/tokyo_eye_v7_lineage_opened.json`](../../../data/gates/tokyo_eye_v7_lineage_opened.json)  
**Design:** [`design.md`](design.md)

## One-line (historical)

Tokyo Eye **v7** made **hyperbolic message passing** the SSOT communication geometry for that experiment. Superseded by **v8** (Equiformer frontend + hyp spine).

## Cutover commands

```bash
make seal-v7-cutover-scaffold   # init .pt for contract (not biology)
make mlflow-register-v7-lineage
make seal-v7-bprime-warmstart   # B′ Fix-1 → TokyoEye (deny Euc trunk)
make train-v7-bprime-health     # Stage-A health micro-run
make train-v7-bprime-health-continue
make train-v7-bprime-disc-continue  # Option B: disc bump + sparsity-style save
make train-v7-bprime-core-floor-continue  # e1 core radial floor
make seal-v7-bprime-healthy              # health bank → v7_healthy_sealed.pt
make train-v7-bprime-uncertainty-heads   # heads-only ale/epi recovery
make grade-v7-bprime-uncertainty-heads
make train-v7-bprime-uncertainty-heads-rematch  # pre-auth rematch-1 if rematch-0 FAIL+disc held
make train-v7 RUN_ID=...
make promote-production-v7 CHECKPOINT=checkpoints/v7/runs/.../v7_best.pt
make verify-v7-production
make grade-v7-forward-smoke
make grade-v7-hyp-mp-telemetry
make grade-v7-hyp-biology-mp-smoke
make train-v7-hyp-biology-mp
make train-v7-cold-hyp-mp-funnel   # cold Hyp MP + Cα + funnel (no Fix-1 warmstart)
make train-v7-cold-hyp-mp-funnel-angfill  # continue: rim/geom + MoE anti-monop + ER>1.5 (MLflow)
make train-v7-phase-a-nucleotide-basin
make grade-v7-phase-a-nucleotide-basin
```

**Healthy restore SSOT (B′ disc Pass):** `checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt` (`experiments.training.v7.healthy_bprime.HEALTHY_V7_CKPT`). Prefer sealed over bare `phase_12.pt` / late epochs.

**Uncertainty:** PARKED after rematch-0/1 FAIL (wrong-order inherited P4 coeffs; no formula benchmark). Unpark checklist: [`bprime-uncertainty-unpark.md`](bprime-uncertainty-unpark.md) · stamp [`data/gates/tokyo_eye_v7_bprime_uncertainty_parked.json`](../../../data/gates/tokyo_eye_v7_bprime_uncertainty_parked.json). Do not rematch-2 under the old heads prereg.

**Investigation metrics (starting defs, frozen Θ):** Allele sensitivity + epistatic coupling on `x_hyp` — [`investigation-allele-epistasis-metrics.md`](investigation-allele-epistasis-metrics.md). Not NIG `ale`/`epi`.

**Biology roadmap:** [`biology-roadmap.md`](biology-roadmap.md) · [`data/gates/tokyo_eye_v7_biology_roadmap.json`](../../../data/gates/tokyo_eye_v7_biology_roadmap.json) — B0 prep → B1 teleconnections → B2 epistasis (B3′ directional; B4/B5 deferred).

B′ prereg: [`bprime-health-warmstart-prereg.md`](bprime-health-warmstart-prereg.md)  
Disc-health continue: [`bprime-disc-health-continue-prereg.md`](bprime-disc-health-continue-prereg.md)  
Core radial floor: [`bprime-core-radial-floor-prereg.md`](bprime-core-radial-floor-prereg.md)  
Uncertainty heads (closed attempt): [`bprime-uncertainty-heads-prereg.md`](bprime-uncertainty-heads-prereg.md)  
Uncertainty unpark: [`bprime-uncertainty-unpark.md`](bprime-uncertainty-unpark.md)  
Allele / epistasis metrics: [`investigation-allele-epistasis-metrics.md`](investigation-allele-epistasis-metrics.md)  
Biology roadmap: [`biology-roadmap.md`](biology-roadmap.md)  
B1 draft: [`b1-teleconnections-prereg-draft.md`](b1-teleconnections-prereg-draft.md) (superseded)  
B1 frozen: [`b1-teleconnections-prereg.md`](b1-teleconnections-prereg.md) · [`data/gates/tokyo_eye_v7_b1_teleconnections_prereg.json`](../../../data/gates/tokyo_eye_v7_b1_teleconnections_prereg.json)

## Boundaries

| Path | Role |
|------|------|
| `science/tokyo_eye/` | Tokyo Eye GNN product (versioned by lineage registry + checkpoints, **stable module name**) |
| `science/dtie/` | Structural biology (historical name). **Do not version the GNN here.** Untouched in v7 open. Future rename → `science/structural-biology/` is a **separate** effort. |
| `checkpoints/v7/` | All v7 runs/diagnostics |
| `experiments/training/v7/` | Train/grade entrypoints for v7 |

## Frozen / parked

- **v6.x** (including Fix-1 sparsity champion under `checkpoints/v66/`): frozen **compare-only** by policy.
- **Chem-MVP**: parked ([`chem-mvp-reengage`](../chem-mvp-reengage/README.md)).
- **July-19 “MP stays Euclidean”** lock: **superseded for v7** ([`EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md)).

## MLflow

| Field | Value |
|-------|--------|
| Experiment | `tokyo-eyes-v7` |
| Registered model | `TokyoEye-v7` |
| Lineage stamp | [`data/gates/tokyo_eye_v7_mlflow_lineage_root.json`](../../../data/gates/tokyo_eye_v7_mlflow_lineage_root.json) |
| Embedding space name | `tokyoeye_v7_hyp128` |
| Register | `make mlflow-register-v7-lineage` |

v6.x experiments (`tokyo-eyes-v66`, `tokyo-eyes-v66-fix1-expand`) stay frozen compare-only — do not attach v7 children there.

## Naming

- **SE(3)** = optional Euclidean local backbone tech — not “S4”.
- Do not label new v7 runs with `*_s4_*` as a technology claim.
- Lineage promotion keeps module path `science.tokyo_eye.TokyoEye`.

## Curvature

Learned `nn.Parameter` on `TokyoEye`; cone depth from `dist0`. No hardcoded `c` on v7 paths. Downstream jobs consume `embedding_space.curvature`.
