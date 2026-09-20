# TokyoEye — Architecture SSOT

> **Standing rule:** Prose without a failing test always loses to a convenient default.
>
> A written requirement that CI or the training harness will not fail is a suggestion, not a lock.

**Audience:** structural biologists and ML engineers joining the project.  
**Role:** newcomer-readable **front door** for the intended TokyoEye training stack.  
**Gate id:** `tokyo_eye_equ_architecture_ssot`  
**Date:** 2026-09-18

---

## How this document relates to other authorities

| Document | Role |
|----------|------|
| **This Architecture SSOT** | Front door: intended stack, forbidden substitutes, scientific questions, discipline. Does **not** restate mutable numeric pins. |
| **[Freeze addendum](superpowers/specs/2026-09-16-tokyo-eye-v8-freeze-addendum.md)** | **Sole owner** of specific technical decisions: wrap threshold, learning rates, canonical MLflow experiment name, signed §2.x dispositions (including §2.6 epsilon-greedy), retired metrics. |
| **[ENFORCEMENT_MATRIX](ENFORCEMENT_MATRIX.md)** | Platform integrity (writes, ingest, curvature literals) plus the TokyoEye training assembly row (GATED when tests land). |
| **[CLAUDE.md](../CLAUDE.md)** | Short operator brief; must mirror this SSOT and point at failing tests. |

**Conflict rule:** If a number or signed clause appears here and in the freeze addendum and drifts, the **freeze addendum wins**. This SSOT only links.

---

## Scientific thesis

A protein graph that combines geometry and local chemistry can reveal underwrapped hydrogen bonds (dehydrons) and related vulnerability structure when residue states live in **hyperbolic** space rather than only Euclidean space—without binding-affinity labels and without classical wrap=19 as a biology number on this trunk.

**Dehydron (operational):** an amide–carbonyl hydrogen bond whose wrapping by non-polar carbons falls at or below the wrap count locked in the freeze addendum. This SSOT names the concept; the addendum owns the integer.

---

## Intended architecture (production stack)

```text
Atomic coords + elements ──► EquiformerV3 pool frontend
Residue graph R0–R5 ───────►        │
                                    ▼
                         geoopt Poincaré ball lift
                                    ▼
                         Hyperbolic graph attention ◄── R0–R5
                                    ▼
                         Hard mixture of experts (4)
                                    ▼
                         Prediction heads (biology on hyperbolic state)
```

| Stage | Required implementation |
|-------|-------------------------|
| Frontend | EquiformerV3 cut before energy pooling (`EquiformerPoolFrontend`). **Cold random init is the only evidence-backed default today:** Stage-0 LOOCV / CA-sized margin probing has not cleared any protein-pretrained candidate (GearNet-Edge was tested and failed). This is not “an approved checkpoint exists but is unloaded.” |
| Lift | `geoopt` Poincaré ball |
| Backbone | Sparse R0–R5 hyperbolic attention |
| Router | Hard MoE (four experts; evidence = evaluation-time argmax assignment) |
| Biology claim | Loss must depend on the post-attention / post-MoE hyperbolic state—not only a Euclidean skip |

## Forbidden on any run this SSOT calls governed

- SE(3)-lite / stub `live_backbone` as a substitute for Equiformer pool
- Affinity-champion (or other legacy alias) init as if it were this biology model
- Claiming hyperbolic biology when the trained path is Euclidean-skip-only

Cold init is required because **no candidate has passed the evidence bar**; when one does, the freeze addendum records it. Never fall back to SE(3)-lite.

### SE(3)-lite pilots (off-path, non-claim)

SE(3)-lite / stub is **not** the governed frontend. A pre-registered SE(3)-lite run is permitted **only** when all of the following hold:

1. Explicit `--allow-off-path-frontend` (harness refuses stub otherwise).
2. Card is pre-registered with `do_not_promote: true` and `not_claims` that name the frontend.
3. Purpose is a **cheap signal-existence / power check** before spending Equiformer-pool compute — not a biology seal.
4. Failure or success on SE(3)-lite is **never** read as evidence about the intended Equiformer-pool stack; a confirmatory pool run is required before any trunk biology claim.

Example: wrap=1 dehydron LOSO arm B on cold SE(3)-lite (`tokyo_eye_equ_wrap1_dehydron_loso_B_result.json`) is a disclaimed pilot; `PENDING_G_FIT` still gates underfit vs no-signal on that pilot, but does not answer pool-frontend biology.

---

## Assembly gate (fail closed)

Three conditions; all must be **ENFORCED** (automatic) when applicable:

1. **Frontend:** Equiformer pool only. SE(3)-lite / stub live backbone → fail (unless an explicit off-path diagnostic flag is set).
2. **Geometry integrity:** live-forward `pure_hyp_pass` (tracer) must pass. Grep-only checks are not sufficient. Skip under off-path allow sets `pure_hyp_checked=false` / `pure_hyp_ok=false` — never treat a skip as a clean seal.
3. **Claim-bearing biology** (when `--claim-bearing-biology`): refuse unless (a) the batch is *structurally* non-leaking — dehydron/SDRP targets do not reconstruct from `edge_type` via the same loader functions — and (b) `--log-biology-grad-sources N>0`. A `--biology-non-leaking-target` flag is advisory only and **cannot** waive a still-leaking batch (same class of gap as `forbid_se3_lite` alone). Incompatible with `--allow-off-path-frontend`.

**Environment:** constructing the pool frontend requires `ase`, `torch-scatter`, `torch-cluster`, `lmdb`, `e3nn` (and related imports) in the **same** environment as the gate. Missing dependencies must fail with an explicit “missing frontend dependencies” message—not look like an architecture veto.

**Code:** `science/tokyo_eye/v8/assembly_gate.py`  
**Harness:** `experiments/training/v8/run_v8_experiment.py` (governed path)  
**Tests:** `tests/v8/test_assembly_gate.py`

---

## Confirmed wiring risks

### Two separate defects on dehydron / SDRP numbers

| Defect | What it is | Status |
|--------|------------|--------|
| **A. Labeling calibration** | Stage-A-12 wrap threshold (wrap_max=1) | **Closed** — `data/gates/tokyo_eye_equ_wrap_threshold.json` |
| **B. Label leakage** | Dehydron target is R2-incidence while `edge_type` is a model input (`loader.dehydron_labels_from_edges`) | **Still open** |

Do not conflate: the wrap AMEND does not fix leakage. See `data/gates/tokyo_eye_equ_nonclaim_disposition.json`.

### Euclidean skip vs hyperbolic state

Dehydron BCE today uses `MechanismScoreHead(h_euc)`. SDRP fuses `z_hyp` with an `exp₀(h_euc)` lift. Before any biology **claim**, log loss gradients broken out by hyperbolic vs Euclidean-skip contribution (`biology_grad_by_source` in the harness).

---

## Scientific agenda (order)

1. Assembly ENFORCED (this gate) + non-claim disposition published.
2. Router reality on Equiformer pool only (sealed §2.6 if pursued).
3. Hyperbolic biology: non-leaking protocol + hyperbolic-state head + gradient split.
4. Only then: MoE on vs off under identical wiring.

---

## Discipline checklist

1. Intended architecture?
2. Answers router reality or hyperbolic biology?
3. Criteria fixed before results?
4. Would a check **fail** on the convenient substitute?
5. Newcomer-readable card?
6. Numbers only in the freeze addendum?

If any answer is no → not trunk science.
