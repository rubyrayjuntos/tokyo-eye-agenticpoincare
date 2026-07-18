# Hierarchical Containment Edges — Ablation / Gates

**Date:** 2026-07-17
**Purpose:** Pre-registered physics + diameter-stratified flow-influence acceptance for containment v2.
**Design:** [`design.md`](design.md) · Hard lock: [`depth-collision.md`](depth-collision.md)

---

## Matched arms (to lock at launch)

| Arm | Intent |
|-----|--------|
| **Baseline** | Same parent as Chem-MVP feeler stack (role ± chem as in the Stage A-12 flow baseline), **no** containment edges |
| **Containment** | Same + `contain_down` / `contain_up` (+2 radial MLPs), parent nodes mean-pooled (Option A) |

Identity checklist before cold run: isolated-seed prototype identity (`max |Δ|=0` except new radial MLP slots); `node_dim` / lineage / role / chem flags matched.

---

## Gate order

1. Isolated-seed check (§5.3 design) — plan Task 4, `max |Δ|=0`
2. Physics non-regression (rim / cone·τ / MoE)
3. Oversmoothing-at-root (T1c-style on parent nodes) — plan Task 8, design §5.2
4. **Primary:** diameter-stratified asymmetry via `jacobian_flow_influence.py` — plan Task 9

Score is sanity only — not the verdict.

---

## Primary acceptance (locked before results)

Baseline numbers from `checkpoints/v66/diagnostics/learned_flow_influence/stage_a12.json`
and diameters from `asymmetry_vs_diameter.json`. Asymmetry floor = **0.05**
(`ASYMMETRY_FLOOR` in the flow probe). Pass-group regression tolerance = **10% relative**
below that structure's own baseline asymmetry.

| Group | Structures | Baseline asym | Requirement |
|-------|------------|---------------|-------------|
| Fail (diam ≥14) | 1BG1, 2Z6H, 1IVO, 2SHP, 1F88 | 0.023–0.032 | Individually clear ≥0.05 on **≥3 / 5** |
| Pass (diam ≤9) | 1UBQ, 1TEN, 1HHP, 1LYZ, 1MBN, 4OBE, 1TIM | 0.058–0.148 | No structure below `baseline × 0.90` |
| Hub-tracking | all 12 | ρ ≈ 0.44–0.69 | No structure below its own baseline Spearman(betweenness) |

### Outcomes

| Outcome | Criteria |
|---------|----------|
| **Win** | Fail-group ≥3/5 clear 0.05; pass-group holds; hub-tracking holds corpus-wide |
| **Partial** | Fail-group any rise but &lt;3/5 clear floor, **or** pass-group mild regression within 10% |
| **Fail** | No fail-group movement, **or** pass-group regression &gt;10%, **or** any hub-tracking drop below own baseline |

No pooling. Per-structure only. Disc layer reported but **not** deciding
([`DISC_PROJECTION_NOT_TRUNK_PROXY`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)).

---

## What this does *not* decide

- Path 2 (explicit directionality reward on diam ≤9) — separate registration after this closes
- Chem-Full — independent, not gated
- Path A ingest extension / train–app hierarchy unification — **explicitly deferred**; Path B is a stopgap only (design §9.2)
- Whether biotite `sse_code` should remain the app SSOT for `ss` features — separate fidelity question (design §9.1)
- Cross-domain parent shortcuts — out of scope for v1
- Dual-loading residue radius — **forbidden** (depth-collision lock)

---

## Status

| Item | Status |
|------|--------|
| Design v2 promotion | **Adopted** 2026-07-17 |
| Acceptance table | **Pre-registered** |
| Hierarchy data coverage | **Audited 2026-07-17** — source SSE 12/12; no governed hierarchy tables |
| Materialization path | **Path B locked** (train-side HELIX/SHEET stopgap; Path A supersedes later) — see design §9.2 |
| SSE fidelity (biotite vs deposited) | **Flagged** independently — design §9.1; side-check when docker is up |
| Implementation / first train | **Unblocked for Path B** — no governed-table writes; parse code not started until design lock read |
| Matched cold Makefile arms | **Registered** — `train-v66-containment-baseline` / `train-v66-containment-pathb` (Stage A-12 chem parent stack) |
| Checkpoint reload (`num_relations=9`) | **Tested** — `radial_mlps.7/8` sniff + metadata in `test_load_checkpoint_restores_containment_num_relations` |
| Stage-runner liveness wiring | **Wired** — `liveness_containment_*` MLflow metrics on `--containment-edge-mp` runs |
| Oversmoothing-at-root (Task 8 T1c parent) | **Diagnostic landed** — `t1c_containment_parent_oversmooth.py` + unit tests; cold-run grade **pending** |
| Flow-influence residue slice (Task 9) | **Landed** — `jacobian_flow_influence.py` honors `n_residue_nodes`; primary asymmetry score **pending cold arms** |
