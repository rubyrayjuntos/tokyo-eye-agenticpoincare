# Hyp biology MP v1 Implementation Plan

> **For agentic workers:** Execute task-by-task. Prereg SSOT: `docs/specs/tokyo-eye-v7/hyp-biology-mp-prereg.md`.

**Goal:** Child lineage where Hyp MP uses only structured biology edges (hbond, dehydron, pi_stack, salt_bridge), fail-closed against Cα, Option A degree-0.

**Architecture:** New `biology_graph.py` builds typed edges; `resolve_hyp_mp_edges` fail-closes when `hyp_biology_mp`; audit counters prove zero Cα leakage; smoke grade under `tokyo_eye_v7_hyp_biology_mp_v1` — never overwrite sealed Θ.

**Tech Stack:** PyTorch Geometric, numpy, existing `ResidueRecord` / wrapping SSOT.

## Global Constraints

- `allow_ca_fallback=false` on this path
- Residual isolates: Option A (self-only)
- Do not overwrite `HEALTHY_V7_CKPT` / `v7_healthy_sealed.pt`
- Default sealed forward unchanged (`hyp_biology_mp` default False)

---

### Task 1: Audit + fail-closed resolve

- [ ] Extend `resolve_hyp_mp_edges` for `hyp_biology_mp` / `allow_ca_fallback`
- [ ] Add `biology_mp_audit.py` counters
- [ ] Unit tests: Cα fallback raises; empty biology graph OK (degree-0)

### Task 2: Biology edge extract

- [ ] `science/tokyo_eye/biology_graph.py`: salt (≤4Å), π (≤5.5Å + dihedral gates), hbond/dehydron via wrapping SSOT
- [ ] `attach_biology_mp_graph(data, ...)` sets `hyperbolic_graph` + typed index
- [ ] Unit tests on synthetic residues

### Task 3: Wire + smoke

- [ ] `TokyoEye.hyp_biology_mp` → audit trail + resolve kwargs
- [ ] `experiments/diagnostics/v7_hyp_biology_mp_smoke.py` + Makefile target
- [ ] Hard-fail if `ca_in_mp` or forbidden ontology present
