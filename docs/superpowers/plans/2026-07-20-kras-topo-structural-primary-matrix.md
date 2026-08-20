# KRAS Topo-Structural Primary Matrix — Implementation Plan

> **For agentic workers:** Execute task-by-task. Primary triad only (`4OBE`/`4DSO`/`5VQ2`); do not block on Section 13 PDB.

**Goal:** Wire Pass/Fail triangulation grade against `FIX1_SPARSITY_CHAMPION_CKPT` per `docs/specs/kras-topo-structural-inference/design.md`.

**Architecture:** Pure helpers in `science/dtie/common/kras_topo_matrix.py`; GPU diagnostic CLI for forward-knockout Spearman; classical ΔE + switch-lock without model weights.

**Tech Stack:** NumPy, NetworkX, PyTorch, Bio.PDB, existing `knockout_scan` / `classical_network_metrics`

## Global Constraints

- Jacobian rankings forbidden under z-norm
- Active primary = `5VQ2` only
- Uncertainty monitor-only
- G12D·GppNHp observational via Section 13 only

---

### Task 1: Helpers + unit tests

- [ ] `align_resseq_vectors`, `spearman_rho`, `edge_symdiff_size`, `switch_lock_verdict`, `primary_matrix_verdict`
- [ ] Tests in `tests/test_kras_topo_matrix.py`

### Task 2: Diagnostic CLI + Make

- [ ] `experiments/diagnostics/kras_topo_structural_matrix.py`
- [ ] `make grade-v66-fix1-sparsity-kras-topo-matrix`
- [ ] Update `docs/specs/routing-entropy-sparsity/next-phase.md`
