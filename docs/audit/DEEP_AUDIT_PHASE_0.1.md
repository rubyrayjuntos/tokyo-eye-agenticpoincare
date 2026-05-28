# Deep Audit Report — Phase 0.1
**Repository:** tokyo-eye-agenticpoincare  
**Date:** May 27, 2026  
**Focus:** Core scientific layer comparison between v3 (mature full pipeline) and v4 (advanced hyperbolic GNN)

---

## 1. Executive Summary of Findings

The two codebases represent **evolutionary stages of the same core idea**, not two independent systems. However, the differences are significant enough that naive merging will create technical debt.

**Key Insight:**  
v4 is not a drop-in upgrade of v3. It is a deliberate architectural shift that keeps more computation inside hyperbolic geometry. This improves scientific quality but changes internal data flow and some output contracts.

---

## 2. GNN Model Comparison (GOSPConeMapper)

Both versions expose a class named `GOSPConeMapper`. This is helpful for naming consistency but masks deep internal changes.

### Shared Elements (Low Friction)
- Input contract is preserved: `node_dim=4` (`rho, tau_flag, ss_type, sasa`)
- SE(3)-equivariant backbone (`EquivariantConv`) is largely the same
- Curvature parameterization (`log_c` + softplus) is identical
- High-level module layout is similar (node_emb → convs → hyperbolic lift → MoE → uncertainty → projections)
- `Data` object expectation (edge_index, edge_attr, clustering)

### Major Differences (High Friction)

| Aspect | v3 (Gnnv3.py) | v4 (Gnnv4.py) | Impact |
|--------|---------------|---------------|--------|
| **Core Philosophy** | Lift to hyperbolic → quickly return to Euclidean for MoE, experts, uncertainty | Stay in hyperbolic / tangent space as long as possible | Fundamental |
| **MoE Routing** | Experts receive Euclidean features after hyperbolic lift | Experts receive `logmap0(x_hyp)` (tangent space) | Changes expert input semantics |
| **Expert Combination** | Standard Euclidean | Weighted sum in tangent space → re-lift via expmap0 | Different geometry |
| **Uncertainty Head** | Receives Euclidean features | Receives `cat([logmap0(x_routed_hyp), depth, cone_width])` — input dim changed | **Breaking change** for EvidentialHead |
| **Projection Heads** | Single Euclidean `nn.Linear` (64-dim) | Dual: Euclidean + native `MobiusLinear` → 2D disc | New output: `hyp_projections` |
| **Exposed Outputs** | Limited hyperbolic state | Explicitly exposes `x_hyp`, `x_routed_hyp`, `hyp_projections` | Richer but different dict |
| **New Parameters** | — | `hyper_scale`, `aleatoric_logvar_*`, `epistemic_temp_scaling` | New knobs |
| **Checkpoint Compatibility** | `robust_experts.pt` works | Explicitly incompatible with v3 checkpoints | Major migration issue |

**Audit Finding:** The uncertainty head and expert routing are the two largest points of incompatibility.

---

## 3. Phase Module Landscape

### v3 (Demensional Investigator)
- **13 phase files** in `DTIE_GNN_ORCHESTRATION/`
- Full coverage: Phases 1 → 6d (including virtual screening, ADMET, state selectivity)
- Orchestrator (`dtie_pipeline.py`) is mature and explicit about state passing
- Strong provenance and artifact management

### v4 (Visualizer)
- Only **2 specialized phase files**:
  - `phase1_witness_embedding_v4.py`
  - `phase3_witness_persistence_v4.py`
- These appear to be targeted improvements rather than a full parallel pipeline
- Training-focused scripts dominate (`train_v4.py`, `retrain_stage2.py`, etc.)

**Audit Finding:** v4 does not have a complete orchestrator equivalent to v3's `dtie_pipeline.py`. The v4 work is currently "GNN + selected improved phases" rather than "full DTIE pipeline v4".

---

## 4. Orchestrator Comparison

**v3 Orchestrator (`dtie_pipeline.py`):**
- ~1100+ lines
- Explicit phased execution with clear handoff of results dict
- Supports full drug discovery flow (screening library, docking, ADMET)
- `run_dtie_full_pipeline(...)` is the primary entry point
- Good logging and artifact writing

**v4 Situation:**
- No equivalent full orchestrator found in the visualizer directory
- Training scripts (`train_v4.py`) are the main "orchestration" currently
- The March 2026 v4 spec (`dtie_v4_source_leak_pipeline_spec.md`) proposes a narrower, source-leak-focused pipeline

**Audit Finding:** This is one of the largest gaps. The v3 orchestrator is production-grade for broad use. v4 currently lacks this layer.

---

## 5. Training vs Inference Separation

This is a major source of future confusion.

- **v3 location:** Training and inference logic are relatively separated. The `DTIE_GNN_ORCHESTRATION/` folder is mostly inference-oriented.
- **v4 location:** Heavy entanglement. `train_v4.py`, `retrain_stage2.py`, `finetune_differential.py` live right next to the model and phase files. Many hardcoded paths and experiment-specific logic.

**Audit Finding:** v4 training code will require significant cleaning before it can live cleanly in `experiments/training/`.

---

## 6. Key Risks Identified

1. **Uncertainty Head Shape Change** — Highest immediate technical risk. Any code assuming the v3 EvidentialHead signature will break.
2. **"GOSPConeMapper" Name Collision** — Same class name, very different internals. Dangerous during gradual migration.
3. **Missing v4 Orchestrator** — We cannot simply "upgrade" the v3 orchestrator to use v4 GNN without substantial new work.
4. **Hyperbolic Output Expectations** — The visualizer frontend and Phase 3 witness code may have different assumptions about what "projections" look like (Euclidean 64-d vs native 2D disc).
5. **Checkpoint Story** — `robust_experts.pt` (v3) vs newer v4 checkpoints. The production `tokyo_eyes_v4.pt` lives in the agent repo checkpoints but its exact training script is scattered.

---

## 7. Preliminary Recommendations (for Decision Records)

- **Do not** attempt a single unified `GOSPConeMapper` in the near term.
- Create `science/dtie/common/interfaces.py` with versioned protocols first.
- Treat v4 as the **future primary scientific path** for new work, while keeping v3 as the "broad pipeline" reference.
- The new orchestrator in `science/dtie/v4/` should be intentionally narrower (source-leak / allostery focused) per the March spec, rather than trying to replicate all of v3's Phase 6 work immediately.
- Move all v4 training scripts into `experiments/training/v4/` with strict separation from inference code.

---

## 8. Additional Findings — EvidentialHead (Uncertainty)

**v3 Signature:**
```python
def __init__(self, hidden_dim: int, out_dim: int = 1)
```

**v4 Signature:**
```python
def __init__(
    self,
    hidden_dim: int,
    extra_input_dim: int = 2,  # depth + cone_width
    ...
)
```

**Impact:** This is a **breaking interface change**. The v4 version deliberately feeds geometric information (how deep the residue is in the learned conformational hierarchy) directly into the uncertainty model. This is scientifically powerful but means:

- Any code that instantiates or calls the uncertainty head directly must be version-aware.
- Checkpoints trained with v3 uncertainty heads cannot be used with v4 models (and vice versa).

This reinforces the earlier recommendation against trying to maintain a single unified GNN class in the short-to-medium term.

---

## 9. Phase Module Observations (Initial)

- v3 Phase 3 (`phase3_witness_persistence.py`): 147 lines — relatively focused.
- v4 Phase 3 (`phase3_witness_persistence_v4.py`): 235 lines — substantially more complex.

This suggests the v4 version of even individual phases contains meaningfully more sophisticated logic (likely tighter integration with hyperbolic distances and the new GNN outputs).

---

## 10. Updated Next Audit Steps (0.2)

- [ ] Full comparison of `TopologicalMoEGate` between v3 and v4
- [ ] Side-by-side analysis of `phase3_witness_persistence.py` vs `phase3_witness_persistence_v4.py` (data flow from GNN outputs)
- [ ] Inventory of all places the visualizer expects `hyp_projections` (2D disc) vs old Euclidean projections
- [ ] Review of `contracts.py` in both locations for shared data models
- [ ] Assessment of how much of the v3 full Phase 6 (virtual screening) work would need to be re-implemented or adapted for a v4 orchestrator

---

**End of Phase 0.1 Initial Audit Pass**

---

*This document is the living record of the Deep Audit. It will be updated as more modules are examined.*