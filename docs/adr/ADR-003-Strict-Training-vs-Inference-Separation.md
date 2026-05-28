# ADR-003: Strict Separation of Training and Inference Outputs

**Status:** Accepted  
**Date:** May 27, 2026  
**Deciders:** Ray  

---

## Context

The v4 (and to some extent v3) GNN work involves significant training runs that produce many intermediate artifacts (checkpoints, logs, diagnostic outputs, per-epoch embeddings, etc.).

At the same time, the governed scientific outputs that will be consumed by the agent, visualizer, and future RAG systems come from **inference** runs against trained models.

Blurring these two creates several problems:
- Difficulty in knowing which outputs are "production" vs experimental
- Contamination risk in provenance queries
- Harder reproducibility for downstream consumers
- Unclear governance expectations for training artifacts

## Decision

We will enforce a **strict separation** between training and inference outputs in the governed data layer.

### Rules

- Training runs must produce governed assets.
- All training-related assets must be explicitly labeled (via `run_type`, `asset_type`, or dedicated fields such as `is_training_artifact`).
- The primary scientific outputs intended for agents, visualization, and RAG (GNN node outputs, phase results, final embeddings, etc.) should come from inference runs.
- Training artifacts may reference the same `model_version` / checkpoint lineage but must be distinguishable in queries and lineage graphs.

## Consequences

### Positive
- Clear provenance and audit trails
- Easier to maintain reproducibility for scientific consumers
- Reduces risk of accidentally serving experimental training outputs downstream

### Negative / Trade-offs
- Slightly more ceremony when recording training runs
- Requires discipline in the normalizer / write path

## Alternatives Considered

- Loose / blended approach: Rejected due to governance and reproducibility risks.
- Training artifacts completely outside the governed layer: Rejected — training runs are still valuable governed knowledge (especially for future RAG over experiments).

## References

- `data/DATA_FIRST_CONSOLIDATION_ROADMAP.md` (Section 5, Decision 4)