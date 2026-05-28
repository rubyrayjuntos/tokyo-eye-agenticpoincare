# Science Code Integration Strategy (v3 and v4) – v0.8

**Status:** Phase 0.2 Complete – Design sufficient to guide Phase 3 execution
**Date:** May 27, 2026
**Related:** Roadmap Phase 3, ADR-002 (Parallel Lineages), ARCHITECTURE.md

## 1. Goal

Bring the actual v3 (full pipeline) and v4 (advanced hyperbolic GNN) science codebases into the new platform while ensuring they emit data into the governed, residue-centric data model.

## 2. High-Level Approach

We will maintain **two parallel integration tracks** (consistent with ADR-002):

- **v3 Track**: Mature, broad DTIE pipeline. Goal = make it emit governed data using the new model with minimal internal changes where possible.
- **v4 Track**: Newer, narrower, higher-quality GNN. Goal = integrate more deeply with the new residue-centric and multi-space model.

Both tracks will ultimately write through the Normalizer (once built).

## 3. Key Integration Principles

- The science code should **not** write directly to Aurora in the long term.
- Science code owns computation; the data layer owns governance, schema enforcement, and provenance.
- v3 and v4 may have different internal data structures. Adapters / normalizers will translate them into the canonical model.
- Historical runs from either codebase should be representable with proper provenance (even if synthetic).

## 4. Proposed Integration Layers (to be built in Phase 3)

### For v3 (Broad Pipeline)
- Create adapter modules under `science/v3/adapters/` that take v3 internal outputs and produce payloads acceptable to the Normalizer.
- Focus first on high-volume paths: GNN output, dehydron, phase 3 persistence, etc.
- Preserve the existing broad phase coverage while routing outputs through governance.

### For v4 (Advanced GNN)
- Deeper integration because v4 already produces richer hyperbolic state (`x_hyp`, native 2D disc projections, etc.).
- Ensure the new `embedding_space` registry and multi-space patterns are used natively.
- Because v4 is narrower, integration scope is smaller but expectations for data quality are higher.

## 5. Phasing within Phase 3

1. **3.1** – Define stable interfaces between science code and Normalizer (versioned payloads).
2. **3.2** – Implement v4 integration first (higher scientific value, narrower scope).
3. **3.3** – Implement v3 integration (broader scope, more adapters needed).
4. **3.4** – Historical backfill tooling for existing runs from both lineages.
5. **3.5** – End-to-end testing with the agent and visualizer.

## 6. Open Questions for Phase 3

- How much refactoring of the existing v3/v4 codebases is acceptable vs. building adapters around them?
- Where will the Normalizer live relative to the science code (same process, separate service, or both)?
- Strategy for handling training runs from v4 (ADR-003) during integration.
- Exact mechanism for passing `run_id` and other provenance context from the agent/orchestrator down into the science code.

## 7. Current Status (Phase 0.2 Close)

- Strategy and principles documented.
- Data model is designed to accept outputs from both lineages without forcing unification.
- No actual integration code or interface definitions written yet (this is expected in Phase 3).

---

This document is considered sufficient to mark item (e) as designed to a Phase 0 complete level. Detailed interface design and implementation will occur in Phase 3.