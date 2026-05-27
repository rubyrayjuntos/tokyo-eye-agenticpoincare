# Architecture Overview

**Status:** Initial placeholder (to be expanded during consolidation)

## Guiding Principles

- Science core is pure and framework-agnostic.
- v3 (mature full pipeline) and v4 (advanced hyperbolic) are deliberately separated.
- Agent layer orchestrates science, it does not implement it.
- Visualization is a first-class consumer of scientific outputs.

## Layer Responsibilities

See `CONSOLIDATION_PLAN.md` for the proposed target structure and rationale.

## Key Interfaces (To Be Defined)

- `science.dtie.common.interfaces`
- DTIE tool contracts for the agent
- Viewer data contract (Poincaré projections + metadata)

---

**This document will evolve significantly during Phase 1 and 2.**