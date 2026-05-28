# ADR-004: Design for RAG, Implement Later

**Status:** Accepted  
**Date:** May 27, 2026  
**Deciders:** Ray  

---

## Context

Vector RAG (and hybrid graph + vector retrieval) is an important future capability for the platform, especially for querying across structures, uncertainty sites, and scientific findings.

However, the immediate priority is building a solid, governed data foundation for the core scientific pipelines (v3 and v4 DTIE/GNN work).

Building full RAG infrastructure too early risks:
- Over-engineering before we understand real usage patterns
- Diverting focus from the residue-centric data model
- Creating RAG assets that are poorly governed or poorly connected to provenance

## Decision

We will **design the core data model to support first-class Vector RAG and hybrid retrieval from the beginning**, but we will defer the actual implementation of RAG-specific assets, chunking strategies, and retrieval infrastructure until Phase 5.

### Guidelines

- The dimensional model (especially residue/site level) and provenance spine must be shaped so RAG assets can be added cleanly later.
- When designing new fact tables or embedding patterns, consider future RAG use cases (e.g., attaching retrieval metadata, context windows, similarity scores).
- Do not build dedicated RAG tables, chunk stores, or retrieval context models in Phases 1–3 unless they are strictly necessary for core science.
- Explicit RAG work is scheduled for Phase 5.

## Consequences

### Positive
- Prevents premature complexity
- Ensures RAG capabilities are built on top of a mature, well-governed foundation
- Keeps focus on the highest-priority item (data governance + residue model)

### Negative / Trade-offs
- RAG capabilities will arrive later than they might in a more RAG-first project
- Some early design decisions may need minor adjustment when RAG work begins

## Alternatives Considered

- Build RAG infrastructure in parallel with core data layer: Rejected — too high risk of distraction from data foundation.
- Ignore RAG entirely during early design: Rejected — would make later integration painful.

## References

- `data/DATA_FIRST_CONSOLIDATION_ROADMAP.md` (Section 5, Decision 5)
- `data/ARCHITECTURE.md` (RAG considerations section)