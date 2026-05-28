# ADR-005: Storage Split with Equal Accessibility

**Status:** Accepted  
**Date:** May 27, 2026  
**Deciders:** Ray  

---

## Context

The platform will store a wide variety of data:
- Governed metadata, provenance records, vector embeddings, graph structure, and small-to-medium scientific outputs
- Large binary artifacts (full graphs, folding trajectories, raw GNN payloads, checkpoints, etc.)

Using a single storage technology for everything is impractical. Aurora (PostgreSQL + pgvector) is excellent for structured data, provenance queries, and vector similarity. Object storage is far better for large binaries.

The risk is creating a two-tier system where one tier is much harder to access than the other, leading to poor developer experience and fragmented data access patterns.

## Decision

We will use a **tiered storage model**:

- **Aurora (PostgreSQL + pgvector)**: Primary store for governed metadata, provenance, vector embeddings, graph metadata/edges, and smaller scientific outputs.
- **Object Storage** (e.g., S3/GCS): Primary store for large binary artifacts.

**Critical constraint (locked):** Both tiers must be **equally accessible** from the governed data layer and from authorized consumers (agent coordinator, visualizer, future RAG systems, etc.).

### Implementation Guidelines

- All large artifacts in object storage must have strong pointers + checksums recorded in the Aurora governed layer.
- Access patterns (queries, joins, provenance traversal) should feel consistent whether the underlying data is in Aurora or object storage.
- Abstraction layers (e.g., storage service, asset resolver) should hide the physical location where reasonable.

## Consequences

### Positive
- Best tool for the job in each case (queryable structured data vs cheap scalable binary storage)
- Meets the explicit requirement for equal accessibility
- Scalable and cost-effective

### Negative / Trade-offs
- Requires good abstraction and tooling to avoid a "leaky" two-tier feel
- Slightly more operational complexity than single-store

## Alternatives Considered

- Everything in Aurora: Rejected — poor fit and expensive for large binaries.
- Everything in object storage: Rejected — loses the power of SQL + pgvector for metadata, provenance, and embeddings.
- Aurora as primary with object storage as "second class": Rejected by the locked decision requiring equal accessibility.

## References

- `data/DATA_FIRST_CONSOLIDATION_ROADMAP.md` (Section 5, Decision 6)
- `data/ARCHITECTURE.md` (Storage Layers section)