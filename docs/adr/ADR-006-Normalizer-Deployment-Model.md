# ADR-006: Normalizer Deployment Model — Library-First, Service-Optional

**Status:** Accepted  
**Date:** May 27, 2026  
**Deciders:** Ray  
**Related ADRs:** ADR-003 (Training vs Inference), ADR-005 (Storage Split)

---

## Context

The Normalizer is the single governed write path for all scientific outputs. The open question was whether to implement it as:

1. An in-process Python library (imported directly by science code)
2. A standalone FastAPI microservice (called over HTTP)
3. Both (library with an optional service wrapper)

Each option has different implications for latency, error handling, deployment complexity, and how training runs flow through the system.

## Decision

**Library-first, service-optional.**

The Normalizer is implemented as a Python library (`data.normalizer.core.Normalizer`) that can be:

1. **Imported directly** by science code running in the same process — this is the primary usage path for v3/v4 pipelines and training scripts.
2. **Wrapped in a thin FastAPI service** for remote callers (agent coordinator, external tools, future microservices) — this is secondary and built on top of the same library.

## Rationale

### Why library-first

- **Latency:** GNN inference produces hundreds of per-residue outputs per structure. HTTP overhead per-node is unacceptable. Batch HTTP is possible but adds complexity for no benefit when the science code and normalizer share a process.
- **Atomicity:** In-process calls can share a database transaction with the science code's provenance creation, ensuring atomic writes without distributed transaction coordination.
- **Simplicity:** No service discovery, no health checks, no retry logic needed for the primary path.
- **Testing:** The library can be tested with a mock database (as demonstrated in the e2e tests) without spinning up infrastructure.

### Why service-optional (not service-never)

- The **agent coordinator** (FastAPI) runs as a separate process and needs to trigger normalizations for agent-driven workflows.
- Future **external tools** or **backfill scripts** running in different environments may need HTTP access.
- The service wrapper is trivial to build on top of the library — it's just a thin FastAPI router that instantiates the Normalizer and calls its methods.

### How training runs are handled (ADR-003 alignment)

Training runs use the same library but:
- Must set `run_type = RunType.TRAINING` in the provenance context
- The Normalizer validates this and tags all resulting assets accordingly
- No special deployment path needed — the library handles the differentiation

## Consequences

### Positive
- Science code gets the simplest possible integration (import and call)
- No infrastructure overhead for the primary write path
- Agent and external tools still get HTTP access when needed
- Single implementation to maintain (library), not two

### Negative
- Science code takes a dependency on the normalizer library (acceptable — it's in the same monorepo)
- The service wrapper must be kept in sync with the library API (mitigated by building it directly on top)

### Neutral
- Database connection management is the caller's responsibility (science code passes its connection to the Normalizer)

## Implementation

```
data/normalizer/
├── __init__.py          # Public API
├── core.py              # Library implementation (Normalizer class)
└── service.py           # FastAPI wrapper (future, Phase 2)
```

Usage from science code:
```python
from data.normalizer import Normalizer

normalizer = Normalizer(db=connection)
result = await normalizer.normalize_gnn_output(payload)
```

Usage from agent (future):
```python
# POST /api/normalize/gnn-output
# Body: GNNOutputPayload JSON
```

---

**This decision is LOCKED for Phase 1 execution.**
