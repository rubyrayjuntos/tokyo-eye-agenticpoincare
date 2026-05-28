# Views, Materialized Views, and Access Patterns – Design (v0.9)

**Status:** Phase 0.2 Complete – Sufficient for Phase 1 detailed design and implementation
**Date:** May 27, 2026
**Related:** ARCHITECTURE.md, Migrations 017/019/020, Dimensional Model Draft

## 1. Purpose

Define the access layer that sits on top of the core dimensions and fact tables. This layer is critical for the agent, visualizer, future RAG systems, and human analysts.

## 2. Layering Philosophy

- **Core Tables** (migrations 001–018+): Authoritative, normalized, provenance-rich. Optimized for correctness and write paths.
- **Access Layer** (views + materialized views): Optimized for common read patterns. Can be rebuilt or refreshed without losing governed data.

## 3. Categories of Access Patterns

### A. Residue-Centric Views (Highest Priority)
Most scientific consumers want data at the residue level.

Examples already prototyped or planned:
- `v_residue_current_state` / `mv_residue_current_state`
- Latest embedding per residue per space
- Residue + site membership
- Residue + dehydron status + uncertainty

### B. Site-Level Views
For higher-order biological constructs (allosteric sites, source leaks, etc.).

- Site summary with aggregated residue metrics
- Site membership + contributing residue details

### C. Provenance & Audit Views
- Full lineage for a given asset or residue
- "What runs contributed to this residue's current embedding?"
- Synthetic vs real run differentiation

### D. RAG / Retrieval Support Views (Future – Phase 5)
- Chunk + embedding + provenance joined views
- Similarity search helpers (once dedicated retrieval spaces exist)

### E. Cross-Structure / Analytical Views
- Residue similarity across structures
- Site pattern discovery

## 4. Materialized View Strategy

Because many queries will be high-frequency (agent decisions, viewport rendering), we will use materialized views for the hottest paths.

Guidelines:
- Start with a small number of high-value materialized views (see migration 020 and 021).
- Define clear refresh strategies (on-demand after major runs vs scheduled).
- Provide "live" views alongside materialized ones for cases where freshness matters more than speed.
- Version or name materialized views clearly so consumers know their freshness guarantees.

## 5. Access Control & Performance

- Row-level or view-level access control aligned with `access_level` on assets.
- Read replicas or connection pooling for high-concurrency consumers (agent, visualizer).
- Query logging and slow-query monitoring from day one.

## 6. Current Status (Phase 0.2 Close)

- Several starter views and materialized view candidates have been created in migrations 017, 019, and 020.
- Pattern is established (residue-centric + provenance-aware).
- No comprehensive catalog of all required views yet (this is expected in Phase 1).
- No production refresh tooling or monitoring for materialized views yet.

## 7. Next Steps (Phase 1)

- Catalog the top 15–20 most important access patterns from the agent and visualizer.
- Design and implement the first production set of views and materialized views.
- Define refresh and invalidation strategies.
- Add query performance monitoring.

---

This document is considered sufficient to mark item (d) as designed to a Phase 0 complete level. Detailed implementation will occur in Phase 1.