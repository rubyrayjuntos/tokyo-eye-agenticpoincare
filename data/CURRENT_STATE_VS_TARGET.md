# Current State vs Target – Data Layer Comparison

**Date:** May 27, 2026  
**Purpose:** Quick reference to understand the gap between existing source material and the locked target architecture.

---

## Summary of Locked Target Model

- **Granularity:** Residue is the primary anchor (`dim_residue` as main join key). Atom supported. Lightweight `dim_site` for higher-order concepts.
- **Lineages:** v3 and v4 maintained as parallel scientific tracks (differentiated by provenance + model version).
- **v4 Scope:** Starts narrow (source-leak / allosteric / hyperbolic focus).
- **Training vs Inference:** Strict separation with clear labeling.
- **RAG:** Designed for from the beginning; first-class implementation deferred to Phase 5.
- **Storage:** Aurora (pgvector) for governed metadata + vectors + graph structure; object storage for large binaries. Equal accessibility required.
- **Core Principle:** All calculated or retrieved data lives in the governed data structure with strong provenance.

---

## Source Location Comparison

### 1. `adk-samples/python/agents/data-science`

| Aspect                    | Current State                          | Alignment with Target | Notes / Gap |
|---------------------------|----------------------------------------|-----------------------|-------------|
| Governance Schemas        | Good JSON schemas + templates          | Medium-High           | Needs evolution toward residue-centric model and embedding space registry |
| Aurora Schemas            | Strongest existing work (DTIE facts, pgvector, Poincaré distance function) | High | Best foundation for `data/aurora/` |
| Dimensional Model         | Mostly structure-centric               | Low-Medium            | Needs explicit `dim_residue` / `dim_site` emphasis |
| Provenance                | Present in schemas                     | Medium                | Needs strengthening and standardization |
| v3/v4 Handling            | Almost no real science code            | N/A                   | Mostly references |
| RAG Readiness             | Not addressed                          | Low                   | Will need design work in Phase 1-2 |

**Overall Assessment:** Best source for governance schemas and Aurora patterns. Should be heavily used but not treated as final.

### 2. `Tokyo-eye-demensional-investigator`

| Aspect                    | Current State                          | Alignment with Target | Notes / Gap |
|---------------------------|----------------------------------------|-----------------------|-------------|
| v3 DTIE Pipeline          | Most complete full pipeline            | High for breadth      | Excellent reference for how a broad pipeline emits data |
| Data Model Thinking       | Secondary to computation code          | Low                   | Not the source of truth for the target model |
| Provenance Practices      | Strong in v3 code                      | Medium-High           | Good patterns to extract |
| v4 Work                     | Only as design spec                    | Low                   | Not the location of current v4 implementation |
| Residue-Level Focus       | Present but not as explicit as other sources | Medium            | Useful but not the strongest example |

**Overall Assessment:** Valuable reference implementation for v3 breadth and provenance discipline. Not the primary driver of the data architecture.

### 3. `tokyo-eyes-visualizer`

| Aspect                    | Current State                          | Alignment with Target | Notes / Gap |
|---------------------------|----------------------------------------|-----------------------|-------------|
| v4 GNN Architecture       | Current best implementation (Gnnv4)    | High (science)        | Critical source for what the data model must represent |
| Per-Residue Hyperbolic Outputs | Strong (hyp_projections, x_hyp, etc.) | High                  | Must be cleanly representable |
| Training Code             | Entangled with model                   | Low                   | Needs cleaning (aligns with strict separation decision) |
| Orchestrator              | Weak / absent                          | Low                   | Consistent with "start narrow" decision |
| Data Governance           | Informal                               | Low                   | Will need to be retrofitted to governed model |

**Overall Assessment:** Highest-signal source for v4 science outputs and requirements. The data model must be able to express its richer hyperbolic state without loss.

### 4. `tokyo-eye-data`

| Aspect                    | Current State                          | Alignment with Target | Notes / Gap |
|---------------------------|----------------------------------------|-----------------------|-------------|
| Dimensional Model         | Most explicit (`dim_structure` → `dim_chain` → `dim_residue` → `dim_atom`) | Very High | Strongest existing example of the desired residue-centric approach |
| Per-Residue Fact Tables   | Clear pattern                          | High                  | Aligns closely with locked granularity decision |
| Vector Embeddings         | Residue-aware proposals (`fact_site_embedding`) | High | Good input for embedding space design |
| Lineage Thinking          | Detailed data-lineage-map              | High                  | Useful reference |
| Overall Philosophy        | Still somewhat structure-centric at top level | Medium-High      | Good foundation but needs evolution toward more flexible "join at any level" model |

**Overall Assessment:** One of the highest-value sources for the target data model philosophy. Should heavily influence the dimensional model and residue-level design.

---

## Overall Gap Summary

**Biggest Gaps to Close in Phase 1:**
- Move from structure-centric to residue-centric as the default mental model.
- Create explicit support for multiple embedding spaces (especially hyperbolic).
- Standardize provenance across all sources.
- Define clear patterns for parallel v3/v4 outputs.
- Establish the "only governed write path" (normalizer-like) pattern.

**What We Can Leverage Immediately:**
- Aurora schema work from the ADK repo
- Dimensional model thinking from `tokyo-eye-data`
- v4 output requirements from the visualizer
- v3 pipeline emission patterns from the demensional-investigator

---

**This document should be updated after each major phase or when significant new source material is evaluated.**