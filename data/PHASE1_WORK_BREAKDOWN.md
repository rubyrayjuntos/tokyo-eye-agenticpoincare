# Phase 1 Work Breakdown – Core Data Model & Schemas

**Date:** May 27, 2026  
**Goal:** Produce versioned, reviewable schemas and a concrete dimensional model that implements the locked architecture.

---

## Major Workstreams

### 1. Dimensional Model Finalization
- Lock primary key strategy for `residue_id`
- Finalize `dim_site` modeling (table vs virtual)
- Define embedding space registry schema
- Document rules for choosing grain (when to attach at structure/chain vs residue)
- Produce v1.0 of the dimensional model documentation

### 2. Governance Schemas (JSON)
- Evolve the schemas from `tokyoeyes-data-governance/` to align with the new model
- Add support for:
  - Multiple embedding spaces
  - `source_type` standardization
  - Stronger provenance fields
  - `dim_site` and residue-level assets
- Version the schemas (e.g., v4 base)

### 3. Aurora Schema (PostgreSQL + pgvector)
- Base migration(s) incorporating:
  - Core dimensions (`dim_structure` through `dim_site`)
  - Key fact tables (GNN node output, phase results, embeddings)
  - Provenance spine tables
  - Embedding space registry
- Include the strong elements from the ADK repo (003_gnn_vector_embeddings.sql and 006_dtie_persistence.sql) where they fit
- Add Poincaré / hyperbolic helper functions as needed

### 4. Provenance Model
- Define the minimum viable provenance record
- How it attaches to assets at different grains
- Run / model_version / checkpoint tracking

### 5. Tooling & Validation
- Schema validation scripts (JSON Schema + SQL migration checks)
- Basic normalizer skeleton or write-path guidelines
- Simple query examples demonstrating residue-centric access

### 6. Documentation
- Updated `data/ARCHITECTURE.md`
- Dimensional Model v1.0 guide
- Migration notes from existing schemas

---

## Dependencies & Inputs

- Locked decisions (May 27, 2026)
- ADR-001 (Residue Anchor)
- Best elements from:
  - ADK governance schemas + Aurora migrations
  - `tokyo-eye-data` dimensional thinking and lineage map
- Insights from v3 and v4 output requirements

---

## Success Criteria for Phase 1

- A new contributor can understand the data model in < 1 day.
- It is clear how both v3 and v4 outputs will be represented.
- Adding a new per-residue computation feels natural and governed by default.
- The model supports the "design for RAG" requirement without forcing early implementation.

---

**This is a working breakdown.** It will be refined as Phase 0.2 completes and we get closer to starting Phase 1 execution.