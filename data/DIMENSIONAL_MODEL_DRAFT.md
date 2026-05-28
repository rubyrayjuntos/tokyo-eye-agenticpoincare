# Dimensional Model Draft (v0.1)

**Date:** May 27, 2026  
**Status:** Early Draft – For internal development use  
**Related:** ADR-001, `data/ARCHITECTURE.md`, `data/CURRENT_STATE_VS_TARGET.md`

---

## 1. Core Philosophy

The data model is built around a **star/snowflake dimensional style** with a strong emphasis on **residue** as the primary analytical grain.

New data types (fact tables, derived views, embeddings, etc.) should attach to the dimensional model at the most natural level (structure, chain, residue, atom, or site) rather than forcing everything through `structure_id`.

---

## 2. Proposed Core Dimensions

### dim_structure
- Primary key: `structure_id` (UUID, generated at ingestion)
- Core fields: pdb_id, method, resolution, deposited_date, source (RCSB / AlphaFold / user), etc.
- Governance fields: access_level, owner, etc.

### dim_chain
- Primary key: `chain_id`
- Foreign key: `structure_id`
- Fields: chain_label, entity_type, etc.

### dim_residue (Primary Analytical Anchor)
- Primary key: `residue_id` (stable, preferably canonical)
- Foreign key: `chain_id`
- Key fields:
  - residue_index
  - residue_name (one-letter + three-letter)
  - sse_code (secondary structure)
  - sasa (computed)
  - other baseline per-residue properties

This is the grain at which most GNN outputs, dehydron data, cone metrics, uncertainty, and phase results should live.

### dim_atom
- Primary key: `atom_id`
- Foreign key: `residue_id`
- Fields: atom_name, element, coordinates (if stored), occupancy, b_factor, etc.

Used when atomic precision is genuinely required (e.g., certain physical calculations or detailed visualizations).

### dim_site (Lightweight / Virtual)
- Conceptual grain for higher-order biological constructs.
- Examples: allosteric sites, source leaks, high-uncertainty regions, glue sites, pharmacophore interaction sites.
- Not a physical grain — it is a derived/analytical construct that can reference one or more residues via bridge tables.

---

## 3. Relationships and Join Patterns

The model follows a classic dimensional pattern with `dim_residue` as the most important join key for scientific analysis.

### Recommended Foreign Key Patterns

- Most scientific fact tables should have:
  - `structure_id` (required for top-level filtering)
  - `residue_id` (preferred for per-residue facts)
  - Optional: `atom_id`, `site_id`

- Embeddings should always carry both:
  - `residue_id` (or `site_id`)
  - `space_id` (reference to `embedding_space`)

This design allows clean queries like:
- "Give me all hyperbolic embeddings for residue X across runs"
- "Show me all source leaks (dim_site) with their associated residue embeddings"
- "Compare v3 vs v4 GNN outputs on the same residues"

---

## 4. Multi-Space Embedding Support

The model is designed from the ground up to handle both Euclidean and Hyperbolic representations without forcing conversion at storage time.

### Embedding Space Registry (already started in 001)

```sql
embedding_space (
    space_id,
    name,
    space_type,           -- 'euclidean' or 'hyperbolic'
    dimensionality,
    curvature,            -- only populated for hyperbolic
    model_name,
    ...
)
```

### Fact Table Pattern

```sql
fact_xxx_embedding (
    ...
    residue_id,
    space_id,             -- FK to embedding_space
    embedding VECTOR,
    ...
)
```

This allows:
- Native 2D Poincaré disc projections from v4 (hyperbolic)
- Higher-dimensional Euclidean embeddings for RAG or other analysis
- Future addition of new spaces without schema changes

---

## 5. Example Query Patterns (Illustrative)

```sql
-- Get all native hyperbolic embeddings for a specific residue across runs
SELECT e.embedding, e.cone_depth, r.run_id, r.model_version
FROM fact_gnn_node_embedding e
JOIN provenance_run r ON r.run_id = e.run_id
WHERE e.residue_id = 'xxx'
  AND e.space_id IN (SELECT space_id FROM embedding_space WHERE space_type = 'hyperbolic')
ORDER BY r.completed_at DESC;

-- Find sites (dim_site) and their associated residue embeddings
SELECT s.site_type, r.residue_index, e.embedding, e.epistemic_uncertainty
FROM dim_site s
JOIN bridge_site_residue bsr ON bsr.site_id = s.site_id
JOIN dim_residue r ON r.residue_id = bsr.residue_id
LEFT JOIN fact_gnn_node_embedding e ON e.residue_id = r.residue_id
WHERE s.site_type = 'source_leak';
```

---

## 6. Next Steps for Phase 1 (Updated)

- Define stable key strategy for `residue_id`
- Finalize `embedding_space` registry schema
- Align with best parts of existing schemas from source repos
- Create a small set of canonical views for common access patterns (residue + embedding + provenance)
- Decide on rules for when data may legitimately live only at `structure_id` level

---

**This is an early working draft (v0.1).** It will be refined during Phase 1. The goal is a shared, concrete mental model that directly supports the locked decisions.
- Can reference one or more `residue_id`s (via a bridge table).
- Not a physical dimension like the others — more of a derived/analytical construct.

---

## 3. Embedding & Representation Support

We need first-class support for multiple representation spaces on entities (especially at the residue/site level).

Proposed approach:

- A `fact_*_embedding` pattern (or similar) that includes:
  - `residue_id` (or `site_id`)
  - `embedding_space_id` (reference to a registry)
  - `embedding` (VECTOR)
  - `space_type` (euclidean | hyperbolic)
  - `curvature` (when hyperbolic)
  - `model_version`
  - `computed_at`

- A small **embedding_space_registry** table to describe available spaces:
  - id, name, space_type, dimensionality, description, curvature (nullable), etc.

This allows clean storage of both classic Euclidean vectors and the native hyperbolic outputs from v4 (e.g., 2D Poincaré disc projections).

---

## 4. Fact Table Pattern

General pattern for scientific outputs:

```sql
fact_<domain>_<computation> (
    id,
    run_id,                    -- strong provenance link
    structure_id,
    residue_id,                -- preferred grain (per ADR-001)
    -- optional: atom_id, site_id
    source_type,               -- deterministic | probabilistic | external | derived
    model_version,
    -- domain-specific columns + JSONB for flexibility
    computed_at
);
```

Examples from existing thinking:
- `fact_gnn_node_output`
- `fact_dehydron`
- `fact_site_embedding` (proposed in tokyo-eye-data)
- Various `fact_phase*_output` tables

New computations should follow this pattern and prefer `residue_id` as the join key.

---

## 5. Example Core Table Sketches (Illustrative)

Real starting-point migrations have been created:

- `data/aurora/migrations/001_core_dimensions.sql`
- `data/aurora/migrations/002_example_fact_tables.sql`

These are **not final schemas** — they are early, concrete implementations of the model to make it actionable. They will be refined during Phase 1.

### Dimensions (Core)

```sql
CREATE TABLE dim_structure (
    structure_id   TEXT PRIMARY KEY,
    pdb_id         TEXT,
    method         TEXT,
    resolution     DOUBLE PRECISION,
    source         TEXT,                    -- RCSB, AlphaFold, user_upload, etc.
    created_at     TIMESTAMPDT DEFAULT NOW()
);

CREATE TABLE dim_chain (
    chain_id       TEXT PRIMARY KEY,
    structure_id   TEXT REFERENCES dim_structure(structure_id),
    chain_label    TEXT,
    entity_type    TEXT
);

CREATE TABLE dim_residue (
    residue_id     TEXT PRIMARY KEY,
    chain_id       TEXT REFERENCES dim_chain(chain_id),
    residue_index  INTEGER,
    residue_name   TEXT,
    sse_code       TEXT,
    sasa           DOUBLE PRECISION,
    created_at     TIMESTAMPDT DEFAULT NOW()
);

CREATE TABLE dim_atom (
    atom_id        TEXT PRIMARY KEY,
    residue_id     TEXT REFERENCES dim_residue(residue_id),
    atom_name      TEXT,
    element        TEXT,
    x              DOUBLE PRECISION,
    y              DOUBLE PRECISION,
    z              DOUBLE PRECISION
);

CREATE TABLE dim_site (   -- lightweight / derived
    site_id        TEXT PRIMARY KEY,
    structure_id   TEXT REFERENCES dim_structure(structure_id),
    site_type      TEXT,                    -- allosteric, source_leak, etc.
    description    TEXT,
    created_at     TIMESTAMPDT DEFAULT NOW()
);

CREATE TABLE bridge_site_residue (
    site_id        TEXT REFERENCES dim_site(site_id),
    residue_id     TEXT REFERENCES dim_residue(residue_id),
    PRIMARY KEY (site_id, residue_id)
);
```

### Example Fact Table (Residue-centric)

```sql
CREATE TABLE fact_gnn_node_output (
    node_output_id TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,           -- provenance
    structure_id   TEXT REFERENCES dim_structure(structure_id),
    residue_id     TEXT REFERENCES dim_residue(residue_id),  -- primary grain
    model_version  TEXT NOT NULL,
    input_features JSONB,
    projections    JSONB,                   -- Euclidean (backward compat)
    hyp_projections JSONB,                  -- Native hyperbolic (v4+)
    x_hyp          VECTOR,                   -- full hyperbolic embedding
    cone_depth     DOUBLE PRECISION,
    cone_width     DOUBLE PRECISION,
    epistemic_uncertainty DOUBLE PRECISION,
    aleatoric_uncertainty DOUBLE PRECISION,
    source_type    TEXT NOT NULL,
    computed_at    TIMESTAMPDT DEFAULT NOW()
);
```

## 6. Relationships and Provenance Integration

- Every fact table should have a direct or indirect link to a `run_id` for strong provenance.
- `dim_residue` and `dim_site` should support many-to-many relationships via bridge tables when needed (e.g., one site spanning multiple residues).
- Embeddings at the residue level should reference both the `residue_id` and an `embedding_space_id` for clarity across Euclidean and Hyperbolic representations.

## 7. Next Steps for Phase 1

- Define stable key generation strategy for `residue_id` (canonical vs surrogate).
- Design the embedding space registry in detail.
- Align this model with the best elements from the existing governance schemas (ADK repo) and dimensional thinking (`tokyo-eye-data`).
- Define rules for when it is acceptable to attach data at `structure_id` or `chain_id` instead of `residue_id`.
- Prototype a small set of core fact tables against this model.
- Define how provenance records link to assets at different grains (especially residue-level embeddings and phase outputs).

## 8. Embedding Space Considerations (Early Notes)

The model must cleanly support multiple embedding spaces:

- Euclidean vectors (various dimensions and models, used for classical similarity / RAG)
- Hyperbolic embeddings (Poincaré ball/disc from GNNs, with associated curvature and depth metrics)
- Potentially graph-derived embeddings or topological signatures

Recommended pattern:
- A lightweight `embedding_space` registry table.
- Fact tables that reference `residue_id` + `embedding_space_id`.
- Separate storage for the vector itself (using pgvector) plus metadata (model, space type, curvature, etc.).

This supports both current v4 hyperbolic outputs and future RAG needs without forcing conversion at write time.

---

---

## 8. Extensibility Mechanisms

The model is designed to support long-term evolution without constant breaking changes.

### Primary Extensibility Patterns

1. **Flexible `fact_computed_property` table** (see migration 014)
   - Allows rapid addition of new per-residue or per-site metrics during research phases.
   - Over time, high-value properties can be promoted to dedicated columns or tables.

2. **JSONB columns** on fact tables for semi-structured data.
3. **New fact tables** that follow the established residue/site + provenance + embedding_space patterns.
4. **New embedding spaces** registered via the `embedding_space` table (migration 013) without schema changes.

### Guidelines for New Data Types

- Prefer joining at `residue_id` (or `site_id`) unless there is a strong reason otherwise.
- Always include a `run_id` for provenance.
- Use the `embedding_space` registry for any new vector representations.
- Document the new type in the Dimensional Model and add it to future migrations.

---

## 9. RAG Integration Patterns (Design Notes)

While full RAG implementation is scheduled for Phase 5, the model is shaped to support it cleanly.

### Planned RAG Assets (Future)

- Chunk-level assets linked to residue/site.
- Retrieval context records (query, retrieved assets, scores, model used).
- Embedding spaces specifically tuned for retrieval (separate from scientific GNN spaces).

### Current Design Enablers

- Residue and site as stable join keys make it easy to attach retrieval metadata later.
- Multiple embedding spaces allow dedicated retrieval vectors without polluting scientific embeddings.
- Strong provenance ensures RAG results can be audited back to source scientific outputs.

---

## 10. Example Materialized Views (Future Candidates)

These are not created yet but illustrate common access patterns the model should support efficiently:

- `mv_residue_latest_embedding` (latest hyperbolic + Euclidean for each residue)
- `mv_site_uncertainty_summary`
- `mv_provenance_chain` (flattened lineage for quick audit queries)

---

---

## 10. Core Table Definitions (Proposed v0.2)

These are the current proposed structures for the primary dimensions, based on migrations 001–018.

### dim_residue (Primary Anchor)
```sql
residue_id          TEXT PRIMARY KEY
chain_id            TEXT NOT NULL
residue_index       INTEGER NOT NULL
residue_name        TEXT
residue_name_3      TEXT
sse_code            TEXT
sasa                DOUBLE PRECISION
created_at          TIMESTAMPTZ
updated_at          TIMESTAMPTZ
```

### dim_site (Lightweight Higher-Order)
```sql
site_id             TEXT PRIMARY KEY
structure_id        TEXT NOT NULL
site_type           TEXT          -- allosteric, source_leak, glue_site, etc.
description         TEXT
created_at          TIMESTAMPTZ
```

### embedding_space
```sql
space_id            TEXT PRIMARY KEY
name                TEXT NOT NULL
space_type          TEXT          -- euclidean | hyperbolic
dimensionality      INTEGER
curvature           DOUBLE PRECISION
model_name          TEXT
is_active           BOOLEAN
```

---

## 11. Extensibility & Evolution Strategy (Phase 0.2 Close)

The model must support years of evolution. The following principles have been established during Phase 0.2:

### Layered Extensibility Approach

1. **Fast iteration layer**: `fact_computed_property` (migration 014) + JSONB columns.
2. **Structured extension layer**: New dedicated fact tables following the established patterns (residue/site + run_id + space_id).
3. **Core model evolution**: Only when patterns stabilize and usage justifies promoting fields into the dimensional core.

### RAG Readiness (Design Complete)

The model is intentionally shaped so that when Phase 5 RAG work begins, the foundation is already there:
- Stable residue and site keys for chunk attachment.
- Native support for multiple embedding spaces (scientific vs retrieval-optimized).
- Full provenance for auditability of retrieved results.

---

## 12. Phase 0.2 Completion Note

As of this sustained pass, the Dimensional Model component of Phase 0.2 is considered **substantially complete** for the purposes of transitioning into Phase 1.

What has been achieved in this pass:
- Clear, decision-aligned philosophy (residue as primary anchor)
- Defined dimensions with supporting bridges
- Multi-space embedding architecture
- Extensibility strategy
- RAG integration considerations
- Concrete examples backed by 20+ migrations
- Documented relationships and query patterns

The model is now ready to serve as the reference for Phase 1 schema finalization and review.

---

**This document is now at v0.3 (Phase 0.2 close).** It will continue to evolve during Phase 1 execution, but the foundational architecture is locked and documented.

### Example Fact Table Pattern (residue-centric)
```sql
fact_xxx (
    id                  TEXT PRIMARY KEY
    run_id              TEXT NOT NULL          -- provenance
    structure_id        TEXT NOT NULL
    residue_id          TEXT NOT NULL          -- primary grain (ADR-001)
    space_id            TEXT                   -- for embeddings
    ... domain fields ...
    source_type         TEXT NOT NULL
    computed_at         TIMESTAMPTZ
)
```

---

## 11. Extensibility Rules (Draft)

1. New per-residue or per-site metrics should first be added via `fact_computed_property` (migration 014).
2. When a property proves valuable and stable, it should be promoted to a dedicated column or new fact table.
3. New embedding spaces are added via the registry (migration 013) — no schema change required.
4. New higher-order concepts should use `dim_site` + bridge tables rather than overloading `dim_residue`.

---

## 12. RAG Readiness Checklist (Early)

- [x] Residue and site as stable, queryable keys
- [x] Multiple embedding spaces supported
- [x] Strong provenance on all assets
- [ ] Chunk-level asset type defined
- [ ] Retrieval context / query log tables designed
- [ ] Materialized views for common similarity searches

---

**This is now an evolving working draft (v0.2).** Further refinement will occur in Phase 1 with stakeholder input.