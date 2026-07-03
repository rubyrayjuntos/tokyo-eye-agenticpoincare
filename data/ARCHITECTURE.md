# Data Architecture

**Repository:** tokyo-eye-agenticpoincare  
**Version:** 1.0 (Phase 0 Complete)  
**Date:** May 27, 2026  
**Status:** Final for Phase 0 – Ready to support Phase 1 execution

---

## 1. Purpose of This Document

This document defines the **foundational data architecture** for the Tokyo Eye Agentic Poincaré platform.

It establishes the principles, model concepts, and strategies that all other layers (science, agent, visualization, infrastructure) must follow. The data layer is treated as the highest-priority concern in this project.

---

## 1.5 Locked Decisions (May 27, 2026)

The following strategic decisions have been locked:

- **Granularity Anchor**: Residue is the primary anchor for most scientific facts. Atom is secondary but supported. A lightweight `dim_site` concept will be introduced for allosteric sites, high-uncertainty hotspots, glue sites, etc.
- **v3 vs v4**: Two parallel scientific lineages will be maintained for the foreseeable future (differentiated by provenance and model version). No forced early convergence.
- **v4 Scope**: The v4 orchestrator will start narrow (source-leak / allosteric / hyperbolic depth first).
- **Training vs Inference**: Strict separation — training artifacts must be clearly labeled.
- **RAG**: Design the model to support it cleanly; implement first-class RAG assets in Phase 5.
- **Storage**: Aurora (with pgvector) for governed metadata + vectors + graph structure; object storage for large binaries. Both must be equally accessible.

These decisions override earlier exploratory text where they conflict.

---

## 2. Core Data Principles

These principles are non-negotiable. All design and implementation decisions must be evaluated against them.

### 2.1 Governed Data by Default

**All data that is calculated or retrieved from external sources must be stored in the governed data structure.**

- No ad-hoc files, notebook outputs, or temporary artifacts should be considered authoritative.
- Every meaningful scientific output (GNN embeddings, phase results, model inferences, retrieval contexts, etc.) must be registered as a governed asset.
- "Sidecar" files are acceptable only when they are explicitly declared as derived artifacts of a governed asset with proper lineage.

### 2.2 Provenance is Mandatory

Every governed asset must carry strong, machine-readable provenance.

Required elements at minimum:
- The run / process that produced it
- The exact model version + checkpoint (with SHA256)
- Upstream data assets (with their identifiers)
- Code version / commit hash
- Timestamp and environment context

Provenance must be queryable and linkable across assets.

### 2.3 Schema-Driven by Default

All data structures (both metadata and payload schemas) must be defined by explicit, versioned schemas.

- JSON Schema for metadata and catalog records
- Database schema migrations for persistent storage (Aurora / PostgreSQL)
- Schema evolution must be deliberate and backward-compatible where possible

### 2.4 Multiple Representation Spaces Must Be First-Class

The platform must natively support multiple geometric and embedding representations without forcing conversion at storage time:

- Euclidean vector embeddings (for classical RAG, similarity search, etc.)
- Hyperbolic embeddings (Poincaré ball / disc representations from GNNs)
- Graph structures (Cα contact graphs, witness complexes, persistence diagrams)
- Hybrid / multi-space assets (an asset can have both Euclidean and hyperbolic projections)

### 2.5 Extensibility and Future-Proofing

The data model must be designed for long-term evolution, including:

- New GNN versions (v5, v6, …)
- New analysis phases
- New embedding models
- New RAG patterns
- Integration with external knowledge bases

Extensibility should be achieved through:
- Versioned schemas
- Extensible metadata (JSONB + controlled vocabularies)
- Clear separation between core governed fields and domain-specific extensions

### 2.6 Separation of Concerns

- **Governed Layer**: Authoritative, immutable (or append-only), queryable, provenance-rich.
- **Working / Intermediate Layer**: Allowed for compute-heavy or experimental work, but must eventually produce governed outputs.
- **Derived / Cached Layer**: Can exist for performance (e.g., materialized views, embedding caches), but must be traceable back to governed sources.

### 2.7 Granularity Model (Locked Decision)

**Residue is the primary granular anchor** for the majority of scientific data.

- `dim_residue` (with stable `residue_id`) is the default join key for GNN node outputs, per-residue features, phase results, uncertainty, embeddings, etc.
- `dim_atom` is supported and linked to residue, but is secondary.
- A lightweight `dim_site` concept will exist for higher-order biological constructs (allosteric sites, source leaks, high-uncertainty regions, etc.). This is a derived/virtual grain that can reference one or more residues.
- New data types should default to joining at the residue level unless there is a clear reason to use a different grain.

This model is intentionally designed to be extensible — new fact types or analysis outputs can join at structure, chain, residue, atom, or site level as appropriate.

### 2.8 v3 vs v4 Scientific Lineages (Locked Decision)

We will maintain **two parallel scientific lineages** (v3 and v4) in the data model for the foreseeable future.

- Differentiation will be handled through `model_version`, `pipeline_version`, and strong provenance rather than forcing a single unified schema prematurely.
- The core dimensional model (especially residue-level anchoring) is shared.
- Significant contract differences (e.g., native hyperbolic outputs in v4) may be represented via extension fields or version-specific fact tables.
- Consumers are expected to be version-aware when necessary. Convergence will be evaluated later based on actual usage and scientific need.

---

## 3. High-Level Data Model Concepts

### 3.1 Core Entity: Governed Asset

The central concept is the **Governed Asset**.

Every significant piece of data in the system is modeled as an asset with:

- `asset_id` (immutable, globally unique)
- `asset_type`
- `storage_uri`
- `provenance` (structured record)
- `schema` + `schema_version`
- `embedding_spaces` (array describing available representations)
- `lineage`
- `quality_metadata` and `governance_metadata`

### 3.2 Provenance Spine

We will maintain a strong **provenance spine**:

- Every asset points to its producing `run_id`
- Runs are linked to `model_version`, `checkpoint`, `code_version`, and input assets
- This forms a directed acyclic graph (DAG) that can be traversed for audit, reproduction, and explanation

### 3.2.1 Operational audit trails (compute runtime)

In addition to provenance on governed assets, the platform maintains **pipeline runtime audit** (`audit_pipeline_events`, migrations 049–050):

- Records geometric contract enforcement, learned-curvature passthrough, precondition failures, and pathway lifecycle events
- Queryable via `GET /api/structures/{id}/audit` and CLI (`make audit-structure`)
- Retention: detailed events roll into `audit_pipeline_daily_summary` after 90 days (configurable)

This is distinct from `normalization_audit` (governed writes). See `docs/audit/PIPELINE_AUDIT.md`.

### 3.3 Embedding & Graph Support

The architecture explicitly models multiple representation types:

- **Vector Embeddings**: Stored with dimensionality, model, space type (`euclidean`, `hyperbolic`), and curvature (when applicable).
- **Graph Structures**: Explicit support for different graph types.
- **Multi-space Assets**: A single scientific output can have multiple associated embeddings.

Special attention will be given to:
- Native support for hyperbolic geometry operations
- Vector RAG use cases (chunking strategy, retrieval metadata, context windows)

### 3.4 Versioning Strategy

- Schema versions are independent of scientific model versions.
- A `model_version` is metadata *on* an asset.
- The data model itself must support multiple model versions coexisting cleanly.

---

## 4. Governance and Storage Strategy

### 4.1 Schema-Driven Governance

- All metadata and catalog records are validated against versioned JSON Schemas.
- Database tables are managed exclusively through migration scripts.
- Schema changes require review against this architecture document.

### 4.2 Storage Layers (Locked)

- **Primary Governed Store**: Aurora PostgreSQL with pgvector (strong consistency, rich querying, provenance joins, vector similarity)
- **Object Storage**: For large binary artifacts (graphs, trajectories, payloads, checkpoints)
- **Catalog / Index Layer**: Lightweight, highly queryable records that point to primary storage

Both layers must be equally accessible to authorized consumers.

### 4.3 Immutability & Lifecycle

- Governed assets should be immutable once created (or append-only for time-series style data).
- Updates are modeled as new versions with explicit supersession links in provenance.

---

## 5. Phase 0 Closure & Readiness for Phase 1

As of the completion of Phase 0.2, this architecture is considered final for the pre-execution phase.

### What Has Been Locked in Phase 0
- All six major strategic decisions (documented in section 1.5)
- Residue-centric granularity model with extensible attachment points
- Parallel v3/v4 scientific lineage approach
- Strict training vs inference separation
- Design-for-RAG posture
- Storage split with equal accessibility requirement

### What Remains for Phase 1 and Beyond
- Final detailed schema definitions and migration scripts (building on the 20+ starter migrations already created)
- Full provenance spine implementation and tooling
- Normalizer / governed write-path design
- Views, materialized views, and access patterns
- Integration with science code (v3 and v4)
- Actual data population and backfill strategies

---

## 6. Governance Operating Model (High Level)

- **Schema-Driven**: All metadata and catalog records validated against versioned JSON Schemas. Database changes managed exclusively through migrations.
- **Provenance by Default**: No governed asset exists without a linked `run_id`.
- **Single Write Path Philosophy**: In production, a normalizer-style component will be the only authorized writer into the governed layer (design to be completed in Phase 1/2).
- **Extensibility with Guardrails**: Fast iteration via flexible tables + JSONB; promotion to structured tables only when patterns stabilize.

---

## 7. Known Risks & Mitigations (Phase 0 View)

- Risk: Premature unification of v3 and v4 outputs → Mitigation: Parallel lineages strategy with clear provenance differentiation.
- Risk: Overly complex schema chasing perfection → Mitigation: Layered extensibility (flexible properties first, structured tables later).
- Risk: Storage tier accessibility gaps → Mitigation: Explicit requirement for equal accessibility between Aurora and object storage.

---

## 8. Success Criteria for This Architecture (Phase 0)

This architecture will be considered successful for Phase 0 if, by the end of Phase 2:
- A new team member can understand the data model and governance rules in under one day.
- Adding a new per-residue or per-site computation feels natural and automatically governed.
- Both v3 and v4 scientific outputs can be represented cleanly without forcing artificial unification.
- The foundation demonstrably supports future Vector RAG and hybrid graph+vector workloads.

---

**This document (v1.0) is now the final reference architecture for Phase 0.** 

It will continue to be referenced throughout Phase 1 and beyond, but the core principles, decisions, and model concepts are considered stable. All future schema and implementation work should align with this document.