# DATA_FIRST_CONSOLIDATION_ROADMAP

**Repository:** tokyo-eye-agenticpoincare  
**Version:** 0.1 – Initial Draft  
**Date:** May 27, 2026  
**Owner:** Ray (with Grok assistance)  
**Status:** For Review

---

## 1. Guiding Principle

**Data is the foundation.** Everything else (science, agents, visualization, RAG) is downstream of a well-governed, extensible, provenance-rich data layer.

The single most important outcome of this consolidation is a **flexible, schema-driven, extensible data platform** that can support the current work and unknown future work for years.

Key non-negotiables:
- Every piece of calculated or externally retrieved data must live in the governed data structure.
- Residue (and to a lesser extent Atom) is the primary granular anchor point. New data types join from whatever level makes sense (structure, chain, residue, atom, or site).
- The system must natively support multiple representation spaces: Euclidean vectors, Hyperbolic embeddings (Poincaré), graph structures, and hybrid combinations.
- Provenance is mandatory and first-class for every asset.
- The model must be schema-driven, versioned, and designed for long-term evolution.

---

## 2. Target Data Model Vision (High Level)

### Core Concepts
- **Dimensional Model** with clear separation between dimensions and facts.
  - `dim_structure`
  - `dim_chain`
  - `dim_residue` (primary join key for most scientific data)
  - `dim_atom`
  - `dim_site` (optional virtual/conceptual grain for allosteric sites, uncertainty hotspots, etc.)

- **Governed Asset** as the central abstraction (see `data/ARCHITECTURE.md` for details).

- **Provenance Spine**: Every asset links back through a consistent provenance model (run, model_version, checkpoint, upstream assets, code version).

- **Multi-Space Embeddings**: Explicit support for multiple embedding representations on the same logical entity (residue/site), including:
  - Hyperbolic (Poincaré disc / ball)
  - Euclidean vectors (various dimensions and models)
  - Graph-based representations

- **Source Type**: Every fact carries `source_type` (deterministic, probabilistic, external, derived, etc.).

- **Extensibility Pattern**: Core dimensions + many attachable fact tables + extensible metadata (JSONB + registered extension schemas). New computations or data types should be addable without breaking existing models.

---

## 3. Source Material Assessment

We have explored four significant source locations. This is a synthesis of what each contributes.

### 3.1 `adk-samples/.../data-science` (Agent Repo)
**Strengths:**
- Strongest existing governance schemas (`tokyoeyes-data-governance/`)
- Most mature Aurora/pgvector schema work (fact tables for DTIE phases, `fact_gnn_node_embedding`, custom `poincare_distance()` function)
- Existing Terraform, data governance templates, and some provenance patterns
- Good documentation of findings and session handoffs

**Gaps:**
- Governance schemas are still relatively high-level and not yet residue-centric enough
- Actual implementation of the data layer is thin (mostly references)
- v3/v4 science code is almost entirely missing

**Verdict:** Best source for governance schemas and Aurora storage patterns. Should be the starting point for `data/governance/` and `data/aurora/`.

### 3.2 `Tokyo-eye-demensional-investigator`
**Strengths:**
- Most complete v3 DTIE full pipeline (orchestrator + phases 1-6d)
- Strong provenance and artifact management patterns in the v3 code
- Ensemble and comparative run thinking

**Gaps:**
- v3-centric (older GNN)
- v4 exists only as a design spec here
- Data model thinking is secondary to the computation code

**Verdict:** Valuable reference for how a full DTIE pipeline should emit governed data, but not the source of truth for the data model itself.

### 3.3 `tokyo-eyes-visualizer`
**Strengths:**
- Current best v4 GNN work (Gnnv4 architecture, loss functions, training scripts)
- Real per-residue hyperbolic embedding outputs and validation
- React Poincaré viewer that consumes residue-level data

**Gaps:**
- Training code is entangled
- Lacks a full orchestrator
- Data emission patterns are less governed than in the v3 location

**Verdict:** Critical source for v4 science outputs and what the data model must be able to represent (native 2D disc projections, `x_hyp`, `x_routed_hyp`, cone metrics, etc.).

### 3.4 `tokyo-eye-data`
**Strengths:**
- Most explicit dimensional modeling with `dim_residue` and `dim_atom` as first-class citizens
- Clear data lineage map showing residue-level joins
- Vector embedding proposals that are residue-aware (`fact_site_embedding`)
- Good thinking on `source_type` and per-residue facts

**Gaps:**
- Still somewhat structure-centric at the top level
- Focused on a specific implementation plan (GCP + normalizer) rather than a fully general extensible model
- Less coverage of RAG and cross-structure scenarios

**Verdict:** One of the highest-signal sources for the **residue-centric + extensible fact table** pattern the user wants. Should heavily influence the target model.

---

## 4. Phased Roadmap

### Phase 0: Data Architecture & Audit (Current – ~3 weeks)
**Goal:** Lock the north star before moving large amounts of code.

**Key Deliverables:**
- `data/ARCHITECTURE.md` (first draft created; needs review + refinement)
- This roadmap (`DATA_FIRST_CONSOLIDATION_ROADMAP.md`) – reviewed and approved
- Deep Audit report (0.1 already started; continue with residue-level data flows and embedding contracts)
- Source Material Assessment (this section) – reviewed
- First set of Architecture Decision Records (ADRs) for major data choices

**Success Criteria:**
- Clear, written agreement on the residue/atom anchor model + extensibility strategy
- Decision on v3 vs v4 coexistence at the data layer

### Phase 1: Core Data Model & Schemas (~4–5 weeks)
**Goal:** Produce versioned, reviewable schemas that implement the vision.

**Key Deliverables:**
- Refined dimensional model (`dim_structure`, `dim_chain`, `dim_residue`, `dim_atom`, `dim_site`)
- Core governance schemas (evolved from ADK + insights from `tokyo-eye-data`)
- Aurora migration scripts (starting from the strong migrations in the ADK repo)
- Provenance schema (v1)
- Embedding space registry concept + initial tables for Euclidean + Hyperbolic
- JSON Schema + migration validation tooling

**Success Criteria:**
- Can represent both v3 and v4 GNN outputs cleanly
- Can attach new fact tables at residue (or other) level without schema changes

### Phase 2: Provenance & Governance Layer (~3 weeks)
**Goal:** Make provenance the default, not an afterthought.

**Key Deliverables:**
- Provenance spine implementation (run, model, checkpoint, lineage)
- Normalizer pattern (or equivalent) as the only write path into governed storage
- Asset registration + catalog patterns
- Basic query patterns for "show me the full provenance of this residue embedding"

### Phase 3: Science Layer Integration (~4–6 weeks)
**Goal:** Bring the actual v3 and v4 science code into the new structure while making them emit governed data.

**Key Deliverables:**
- `science/dtie/common/interfaces.py` (versioned contracts)
- v3 pipeline ported/adapted with governed emission
- v4 GNN + training harness ported with governed emission
- Adapters between v3/v4 outputs and the canonical data model where useful

### Phase 4: Agent + Visualizer Integration (~3–4 weeks)
**Goal:** Wire the Coordinator and Poincaré viewer to the new data layer.

### Phase 5: Extensibility, RAG & Hardening (~4+ weeks, ongoing)
**Goal:** Deliver on the "flexible and future-proof" promise.

- Vector RAG support (chunking, retrieval context assets, cross-structure queries)
- Hyperbolic graph query support
- Extension mechanism for new data types
- Documentation, examples, and contributor guides

---

## 5. Key Decision Points — LOCKED (May 27, 2026)

These decisions were made in conversation and are now considered locked for the next major phase of work.

1. **Granularity Anchor**  
   **Decision:** Yes — Residue is the primary anchor for most scientific facts. Atom is secondary but supported. We will introduce a lightweight `dim_site` (or virtual concept) for allosteric sites, high-uncertainty hotspots, glue sites, etc.

2. **v3 vs v4 Strategy**  
   **Decision:** Maintain **two parallel scientific lineages** in the data model for the foreseeable future (at least 6–12 months). Differentiate via `model_version` / `pipeline_version` + strong provenance. Do not force convergence early.

3. **Scope of v4 Orchestrator**  
   **Decision:** Start **narrow**. Focus on source-leak / allosteric depth and hyperbolic analysis first. Do not attempt to replicate the full breadth of v3 Phase 6 (virtual screening, ADMET, state selectivity) in the initial v4 orchestrator.

4. **Training vs Inference Data**  
   **Decision:** Be **strict**. Training runs must produce governed assets but must be clearly labeled (different `run_type` or `asset_type`). Governed scientific outputs consumed by agents/visualizers/RAG should primarily come from inference runs.

5. **RAG Priority**  
   **Decision:** **Design for, but implement later.** The core data model must be shaped so first-class Vector RAG assets (and hybrid graph+vector retrieval) can be added cleanly in Phase 5 without major rework. We will not build full RAG infrastructure in Phases 1–3.

6. **Storage Split**  
   **Decision:** The proposed split is accepted **with the condition that Aurora and object storage must be equally accessible** from the governed layer and from consumers (agent, visualizer, future RAG systems). Large binary artifacts go to object storage; metadata, vectors, and graph structure live primarily in Aurora (pgvector).

---

## 6. Success Criteria (Overall)

By the end of Phase 2, a new person should be able to:
- Understand the data model in < 1 day of reading
- Add a new per-residue computation and have it automatically governed and queryable
- Trace the full provenance of any residue-level embedding or phase result

By the end of Phase 5:
- The system feels extensible rather than "finished"
- Adding support for a new embedding space or analysis type feels natural, not painful

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data model becomes too complex chasing perfection | Medium | High | Ruthless focus on "good enough for the next 2–3 years" + clear extension points |
| v3/v4 science teams fight the data model instead of embracing it | Medium | Medium | Involve science contributors early in Phase 1 reviews |
| Provenance overhead slows down research velocity | High | Medium | Make the normalizer extremely ergonomic; provide convenience helpers |
| We under-invest in RAG because "science comes first" | Medium | High | Explicit RAG milestone in Phase 5 with dedicated time |

---

## 8. Immediate Next 1–2 Weeks (Concrete Actions)

**Status as of May 27, 2026 (autonomous execution):**

- [x] Review & Approve this Roadmap (user confirmed direction)
- [x] Review and iterate on `data/ARCHITECTURE.md` (v0.2 completed with locked decisions)
- [x] Complete Phase 0.1 Deep Audit (initial GNN + EvidentialHead analysis done; more to follow)
- [x] Create first 5 Architecture Decision Records (all locked decisions covered)
- [x] `data/DIMENSIONAL_MODEL_DRAFT.md` (v0.1) started with concrete table sketches and residue-centric patterns
- [x] Produce "Current State vs Target" comparison (`data/CURRENT_STATE_VS_TARGET.md`)
- [ ] Schedule a focused review session on the dimensional model (pending user availability)
- [x] ADR directory + README established

**Remaining in this window:**
- Phase 0.2 is now well advanced. The focus can shift toward completing the Dimensional Model (v1.0) and transitioning into Phase 1 execution when ready.
- Continue refining coordination documents as needed.

**Notable autonomous output (sustained work — final Phase 0.2 pass):**
- Real SQL migration files created (now 025 total, including core dimensions, provenance, embeddings, DTIE phases, extensibility scaffolding, RAG/write-path markers, and Phase 1 refinement placeholder)
- `data/aurora/MIGRATIONS.md` kept current with full set and notes
- `data/DIMENSIONAL_MODEL_DRAFT.md` significantly expanded to a state ready to support Phase 1 (v0.2 / Phase 0.2 close)
- `data/ARCHITECTURE.md` upgraded to v1.0 (Phase 0 Complete) with full strategic sections, Phase 0 closure, governance operating model, risks, and success criteria
- Detailed design documents created for:
  - Normalizer / governed write-path (c) → NORMALIZER_DESIGN.md
  - Views, Materialized Views & Access Patterns (d) → VIEWS_AND_ACCESS_PATTERNS.md
  - Science Code Integration (e) → INTEGRATION_STRATEGY.md
  - Data Population & Backfill (f) → DATA_POPULATION_AND_BACKFILL_STRATEGY.md
- Real migration count now at 25 (including Phase 1 refinement marker)
- Phase 0.2 is now considered **100% complete** for the data architecture foundation.
- All coordination documents updated in this final pass.

---

**This document is intended to be the single point of coordination for the data-first consolidation.**

It will be updated after every major phase or significant decision. All other work should be traceable back to this roadmap.

---

*End of Roadmap v0.1*