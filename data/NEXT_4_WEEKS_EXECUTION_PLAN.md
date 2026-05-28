# Next 4 Weeks Execution Plan (Locked Decisions Applied)

**Date:** May 27, 2026  
**Based on:** DATA_FIRST_CONSOLIDATION_ROADMAP.md (decisions locked)

## Goal for Next 4 Weeks
Move from "decided roadmap" to "concrete, reviewable artifacts" so real implementation can begin with minimal further discussion.

---

## Week 1–2: Phase 0.2 – Solidify Architecture & First ADRs

### Primary Deliverables
1. **Refined `data/ARCHITECTURE.md`** (v0.2)
   - Incorporate all locked decisions
   - Add explicit section on the residue/atom/site dimensional model
   - Add section on how parallel v3/v4 lineages will be represented

2. **First Architecture Decision Records** (in `docs/adr/`)
   - ADR-001: Residue as Primary Granular Anchor
   - ADR-002: Parallel v3/v4 Scientific Lineages
   - ADR-003: Strict Training vs Inference Separation

3. **"Current State vs Target" One-Pager**
   - Simple table showing how the four source locations map (or don't map) to the locked target model.

### Secondary Deliverables
- Update `DATA_FIRST_CONSOLIDATION_ROADMAP.md` with any refinements that came out of the ADR process.
- Create `docs/adr/README.md` with the decision record template.

---

## Week 3–4: Phase 1 Kickoff Preparation

### Primary Deliverables
1. **Dimensional Model Draft** (residue-centric)
   - Proposed table structures for:
     - `dim_structure`, `dim_chain`, `dim_residue`, `dim_atom`, `dim_site`
     - Core `fact_*` tables that will join primarily on `residue_id`
   - Initial proposal for embedding space metadata (how we will distinguish Euclidean vs Hyperbolic representations)

2. **Provenance Model v0.1**
   - Minimum viable provenance record structure
   - How it will attach to assets at different grains (especially residue level)

3. **Gap Analysis Output (A)**
   - Structured comparison of the best existing schemas/migrations against the locked architecture (drawing heavily from `tokyo-eye-data` and the ADK governance work).

### Success Criteria for Week 4
- Someone new to the project can read the Dimensional Model Draft + ARCHITECTURE.md and understand the intended shape of the data layer.
- The major technical risks around v3/v4 coexistence and residue-level data have been explicitly called out with mitigation ideas.

---

## Working Constraints (Based on Locked Decisions)

- Residue-first design
- Parallel v3/v4 lineages (no unification pressure yet)
- v4 starts narrow
- Strict training/inference separation
- RAG designed for, not built yet
- Aurora + object storage with equal accessibility

All work in the next 4 weeks must be consistent with these constraints.

---

## What I Will Do With Minimal Interruption

Once you confirm this plan (or give adjustments), I can work through the above deliverables with only occasional check-ins at natural review points (e.g., after first ADR drafts, after dimensional model v0.1).

Would you like me to begin **Week 1** work immediately (starting with refining ARCHITECTURE.md + creating the first two ADRs)?