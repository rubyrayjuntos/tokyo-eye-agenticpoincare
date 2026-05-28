# ADR-001: Residue as Primary Granular Anchor

**Status:** Accepted  
**Date:** May 27, 2026  
**Deciders:** Ray  
**Related ADRs:** ADR-002 (Parallel v3/v4 Lineages)

---

## Context

The project involves multiple layers of scientific computation on protein structures (GNN embeddings, topological analysis, dehydron detection, uncertainty quantification, etc.). Different computations naturally live at different levels of granularity:

- Some data is truly structural (e.g., overall metadata, global properties).
- Much of the meaningful scientific signal lives at the **residue** level (dehydrons, cone depth, per-residue embeddings, uncertainty, allosteric effects).
- Some data lives at the **atom** level (precise coordinates, certain physical calculations).
- Higher-level concepts (allosteric sites, "source leaks", glue sites, uncertainty hotspots) are often aggregations or virtual concepts built on top of residues.

Previous codebases had inconsistent anchoring — some data was only attached at `structure_id`, making residue-level analysis and cross-structure comparison difficult.

## Decision

We will adopt a **dimensional model** where `dim_residue` is the **primary granular anchor** for the majority of scientific facts in the governed data layer.

### Specifics

- `dim_structure` remains the top-level dimension.
- `dim_chain` exists as an intermediate dimension.
- `dim_residue` (with `residue_id` as the stable key) is the main join point for:
  - GNN node outputs and embeddings
  - Per-residue features (ρ, SASA, secondary structure, etc.)
  - Phase results (witness embeddings, persistence, flux, etc.)
  - Uncertainty metrics
  - Most analysis outputs
- `dim_atom` is supported and linked to `dim_residue`, but is secondary. It is used when atomic precision is genuinely required.
- A lightweight `dim_site` (or virtual/site concept) will be introduced for higher-level biological concepts (allosteric sites, high-uncertainty regions, etc.). This is **not** a physical grain like residue/atom but a derived one that can reference one or more residues.

All new data types should default to joining at the `residue` level unless there is a strong reason to attach at a different grain.

## Consequences

### Positive
- Enables clean per-residue and cross-structure analysis (critical for the scientific goals).
- Matches the natural shape of the GNN outputs and most DTIE phases.
- Provides a clear, consistent pattern for future extensions.
- Aligns with the best thinking found in the `tokyo-eye-data` source (explicit `dim_residue` / `dim_atom` dimensions and residue-keyed fact tables).

### Negative / Trade-offs
- Slightly more complex schema than a purely `structure_id`-centric model.
- Some legacy data that was only attached at structure level will need migration or bridging logic.
- Requires discipline when new data types are added (people must think about the right grain).

### Neutral
- Object storage and large binary artifacts are unaffected (they will still be referenced from the governed layer).

## Alternatives Considered

- **Structure-only model**: Rejected. Too coarse for the actual scientific work.
- **Atom as primary**: Rejected. Most analysis (including GNN node features and topological methods) operates at residue level. Atomic data is important but secondary.
- **Fully flexible / no preferred grain**: Rejected. Leads to inconsistent modeling and harder querying over time.

## References

- `data/DATA_FIRST_CONSOLIDATION_ROADMAP.md` (Section 5, Decision 1)
- `data/ARCHITECTURE.md`
- Insights from `/home/rswan/Documents/tokyo-eye-data/docs/data-lineage-map.md` and dimensional model thinking.