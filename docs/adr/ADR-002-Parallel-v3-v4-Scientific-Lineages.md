# ADR-002: Parallel v3 and v4 Scientific Lineages

**Status:** Accepted  
**Date:** May 27, 2026  
**Deciders:** Ray  
**Related ADRs:** ADR-001 (Residue Anchor)

---

## Context

The codebase contains two distinct generations of the core GNN + DTIE science:

- **v3**: Mature, full-pipeline implementation (complete phases 1–6d, broad drug discovery coverage). Uses `GOSPConeMapper` (v3) with more Euclidean post-processing after the hyperbolic lift.
- **v4**: More advanced GNN architecture (`Gnnv4`). Keeps significantly more computation inside hyperbolic/tangent space. Produces native 2D Poincaré disc projections (`hyp_projections`), exposes `x_hyp` and `x_routed_hyp`, and has different uncertainty head inputs. Currently narrower in pipeline scope.

These two versions have non-trivial differences in:
- Internal data flow (especially MoE routing and uncertainty)
- Output contracts (projection heads, exposed hyperbolic state)
- Training requirements (v4 generally requires retraining)

Forcing early unification in the data model would either slow down v4 progress or compromise the maturity of v3.

## Decision

We will maintain **two parallel scientific lineages** in the governed data model for the foreseeable future (minimum 6–12 months).

### Implementation Guidelines

- Differentiation will occur primarily through:
  - `model_version` / `pipeline_version` fields
  - Strong, structured provenance records
  - Explicit `source_type` where relevant
- The core dimensional model (especially `dim_residue`) remains shared.
- Different fact tables or extension fields may be used when contracts diverge significantly (e.g., v4 native hyperbolic outputs).
- Consumers (agents, visualizers, future RAG systems) are expected to be version-aware when necessary.
- Convergence will only be pursued later when there is clear scientific justification and after real usage data exists.

## Consequences

### Positive
- Allows v4 development to proceed at full speed without being constrained by v3 compatibility.
- Preserves the production maturity and breadth of the v3 pipeline.
- Reduces risk of premature abstraction or incorrect unification.
- Aligns with observed reality across the source repositories.

### Negative / Trade-offs
- Some duplication in data structures and consumer logic in the short-to-medium term.
- Consumers must handle version differences (mitigated by good provenance and documentation).
- Slightly higher long-term maintenance cost until (and unless) convergence occurs.

### Neutral
- The residue-centric anchor model (ADR-001) applies equally to both lineages.

## Alternatives Considered

- **Single unified model now**: Rejected. Too high risk of slowing down the better science (v4) or compromising the more complete pipeline (v3).
- **Completely separate data models**: Rejected. Would lose the benefits of shared dimensions, shared provenance spine, and consistent governance.

## References

- `data/DATA_FIRST_CONSOLIDATION_ROADMAP.md` (Section 5, Decision 2)
- Deep Audit findings in `docs/audit/DEEP_AUDIT_PHASE_0.1.md` (GNN and EvidentialHead differences)
- Source code comparisons between `Gnnv3.py` and `Gnnv4.py`