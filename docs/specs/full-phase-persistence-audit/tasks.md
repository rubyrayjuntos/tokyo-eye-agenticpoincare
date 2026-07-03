# Implementation Plan: Full Phase Persistence Audit

## Overview

Closes the persistence gap for phases 2, 3.5, 4, 5, and 6 by creating adapters, payload models, Normalizer methods, database tables, orchestrator wiring, and hydration endpoint queries. Follows the proven pattern from the source-leak and allosteric site adapters.

## Tasks

- [x] 1. Database migration for new fact tables
  - [x] 1.1 Create migration `039_phase_persistence_tier2.sql` with tables for phases 2, 3.5, 4, 5, 6
    - `fact_phase2_vulnerability` with unique index on (run_id, residue_id)
    - `fact_topological_lift` with unique index on (run_id, site_index)
    - `fact_resistance_pathway` with unique index on (run_id, source_node, target_node)
    - `fact_resistance_spectral` with unique index on (run_id)
    - `fact_pharmacophore` with unique index on (run_id, pocket_index)
    - `fact_drug_candidate` with unique index on (run_id, pocket_index)
    - _Requirements: 1.2, 2.2, 3.2, 4.2, 5.2_

- [x] 2. Pydantic payload models
  - [x] 2.1 Add Phase2VulnerabilityPayload, TopologicalLiftPayload, ResistancePathwayPayload, PharmacophorePayload, and DrugCandidatePayload to `normalizer_payloads.py`
    - Each with ProvenanceContext, computed_at, and phase-specific fields
    - _Requirements: 1.2, 2.2, 3.2, 4.2, 5.2_

- [x] 3. Normalizer methods
  - [x] 3.1 Add `normalize_phase2_vulnerability()` to Normalizer
    - Validate residue_ids, atomic upsert into fact_phase2_vulnerability, register governed assets
    - _Requirements: 1.1, 1.2_
  - [x] 3.2 Add `normalize_topological_lift()` to Normalizer
    - Atomic upsert into fact_topological_lift, register governed assets
    - _Requirements: 2.1, 2.2_
  - [x] 3.3 Add `normalize_resistance_pathways()` to Normalizer
    - Atomic upsert into fact_resistance_pathway + fact_resistance_spectral, register governed assets
    - _Requirements: 3.1, 3.2_
  - [x] 3.4 Add `normalize_pharmacophores()` to Normalizer
    - Atomic upsert into fact_pharmacophore, register governed assets
    - _Requirements: 4.1, 4.2_
  - [x] 3.5 Add `normalize_drug_candidates()` to Normalizer
    - Atomic upsert into fact_drug_candidate, register governed assets
    - _Requirements: 5.1, 5.2_

- [x] 4. Persistence adapters
  - [x] 4.1 Create `phase2_vulnerability_adapter.py` implementing PhasePersistenceAdapter
    - Extract doorways from PhaseResult outputs, build residue_ids from structure_id:chain:index
    - _Requirements: 1.1, 1.2_
  - [x] 4.2 Create `phase35_lift_adapter.py` implementing PhasePersistenceAdapter
    - Extract lifted_sites from PhaseResult outputs
    - _Requirements: 2.1, 2.2_
  - [x] 4.3 Create `phase4_resistance_adapter.py` implementing PhasePersistenceAdapter
    - Extract pathways and spectral data from PhaseResult outputs
    - _Requirements: 3.1, 3.2_
  - [x] 4.4 Create `phase5_pharmacophore_adapter.py` implementing PhasePersistenceAdapter
    - Extract pharmacophores from PhaseResult outputs
    - _Requirements: 4.1, 4.2_
  - [x] 4.5 Create `phase6_drug_discovery_adapter.py` implementing PhasePersistenceAdapter
    - Extract scored_pockets from PhaseResult outputs
    - _Requirements: 5.1, 5.2_

- [x] 5. Registry and orchestrator wiring
  - [x] 5.1 Update `ensure_adapters_registered()` in `phase_persistence.py` to register all 5 new adapters
    - _Requirements: 6.2_
  - [x] 5.2 Add `_persist_phase_result` calls in orchestrator after phases 2, 3.5, 4, 5, and 6
    - Follow the same pattern as source-leak and allosteric site persistence calls
    - _Requirements: 6.1, 1.1, 2.1, 3.1, 4.1, 5.1_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Property tests for round-trip persistence
  - [x] 7.1 Write property test for Phase 2 vulnerability round-trip
    - **Property 3: Phase 2 vulnerability round-trip**
    - **Validates: Requirements 1.2**
  - [x] 7.2 Write property test for topological lift round-trip
    - **Property 4: Topological lift round-trip**
    - **Validates: Requirements 2.2**
  - [x] 7.3 Write property test for resistance pathway round-trip
    - **Property 5: Resistance pathway round-trip**
    - **Validates: Requirements 3.2**
  - [x] 7.4 Write property test for pharmacophore round-trip
    - **Property 6: Pharmacophore round-trip**
    - **Validates: Requirements 4.2**
  - [x] 7.5 Write property test for drug candidate round-trip
    - **Property 7: Drug candidate round-trip**
    - **Validates: Requirements 5.2**

- [x] 8. Property tests for orchestrator behavior
  - [x] 8.1 Write property test for universal persistence invocation
    - **Property 1: Universal persistence invocation**
    - **Validates: Requirements 1.1, 2.1, 3.1, 4.1, 5.1, 6.1**
  - [x] 8.2 Write property test for metadata flag
    - **Property 2: Metadata flag set on successful persistence**
    - **Validates: Requirements 1.3, 2.3, 3.3, 4.3, 5.3**
  - [x] 8.3 Write property test for idempotent upserts
    - **Property 8: Idempotent upserts across all new tables**
    - **Validates: Requirements 6.1**

- [x] 9. Hydration endpoint updates
  - [x] 9.1 Add queries for Phase 2, Phase 4, Phase 5, and Phase 6 data to the hydration endpoint
    - Use `asyncio.gather` with `return_exceptions=True` for error isolation
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 10. Hydration endpoint property test
  - [x] 10.1 Write property test for hydration completeness and error isolation
    - **Property 9: Hydration endpoint returns persisted data**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

- [x] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required (comprehensive testing from start)
- Each task references specific requirements for traceability
- The existing `_persist_phase_result` method handles all the metadata-setting and error-handling logic — adapters just need to build payloads and call the normalizer
- Phase 3.5 is Tier 2 (won't fail the pipeline if persistence fails) since it's derived from Phase 3 data
- All other new phases are Tier 1 (will fail the pipeline if `enforce_governed_outputs` is True)
