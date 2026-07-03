# Implementation Plan: Phase Output Persistence

## Overview

Implement Tier 1 phase output persistence for the DTIE v5 pipeline, starting with the adapter protocol/registry, then wiring Phase 3, source-leak detection, and allosteric site persistence through the Normalizer with full provenance.

## Tasks

- [x] 1. Create the phase persistence protocol and registry
  - [x] 1.1 Create `science/dtie/common/phase_persistence.py` with `PhasePersistenceSpec`, `PhasePersistenceAdapter` protocol, `PERSISTENCE_ADAPTERS` registry, and `register_adapter` function
    - _Requirements: 4.1, 4.2, 4.4_
  - [x] 1.2 Write property test for adapter registry lookup
    - **Property 10: Orchestrator adapter invocation**
    - **Validates: Requirements 4.3**

- [x] 2. Create database migration for new fact tables
  - [x] 2.1 Create `data/aurora/migrations/032_phase_persistence_tier1.sql` with `fact_source_leak`, `fact_allosteric_site`, `fact_allosteric_site_residue` tables, indexes, and unique constraints
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 3. Define new Normalizer payload models
  - [x] 3.1 Add `SourceLeakResidue`, `SourceLeakPayload`, `AllostericSiteRecord`, `AllostericSitePayload` to `science/dtie/common/normalizer_payloads.py`
    - _Requirements: 5.1, 5.2_
  - [x] 3.2 Write property test for payload validation
    - **Property 7: Payload validation rejects invalid residue_ids**
    - **Validates: Requirements 5.3**

- [x] 4. Implement Normalizer write methods
  - [x] 4.1 Add `normalize_source_leaks(payload: SourceLeakPayload)` to `data/normalizer/core.py` with atomic writes, idempotent upserts, provenance enforcement, and audit logging
    - _Requirements: 2.2, 2.3, 6.5_
  - [x] 4.2 Add `normalize_allosteric_sites(payload: AllostericSitePayload)` to `data/normalizer/core.py` with atomic writes to both `fact_allosteric_site` and `fact_allosteric_site_residue`, idempotent upserts, and audit logging
    - _Requirements: 3.2, 3.3, 6.5_
  - [x] 4.3 Write property test for source-leak round-trip persistence
    - **Property 2: Source-leak persistence round-trip**
    - **Validates: Requirements 2.2**
  - [x] 4.4 Write property test for allosteric site round-trip persistence
    - **Property 3: Allosteric site persistence round-trip**
    - **Validates: Requirements 3.2, 3.3**
  - [x] 4.5 Write property test for idempotent upserts
    - **Property 6: Idempotent upserts**
    - **Validates: Requirements 6.5**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement persistence adapters
  - [x] 6.1 Create `science/dtie/common/adapters/source_leak_adapter.py` implementing `PhasePersistenceAdapter` for source-leak detection
    - _Requirements: 2.1, 2.3, 8.1_
  - [x] 6.2 Create `science/dtie/common/adapters/allosteric_site_adapter.py` implementing `PhasePersistenceAdapter` for allosteric site identification
    - _Requirements: 3.1, 3.3, 8.1_
  - [x] 6.3 Register Phase3Adapter, SourceLeakAdapter, and AllostericSiteAdapter in the persistence registry
    - _Requirements: 4.2_

- [x] 7. Wire persistence into the orchestrator
  - [x] 7.1 Add `enforce_governed_outputs: bool = True` to `PipelineConfig`
    - _Requirements: 7.1_
  - [x] 7.2 Implement `_persist_phase_result` method in `DTIEOrchestrator` that looks up the adapter registry, constructs ProvenanceContext with parent_run_id, calls the adapter, and sets metadata on PhaseResult
    - _Requirements: 1.1, 2.1, 3.1, 4.3, 8.1_
  - [x] 7.3 Wire Phase 3 persistence call after `_run_phase3()` completes successfully
    - _Requirements: 1.1, 1.3_
  - [x] 7.4 Wire source-leak persistence call after `_detect_source_leaks()` completes successfully
    - _Requirements: 2.1, 2.4_
  - [x] 7.5 Wire allosteric site persistence call after `_identify_allosteric_sites()` completes successfully
    - _Requirements: 3.1, 3.4_
  - [x] 7.6 Implement `_validate_persistence_requirements` and call it at the end of `run()`
    - _Requirements: 7.2, 7.4_
  - [x] 7.7 Write property test for metadata set on successful persistence
    - **Property 4: Metadata set on successful persistence**
    - **Validates: Requirements 1.3, 2.4, 3.4**
  - [x] 7.8 Write property test for Tier 1 failure on persistence error
    - **Property 5: Tier 1 failure on persistence error with enforcement**
    - **Validates: Requirements 1.4, 2.5, 3.5**
  - [x] 7.9 Write property test for dry-run mode
    - **Property 9: Dry-run mode allows completion without persistence**
    - **Validates: Requirements 7.3**

- [x] 8. Implement provenance linkage
  - [x] 8.1 Ensure all adapters set `parent_run_id` to the GNN inference run_id in ProvenanceContext
    - _Requirements: 8.1, 8.2_
  - [x] 8.2 Write property test for parent_run_id provenance linkage
    - **Property 8: Parent run_id provenance linkage**
    - **Validates: Requirements 8.1, 8.2**

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks including property tests are required
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis (minimum 100 iterations)
- Unit tests validate specific examples and edge cases
- The existing Phase3Adapter in `science/dtie/common/adapters.py` should be refactored to conform to the new `PhasePersistenceAdapter` protocol
