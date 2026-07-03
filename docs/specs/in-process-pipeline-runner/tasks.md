# Implementation Plan: In-Process Pipeline Runner

## Overview

Implement the `InferenceEngine` class and `flush_stale` utility as a new module `scripts/pipeline_runner.py`, composing existing V5GNNRunner and DTIEOrchestrator with checkpoint version hashing and stale-data management. Python with Hypothesis for property-based testing.

## Tasks

- [x] 1. Create InferenceEngine core class
  - [x] 1.1 Create `scripts/pipeline_runner.py` with `InferenceEngine` class skeleton
    - Constructor accepting checkpoint_path, device, db, force_overwrite
    - Implement `_compute_checkpoint_hash()` using hashlib.sha256
    - Expose `checkpoint_version_hash` property
    - _Requirements: 1.1, 1.3_

  - [ ]* 1.2 Write property test for checkpoint hash determinism
    - **Property 1: Checkpoint hash determinism**
    - **Validates: Requirements 1.3**

  - [x] 1.3 Implement `InferenceEngine.run()` method
    - Open async DB connection (or use injected db)
    - Build graph via GraphBuilder
    - Run V5GNNRunner inference
    - Pass GNNInferenceResult to DTIEOrchestrator
    - Populate ProvenanceContext with checkpoint_version_hash and checkpoint_uri
    - Return PipelineResult with checkpoint metadata
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.4, 4.3, 5.1, 5.2_

  - [ ]* 1.4 Write property test for checkpoint metadata propagation
    - **Property 2: Checkpoint metadata propagation**
    - **Validates: Requirements 3.1, 3.2, 3.4**

  - [ ]* 1.5 Write property test for run ID uniqueness
    - **Property 5: Run ID uniqueness**
    - **Validates: Requirements 4.3**

- [x] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implement flush_stale utility
  - [x] 3.1 Implement `flush_stale()` function in `scripts/pipeline_runner.py`
    - Query provenance_run for stale run_ids (hash mismatch)
    - Delete from fact_source_leak, fact_allosteric_site_residue, fact_allosteric_site, fact_phase3_persistence, governed_asset
    - Return FlushResult with per-table counts
    - Raise ValueError if structure_id is None
    - _Requirements: 7.1, 7.2, 7.4_

  - [ ]* 3.2 Write property test for flush stale correctness
    - **Property 8: Flush stale correctness**
    - **Validates: Requirements 7.1, 7.2**

  - [x] 3.3 Implement force_overwrite logic in `InferenceEngine.run()`
    - When force_overwrite=True, call flush_stale before pipeline execution
    - Wire flush_before_run CLI flag
    - _Requirements: 4.4, 7.3_

  - [ ]* 3.4 Write property test for force overwrite
    - **Property 9: Force overwrite removes prior results**
    - **Validates: Requirements 4.4**

- [x] 4. Implement CLI interface
  - [x] 4.1 Add argparse CLI to `scripts/pipeline_runner.py`
    - Arguments: --structure (required), --checkpoint, --device, --force-overwrite, --source-leak-only, --flush-before-run
    - JSON output on success (run_id, structure_id, checkpoint_version_hash, num_nodes, phases_run, assets_created)
    - JSON error output on failure with non-zero exit
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 4.2 Write property test for CLI argument parsing
    - **Property 6: CLI argument parsing**
    - **Validates: Requirements 6.1**

  - [ ]* 4.3 Write property test for CLI output completeness
    - **Property 7: CLI output completeness**
    - **Validates: Requirements 6.2**

- [x] 5. Implement DB connection management and error handling
  - [x] 5.1 Implement connection lifecycle in `InferenceEngine.run()`
    - Read DATABASE_URL from env with fallback
    - Open async connection, commit on success, close on exit
    - Credential redaction in ConnectionError messages
    - Support injected db parameter for testing
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 5.2 Write unit tests for error handling
    - Test FileNotFoundError for missing checkpoint
    - Test ConnectionError with credential redaction
    - Test ValueError for flush_stale without structure_id
    - _Requirements: 1.4, 5.3, 7.4_

- [x] 6. Integration wiring and pipeline result validation
  - [x] 6.1 Wire source_leak_only config propagation
    - Ensure PipelineConfig.source_leak_only maps correctly from InferenceEngine parameter
    - Verify phase_results only contains expected phases
    - _Requirements: 2.3_

  - [ ]* 6.2 Write property test for pipeline result completeness
    - **Property 3: Pipeline result completeness matches config**
    - **Validates: Requirements 2.2, 2.3**

  - [ ]* 6.3 Write property test for idempotent upsert safety
    - **Property 4: Idempotent upsert safety**
    - **Validates: Requirements 4.1, 4.2**

- [x] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The InferenceEngine composes existing V5GNNRunner and DTIEOrchestrator — no reimplementation
- Property tests use Hypothesis with mock DB (existing DatabaseConnection protocol)
- Integration tests require docker-compose PostgreSQL and real checkpoints
