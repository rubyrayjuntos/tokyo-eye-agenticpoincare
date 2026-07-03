# Implementation Plan: Cryptic Site Calibration System

## Overview

Build the calibration and scientific validation layer for the cryptic site discovery system. Proceeds bottom-up: schema extensions → signal provenance → prediction classifier → confidence calculator → calibration pipeline → benchmark dataset → integration.

## Tasks

- [x] 1. Extend schema with provenance and versioning fields
  - [x] 1.1 Add `signal_provenance_summary`, `provenance_gate`, and `heuristic_version` fields to `CrypticBindingSiteSpec` in `science/dtie/common/cryptic_payloads.py`
    - Add `signal_provenance_summary: dict[str, str]` with default_factory=dict
    - Add `provenance_gate: str` with default="hybrid-gated"
    - Add `heuristic_version: str` with default="1.0"
    - _Requirements: 3.2, 5.1_

  - [x] 1.2 Add `validation_class` field to `FalsifiablePrediction`
    - Add `validation_class: str` with default="model_self_consistency"
    - Add Pydantic validator constraining to {"independent_physics", "model_self_consistency", "experimental"}
    - _Requirements: 4.1_

  - [x] 1.3 Add `calibration_metadata` to `DEFAULT_SUCCESS_CRITERIA` in `agent/tools/cryptic/md_validator.py`
    - Add nested dict with keys: source, date, reference_sites, pulling_rate_nm_per_ns, force_constant_kJ_mol_nm2, notes
    - Mark all current thresholds as `"source": "provisional"`
    - _Requirements: 2.4, 2.5_

  - [x] 1.4 Write property tests for schema extensions (Properties 8, 11)
    - **Property 8: Validation Class Schema Constraint**
    - **Property 11: Heuristic Version Round-Trip**
    - **Validates: Requirements 4.1, 5.1**

- [x] 2. Implement Signal Provenance Registry
  - [x] 2.1 Create `agent/tools/cryptic/signal_provenance.py`
    - Define `SignalProvenance` enum (GNN_LEARNED, STRUCTURAL_PHYSICS)
    - Define `SIGNAL_PROVENANCE` mapping for all keys in DEFAULT_WEIGHTS
    - Implement `classify_provenance_gate(load_bearing_signals)` returning "GNN-gated", "physics-gated", or "hybrid-gated"
    - _Requirements: 3.1, 3.3, 3.4_

  - [x] 2.2 Wire provenance into mapper output
    - After scoring in `mapper.py`, call `classify_provenance_gate` with the active signals
    - Populate `signal_provenance_summary` and `provenance_gate` on the spec
    - _Requirements: 3.2, 3.5_

  - [x] 2.3 Write property tests for signal provenance (Properties 6, 7)
    - **Property 6: Signal Provenance Annotation Completeness**
    - **Property 7: Provenance Classification Correctness**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

- [x] 3. Implement Prediction Independence Classifier
  - [x] 3.1 Create `agent/tools/cryptic/prediction_classifier.py`
    - Define `ValidationClass` enum
    - Define GNN_TOOLS, MD_TOOLS, EXPERIMENTAL_TOOLS pattern sets
    - Implement `classify_prediction_independence(test_tool: str) -> ValidationClass`
    - _Requirements: 4.2, 4.3_

  - [x] 3.2 Wire classifier into MD validator
    - After creating falsifiable predictions in mapper or when evaluating in md_validator, auto-assign validation_class
    - _Requirements: 4.2, 4.3_

  - [x] 3.3 Write property test for prediction classification (Property 9)
    - **Property 9: Validation Class Auto-Assignment**
    - **Validates: Requirements 4.2, 4.3**

- [x] 4. Implement Confidence Calculator
  - [x] 4.1 Create `agent/tools/cryptic/confidence.py`
    - Implement `compute_site_validation_confidence(predictions)` that excludes model_self_consistency
    - Returns 0.0 when all predictions are self-consistency or when list is empty
    - _Requirements: 4.5_

  - [x] 4.2 Write property test for confidence exclusion (Property 10)
    - **Property 10: Confidence Excludes Self-Consistency Predictions**
    - **Validates: Requirements 4.5**

- [x] 5. Checkpoint - Verify schema extensions and classifiers
  - Run all property tests (P6-P11) and confirm they pass
  - Verify provenance classification returns correct results for the current DEFAULT_WEIGHTS
  - Verify prediction classifier correctly identifies GNN tools vs MD tools
  - Ask the user if questions arise

- [x] 6. Implement provisional threshold warning
  - [x] 6.1 Update `validate_cryptic_site_md()` to check calibration_metadata.source
    - When source == "provisional", add warning to result dict
    - Warning text: "MD threshold is provisional (uncalibrated) — results should be interpreted cautiously"
    - _Requirements: 2.6_

  - [x] 6.2 Write property test for provisional warning (Property 5)
    - **Property 5: Provisional Threshold Warning Emission**
    - **Validates: Requirements 2.6**

- [x] 7. Implement calibration report generation
  - [x] 7.1 Create `scripts/calibrate_cryptic.py` with report generation functions
    - Implement `compute_heuristic_metrics(predictions, ground_truth)` → precision/recall/ROC-AUC per site_type
    - Implement `compute_threshold_recommendation(positive_values, negative_values)` → optimal threshold via Youden's J
    - Implement `generate_calibration_report(heuristic_metrics, threshold_metrics, previous_report)` → structured JSON
    - Implement `detect_regressions(current, previous)` → list of regression flags
    - _Requirements: 1.4, 2.3, 6.3, 6.5_

  - [x] 7.2 Write property tests for calibration (Properties 1, 2, 3, 4, 12)
    - **Property 1: Calibration Report Structure Completeness**
    - **Property 2: Precision Threshold Flagging**
    - **Property 3: Threshold Selection Separates Positives from Negatives**
    - **Property 4: Calibration Metadata Structural Completeness**
    - **Property 12: Calibration Report Regression Detection**
    - **Validates: Requirements 1.4, 1.5, 1.6, 2.3, 2.4, 2.5, 6.3, 6.5**

- [x] 8. Curate benchmark dataset
  - [x] 8.1 Create `data/calibration/benchmark_cryptic_sites.json`
    - Include 20+ known cryptic sites from PocketMiner/CryptoSite literature
    - Include 10+ known-negative surface pockets
    - Include 3+ internal validated sites (KRAS, SPOP candidates)
    - Add metadata section with counts and curation date
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 8.2 Create benchmark loading utility in `scripts/calibrate_cryptic.py`
    - Implement `load_benchmark(path: str) -> list[BenchmarkSite]`
    - Validate structure and report missing fields
    - _Requirements: 1.1_

- [x] 9. Wire calibration into build system
  - [x] 9.1 Add `make calibrate-cryptic` target to Makefile
    - Run `PYTHONPATH=. python scripts/calibrate_cryptic.py --benchmark data/calibration/benchmark_cryptic_sites.json --output data/calibration/report.json`
    - _Requirements: 6.4_

  - [x] 9.2 Add database migration for new columns
    - Add `heuristic_version` and `provenance_gate` columns to `fact_cryptic_site`
    - _Requirements: 5.3_

- [x] 10. Integration - Wire provenance and confidence into ToolResult
  - [x] 10.1 Update `discover_and_validate_cryptic_site` in `tool.py`
    - Include `provenance_gate` in ToolResult.data
    - Include confidence score (excluding self-consistency) in ToolResult.data
    - When site_type has precision below threshold (from calibration), add warning to ToolResult
    - _Requirements: 1.6, 3.5, 4.5_

- [x] 11. Final checkpoint - Full calibration system verification
  - Run all 12 property tests and confirm they pass
  - Run `make calibrate-cryptic` against the benchmark dataset and verify report structure
  - Verify provenance flows through the full pipeline (mapper → tool result)
  - Verify provisional threshold warnings appear in MD validation results
  - Ask the user if questions arise

## Notes

- All tasks including property tests are required for comprehensive correctness
- Each task references specific requirements for traceability
- Checkpoints include specific validation goals for review
- Property tests use Hypothesis library (100+ iterations each)
- The benchmark dataset (task 8) requires literature research — initial version can start with PocketMiner's published 39-site set
- Calibration requires running real MD against known sites — task 7 builds the framework, actual calibration runs are a separate operational step
- The heuristic_version starts at "1.0" matching the current if/elif/else tree in mapper.py
