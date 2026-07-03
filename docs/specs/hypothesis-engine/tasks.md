# Implementation Plan: Hypothesis Engine

## Overview

Implement the hypothesis engine in Python, following the existing project patterns: Pydantic models for validation, Normalizer for writes, ToolResult for tool responses, and Hypothesis (library) for property-based testing.

## Tasks

- [x] 1. Create data models and confidence logic
  - [x] 1.1 Create `agent/tools/hypothesis/__init__.py` and `agent/tools/hypothesis/models.py`
    - Define `HypothesisStatus` enum, `Prediction`, `Evidence`, `Hypothesis` Pydantic models
    - Include JSON serialization/deserialization methods
    - _Requirements: 1.1, 8.1, 8.2, 8.3_

  - [x] 1.2 Create `agent/tools/hypothesis/confidence.py`
    - Implement `calculate_confidence(evidence)` — weighted ratio formula
    - Implement `apply_decay(confidence, days_stale)` — 10% per day toward 0.5
    - Implement `determine_status(confidence, evidence_count, current_status)` — threshold transitions
    - _Requirements: 3.3, 4.1, 4.2, 4.3, 7.1, 7.2, 7.3_

  - [x] 1.3 Create `agent/tools/hypothesis/threshold.py`
    - Implement `evaluate_threshold(threshold_expr, result_value)` — parse and evaluate comparison expressions
    - Support operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
    - _Requirements: 2.2_

  - [x] 1.4 Write property tests for confidence calculation (Property 2)
    - **Property 2: Confidence calculation formula**
    - **Validates: Requirements 3.3, 7.1, 7.2, 7.3**

  - [x] 1.5 Write property tests for status transitions (Property 3)
    - **Property 3: Status transitions based on confidence and evidence count**
    - **Validates: Requirements 4.1, 4.2**

  - [x] 1.6 Write property tests for confidence decay (Property 4)
    - **Property 4: Confidence decay toward 0.5 over time**
    - **Validates: Requirements 4.3**

  - [x] 1.7 Write property tests for threshold evaluation (Property 5)
    - **Property 5: Threshold evaluation correctness**
    - **Validates: Requirements 2.2**

  - [x] 1.8 Write property tests for serialization round-trip (Property 8)
    - **Property 8: Serialization round-trip**
    - **Validates: Requirements 8.1, 8.2**

  - [x] 1.9 Write property tests for creation defaults and falsifiability (Properties 1, 9)
    - **Property 1: Falsifiability guardrail rejects hypotheses without predictions**
    - **Property 9: Hypothesis creation produces correct defaults**
    - **Validates: Requirements 1.1, 1.2**

- [x] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Create database migration and normalizer path
  - [x] 3.1 Create `data/aurora/migrations/032_hypothesis_engine.sql`
    - Define `hypothesis`, `hypothesis_prediction`, `hypothesis_evidence` tables
    - Add indexes for structure_id, status, hypothesis_id lookups
    - Follow existing migration conventions (comments, requirements refs)
    - _Requirements: 1.3_

  - [x] 3.2 Add `normalize_hypothesis` method to `data/normalizer/core.py`
    - Follow the `normalize_graph_topology` pattern: provenance → validate → transaction → upsert → register assets → commit → audit
    - Handle hypothesis creation, prediction upsert, evidence insertion
    - _Requirements: 1.3, 3.1_

  - [x] 3.3 Add hypothesis payload models to `science/dtie/common/normalizer_payloads.py`
    - Define `HypothesisPayload` with `ProvenanceContext`, hypothesis fields, nested predictions
    - Define `EvidencePayload` for adding evidence
    - _Requirements: 1.3, 3.1_

- [x] 4. Implement hypothesis tools
  - [x] 4.1 Create `agent/tools/hypothesis/tools.py` with `propose_hypothesis`
    - Validate falsifiability (at least one prediction)
    - Generate hypothesis_id, set defaults (status=proposed, confidence=0.5)
    - Write through normalizer
    - Return ToolResult with hypothesis_id
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 4.2 Implement `test_hypothesis` tool
    - Load hypothesis and predictions from DB
    - Transition status to `gathering`
    - For each prediction: call the referenced tool, evaluate threshold, record result
    - Create evidence from results, recalculate confidence, update status
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 4.4_

  - [x] 4.3 Implement `get_hypotheses` tool
    - Query by structure_id and/or status
    - Include evidence counts and prediction pass/fail summary
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 4.4 Implement `add_evidence` tool
    - Store evidence record
    - Recalculate confidence and update status
    - _Requirements: 3.1, 3.2_

  - [x] 4.5 Implement `evaluate_confidence` tool
    - Reload all evidence, recalculate, apply decay if stale, update status
    - _Requirements: 4.3, 7.1_

  - [x] 4.6 Write property tests for hypothesis retrieval (Property 6)
    - **Property 6: Hypothesis retrieval with filtering**
    - **Validates: Requirements 5.1, 5.2**

- [x] 5. Wire into agent coordinator
  - [x] 5.1 Add `HYPOTHESIS_TOOLS` definitions to `agent/llm/agents.py`
    - Define ToolDefinition entries for all 5 hypothesis tools
    - Follow existing pattern (name, description, parameters JSON schema, handler=None)
    - _Requirements: 1.1, 2.1, 3.1, 5.1_

  - [x] 5.2 Wire hypothesis tool handlers in `create_coordinator`
    - Import hypothesis tools, add to handler map with db injection
    - _Requirements: 1.1, 2.1, 3.1, 5.1_

  - [x] 5.3 Register hypothesis tools in `agent/tools/diagnostics.py`
    - Add hypothesis tools to the startup diagnostic check
    - _Requirements: 1.1_

- [x] 6. Implement contradiction detection
  - [x] 6.1 Add contradiction check hook to normalizer or pipeline completion
    - After new pipeline results are written for a structure, check active hypotheses
    - Re-evaluate predictions against new data
    - Add contradicting evidence if a previously-passing prediction now fails
    - _Requirements: 6.1, 6.2_

  - [x] 6.2 Write property test for contradiction detection (Property 7)
    - **Property 7: Contradiction detection on prediction flip**
    - **Validates: Requirements 6.2**

- [x] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required including property tests
- All writes go through the Normalizer — no direct DB inserts in tool code
- The hypothesis engine calls existing tools (get_graph_metrics, compare_graphs, etc.) to test predictions
- Property tests use the Hypothesis library (already in the project)
- Python is the implementation language (matching the existing codebase)
