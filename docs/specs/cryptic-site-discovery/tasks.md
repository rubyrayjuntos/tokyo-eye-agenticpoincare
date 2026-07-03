# Implementation Plan: Cryptic Binding Site Discovery System

## Overview

Build the cryptic site discovery engine as a new agent tool family, following the existing Tokyo Eye patterns. Implementation proceeds bottom-up: schema → mapper → MD validator → fragment screener → orchestrator tool → persistence → registration. All property-based tests are required.

## v1 Scope Reminder

- Full schema (all site_types defined), Mapper + graph enrichment for `cryptic_wedge` and `structural_stent`
- Synchronous MD validation (dispatch + wait), SMD stub returning synthetic results
- Basic fragment screener with built-in fragment set
- 17 property-based tests via Hypothesis (100+ iterations each)

## Tasks

- [x] 1. Define core schema and Pydantic models
  - [x] 1.1 Create `science/dtie/common/cryptic_payloads.py` with all Pydantic models
    - Implement `SiteType` enum, `PharmacophoreAnchor`, `DisplacementTarget`, `BedrockNode`, `GeometricConstraints`, `FalsifiablePrediction`, `MDValidationState`, `CrypticBindingSiteSpec`
    - Implement `requires_md_validation()` and `get_md_protocol()` methods
    - Add `CrypticSitePayload` (with `ProvenanceContext`) for Normalizer integration
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 1.2 Write property tests for schema (Properties 1-3)
    - **Property 1: Schema Round-Trip Consistency**
    - **Property 2: Invalid Site Type Rejection**
    - **Property 3: Schema Backward Compatibility**
    - **Validates: Requirements 1.4, 1.5, 1.6**

- [x] 2. Implement the Mapper module
  - [x] 2.1 Create `agent/tools/cryptic/__init__.py` and `agent/tools/cryptic/mapper.py`
    - Implement `map_cryptic_site()` async function
    - Implement multi-signal scoring (uncertainty, cone_depth, dehydron density, betweenness, is_bridge)
    - Implement graph-based expansion from seed residues via `fact_graph_edge`
    - Implement residue classification (anchors vs displacement targets vs bedrock)
    - Implement `_infer_site_type()` heuristic returning `(SiteType, confidence)`
    - Prerequisites: Task 1.1 complete (schema models available)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 2.2 Write property tests for Mapper (Properties 4-7)
    - **Property 4: Mapper Threshold Filtering**
    - **Property 5: Mapper Graph Expansion Connectivity**
    - **Property 6: Multi-Signal Scoring Monotonicity**
    - **Property 7: Site Type Inference Determinism**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3. Implement graph topology enrichment
  - [x] 3.1 Add graph enrichment logic to `agent/tools/cryptic/mapper.py`
    - Compute betweenness_centrality, clustering_coefficient, degree, is_bridge for bedrock nodes
    - Use existing `fact_graph_node_metrics` when available, compute fresh via NetworkX when not
    - Populate `bedrock_clique` field on the spec
    - Prerequisites: Task 2.1 complete (mapper module exists)
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 3.2 Write property test for graph metrics (Property 8)
    - **Property 8: Graph Metrics Correctness**
    - **Validates: Requirements 3.1, 3.2**

- [x] 4. Checkpoint - Verify Mapper quality
  - Run all property tests (P1-P8) and confirm they pass
  - Manually verify Mapper output on 5WHA seeds ["A:78", "A:82"] produces sensible spec structure
  - Confirm site_type inference returns `cryptic_wedge` with confidence > 0.5 for the 5WHA example
  - Ask the user if questions arise

- [x] 5. Implement MD Validator module
  - [x] 5.1 Create `agent/tools/cryptic/md_validator.py`
    - Implement `validate_cryptic_site_md()` async function
    - Implement protocol dispatch mapping (site_type → protocol name)
    - Implement dispatch via `start_science_job()` + `wait_for_science_job()`
    - Implement default success criteria per protocol: SMD_three_phase → work < 25 kcal/mol, SMD_stent_stabilization → strain_delta < -1.5
    - Implement falsifiable prediction evaluation against MD results
    - Prerequisites: Task 1.1 complete (schema), science dispatch module available
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4_

  - [x] 5.2 Write property tests for MD gating (Properties 9-12)
    - **Property 9: MD Validation Gating**
    - **Property 10: Force MD Override**
    - **Property 11: Protocol Dispatch Mapping**
    - **Property 12: Falsifiable Prediction Evaluation Completeness**
    - **Validates: Requirements 4.1, 4.3, 4.4, 4.5, 5.1, 5.4, 10.3**

- [x] 6. Implement Fragment Screener module
  - [x] 6.1 Create `agent/tools/cryptic/fragment_screener.py`
    - Implement `screen_fragments()` async function
    - Implement geometric filtering (bounding sphere, hull padding)
    - Implement basic pharmacophore complementarity scoring
    - Implement result ranking by score descending
    - Include built-in fragment set (10-20 curated fragments for v1 testing)
    - Prerequisites: Task 1.1 complete (schema with GeometricConstraints)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 6.2 Write property tests for Fragment Screener (Properties 13-14)
    - **Property 13: Fragment Screening Monotonicity**
    - **Property 14: Fragment Result Ordering**
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 7. Checkpoint - Verify all components independently
  - Run all property tests (P1-P14) and confirm they pass
  - Verify MD gating logic works for all 5 site_types (True for wedge/stent/lid/clamp, fallback for strain_relief)
  - Verify fragment screener returns ordered results for a test spec
  - Ask the user if questions arise

- [x] 8. Implement orchestrator tool - core pipeline
  - [x] 8.1 Create `agent/tools/cryptic/tool.py` with `discover_and_validate_cryptic_site()`
    - Wire Mapper → MD Validator → Fragment Screener pipeline
    - Implement stage tracking (mapping, md_validation, fragment_screening, complete)
    - Return ToolResult with data containing: spec, md_result, fragments, stage, next_actions
    - Prerequisites: Tasks 2.1, 5.1, 6.1 complete (all components available)
    - _Requirements: 7.1, 7.2, 7.4, 8.1_

  - [x] 8.2 Implement error handling and next_actions in orchestrator
    - Handle all error cases: no DB, no GNN data, mapping failure, MD failure, screening failure
    - Implement contextual `next_actions` generation based on outcome
    - When MD fails: suggest different site_type, geometric constraint adjustment, skip MD
    - When < 3 fragments: suggest alternative site_types for the region
    - _Requirements: 7.5, 7.6, 8.2, 8.3, 8.4, 8.5_

  - [x] 8.3 Implement ViewportDirective emission in orchestrator
    - Emit highlight directives for discovered residues (pharmacophore anchors in green, displacement targets in orange, bedrock in blue)
    - Use site_type-appropriate labels and colors
    - _Requirements: 7.3_

  - [x] 8.4 Write property tests for orchestrator tool (Properties 15-16)
    - **Property 15: ToolResult Structural Completeness**
    - **Property 16: Contextual Next Actions**
    - **Validates: Requirements 7.2, 7.4, 8.1, 8.2, 8.3, 8.4, 8.5**

- [x] 9. Implement persistence and provenance
  - [x] 9.1 Add database migration for `fact_cryptic_site` table
    - Create migration file in `data/aurora/migrations/`
    - Define table: cryptic_site_id (UUID PK), structure_id, run_id, site_id, site_type, chain, residue_ids (TEXT[]), zone_classification, accessibility_mode, spec_json (JSONB), md_validation_status, md_job_id, fragments_screened, hypothesis_status, created_at, updated_at
    - Add UNIQUE(structure_id, site_id) constraint
    - Add indexes on structure_id, site_type, md_validation_status
    - _Requirements: 9.1, 9.3_

  - [x] 9.2 Add Normalizer write path for cryptic sites
    - Add `normalize_cryptic_site()` method to `data/normalizer/core.py`
    - Implement idempotent upsert (ON CONFLICT on structure_id + site_id → UPDATE)
    - Register as governed_asset with asset_type "cryptic_binding_site"
    - Wire persistence call into `tool.py` on successful discovery
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 9.3 Write property test for idempotent persistence (Property 17)
    - **Property 17: Idempotent Upsert**
    - **Validates: Requirements 9.3**

- [x] 10. Register tool with agent system
  - [x] 10.1 Register `discover_and_validate_cryptic_site` in agent tool definitions
    - Add ToolDefinition to DTIE_TOOLS or a new CRYPTIC_TOOLS list in `agent/llm/agents.py`
    - Add `"cryptic"` to the `research_scientist` specialist's owned_families
    - Define JSON Schema input: structure_id (required string), seed_residues (required array of strings), site_type (optional enum), force_md_validation (optional boolean, default false)
    - Wire handler to the async function in `agent/tools/cryptic/tool.py`
    - _Requirements: 7.1_

- [x] 11. Create SMD runner stub for Science Container
  - [x] 11.1 Create `science/dtie/cryptic/__init__.py` and `science/dtie/cryptic/smd_runner.py`
    - Implement CLI: `python -m science.dtie.cryptic.smd_runner --spec-json <path> --protocol <name>`
    - **Stub contract**: Always returns JSON on stdout with:
      - `{"success": true, "status": "passed", "protocol": "<name>", "work_kcal_mol": 12.4, "strain_delta": -2.1, "duration_ms": 500, "notes": "Stub SMD result for testing"}`
    - For `SMD_three_phase`: return work_kcal_mol between 8-20 (synthetic, below default threshold of 25)
    - For `SMD_stent_stabilization`: return strain_delta between -1.8 and -2.5 (below default threshold of -1.5)
    - Exit code 0 on success, non-zero on invalid args
    - _Requirements: 5.1, 5.2, 5.3, 11.1_

- [x] 12. Final checkpoint - Full integration verification
  - Run all 17 property tests and confirm they pass
  - Run the full tool end-to-end: `discover_and_validate_cryptic_site("5wha", ["A:78", "A:82"])` → verify ToolResult shape
  - Confirm ToolResult.data contains spec with site_type, md_result with "passed", fragments list, stage="complete", non-empty next_actions
  - Verify the tool is callable from the agent tool registry
  - Ask the user if questions arise

## Notes

- All tasks including property tests are required for comprehensive correctness
- Each task references specific requirements for traceability
- Checkpoints include specific validation goals for review
- Property tests use Hypothesis library (100+ iterations each)
- The SMD runner (task 11) is a stub in v1 — real OpenMM implementation is a separate effort
- Unit tests for edge cases are covered within implementation tasks (error handling paths)
- Some properties may be initially prototyped as simpler unit tests, then upgraded to full Hypothesis generators as the domain understanding matures
