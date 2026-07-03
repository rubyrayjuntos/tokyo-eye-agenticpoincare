# Requirements Document

## Introduction

The DTIE v5 pipeline runs 8+ phases that each produce rich scientific outputs, but only a subset of these outputs are persisted to the database. The pipeline reports success for phases that compute data in-memory but never write it to governed fact tables. This creates a "data black hole" where computed results are lost after the pipeline response, making the frontend tool panels empty and scientific discoveries non-reproducible. This spec addresses the complete persistence gap by auditing every phase output and ensuring all computed data flows to the database.

## Glossary

- **Phase_Output**: The computed result of a pipeline phase (e.g., Phase 2 vulnerability doorways, Phase 4 resistance pathways, Phase 5 pharmacophores, Phase 6 drug candidates)
- **Persistence_Adapter**: A component that converts phase outputs into Normalizer payloads for governed database writes
- **Fact_Table**: A database table storing computed scientific results with provenance linkage
- **Governed_Write**: A database write that goes through the Normalizer with validation, provenance, and audit trail
- **Phase_2_Vulnerability**: Identifies "doorway" residues with high epistemic uncertainty at the boundary of the hyperbolic cone
- **Phase_35_Topological_Lift**: Lifts allosteric sites from 2D disc to 3D ball coordinates
- **Phase_4_Resistance**: Maps resistance pathways through the contact graph using spectral analysis
- **Phase_5_Pharmacophore**: Identifies druggable pharmacophore features from cone geometry
- **Phase_6_Drug_Discovery**: Scores pockets, runs ADMET filtering, identifies state-selective candidates

## Requirements

### Requirement 1: Phase 2 Vulnerability Persistence

**User Story:** As a researcher, I want Phase 2 vulnerability doorway results persisted, so that I can query which residues were identified as structural vulnerabilities across runs.

#### Acceptance Criteria

1. WHEN Phase 2 completes successfully, THE Orchestrator SHALL persist doorway residues, epistemic medians, and depth thresholds through the Normalizer
2. THE System SHALL store Phase 2 outputs in a `fact_phase2_vulnerability` table with residue_id, doorway_score, epistemic_uncertainty, depth_threshold, and run_id
3. WHEN Phase 2 persistence succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` on the PhaseResult

### Requirement 2: Phase 3.5 Topological Lift Persistence

**User Story:** As a researcher, I want topological lift results persisted, so that lifted allosteric site coordinates are queryable.

#### Acceptance Criteria

1. WHEN Phase 3.5 completes successfully, THE Orchestrator SHALL persist lifted site coordinates through the Normalizer
2. THE System SHALL store Phase 3.5 outputs in a `fact_topological_lift` table with site_id, lifted_x, lifted_y, lifted_z, method, and run_id
3. WHEN Phase 3.5 persistence succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` on the PhaseResult

### Requirement 3: Phase 4 Resistance Pathway Persistence

**User Story:** As a researcher, I want resistance pathway data persisted, so that spectral analysis results and pathway residues are queryable.

#### Acceptance Criteria

1. WHEN Phase 4 completes successfully, THE Orchestrator SHALL persist resistance pathways and spectral data through the Normalizer
2. THE System SHALL store Phase 4 outputs in a `fact_resistance_pathway` table with pathway_id, residue_ids (array), spectral_score, graph_nodes, graph_edges, and run_id
3. WHEN Phase 4 persistence succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` on the PhaseResult

### Requirement 4: Phase 5 Pharmacophore Persistence

**User Story:** As a drug discovery scientist, I want pharmacophore predictions persisted, so that druggable features are queryable across structures.

#### Acceptance Criteria

1. WHEN Phase 5 completes successfully, THE Orchestrator SHALL persist pharmacophore features through the Normalizer
2. THE System SHALL store Phase 5 outputs in a `fact_pharmacophore` table with pharmacophore_id, residue_ids, druggability_score, feature_type, and run_id
3. WHEN Phase 5 persistence succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` on the PhaseResult

### Requirement 5: Phase 6 Drug Discovery Persistence

**User Story:** As a drug discovery scientist, I want scored pockets and drug candidates persisted, so that I can compare candidates across structures and track how predictions evolve.

#### Acceptance Criteria

1. WHEN Phase 6 completes successfully, THE Orchestrator SHALL persist scored pockets, ADMET results, and state-selective candidates through the Normalizer
2. THE System SHALL store Phase 6 outputs in a `fact_drug_candidate` table with candidate_id, pocket_residues, admet_score, state_selectivity_score, and run_id
3. WHEN Phase 6 persistence succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` on the PhaseResult

### Requirement 6: Universal Phase Persistence Wiring

**User Story:** As a developer, I want every phase that produces scientific outputs to have its persistence wired into the orchestrator, so that no computed data is silently discarded.

#### Acceptance Criteria

1. THE Orchestrator SHALL call `_persist_phase_result` for every phase that completes successfully and has a registered persistence adapter
2. THE System SHALL register persistence adapters for phases 2, 3.5, 4, 5, and 6 in the adapter registry
3. WHEN `enforce_governed_outputs` is True, THE Orchestrator SHALL mark the pipeline as degraded if any Tier 1 phase fails to persist
4. THE System SHALL log a warning for any phase that produces outputs but has no registered persistence adapter

### Requirement 7: Hydration Endpoint Completeness

**User Story:** As a frontend developer, I want the hydration endpoint to return all persisted phase data, so that tool panels can display complete analysis results.

#### Acceptance Criteria

1. THE Hydration endpoint SHALL query and return Phase 2 vulnerability data when available
2. THE Hydration endpoint SHALL query and return Phase 4 resistance pathway data when available
3. THE Hydration endpoint SHALL query and return Phase 5 pharmacophore data when available
4. THE Hydration endpoint SHALL query and return Phase 6 drug candidate data when available
5. THE Hydration endpoint SHALL use `return_exceptions=True` in asyncio.gather so individual query failures don't block other data
