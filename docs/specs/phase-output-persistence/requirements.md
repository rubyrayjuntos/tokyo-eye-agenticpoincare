# Requirements Document

## Introduction

The DTIE v5 pipeline orchestrator produces rich scientific outputs across 13+ phases, but only GNN node embeddings are reliably persisted to the governed data layer. All other phase results (source-leak detection, allosteric sites, Phase 3 persistence, pharmacophore models) live ephemerally in memory and are discarded after the API response. This removes true provenance of a run and makes scientific discoveries non-reproducible and non-queryable. This spec addresses the immediate priorities: wiring Phase 3 persistence, and creating persistence for source-leak detection and allosteric site identification.

## Glossary

- **Orchestrator**: The `DTIEOrchestrator` class in `science/dtie/v5/orchestrator/pipeline.py` that runs the full DTIE pipeline
- **Normalizer**: The single governed write path (`data/normalizer/core.py`) that validates, persists, and audits all data writes
- **PhaseResult**: A dataclass holding the outputs of a single pipeline phase (currently ephemeral)
- **ProvenanceContext**: Required metadata (run_id, structure_id, model_version, etc.) that must accompany every Normalizer write
- **Tier_1_Phase**: A phase whose outputs represent core scientific claims and must be persisted with full provenance
- **PhasePersistenceAdapter**: A protocol that converts a PhaseResult into one or more Normalizer payloads
- **Source_Leak**: A residue identified by the pipeline as having anomalously high epistemic uncertainty and cone depth, indicating potential information leakage from training data
- **Allosteric_Site**: A spatial cluster of source-leak residues that may represent a functionally significant allosteric pocket
- **Governed_Asset**: A data record written through the Normalizer with full provenance, audit trail, and idempotent upsert semantics

## Requirements

### Requirement 1: Phase 3 Persistence Wiring

**User Story:** As a computational biologist, I want Phase 3 (witness persistence) outputs to be automatically persisted when the pipeline runs, so that topological analysis results are queryable and traceable to their producing run.

#### Acceptance Criteria

1. WHEN the orchestrator completes Phase 3 successfully, THE Orchestrator SHALL call the Phase3 persistence adapter to write results through the Normalizer
2. WHEN Phase 3 persistence data is written, THE Normalizer SHALL store barcodes, max_alpha, residue contributions, and curvature in `fact_phase3_persistence` with a valid `run_id` reference
3. WHEN Phase 3 persistence write succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` and `metadata["phase3_id"]` on the PhaseResult
4. IF Phase 3 persistence write fails and `enforce_governed_outputs` is True, THEN THE Orchestrator SHALL mark the phase as failed and include the error in warnings

### Requirement 2: Source-Leak Detection Persistence

**User Story:** As a researcher, I want source-leak detection results persisted with full provenance, so that I can query which residues were flagged as source leaks across multiple runs and structures.

#### Acceptance Criteria

1. WHEN source-leak detection completes successfully, THE Orchestrator SHALL persist each identified source-leak residue through the Normalizer
2. THE Normalizer SHALL store source-leak candidates in a dedicated `fact_source_leak` table with residue_id, epistemic_uncertainty, cone_depth, leak_score, and run_id
3. WHEN persisting source-leak results, THE Normalizer SHALL link each record to the parent GNN run via `parent_run_id` in the ProvenanceContext
4. WHEN source-leak persistence write succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` and `metadata["source_leak_count"]` on the PhaseResult
5. IF source-leak persistence write fails and `enforce_governed_outputs` is True, THEN THE Orchestrator SHALL mark the phase as failed

### Requirement 3: Allosteric Site Persistence

**User Story:** As a drug discovery scientist, I want allosteric site predictions persisted, so that I can compare predicted allosteric pockets across structures and track how predictions evolve with model improvements.

#### Acceptance Criteria

1. WHEN allosteric site identification completes successfully, THE Orchestrator SHALL persist each identified site through the Normalizer
2. THE Normalizer SHALL store allosteric site predictions in a dedicated `fact_allosteric_site` table with site_id, constituent residue_ids, centroid coordinates, confidence_score, and run_id
3. WHEN persisting allosteric sites, THE Normalizer SHALL link each site to its constituent residues via the canonical `residue_id` foreign key
4. WHEN allosteric site persistence write succeeds, THE Orchestrator SHALL set `metadata["persisted"] = True` and `metadata["site_count"]` on the PhaseResult
5. IF allosteric site persistence write fails and `enforce_governed_outputs` is True, THEN THE Orchestrator SHALL mark the phase as failed

### Requirement 4: Phase Persistence Adapter Protocol

**User Story:** As a developer, I want a consistent adapter protocol for phase persistence, so that adding persistence for new phases follows a predictable pattern without one-off code.

#### Acceptance Criteria

1. THE System SHALL define a `PhasePersistenceAdapter` protocol with a `to_payloads` method that accepts a PhaseResult and ProvenanceContext and returns a list of Normalizer payloads
2. THE System SHALL maintain a registry mapping phase names to their persistence adapters
3. WHEN a phase completes successfully and has a registered adapter, THE Orchestrator SHALL invoke the adapter and write all returned payloads through the Normalizer
4. THE System SHALL define a `PhasePersistenceSpec` dataclass declaring each phase's tier, whether it produces residue-level data, and its schema version

### Requirement 5: Normalizer Payload Definitions

**User Story:** As a developer, I want well-defined Pydantic payload schemas for source-leak and allosteric site data, so that the Normalizer can validate inputs and the data layer has a clear contract.

#### Acceptance Criteria

1. THE System SHALL define a `SourceLeakPayload` Pydantic model containing ProvenanceContext, a list of source-leak residue records (residue_id, epistemic_uncertainty, cone_depth, leak_score, is_confirmed), and aggregate metadata
2. THE System SHALL define an `AllostericSitePayload` Pydantic model containing ProvenanceContext, a list of site records (site_id, residue_ids, centroid_x/y/z, confidence_score, cluster_method), and aggregate metadata
3. WHEN a payload is submitted to the Normalizer, THE Normalizer SHALL validate all residue_id references against the canonical format before writing

### Requirement 6: Database Schema for New Fact Tables

**User Story:** As a data engineer, I want dedicated fact tables for source-leak and allosteric site data, so that these scientific outputs are queryable with proper indexing and foreign key integrity.

#### Acceptance Criteria

1. THE System SHALL create a `fact_source_leak` table with columns: leak_id (PK), run_id (FK to provenance_run), structure_id (FK to dim_structure), residue_id (FK to dim_residue), epistemic_uncertainty, cone_depth, leak_score, is_confirmed, computed_at
2. THE System SHALL create a `fact_allosteric_site` table with columns: site_id (PK), run_id (FK to provenance_run), structure_id (FK to dim_structure), centroid_x, centroid_y, centroid_z, confidence_score, cluster_method, n_residues, computed_at
3. THE System SHALL create a `fact_allosteric_site_residue` junction table linking site_id to residue_id for many-to-many site membership
4. THE System SHALL create indexes on run_id, structure_id, and residue_id columns for efficient querying
5. THE System SHALL use idempotent upsert semantics (ON CONFLICT DO NOTHING or UPDATE) keyed on natural keys (run_id + residue_id for source leaks, run_id + site_id for allosteric sites)

### Requirement 7: Orchestrator Guardrails

**User Story:** As a system architect, I want the orchestrator to enforce that Tier 1 phases cannot silently discard their outputs, so that the governed data mandate is upheld.

#### Acceptance Criteria

1. THE PipelineConfig SHALL include an `enforce_governed_outputs` boolean field defaulting to True
2. WHEN `enforce_governed_outputs` is True and a Tier 1 phase completes without successful persistence, THE Orchestrator SHALL mark the overall pipeline run as degraded and include a warning
3. WHEN `enforce_governed_outputs` is False, THE Orchestrator SHALL allow phases to complete without persistence (dry-run mode for exploratory work)
4. THE Orchestrator SHALL run a `_validate_persistence_requirements` step after all phases complete, checking that all Tier 1 phases have `metadata["persisted"] == True`

### Requirement 8: Provenance Linkage

**User Story:** As a researcher, I want all persisted phase outputs linked to their parent GNN inference run, so that I can trace any scientific claim back to the exact model checkpoint and input features that produced it.

#### Acceptance Criteria

1. WHEN persisting any phase output, THE Adapter SHALL set `parent_run_id` in the ProvenanceContext to the GNN inference run_id that produced the input embeddings
2. THE System SHALL store the `parent_run_id` in the `provenance_run` table for each phase persistence run
3. WHEN querying phase outputs, THE System SHALL support joining back to the parent GNN run to retrieve model_version, checkpoint_uri, and input features
