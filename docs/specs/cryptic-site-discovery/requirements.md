# Requirements Document

## Introduction

The Cryptic Binding Site Discovery System is a general-purpose engine for discovering, characterizing, and validating non-pocket binding sites in proteins. Unlike traditional pocket-based approaches, this system identifies cryptic attachment points (compressed zones, stents, lids, clamps, strain voids) using Tokyo Eye's full signal repertoire — epistemic uncertainty, cone depth, dehydron density, resistance topology, and graph metrics. The system enforces MD validation before fragment screening for novel site types, ensuring physical plausibility before committing compute. It integrates as an agent tool within the existing Tokyo Eye coordinator, following the governed data layer and provenance-tracked architecture.

## Glossary

- **Cryptic_Site**: A non-pocket binding region that is not surface-accessible by classical methods but is detectable via strain, uncertainty, or topological signals
- **Site_Type**: A classification enum for the mechanism of the cryptic site (e.g., cryptic_wedge, structural_stent, dynamic_lid, allosteric_clamp, strain_relief_insert)
- **CrypticBindingSiteSpec**: The typed, extensible Pydantic schema describing a discovered cryptic site including anchors, displacement targets, geometric constraints, and validation state
- **Mapper**: The component that takes a structure + seed residues and identifies cryptic regions using Tokyo Eye signals (epistemic uncertainty, cone depth, dehydron density, resistance topology, graph centrality)
- **MD_Validator**: The component that runs steered molecular dynamics (SMD) protocols to physically validate proposed sites before fragment screening
- **Fragment_Screener**: The component that evaluates candidate fragments against the geometric and pharmacophore constraints of a validated site
- **SMD_Protocol**: A steered molecular dynamics simulation protocol (e.g., three-phase pull/equilibrate/release) used to validate site accessibility
- **Bedrock_Clique**: The set of load-bearing residues (high graph centrality, bridge nodes) surrounding a cryptic site
- **Pharmacophore_Anchor**: A residue providing a chemical interaction feature (π-stacking, H-bond donor/acceptor) that serves as a fragment attachment point
- **Displacement_Target**: A residue that must be moved or rearranged for a fragment to occupy the cryptic site
- **GeometricConstraints**: The spatial boundary definitions (hull padding, inclusion vectors, bounding spheres) that define the physical envelope of a site
- **Falsifiable_Prediction**: A testable hypothesis about the site (e.g., "insertion reduces epistemic uncertainty by Δ > 2.5") with a defined test_tool and threshold
- **Agent**: The LLM-driven coordinator that orchestrates tool calls in Tokyo Eye
- **ToolResult**: The standard return type for all agent tools (success, data, message, viewport_directives, warnings)
- **Normalizer**: The single governed write path for persisting data to the database
- **Science_Container**: The GPU-enabled Docker container where heavy MD and GNN computation runs
- **Science_Job**: An asynchronous computation dispatched to the Science_Container, tracked by job_id with status polling

## Requirements

### Requirement 1: Site Type Schema and Extensibility

**User Story:** As a computational biologist, I want a typed and extensible schema for describing cryptic binding sites, so that newly discovered attachment mechanisms can be added without breaking existing logic.

#### Acceptance Criteria

1. THE CrypticBindingSiteSpec SHALL define schema_version, site_id, structure_id, chain, residue_ids, site_type, zone_classification, and accessibility_mode as required fields
2. THE CrypticBindingSiteSpec SHALL support at minimum the following site_types: cryptic_wedge, structural_stent, dynamic_lid, allosteric_clamp, strain_relief_insert
3. THE CrypticBindingSiteSpec SHALL include an extension_fields dictionary for future site-type-specific parameters
4. WHEN a CrypticBindingSiteSpec is serialized to JSON and deserialized back, THE system SHALL produce an equivalent object
5. WHEN an invalid site_type value is provided, THE CrypticBindingSiteSpec SHALL reject it with a validation error
6. WHEN a CrypticBindingSiteSpec with an older schema_version is loaded, THE system SHALL apply forward-compatible defaults for any new fields introduced in later versions

### Requirement 2: Cryptic Site Mapping

**User Story:** As a researcher, I want to identify cryptic binding regions by providing seed residues and a structure, so that I can discover non-obvious therapeutic targets.

#### Acceptance Criteria

1. WHEN the Mapper receives a structure_id and seed_residues, THE Mapper SHALL query residues with epistemic_uncertainty above a configurable threshold AND cone_depth above a configurable threshold within a spatial neighborhood of the seeds
2. WHEN the Mapper detects a candidate region, THE Mapper SHALL expand from seed residues using the existing graph topology (contact graph edges) to include connected residues that meet strain criteria
3. THE Mapper SHALL incorporate dehydron density, resistance topology (hinge scores), and pharmacophore-compatible features as additional signals for region scoring
4. WHEN no site_type is explicitly provided, THE Mapper SHALL infer site_type using a v1 heuristic: if displacement_targets exist with strain_signal EXTREME then cryptic_wedge; if high betweenness but low displacement then structural_stent; if residues are in a flexible loop region then dynamic_lid; otherwise default to strain_relief_insert
5. THE Mapper SHALL populate pharmacophore_anchors, displacement_targets, and bedrock_clique fields from the detected region
6. WHEN the Mapper cannot find residues meeting the threshold criteria near the seed residues, THE Mapper SHALL return a failure result with an actionable message suggesting alternative seeds or threshold relaxation

### Requirement 3: Graph Topology Enrichment

**User Story:** As a researcher, I want the discovered site to include graph-level metrics for surrounding residues, so that I can understand the structural context and load-bearing properties of the cryptic site.

#### Acceptance Criteria

1. WHEN a cryptic site is mapped, THE system SHALL compute betweenness_centrality, clustering_coefficient, graph_degree, and is_bridge for each residue in the bedrock_clique
2. THE system SHALL identify bridge nodes (residues whose removal disconnects the local graph) surrounding the cryptic site
3. WHEN graph metrics are computed, THE system SHALL persist them as part of the CrypticBindingSiteSpec bedrock_clique field

### Requirement 4: MD Validation Gating

**User Story:** As a researcher, I want novel or high-risk site types to be physically validated via molecular dynamics before fragment screening, so that compute is not wasted on physically implausible sites.

#### Acceptance Criteria

1. THE CrypticBindingSiteSpec SHALL expose a requires_md_validation method that returns True for cryptic_wedge, structural_stent, dynamic_lid, and allosteric_clamp site types
2. WHEN a site requires MD validation, THE MD_Validator SHALL dispatch the appropriate SMD protocol as an asynchronous Science_Job to the Science_Container
3. WHEN MD validation fails, THE system SHALL return the spec with md_validation status set to "failed" and SHALL NOT proceed to fragment screening
4. WHEN MD validation passes, THE system SHALL update the spec with md_validation status set to "passed" and proceed to fragment screening
5. WHEN force_md_validation is set to True, THE system SHALL run MD validation regardless of site_type
6. WHEN the spec does not include explicit falsifiable_predictions, THE MD_Validator SHALL apply a default success criterion per protocol (e.g., work < 25 kcal/mol for three_phase, strain_delta < -1.5 for stent_stabilization)

### Requirement 5: SMD Protocol Dispatch

**User Story:** As a researcher, I want each site type to have an appropriate molecular dynamics protocol, so that validation is scientifically meaningful for the specific mechanism being tested.

#### Acceptance Criteria

1. THE MD_Validator SHALL support protocol dispatch based on the site_type: SMD_three_phase for cryptic_wedge, SMD_stent_stabilization for structural_stent, SMD_lid_restraint for dynamic_lid, SMD_clamp_stabilization for allosteric_clamp, SMD_strain_relief for strain_relief_insert
2. WHEN a new protocol is needed but not yet implemented, THE system SHALL fall back to SMD_three_phase as a default with a warning indicating the fallback
3. THE MD_Validator SHALL return structured results including status (passed/failed), work_kcal_mol, protocol-specific metrics, and duration
4. THE MD_Validator SHALL evaluate falsifiable_predictions defined in the spec against the MD simulation output and record per-prediction pass/fail

### Requirement 6: Fragment Screening Against Spec

**User Story:** As a drug discovery researcher, I want to screen fragment libraries against the geometric and pharmacophore constraints of a validated cryptic site, so that I can identify candidate molecules for the non-pocket binding mechanism.

#### Acceptance Criteria

1. WHEN fragment screening is invoked, THE Fragment_Screener SHALL filter candidates using the geometric_constraints (hull padding, inclusion vectors, bounding sphere)
2. THE Fragment_Screener SHALL score candidates based on pharmacophore complementarity with the defined anchors
3. THE Fragment_Screener SHALL return ranked candidates with SMILES, score, and reason fields
4. WHEN no candidates pass the geometric filter, THE Fragment_Screener SHALL return an empty list with a message suggesting constraint relaxation
5. THE Fragment_Screener SHALL accept a pluggable compound_database configuration within fragment_screening_profile, defaulting to a built-in curated fragment set

### Requirement 7: Agent Tool Integration

**User Story:** As the Tokyo Eye agent, I want to call the cryptic site discovery as a standard tool, so that I can orchestrate multi-step workflows involving site discovery, validation, and screening.

#### Acceptance Criteria

1. THE system SHALL expose discover_and_validate_cryptic_site as a registered agent tool with structure_id (required), seed_residues (required), site_type (optional), and force_md_validation (optional) parameters
2. THE tool SHALL return a ToolResult where data contains: spec (full CrypticBindingSiteSpec dict), md_result (dict or null), fragments (list), stage (string), and next_actions (list of strings)
3. WHEN the tool completes successfully, THE tool SHALL emit ViewportDirectives highlighting the discovered residues with site_type-appropriate colors and labels
4. THE tool SHALL follow the existing ToolResult pattern (success, data, message, viewport_directives, warnings)
5. IF the database connection is unavailable, THEN THE tool SHALL return a ToolResult with success=False and an actionable error message
6. IF the structure has not been previously analyzed (no GNN results), THEN THE tool SHALL return a ToolResult with success=False and a message suggesting running the DTIE pipeline first

### Requirement 8: Multi-Step Workflow Support

**User Story:** As a researcher using the agent, I want clear stage tracking and next-action recommendations, so that I can iteratively refine cryptic site discovery across multiple turns.

#### Acceptance Criteria

1. THE tool result SHALL include a stage field indicating the current workflow position (mapping, md_validation, fragment_screening, complete)
2. THE tool result SHALL include a next_actions list with contextual suggestions based on the current outcome
3. WHEN MD validation fails, THE next_actions SHALL suggest trying a different site_type, adjusting geometric constraints, or skipping MD validation
4. WHEN fragment screening yields fewer than 3 candidates, THE next_actions SHALL suggest trying alternative site_types for the same region
5. WHEN auto_propose_new_site_type is True and the current site_type produces poor results, THE system SHALL include a suggested alternative site_type in next_actions

### Requirement 9: Provenance and Persistence

**User Story:** As a researcher, I want all cryptic site discoveries to be persisted with full provenance, so that I can trace results back to their inputs, parameters, and model versions.

#### Acceptance Criteria

1. WHEN a cryptic site is successfully discovered, THE system SHALL persist the CrypticBindingSiteSpec through the Normalizer write path
2. THE persisted record SHALL include a provenance_run linking to the structure_id, seed_residues, site_type parameters, and code version
3. WHEN the same structure and seed_residues are re-analyzed, THE system SHALL perform an idempotent upsert preserving the latest result
4. THE system SHALL register the output as a governed_asset with asset_type "cryptic_binding_site"

### Requirement 10: Falsifiable Predictions

**User Story:** As a researcher, I want each discovered site to include falsifiable predictions, so that the hypothesis can be tested against experimental data or further computation.

#### Acceptance Criteria

1. THE CrypticBindingSiteSpec SHALL include a falsifiable_predictions list within validation_metrics
2. EACH falsifiable prediction SHALL contain a prediction_id, statement, test_tool, and threshold
3. WHEN MD validation is run, THE system SHALL evaluate predictions against the simulation output and record per-prediction pass/fail status
4. THE Agent SHALL present falsifiable predictions to the user as testable hypotheses with clear next steps

### Requirement 11: Asynchronous Execution and Job Tracking

**User Story:** As a researcher, I want long-running MD validations to execute asynchronously with progress tracking, so that I can continue working while physics simulations run.

#### Acceptance Criteria

1. WHEN MD validation is dispatched, THE system SHALL create a Science_Job with a unique job_id and initial status "pending"
2. THE system SHALL support polling the job status via the agent (pending, running, passed, failed, timeout)
3. WHEN a Science_Job exceeds a configurable timeout (default 30 minutes), THE system SHALL mark the job as "timeout" and return a failure with the partial result
4. THE tool result SHALL include job_id when MD validation is dispatched asynchronously, enabling the agent to check status in a subsequent turn

### Requirement 12: Observability and Logging

**User Story:** As a system operator, I want key decisions and metrics from the cryptic site discovery process to be logged, so that I can monitor system health and debug issues.

#### Acceptance Criteria

1. THE system SHALL log (structured JSON) each major decision point: site mapping start/complete, MD validation dispatch/result, fragment screening start/complete
2. THE system SHALL emit timing metrics for each stage (mapping_duration_ms, md_duration_ms, screening_duration_ms)
3. WHEN the Mapper infers a site_type, THE system SHALL log the inference reasoning (which heuristic matched)
4. WHEN MD validation fails, THE system SHALL log the specific failure reason and the evaluation of each falsifiable prediction
