# Requirements Document

## Introduction

The Binding Site Scan at Ingestion feature transforms the cryptic site discovery tool from a hypothesis-driven manual exploration into an exhaustive pre-computed asset. When a structure is ingested and the DTIE pipeline completes, the system will automatically scan the entire structure, identify all candidate binding sites (cryptic and non-cryptic), classify them, rank them by druggability, and persist the results. Downstream queries for binding sites become instant database lookups rather than on-demand computation.

Additionally, this spec covers the operational calibration gap: running benchmark sites through real MD on the GPU-enabled science container to produce calibrated thresholds via Youden's J statistic, replacing the current provisional values.

## Glossary

- **Full_Structure_Scan**: An exhaustive analysis pass that identifies all candidate binding pockets across every residue cluster in a structure, combining GNN-derived signals and geometry-based pocket detection
- **Candidate_Site**: A spatially-clustered group of residues identified during the scan as having binding site potential (cryptic or surface pocket)
- **Surface_Pocket**: A traditional druggable pocket detectable from geometry alone (cavity volume, solvent accessibility) without requiring conformational change
- **Cryptic_Site**: A binding site that is occluded in the resting conformation and only accessible upon conformational rearrangement
- **Druggability_Score**: A composite numeric ranking [0, 1] indicating the likelihood that a candidate site can bind a small molecule with therapeutic relevance
- **Scan_Phase**: A new pipeline phase that runs after GNN inference and graph topology computation, performing exhaustive binding site enumeration
- **Calibration_Run**: Execution of the full SMD protocol against a benchmark site on real GPU hardware to obtain empirical work values for threshold determination
- **Calibrated_Threshold**: A success/failure cutoff derived from Youden's J statistic applied to real MD work values from known positive and negative sites
- **Site_Rank**: The position of a candidate site in the structure-wide druggability ranking (1 = most druggable)

## Requirements

### Requirement 1: Exhaustive Seed Generation from GNN Output

**User Story:** As a computational biologist, I want the system to automatically identify all high-signal residue clusters across the entire structure, so that no potential binding site is missed due to manual seed selection.

#### Acceptance Criteria

1. WHEN GNN inference completes for a structure, THE Scan_Phase SHALL identify all residues where epistemic_uncertainty exceeds the qualifying threshold AND cone_depth exceeds the qualifying threshold
2. THE Scan_Phase SHALL spatially cluster qualifying residues using DBSCAN with a configurable distance parameter (default 8.0 Å), producing one Candidate_Site per cluster
3. WHEN a cluster contains fewer than 3 qualifying residues, THE Scan_Phase SHALL discard that cluster as noise
4. THE Scan_Phase SHALL generate at most 25 candidate sites per structure to bound compute time
5. WHEN more than 25 clusters are identified, THE Scan_Phase SHALL retain the top 25 by composite score and discard the remainder with a logged warning

### Requirement 2: Surface Pocket Detection Integration

**User Story:** As a drug discovery researcher, I want the system to detect both cryptic sites (from GNN signals) and surface pockets (from geometry), so that the binding site inventory is complete regardless of pocket type.

#### Acceptance Criteria

1. THE Scan_Phase SHALL run fpocket (or equivalent geometry-based pocket detector) on the structure to identify surface pockets
2. THE Scan_Phase SHALL merge geometry-detected pockets with GNN-derived cryptic site candidates into a unified ranked list
3. WHEN a geometry-detected pocket overlaps spatially (centroid within 5.0 Å) with a GNN-derived candidate, THE Scan_Phase SHALL merge them into a single Candidate_Site with signals from both sources
4. WHEN a geometry-detected pocket has no GNN overlap, THE Scan_Phase SHALL include it as a Surface_Pocket with site_type "surface_pocket"
5. THE Scan_Phase SHALL annotate each Candidate_Site with its discovery_method: "gnn_strain", "geometry", or "hybrid" (both sources)

### Requirement 3: Classification and Ranking

**User Story:** As a researcher, I want all discovered sites classified by type and ranked by druggability in a single pass, so that I can immediately prioritize which sites to investigate.

#### Acceptance Criteria

1. THE Scan_Phase SHALL classify each Candidate_Site using the site_type heuristic (cryptic_wedge, structural_stent, dynamic_lid, allosteric_clamp, strain_relief_insert, or surface_pocket)
2. THE Scan_Phase SHALL compute a Druggability_Score for each Candidate_Site combining: composite GNN score, fpocket druggability score (if available), volume, and accessibility mode
3. THE Scan_Phase SHALL assign a Site_Rank to each Candidate_Site ordered by Druggability_Score descending (rank 1 = best)
4. THE Scan_Phase SHALL persist all ranked sites to fact_cryptic_site with md_validation_status="pending"
5. WHEN a structure has no qualifying residues and no geometry-detected pockets, THE Scan_Phase SHALL persist a scan_metadata record indicating "no_sites_found" with the scan parameters used

### Requirement 4: Ingestion Pipeline Integration

**User Story:** As a developer, I want the binding site scan to execute automatically as part of the existing DTIE pipeline, so that no manual intervention is needed after structure ingestion.

#### Acceptance Criteria

1. THE Scan_Phase SHALL execute automatically after Phase 3 (GNN inference) and graph topology computation complete
2. WHEN the pipeline runs, THE Scan_Phase SHALL produce a list of Candidate_Sites without requiring user-provided seed residues
3. THE Scan_Phase SHALL complete within 60 seconds for structures up to 500 residues (excluding MD validation)
4. THE system SHALL persist scan provenance (run_id, model_version, scan parameters, heuristic_version) alongside results
5. WHEN a structure is re-ingested with a newer heuristic_version or GNN model, THE Scan_Phase SHALL replace the previous scan results for that structure

### Requirement 5: Query-Time Binding Site Retrieval

**User Story:** As a user interacting with the agent, I want to ask "what binding sites does this structure have?" and receive an instant answer from pre-computed results, so that no computation is needed at interaction time.

#### Acceptance Criteria

1. THE system SHALL provide a query interface that returns all pre-computed Candidate_Sites for a structure ordered by Site_Rank
2. WHEN the agent tool is invoked for a structure that has been scanned, THE tool SHALL return results from the database without triggering new computation
3. THE query result SHALL include: site_id, site_type, residue_ids, druggability_score, site_rank, discovery_method, provenance_gate, md_validation_status, and heuristic_version
4. WHEN the agent tool is invoked for a structure that has NOT been scanned, THE tool SHALL return a clear message indicating the pipeline must be run first
5. THE system SHALL support filtering pre-computed results by site_type, minimum druggability_score, and md_validation_status

### Requirement 6: Real MD Calibration Execution

**User Story:** As a researcher, I want the system to run actual steered molecular dynamics against benchmark sites using the GPU science container, so that thresholds are empirically derived rather than provisional.

#### Acceptance Criteria

1. THE system SHALL dispatch real SMD jobs to the science container for each benchmark site in the calibration dataset
2. THE system SHALL run at minimum 5 known true-positive cryptic sites through each SMD protocol and record work values
3. THE system SHALL run at minimum 5 known true-negative (non-binding) sites through each SMD protocol and record work values
4. WHEN all calibration MD runs complete, THE system SHALL apply Youden's J statistic to determine the optimal threshold separating positives from negatives
5. WHEN calibrated thresholds are determined, THE system SHALL update DEFAULT_SUCCESS_CRITERIA with source="calibrated", the derived threshold, pulling rate, force constant, and reference sites used
6. THE system SHALL persist the calibration run provenance (run_id, benchmark version, MD parameters, compute time) for reproducibility

### Requirement 7: Heuristic v1 Calibration Against Benchmark

**User Story:** As a computational biologist, I want the site_type classification heuristic validated against ground truth labels, so that I know its precision per site_type and can flag unreliable classifications.

#### Acceptance Criteria

1. THE system SHALL run all benchmark sites through the mapper's site_type inference heuristic and compare predicted types against Ground_Truth_Labels
2. THE system SHALL compute precision, recall, and F1 per site_type from the benchmark evaluation
3. WHEN a site_type has precision below 0.7, THE system SHALL flag it in the calibration report and emit a warning when that type is assigned to new discoveries
4. THE system SHALL output updated precision_by_site_type values that the ToolResult uses for confidence warnings
5. WHEN the heuristic is updated (thresholds or structure change), THE system SHALL re-run benchmark evaluation and update the calibration report

### Requirement 8: On-Demand MD Validation for Ranked Sites

**User Story:** As a researcher, I want to trigger MD validation for specific pre-computed sites without re-running the full scan, so that I can selectively validate the most promising candidates.

#### Acceptance Criteria

1. WHEN a user requests MD validation for a specific site_id, THE system SHALL dispatch the appropriate SMD protocol to the science container
2. THE system SHALL update the fact_cryptic_site record with md_validation_status transitioning from "pending" to "running" to "passed" or "failed"
3. WHEN MD validation completes, THE system SHALL update the Candidate_Site's confidence score based on independent physics predictions only
4. THE system SHALL support batch MD validation (validate top N sites by rank) as a single command

</content>
</invoke>