# Requirements Document

## Introduction

The DTIE pipeline computes extensive per-residue and per-structure data across multiple analysis phases (embeddings, graph topology, vulnerability doorways, resistance pathways, pharmacophore pockets, and drug candidates). Currently most of this data is hidden — only accessible through direct API calls or agent tool queries. Scientists have no way to browse, search, or export the full computed dataset. This feature adds a Data Inspector panel that makes all computed values visible, searchable, filterable, and exportable, closing the "black box" gap.

## Glossary

- **Data_Inspector**: A new tool panel providing a unified view of all computed data for the active structure
- **Phase_Data**: Results from specific pipeline phases (Phase 2: vulnerability, Phase 4: resistance, Phase 5: pharmacophore, Phase 6: drug candidates)
- **Residue_Table**: A sortable, filterable table showing per-residue metrics from all available phases
- **Structure_Summary**: An overview card showing what data is available, pipeline phase completion, and aggregate statistics
- **Pocket_View**: A visualization of Phase 5 pharmacophore pockets and Phase 6 drug candidate binding sites in the 3D viewer
- **Persistence_Status**: Flags indicating which phases have been computed and persisted for the active structure

## Requirements

### Requirement 1: Data Availability Overview

**User Story:** As a scientist, I want to see at a glance what data has been computed for my structure, so that I know which analyses are available and which need to be run.

#### Acceptance Criteria

1. WHEN the Data_Inspector panel is opened, THE panel SHALL display a persistence status summary showing which phases are computed (embeddings, graph, sites, phase2, phase4, phase5, phase6, resistance)
2. WHEN a phase is computed, THE status indicator SHALL display a green check with the count of records (e.g., "537 residues" for embeddings, "12 pockets" for phase5)
3. WHEN a phase is not computed, THE status indicator SHALL display a gray indicator with "Not computed" text
4. THE Data_Inspector SHALL display aggregate statistics: total residues, total source leaks, total hypotheses, total provenance runs, total annotations

### Requirement 2: Per-Residue Metrics Table

**User Story:** As a scientist, I want to see all computed values for each residue in a single table, so that I can identify residues of interest across multiple analysis dimensions simultaneously.

#### Acceptance Criteria

1. THE Residue_Table SHALL display one row per residue with columns including: residue_id, chain, index, cone_depth, epistemic_uncertainty, aleatoric_uncertainty, sensitivity_score, classification, leak_score, betweenness, degree, clustering_coefficient, is_bridge
2. WHEN a column header is clicked, THE Residue_Table SHALL sort rows by that column in ascending or descending order
3. WHEN the user enters a filter expression, THE Residue_Table SHALL filter to show only rows matching the criteria (chain filter, uncertainty range, depth range, classification filter)
4. WHEN a row in the Residue_Table is clicked, THE Dashboard SHALL highlight that residue in both the Poincaré disc and 3D viewer
5. THE Residue_Table SHALL merge data from embeddings, graph metrics, source leaks, and resistance data into a unified per-residue view using residue_id as the join key

### Requirement 3: Phase 5 Pharmacophore Data

**User Story:** As a scientist, I want to see the identified drug-binding pockets and their properties, so that I can evaluate druggability and select pockets for further study.

#### Acceptance Criteria

1. WHEN Phase 5 data is available, THE Data_Inspector SHALL display a pocket table with columns: pocket_index, druggability_score, residue_count, volume_estimate, allosteric_coupling, center coordinates
2. WHEN a pocket row is clicked, THE 3D Molecular Viewer SHALL highlight the pocket residues and center the camera on the pocket centroid
3. THE pocket table SHALL be sorted by druggability_score in descending order by default
4. WHEN Phase 5 data is not available, THE Data_Inspector SHALL show a message indicating the pharmacophore phase has not been run

### Requirement 4: Phase 6 Drug Candidate Data

**User Story:** As a scientist, I want to see the drug candidate evaluation results including ADMET and selectivity, so that I can prioritize candidates for experimental validation.

#### Acceptance Criteria

1. WHEN Phase 6 data is available, THE Data_Inspector SHALL display a candidate table with columns: pocket_index, combined_druggability, accessibility_score, binding_potential, admet_pass, selectivity_ratio, is_state_selective
2. WHEN a candidate row is clicked, THE 3D Molecular Viewer SHALL highlight the candidate pocket and display its binding site residues
3. THE candidate table SHALL allow filtering by admet_pass (pass/fail) and is_state_selective (yes/no)
4. THE Data_Inspector SHALL display summary counts: total candidates, ADMET-passed count, state-selective count

### Requirement 5: Data Export

**User Story:** As a scientist, I want to export all computed values for my structure in standard formats, so that I can use them in external tools (PyMOL, ChimeraX, spreadsheets, publication figures).

#### Acceptance Criteria

1. THE Data_Inspector SHALL provide export buttons for CSV and JSON formats
2. WHEN the user exports data, THE export SHALL include ALL available per-residue fields from all computed phases (embeddings, graph, resistance, source leaks) merged by residue_id
3. WHEN Phase 5 or Phase 6 data exists, THE export SHALL include a separate pockets/candidates section or file
4. THE export SHALL include metadata: structure_id, pdb_id, export timestamp, pipeline version, phase completion status

### Requirement 6: 3D Viewer Pocket Visualization

**User Story:** As a scientist, I want to see pharmacophore pockets and drug candidate sites rendered in the 3D molecular viewer, so that I can understand their spatial context in the protein structure.

#### Acceptance Criteria

1. WHEN Phase 5 data is available, THE 3D Molecular Viewer SHALL offer a "Pockets" color mode that highlights pocket residues colored by druggability_score
2. WHEN a pocket is selected in the Data_Inspector, THE 3D Molecular Viewer SHALL render a translucent sphere at the pocket centroid with radius proportional to volume_estimate
3. WHEN Phase 6 data is available, THE 3D Molecular Viewer SHALL offer a "Drug Candidates" color mode that highlights candidate binding sites colored by combined_druggability
4. WHEN a color mode involves pocket or candidate data, THE Molecular Viewer SHALL display the pocket index label near the centroid

### Requirement 7: Agent Context Integration

**User Story:** As a scientist, I want the agent to know what data phases are available and their summary stats, so that it can suggest relevant analyses and answer questions about computed results.

#### Acceptance Criteria

1. WHEN the Data_Inspector panel is active, THE agent context SHALL include which phases are computed and their record counts
2. WHEN Phase 5 or Phase 6 data exists, THE agent context SHALL include the top druggability pocket and top drug candidate scores
3. THE agent context SHALL include the active panel name as "data_inspector" when this panel is open
