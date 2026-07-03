# Requirements Document

## Introduction

Surface all existing agent tools as user-interactive frontend panels in the Tokyo Eye dashboard workbench. When a user loads a structure, the workbench hydrates with all available data (embeddings, graph metrics, allosteric sites, source leaks, hypotheses, provenance, annotations). Users can interact directly via UI controls OR ask the agent — both paths hit the same backend endpoints. All operations write through the Normalizer for persistence and auditability.

This spec covers: RCSB search (4 modes), graph topology analysis (5 tools), hypothesis engine (5 tools), data tools (8 tools), plotting/export, and direct visualization controls.

## Glossary

- **Workbench**: The full dashboard application with all tool panels active
- **Structure_Context**: The currently active structure and all its loaded data
- **Hydration**: Automatic loading of all available data when a structure is selected
- **Tool_Panel**: A collapsible UI section that exposes a specific tool group
- **Viewport_Directive**: A command from backend to frontend to highlight/focus/annotate residues in the viewer
- **Normalizer**: The governed write path that ensures all data is persisted with provenance
- **RCSB_API**: The RCSB Protein Data Bank programmatic interface (search, data, model queries)

## Requirements

### Requirement 1: Structure Hydration on Load

**User Story:** As a researcher, I want all available data for a structure to load automatically when I select it, so that I can immediately explore without manually triggering each query.

#### Acceptance Criteria

1. WHEN the user selects a structure from the results table, THE Workbench SHALL fetch and display: embeddings, graph metrics, allosteric sites, source leaks, hypotheses, provenance runs, and annotations for that structure
2. WHILE data is loading, THE Workbench SHALL display loading indicators for each panel without blocking interaction with already-loaded panels
3. IF any hydration query returns no data for a panel, THEN THE Workbench SHALL display an empty state message indicating the data has not been computed yet
4. WHEN hydration completes, THE Workbench SHALL update the agent chat context with a summary of all loaded data (residue count, source leak count, hypothesis count, top uncertainty residues)

### Requirement 2: RCSB Search Panel

**User Story:** As a researcher, I want to search RCSB by keyword, sequence similarity, structure similarity, and functional annotation, so that I can discover and ingest relevant structures without leaving the dashboard.

#### Acceptance Criteria

1. THE Workbench SHALL display an RCSB search panel with four search modes: text/keyword, sequence similarity, structure similarity, and functional annotation
2. WHEN a user performs a text search, THE Backend_API SHALL query RCSB with keyword, organism, and resolution filters and return matching PDB entries with title, resolution, and organism
3. WHEN a user performs a sequence similarity search, THE Backend_API SHALL accept a sequence string or PDB chain reference and return structures with similar sequences ranked by identity percentage
4. WHEN a user performs a structure similarity search, THE Backend_API SHALL accept a PDB ID and return structurally similar entries ranked by RMSD or TM-score
5. WHEN search results are displayed, THE Workbench SHALL show a preview card for each result with title, resolution, method, and an "Ingest" button
6. WHEN the user clicks Ingest on a search result, THE Workbench SHALL call the ingest endpoint and add the structure to the results table upon completion
7. IF RCSB is unreachable, THEN THE Backend_API SHALL return a structured error and THE Workbench SHALL display a connectivity warning

### Requirement 3: Graph Topology Panel

**User Story:** As a researcher, I want to explore the molecular contact graph topology (metrics, bridges, H-bonds, paths, comparisons), so that I can identify allosteric communication pathways and structural vulnerabilities.

#### Acceptance Criteria

1. WHEN a structure is loaded with graph data, THE Workbench SHALL display per-residue graph metrics (degree, betweenness, clustering coefficient, closeness, eigenvector centrality, bridge status, conductance) in a sortable table
2. WHEN the user clicks "Find Bridges", THE Backend_API SHALL return bridge/articulation-point residues and THE Workbench SHALL highlight them in the molecular viewer with a distinct color
3. WHEN the user clicks "H-Bond Network", THE Backend_API SHALL return the H-bond subgraph and THE Workbench SHALL render it as a network overlay or highlight participating residues
4. WHEN the user selects two residues and clicks "Shortest Path", THE Backend_API SHALL compute the shortest path in the contact graph and THE Workbench SHALL highlight the path residues sequentially in the viewer
5. WHEN the user selects two structures and clicks "Compare Graphs", THE Backend_API SHALL return edge diff (gained/lost/changed) and metric deltas, and THE Workbench SHALL display a diff view with gained edges in green and lost edges in red
6. THE Backend_API SHALL persist all graph metric computations through the Normalizer with provenance tracking

### Requirement 4: Hypothesis Engine Panel

**User Story:** As a researcher, I want to propose, test, and track scientific hypotheses about protein structures with a falsifiability guardrail, so that I can maintain a rigorous record of my reasoning and evidence.

#### Acceptance Criteria

1. THE Workbench SHALL display a hypothesis panel showing all hypotheses for the active structure with their status (proposed, gathering, supported, contradicted, inconclusive) and confidence score
2. WHEN the user clicks "New Hypothesis", THE Workbench SHALL present a form requiring: statement, at least one testable prediction (with optional test_tool, test_params, threshold), and optional mechanism
3. IF the user submits a hypothesis without at least one prediction, THEN THE Backend_API SHALL reject it with a falsifiability error message
4. WHEN the user clicks "Test" on a hypothesis, THE Backend_API SHALL execute each prediction by calling the referenced tools, evaluate thresholds, create evidence records, recalculate confidence, and update status
5. WHEN the user clicks "Add Evidence" on a hypothesis, THE Workbench SHALL present a form for source_tool, supports/contradicts toggle, description, and strength slider (0-1)
6. WHEN confidence changes, THE Workbench SHALL update the hypothesis card with the new confidence score and status badge
7. THE Backend_API SHALL persist all hypotheses, predictions, and evidence through the Normalizer with full provenance

### Requirement 5: Data Tools Panel

**User Story:** As a researcher, I want to search/filter residues, view allosteric sites, export data, annotate structures, and inspect provenance lineage, so that I can perform detailed analysis and share findings.

#### Acceptance Criteria

1. THE Workbench SHALL display a residue search/filter panel with inputs for: chain, residue name, uncertainty range (min/max), cone depth range (min/max), uncertainty type selector, and result limit
2. WHEN the user applies filters, THE Backend_API SHALL return matching residues and THE Workbench SHALL highlight them in the viewer and display them in a results list
3. THE Workbench SHALL display allosteric site clusters for the active structure with site membership, confidence scores, and residue lists, highlighting site residues in the viewer
4. WHEN the user clicks "Export", THE Backend_API SHALL generate a CSV or JSON file of the active structure's pipeline results and return a download link
5. WHEN the user adds an annotation (finding, hypothesis, note, or warning) to a structure or residue selection, THE Backend_API SHALL persist it through the Normalizer and display it in an annotations timeline
6. THE Workbench SHALL display a provenance panel showing the run history for the active structure: run_id, pipeline_name, model_version, parent_run_id, started_at, and asset counts
7. WHEN the user clicks a run in the provenance panel, THE Workbench SHALL display the run summary (phases run, duration, assets created, errors)
8. WHEN the user selects two runs and clicks "Compare", THE Backend_API SHALL return per-residue deltas in cone depth and uncertainty, and THE Workbench SHALL display a diff visualization

### Requirement 6: Plotting and Export

**User Story:** As a researcher, I want to generate publication-quality matplotlib figures from pipeline data, so that I can include them in papers and presentations without external tools.

#### Acceptance Criteria

1. THE Workbench SHALL display a "Generate Plot" panel with a plot type selector (poincare_disc, uncertainty_profile, cone_depth_histogram, wt_vs_mutant, persistence_barcode, source_leak_map) and type-specific parameter inputs
2. WHEN the user clicks "Generate", THE Backend_API SHALL render the matplotlib figure server-side and return the PNG image path
3. THE Workbench SHALL display the generated plot inline with a download button
4. WHEN generating a wt_vs_mutant plot, THE Workbench SHALL require the user to select a second structure for comparison
5. IF plot generation fails due to missing data, THEN THE Backend_API SHALL return a structured error indicating which data is needed

### Requirement 7: Direct Visualization Controls

**User Story:** As a researcher, I want to directly control the molecular viewer (highlight residues, change coloring metric, focus camera, clear highlights) without going through the agent, so that I can quickly explore the structure visually.

#### Acceptance Criteria

1. THE Workbench SHALL display a visualization toolbar with buttons for: highlight selection, change color metric, focus on selection, and clear all highlights
2. WHEN the user selects residues (from any panel: search results, bridge list, path, etc.) and clicks "Highlight", THE Workbench SHALL highlight those residues in the viewer with a user-chosen color and style
3. WHEN the user changes the color metric dropdown (cone_depth, epistemic, aleatoric, total_uncertainty), THE Workbench SHALL re-color all residues in the Poincaré scatter and molecular viewer by that metric
4. WHEN the user clicks "Focus", THE Workbench SHALL animate the camera to center on the currently selected residues
5. WHEN the user clicks "Clear", THE Workbench SHALL remove all highlights and reset to default coloring
6. WHEN the agent returns viewport_directives in a chat response, THE Workbench SHALL apply those directives (highlight, focus, annotate) to the viewer automatically

### Requirement 8: Persistence Verification

**User Story:** As a researcher, I want confidence that all computations and findings are being persisted to the database, so that I don't lose work and can reproduce results.

#### Acceptance Criteria

1. WHEN any tool writes data through the Normalizer, THE Backend_API SHALL return the provenance run_id and asset count in the response
2. THE Workbench SHALL display a persistence status indicator (green checkmark or red warning) next to each data panel showing whether the displayed data has been persisted
3. IF a write operation fails, THEN THE Backend_API SHALL return the error and THE Workbench SHALL display a warning banner with the failure details and a "Retry" button
4. THE Workbench SHALL display a system health panel (accessible from the nav bar) showing: DB connection status, Normalizer audit trail (last 10 writes), and any failed persistence operations
5. THE Control Console SHALL display pipeline runtime audit for the active structure: `geometric_readiness` and recent events from `GET /api/structures/{id}/audit` (see `docs/audit/PIPELINE_AUDIT.md`)
