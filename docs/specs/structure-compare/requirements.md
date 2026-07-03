# Requirements Document

## Introduction

The DTIE dashboard currently supports analyzing one structure at a time. Researchers frequently need to compare two protein structures (e.g., wild-type vs mutant, or two conformational states) to identify functionally significant differences in hyperbolic embedding, graph topology, and per-residue metrics. This feature brings back the compare mode by adding a structured workflow for selecting two proteins, viewing their differences across multiple analysis dimensions, and navigating to specific residues of interest.

## Glossary

- **Compare_Mode**: A dashboard state where two structures are loaded simultaneously for side-by-side or overlay analysis
- **Primary_Structure**: The first (reference) structure — typically the structure already active in the dashboard
- **Secondary_Structure**: The second structure selected for comparison
- **Displacement**: The hyperbolic distance between a residue's embedding in the primary structure vs the secondary structure, computed by aligning residues by canonical position (chain + index)
- **Edge_Diff**: The set of graph edges gained, lost, or weight-changed between two structures' contact networks
- **Metric_Delta**: The per-residue difference in computed graph metrics (betweenness, degree, clustering coefficient) between two structures
- **Canonical_Position**: A residue's alignment key consisting of chain_label and residue_index, used to match corresponding residues across structures
- **Compare_Panel**: The dedicated tool panel in the sidebar that contains comparison controls, results tables, and visualization toggles

## Requirements

### Requirement 1: Compare Mode Activation

**User Story:** As a scientist, I want to select a second structure for comparison from the discovery ledger, so that I can analyze differences between two conformations or mutant states.

#### Acceptance Criteria

1. WHEN a user clicks a "Compare" action on a structure in the discovery ledger, THE Dashboard SHALL enter Compare_Mode with that structure as the Secondary_Structure and the currently active structure as the Primary_Structure
2. WHEN Compare_Mode is active, THE Dashboard SHALL display a compare indicator showing both structure PDB IDs
3. WHEN a user clicks an "Exit Compare" button, THE Dashboard SHALL return to single-structure mode and clear all comparison data
4. IF no Primary_Structure is active when a user attempts to enter Compare_Mode, THEN THE Dashboard SHALL prevent activation and display a message indicating a structure must be selected first
5. WHEN Compare_Mode is activated, THE Dashboard SHALL verify both structures have embeddings available before proceeding

### Requirement 2: Embedding Displacement Comparison

**User Story:** As a scientist, I want to see which residues moved most in hyperbolic space between the two structures, so that I can identify conformationally significant shifts.

#### Acceptance Criteria

1. WHEN Compare_Mode is active, THE Compare_Panel SHALL display a sorted table of per-residue displacements between the Primary_Structure and Secondary_Structure embeddings
2. THE displacement table SHALL include columns for residue_id, chain, index, primary cone_depth, secondary cone_depth, depth_delta, and hyperbolic displacement magnitude
3. WHEN a user clicks a row in the displacement table, THE Poincaré disc SHALL highlight that residue in both structures
4. THE Compare_Panel SHALL display summary statistics including mean displacement, max displacement, and count of residues with displacement above a configurable threshold

### Requirement 3: Graph Topology Comparison

**User Story:** As a scientist, I want to see which contacts were gained or lost between the two structures, so that I can understand changes in the interaction network.

#### Acceptance Criteria

1. WHEN Compare_Mode is active, THE Compare_Panel SHALL display edge diff counts: gained edges, lost edges, and weight-changed edges
2. WHEN a user requests edge details, THE Compare_Panel SHALL show a list of gained and lost edges with their source and target residue IDs and edge type
3. WHEN Compare_Mode is active, THE Compare_Panel SHALL display a per-residue metric delta table showing changes in betweenness, degree, and clustering coefficient
4. WHEN gained or lost edges involve hydrogen bonds, THE Compare_Panel SHALL separately report H-bond gains and losses

### Requirement 4: Poincaré Disc Overlay

**User Story:** As a scientist, I want to see both structures' embeddings overlaid on the same Poincaré disc, so that I can visually identify where residues shifted in hyperbolic space.

#### Acceptance Criteria

1. WHEN Compare_Mode is active, THE Poincaré disc SHALL offer an overlay toggle that displays residues from both structures simultaneously
2. WHEN overlay is enabled, THE Primary_Structure residues SHALL render in the existing color scheme and the Secondary_Structure residues SHALL render with a distinct marker style (hollow circles or different opacity)
3. WHEN overlay is enabled and a residue is selected, THE Poincaré disc SHALL draw a displacement vector (line) from the primary position to the secondary position for that residue
4. WHEN overlay is disabled, THE Poincaré disc SHALL display only the Primary_Structure as normal

### Requirement 5: Comparison Results Navigation

**User Story:** As a scientist, I want to navigate from comparison results to specific residues in the viewers, so that I can inspect differences in 3D context.

#### Acceptance Criteria

1. WHEN a user clicks a residue in any comparison table, THE 3D Molecular Viewer SHALL center and highlight that residue
2. WHEN a user clicks a residue in any comparison table, THE Poincaré disc SHALL select and highlight that residue
3. WHEN a user selects "Top Movers" in the Compare_Panel, THE Dashboard SHALL highlight the top 10 highest-displacement residues across all viewers simultaneously
4. THE Compare_Panel SHALL provide a "Copy to Clipboard" action for selected comparison results in CSV format

### Requirement 6: Compare Panel Integration

**User Story:** As a system developer, I want the compare functionality to live in a dedicated tool panel, so that it follows existing dashboard patterns and integrates with the agent context.

#### Acceptance Criteria

1. THE Compare_Panel SHALL appear as a new entry in the ToolPanelSidebar alongside existing panels (hypothesis, graph_topology, etc.)
2. WHEN Compare_Mode is active, THE agent context payload SHALL include both structure IDs and the comparison summary (top movers, edge diff counts)
3. WHEN the user opens the Compare_Panel without Compare_Mode active, THE panel SHALL display instructions for how to activate comparison from the discovery ledger
4. THE Compare_Panel SHALL organize results into tabs: "Embeddings", "Graph", and "Summary"
