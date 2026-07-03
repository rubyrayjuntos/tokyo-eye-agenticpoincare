# Implementation Plan: Structure Compare

## Overview

Adds structure comparison to the dashboard: thin backend REST endpoints wrapping existing tools, a ComparePanel in the sidebar, Poincaré overlay, and agent context integration. Built incrementally — backend endpoints first, then frontend state + panel, then overlay and polish.

## Tasks

- [x] 1. Backend: Add comparison REST endpoints
  - [x] 1.1 Add GET /api/compare/embeddings/{id_a}/{id_b} endpoint
    - Wrap existing compare_wt_mutant tool in a direct REST endpoint
    - Return sorted displacement rows with all required fields
    - Include summary stats (mean, max, count above threshold)
    - _Requirements: 2.1, 2.2, 2.4_

  - [x] 1.2 Add GET /api/compare/graphs/{id_a}/{id_b} endpoint
    - Wrap existing compare_graphs tool in a direct REST endpoint
    - Return edge diff counts and per-residue metric deltas
    - Include H-bond gain/loss breakdown
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 1.3 Write property tests for displacement sorting and summary stats
    - **Property 1: Displacement table sorted by magnitude**
    - **Property 2: Summary statistics accuracy**
    - **Validates: Requirements 2.1, 2.2, 2.4, 5.3**

  - [x] 1.4 Write property tests for edge diff and metric delta accuracy
    - **Property 3: Edge diff counts consistency**
    - **Property 4: H-bond subset accuracy**
    - **Property 5: Metric delta arithmetic**
    - **Validates: Requirements 3.1, 3.3, 3.4**

- [x] 2. Frontend: Compare state and activation
  - [x] 2.1 Add CompareState to App.tsx and DashboardContext
    - Add secondaryStructure, displacements, graphDiff, compareLoading state
    - Add enterCompareMode and exitCompareMode functions
    - Pass compare state to context provider
    - _Requirements: 1.1, 1.2, 1.3, 1.5_

  - [x] 2.2 Add compare button to ResultsTable rows
    - Add "⇔" icon button per row
    - Disable when row is activeStructure or has_embeddings is false
    - On click: call enterCompareMode(structure)
    - _Requirements: 1.1, 1.4, 1.5_

  - [x] 2.3 Add compare indicator bar
    - Show when compare mode is active
    - Display both PDB IDs and an "Exit Compare" button
    - _Requirements: 1.2, 1.3_

- [x] 3. Frontend: ComparePanel component
  - [x] 3.1 Create ComparePanel with tab structure
    - New panel in ToolPanelSidebar: "Compare"
    - Three tabs: Embeddings, Graph, Summary
    - Show instructions when compare mode is not active
    - _Requirements: 6.1, 6.3, 6.4_

  - [x] 3.2 Implement Embeddings tab
    - Sortable table of displacement rows
    - Click row to highlight residue via emitDirective
    - "Top Movers" button highlights top 10
    - _Requirements: 2.1, 2.2, 2.3, 5.1, 5.2, 5.3_

  - [x] 3.3 Implement Graph tab
    - Edge diff summary card (gained/lost/changed/hbond counts)
    - Metric delta table (sortable by any delta column)
    - Click row to highlight residue
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 5.1, 5.2_

  - [x] 3.4 Implement Summary tab
    - Summary stats card (mean, max, threshold controls)
    - Top movers list
    - Copy to CSV button
    - _Requirements: 2.4, 5.3, 5.4_

  - [x] 3.5 Write property test for CSV export completeness
    - **Property 7: CSV export completeness**
    - **Validates: Requirements 5.4**

- [x] 4. Frontend: Poincaré overlay mode
  - [x] 4.1 Add overlay toggle and secondary point rendering
    - Fetch secondary embeddings when compare mode active
    - Render secondary points as hollow circles with distinct opacity
    - Share color normalization across both structures
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 4.2 Add displacement vector on selection
    - When a residue is selected in overlay mode, draw line from primary to secondary position
    - _Requirements: 4.3_

- [x] 5. Frontend: Agent context integration
  - [x] 5.1 Extend buildContext with compare data
    - Add compare section to ViewportState when compare mode is active
    - Include secondary_structure_id, top_movers, edge_diff counts
    - _Requirements: 6.2_

  - [x] 5.2 Write property test for compare context payload completeness
    - **Property 6: Compare context payload completeness**
    - **Validates: Requirements 6.2**

- [x] 6. Checkpoint - Verify integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required (comprehensive testing from start)
- The backend comparison tools already exist (compare_wt_mutant, compare_graphs) — we just add REST wrappers
- Residue alignment uses canonical position (chain + index), already implemented in backend
- The ComparePanel follows the same pattern as GraphTopologyPanel and HypothesisPanel
- Property tests validate backend logic; frontend tests would be unit tests for state transitions
