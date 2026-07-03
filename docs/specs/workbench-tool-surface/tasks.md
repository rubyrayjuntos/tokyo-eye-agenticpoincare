# Implementation Plan: Workbench Tool Surface

## Overview

Surface all existing agent tools as REST endpoints and frontend panels. Implementation order: backend routers first (they wrap existing tool functions), then hydration endpoint, then frontend panels. Each router is independent so they can be built in parallel.

## Tasks

- [x] 1. Backend: RCSB Search Router
  - [x] 1.1 Create `agent/coordinator/routers/rcsb.py` with 4 endpoints
    - `POST /api/rcsb/search` — text/keyword search (wraps `search_rcsb`)
    - `POST /api/rcsb/sequence-search` — sequence similarity via `SequenceQuery`
    - `POST /api/rcsb/structure-search` — structure similarity via Alignment API
    - `GET /api/rcsb/info/{pdb_id}` — metadata fetch (wraps `fetch_structure_info`)
    - _Requirements: 2.2, 2.3, 2.4, 2.7_

  - [x] 1.2 Register the RCSB router in `agent/coordinator/app.py`
    - _Requirements: 2.1_

  - [x] 1.3 Write property test for RCSB search result ordering
    - **Property 3: RCSB search results are ordered by mode-appropriate metric**
    - **Validates: Requirements 2.2, 2.3, 2.4**

- [x] 2. Backend: Graph Topology Router
  - [x] 2.1 Create `agent/coordinator/routers/graph.py` with 5 endpoints
    - `GET /api/graph/{structure_id}/metrics` — wraps `get_graph_metrics`
    - `GET /api/graph/{structure_id}/bridges` — wraps `find_graph_bridges`
    - `GET /api/graph/{structure_id}/hbonds` — wraps `get_hbond_network`
    - `POST /api/graph/{structure_id}/shortest-path` — wraps `get_shortest_paths`
    - `POST /api/graph/compare` — wraps `compare_graphs`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 2.2 Register the graph router in `agent/coordinator/app.py`
    - _Requirements: 3.1_

  - [x] 2.3 Write property tests for graph endpoints
    - **Property 4: Graph metrics endpoint returns complete metric set**
    - **Property 5: Bridge endpoint returns only bridge residues**
    - **Property 6: H-bond endpoint returns only H-bond edges**
    - **Property 7: Shortest path is valid**
    - **Property 8: Graph compare edge sets are consistent**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

- [x] 3. Backend: Hypothesis Engine Router
  - [x] 3.1 Create `agent/coordinator/routers/hypotheses.py` with 5 endpoints
    - `GET /api/hypotheses` — wraps `get_hypotheses` (query params: structure_id, status)
    - `POST /api/hypotheses` — wraps `propose_hypothesis`
    - `POST /api/hypotheses/{id}/test` — wraps `test_hypothesis`
    - `POST /api/hypotheses/{id}/evidence` — wraps `add_evidence`
    - `POST /api/hypotheses/{id}/evaluate` — wraps `evaluate_confidence`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7_

  - [x] 3.2 Register the hypotheses router in `agent/coordinator/app.py`
    - _Requirements: 4.1_

  - [x] 3.3 Write property tests for hypothesis engine
    - **Property 9: Hypothesis falsifiability guardrail**
    - **Property 10: Confidence is bounded after testing**
    - **Validates: Requirements 4.3, 4.4**

- [x] 4. Backend: Data Tools Router
  - [x] 4.1 Create `agent/coordinator/routers/data.py` with 8 endpoints
    - `POST /api/data/{structure_id}/search-residues` — wraps `search_residues`
    - `GET /api/data/{structure_id}/allosteric-sites` — wraps `get_allosteric_sites`
    - `GET /api/data/{structure_id}/provenance` — wraps `get_provenance_lineage`
    - `POST /api/data/{structure_id}/export` — wraps `export_structure_data`
    - `GET /api/data/{structure_id}/annotations` — lists annotations
    - `POST /api/data/{structure_id}/annotations` — wraps `annotate_structure`
    - `GET /api/data/runs/{run_id}/summary` — wraps `get_run_summary`
    - `POST /api/data/runs/compare` — wraps `compare_runs`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x] 4.2 Register the data router in `agent/coordinator/app.py`
    - _Requirements: 5.1_

  - [x] 4.3 Write property tests for data tools
    - **Property 11: Residue search filter correctness**
    - **Property 12: Export round-trip**
    - **Property 13: Annotation persistence round-trip**
    - **Validates: Requirements 5.2, 5.4, 5.5**

- [x] 5. Backend: Plots Router
  - [x] 5.1 Create `agent/coordinator/routers/plots.py` with 2 endpoints
    - `POST /api/plots/generate` — wraps `generate_plot`, returns file path + download URL
    - `GET /api/plots/{filename}` — serves generated PNG from PLOT_OUTPUT_DIR
    - _Requirements: 6.1, 6.2, 6.5_

  - [x] 5.2 Register the plots router in `agent/coordinator/app.py`
    - _Requirements: 6.1_

  - [x] 5.3 Write property test for plot generation
    - **Property 14: Plot generation produces a file**
    - **Validates: Requirements 6.2**

- [x] 6. Backend: Hydration Endpoint
  - [x] 6.1 Add `GET /api/structures/{id}/hydrate` to the dashboard router
    - Calls embeddings, graph metrics, allosteric sites, source leaks, hypotheses, provenance, annotations in parallel
    - Returns HydrationResponse with null for missing data types
    - Includes persistence_status flags
    - _Requirements: 1.1, 1.4, 8.1_

  - [x] 6.2 Write property tests for hydration
    - **Property 1: Hydration returns all available data types**
    - **Property 2: Agent context reflects hydrated data**
    - **Property 15: Write operations return provenance metadata**
    - **Validates: Requirements 1.1, 1.4, 8.1**

- [x] 7. Checkpoint — Backend complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Frontend: Hydration context and panel infrastructure
  - [x] 8.1 Create `src/context/HydrationProvider.tsx`
    - React context that fetches `/api/structures/{id}/hydrate` on structure selection
    - Exposes hydrated data to all child panels
    - Tracks loading state per data type
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 8.2 Create `src/components/controls/PersistenceIndicator.tsx`
    - Green checkmark when data is persisted, red warning when not
    - Reads from hydration persistence_status
    - _Requirements: 8.2_

  - [x] 8.3 Create `src/components/controls/VisualizationToolbar.tsx`
    - Highlight button, color metric dropdown, focus button, clear button
    - Emits viewport directives to the viewer components
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 8.4 Update `src/lib/api.ts` with all new endpoint wrappers
    - Add methods for: rcsb search (4 modes), graph (5 endpoints), hypotheses (5), data (8), plots (2), hydrate (1)
    - _Requirements: all_

  - [x] 8.5 Update `src/lib/types.ts` with all new TypeScript interfaces
    - Add: RCSBSearchRequest, GraphMetricsData, Hypothesis, Prediction, Evidence, Annotation, PlotRequest, ViewportDirective, HydrationResponse
    - _Requirements: all_

- [x] 9. Frontend: RCSB Search Panel
  - [x] 9.1 Create `src/components/panels/RCSBSearchPanel.tsx`
    - Tab interface for 4 search modes (text, sequence, structure, functional)
    - Results cards with title, resolution, method, Ingest button
    - Loading/error states
    - _Requirements: 2.1, 2.5, 2.6, 2.7_

- [x] 10. Frontend: Graph Topology Panel
  - [x] 10.1 Create `src/components/panels/GraphTopologyPanel.tsx`
    - Sortable metrics table (7 columns)
    - "Find Bridges" button → highlights in viewer
    - "H-Bond Network" button → highlights in viewer
    - Shortest path: two-residue selector + "Find Path" button
    - Compare mode: structure selector + diff view
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 11. Frontend: Hypothesis Panel
  - [x] 11.1 Create `src/components/panels/HypothesisPanel.tsx`
    - Hypothesis cards with status badge, confidence bar, prediction summary
    - "New Hypothesis" form with statement, predictions array, mechanism
    - "Test" button per hypothesis
    - "Add Evidence" form (supports/contradicts toggle, strength slider)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 12. Frontend: Data Tools Panel
  - [x] 12.1 Create `src/components/panels/DataToolsPanel.tsx`
    - Residue search/filter form (chain, name, uncertainty range, depth range)
    - Allosteric sites display with residue lists
    - Export button (CSV/JSON selector)
    - Annotations timeline with "Add" form
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 12.2 Create `src/components/panels/ProvenanceTree.tsx`
    - Run history list with parent/child relationships
    - Click to expand run summary
    - Compare two runs button
    - _Requirements: 5.6, 5.7, 5.8_

- [x] 13. Frontend: Plot Generator Panel
  - [x] 13.1 Create `src/components/panels/PlotGeneratorPanel.tsx`
    - Plot type dropdown (6 types)
    - Dynamic parameter inputs per plot type
    - Generate button → inline image display + download link
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 14. Frontend: Integration and wiring
  - [x] 14.1 Wire all panels into App.tsx layout
    - Add collapsible panel sections to the sidebar/content area
    - Connect HydrationProvider to structure selection
    - Wire viewport directives from agent chat to viewer
    - _Requirements: 1.1, 7.6_

  - [x] 14.2 Wire viewport directives from all panels to viewer
    - Graph bridges → highlight in MolecularViewer + PoincareScatter
    - Hypothesis residue references → highlight
    - Search results → highlight
    - Shortest path → sequential highlight
    - _Requirements: 7.2, 7.6_

- [x] 15. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.
  - Verify: hydration loads all data on structure select
  - Verify: each panel calls its endpoints and displays results
  - Verify: viewport directives flow from panels to viewer

## Notes

- All new routers wrap existing tool functions — no new business logic needed
- The existing `agent/coordinator/routers/tools.py` remains for backward compatibility (agent tool-calling path)
- Frontend panels are collapsible — the layout doesn't get overwhelming
- Hydration is a single GET call that parallelizes all sub-queries server-side
- The agent chat context is automatically enriched with hydrated data
- Plot images are served as static files from the PLOT_OUTPUT_DIR
- All tasks including property tests are required

