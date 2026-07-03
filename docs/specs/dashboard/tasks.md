# Implementation Plan: Tokyo Eye Dashboard

## Overview

Build the production dashboard by: (1) adding direct REST API endpoints to the backend, (2) rebuilding the frontend as a dashboard workbench, (3) wiring the agent as a lightweight context-aware sidebar. The existing agent coordinator, science dispatch, and data layer remain unchanged — we're adding a new router and replacing the frontend.

## Tasks

- [x] 1. Backend: Dashboard API router
  - [x] 1.1 Create `agent/coordinator/routers/dashboard.py` with all 8 endpoints
    - `POST /api/ingest` — calls existing `ingest_structure` + `parse_and_populate` directly
    - `GET /api/structures` — queries `dim_structure` with join counts
    - `GET /api/structures/{id}/embeddings` — queries `fact_gnn_node_embedding` for Poincaré coords
    - `GET /api/structures/{id}/metrics` — queries graph topology views
    - `GET /api/kpis` — aggregate queries (COUNT, AVG, system checks)
    - `POST /api/pipeline/run` — dispatches to science container, stores job state
    - `GET /api/pipeline/status/{job_id}` — returns job progress
    - `POST /api/agent/chat` — lightweight agent (no tools, context-only)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 1.2 Add pipeline job state management (in-memory dict or DB table)
    - Track job_id → {status, current_step, progress, started_at, error}
    - Background task runs science container and updates state
    - _Requirements: 1.2, 1.3, 3.3_

  - [x] 1.3 Register the dashboard router in `agent/coordinator/app.py`
    - _Requirements: 1.1_

  - [x] 1.4 Write property tests for API contracts
    - **Property 1: Ingest returns complete metadata**
    - **Property 3: Structures endpoint reflects ingested count**
    - **Property 5: Error responses are structured JSON**
    - **Validates: Requirements 1.1, 1.4, 1.8**

- [x] 2. Backend: Context-aware agent endpoint
  - [x] 2.1 Implement `POST /api/agent/chat` with no-tools agent
    - Create a minimal Agent with empty tool list
    - System prompt focused on interpretation/analysis only
    - Accept context payload (active structure, metrics summary)
    - Inject context into system prompt, not as tool calls
    - Cap output at 1024 tokens (max_tokens on provider)
    - _Requirements: 5.2, 5.3, 5.4_

  - [x] 2.2 Implement session history for agent chat
    - Store last 10 exchanges per session_id (in-memory)
    - Include compact history in system prompt
    - _Requirements: 5.6_

  - [x] 2.3 Write property tests for agent contract
    - **Property 6: Context includes active structure data**
    - **Property 7: Agent has no execution tools**
    - **Property 8: Agent responses bounded at 2000 tokens**
    - **Property 9: Conversation history persists**
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.6**

- [x] 3. Checkpoint — Backend complete
  - Ensure all API endpoints work with `curl` or httpie
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Frontend: Project setup and layout shell
  - [x] 4.1 Set up new React app (or refactor existing `visualizer/frontend/`)
    - Keep Vite + Tailwind + TypeScript
    - Add dependencies: chart.js, three, 3dmol
    - Create `src/lib/api.ts` with typed fetch wrappers for all endpoints
    - Create `src/lib/types.ts` with TypeScript interfaces from design
    - _Requirements: 2.1, 2.6_

  - [x] 4.2 Build layout shell: NavBar + KPIBar + Sidebar + Content grid
    - NavBar: branding, DB status dot (polls `/api/kpis`), user info
    - KPIBar: 5 metric cards from `/api/kpis`
    - Sidebar: PipelineControls component
    - Content: 2x2 grid placeholders + ResultsTable
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 5. Frontend: Pipeline controls and results table
  - [x] 5.1 Build PipelineControls component
    - PDB ID input + Ingest button → calls `POST /api/ingest`
    - Target selector dropdown
    - Pipeline module checkboxes
    - Execute Pipeline button → calls `POST /api/pipeline/run` + polls status
    - Status indicator with step-by-step progress
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 5.2 Build ResultsTable component
    - Fetches from `GET /api/structures`
    - Columns: ID, target, status, uncertainty, actions (2D/3D/download)
    - Click row → sets active structure in React context
    - Auto-refreshes after pipeline completion
    - _Requirements: 2.4, 3.2, 3.5_

- [x] 6. Frontend: Visualizations
  - [x] 6.1 Build PoincareScatter component (Chart.js scatter)
    - Fetches from `GET /api/structures/{id}/embeddings`
    - Plots residues as (x, y) colored by cone_depth or uncertainty
    - Resample button re-fetches data
    - _Requirements: 4.1, 4.5, 4.6_

  - [x] 6.2 Build ResidueBarChart component (Chart.js bar)
    - Fetches from `GET /api/structures/{id}/metrics`
    - Shows per-residue dehydron score or centrality
    - Updates when active structure changes
    - _Requirements: 4.2, 4.5_

  - [x] 6.3 Build LatentSpace3D component (Three.js or Plotly)
    - Renders 3D point cloud from embedding data
    - Rotation, zoom, color by metric
    - _Requirements: 4.3_

  - [x] 6.4 Build MolecularViewer component (3Dmol.js)
    - Loads PDB from RCSB by ID
    - Cartoon representation with spectrum coloring
    - Highlights regions referenced by agent or selected in table
    - _Requirements: 4.4, 5.5_

- [x] 7. Frontend: Agent chat sidebar
  - [x] 7.1 Build AgentChat component
    - Collapsible panel (button in nav to toggle)
    - Sends messages to `POST /api/agent/chat` with current context
    - Displays responses with markdown rendering
    - Shows "thinking" indicator while waiting
    - Parses residue references from response → emits highlight events
    - _Requirements: 5.1, 5.2, 5.5, 5.6_

- [x] 8. Integration and wiring
  - [x] 8.1 Wire React context for active structure
    - When user clicks a row in ResultsTable, all viz components re-fetch
    - AgentChat includes active structure in context payload
    - _Requirements: 4.5, 4.6, 5.2_

  - [x] 8.2 Wire status polling and auto-refresh
    - After pipeline completes, trigger ResultsTable + viz refresh
    - KPIBar polls `/api/kpis` every 10s
    - _Requirements: 3.5, 4.6_

- [x] 9. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.
  - Verify: ingest → pipeline → visualize → ask agent flow works end-to-end

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The existing `agent/coordinator/routers/chat.py` remains for backward compatibility
- The science container dispatch (`agent/tools/science_dispatch.py`) is reused by the pipeline endpoint
- No changes to the data layer, normalizer, or science code
- Frontend can be developed in parallel with backend (mock API responses)
