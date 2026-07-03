# Requirements Document

## Introduction

A production research dashboard for the Tokyo Eye platform that separates concerns: the frontend handles all direct actions (ingestion, pipeline execution, DB queries, visualization) via REST API calls, while the AI agent operates as a context-aware analyst sidebar that interprets results and answers questions without consuming tokens on mechanical operations.

Reference designs: `.kiro/specs/dashboard/mock1.html` and `.kiro/specs/dashboard/mock2.html`

## Glossary

- **Dashboard**: The React frontend application serving as the primary research workbench
- **Backend_API**: FastAPI REST endpoints that the frontend calls directly (no LLM involved)
- **Agent_Sidebar**: A chat panel that receives current dashboard context and provides scientific analysis
- **Pipeline**: The DTIE v5 computation chain (ingest → parse → GNN → phases → results)
- **Science_Container**: Docker container with torch/GPU that executes heavy compute
- **KPI_Bar**: Top-level metrics strip showing system-wide statistics
- **Discovery_Ledger**: Table of pipeline results with per-structure metrics

## Requirements

### Requirement 1: Backend API Endpoints (Direct Frontend Access)

**User Story:** As a researcher, I want the frontend to call backend APIs directly for all mechanical operations, so that no LLM tokens are wasted on ingestion, pipeline execution, or data retrieval.

#### Acceptance Criteria

1. THE Backend_API SHALL expose a `POST /api/ingest` endpoint that accepts a PDB ID, fetches the structure from RCSB, parses the CIF, populates dimensional tables, and returns structure metadata
2. THE Backend_API SHALL expose a `POST /api/pipeline/run` endpoint that dispatches the full DTIE pipeline to the Science_Container and returns a job ID with status polling URL
3. THE Backend_API SHALL expose a `GET /api/pipeline/status/{job_id}` endpoint that returns current pipeline step, progress percentage, and completion status
4. THE Backend_API SHALL expose a `GET /api/structures` endpoint that returns all ingested structures with their metadata, residue counts, and pipeline run history
5. THE Backend_API SHALL expose a `GET /api/structures/{structure_id}/embeddings` endpoint that returns per-residue Poincaré disc coordinates, cone depth, and uncertainty values
6. THE Backend_API SHALL expose a `GET /api/structures/{structure_id}/metrics` endpoint that returns graph topology metrics (centrality, bridges, clustering coefficients)
7. THE Backend_API SHALL expose a `GET /api/kpis` endpoint that returns system-wide statistics (total structures, mean uncertainty, active jobs, model status)
8. WHEN any Backend_API endpoint encounters an error, THE Backend_API SHALL return a structured JSON error with a human-readable message and HTTP status code

### Requirement 2: Dashboard Layout and Navigation

**User Story:** As a researcher, I want a dashboard with a pipeline controls sidebar, KPI bar, visualization grid, and results table, so that I can drive the entire research workflow from one screen.

#### Acceptance Criteria

1. THE Dashboard SHALL display a sticky top navigation bar with branding, system status indicator (DB green/red, Science Container green/red), and user info
2. THE Dashboard SHALL display a KPI metrics strip below the nav showing: proteins ingested, mean uncertainty, average inference time, active jobs, and model status
3. THE Dashboard SHALL display a left sidebar with pipeline controls: PDB ID input with ingest button, target selector, pipeline module toggles, and Execute Pipeline button with live status
4. THE Dashboard SHALL display a main content area with a results table and a 2x2 visualization grid (2D scatter, 2D bar, 3D latent space, 3D molecular viewer)
5. WHEN the user resizes the browser window, THE Dashboard SHALL responsively adjust the grid layout without breaking visualizations
6. THE Dashboard SHALL use a dark biotech aesthetic consistent with the mock2.html reference design

### Requirement 3: Pipeline Execution (Frontend-Driven)

**User Story:** As a researcher, I want to ingest structures and run the pipeline directly from the dashboard controls, so that I don't need the agent for mechanical operations.

#### Acceptance Criteria

1. WHEN the user enters a PDB ID and clicks Ingest, THE Dashboard SHALL call `POST /api/ingest` and display a progress indicator until completion
2. WHEN ingestion completes, THE Dashboard SHALL add the new structure to the results table and update KPIs
3. WHEN the user clicks Execute Pipeline, THE Dashboard SHALL call `POST /api/pipeline/run` with the selected structure and modules, then poll status until completion
4. WHILE the pipeline is running, THE Dashboard SHALL display step-by-step progress (ingestion → graph build → GNN forward pass → DTIE decomposition → complete)
5. WHEN the pipeline completes, THE Dashboard SHALL refresh the results table, update KPIs, and refresh all visualizations with new data
6. IF the pipeline fails, THEN THE Dashboard SHALL display the error message in the status area without crashing

### Requirement 4: Visualizations (Data-Driven, No Agent)

**User Story:** As a researcher, I want interactive 2D and 3D visualizations that update automatically when new data is available, so that I can explore results visually without asking the agent.

#### Acceptance Criteria

1. THE Dashboard SHALL render a 2D Poincaré disc scatter plot showing per-residue hyperbolic embeddings colored by cluster or uncertainty
2. THE Dashboard SHALL render a 2D bar chart showing per-residue dehydron scores or centrality metrics for the active structure
3. THE Dashboard SHALL render a 3D latent space visualization (Poincaré ball or point cloud) with rotation and zoom
4. THE Dashboard SHALL render a 3D molecular viewer (3Dmol.js or NGL) showing the protein structure with cartoon representation and highlighted regions
5. WHEN the user selects a different structure from the results table, THE Dashboard SHALL update all four visualizations to reflect that structure's data
6. WHEN new pipeline results arrive, THE Dashboard SHALL refresh visualizations without requiring a page reload

### Requirement 5: Context-Aware Agent Sidebar

**User Story:** As a researcher, I want a chat sidebar where I can ask the agent questions about my current results, so that the agent provides scientific interpretation without wasting tokens on data retrieval or pipeline execution.

#### Acceptance Criteria

1. THE Dashboard SHALL include a collapsible chat sidebar (or modal) for conversing with the agent
2. WHEN the user sends a message, THE Dashboard SHALL include current context in the request: active structure ID, loaded metrics summary (top-5 uncertainty residues, source leak count, cone depth range), and current visualization state
3. THE Agent_Sidebar SHALL NOT have access to pipeline execution or ingestion tools — it only receives pre-computed context and answers questions
4. THE Agent_Sidebar SHALL respond with scientific interpretation, pattern analysis, and suggestions in under 2000 tokens per response
5. WHEN the agent references specific residues, THE Dashboard SHALL highlight those residues in the active visualization
6. THE Agent_Sidebar SHALL maintain conversation history within the session so follow-up questions have context
