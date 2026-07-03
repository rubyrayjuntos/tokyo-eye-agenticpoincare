# Requirements Document

## Introduction

The dashboard currently launches with Poincaré Manifold, Molecular Structure, Agent Chat, and Agent Telemetry tabs — but no way to search RCSB, ingest structures, run the pipeline, or select already-ingested structures. This feature adds a unified "Structure Onboarding" panel visible on initial load that provides RCSB search, direct PDB ID ingest, pipeline execution, and a structure selector — with automatic normalize→inference→pipeline execution after any RCSB ingest.

## Glossary

- **Structure_Onboarding_Panel**: A tab-based UI panel providing RCSB search, PDB ingest, pipeline controls, and structure selection in the initial dashboard layout.
- **Auto_Pipeline**: The automatic triggering of normalize + inference + pipeline steps immediately after a structure is ingested from RCSB.
- **IdeLayout**: The flexlayout-react based docked panel system that renders the dashboard's main content area.
- **Dashboard_Context**: The React context providing shared state (activeStructure, refreshKey, etc.) across all panels.

## Requirements

### Requirement 1: Structure Onboarding Panel in Default Layout

**User Story:** As a researcher, I want to see a structure management panel when the dashboard first loads, so that I can immediately begin searching, ingesting, or selecting structures without navigating through hidden menus.

#### Acceptance Criteria

1. WHEN the dashboard loads for the first time, THE IdeLayout SHALL display a "Structures" tab in the right-side tabset alongside Agent Chat and Agent Telemetry.
2. THE Structure_Onboarding_Panel SHALL contain four sections: RCSB Search, Direct Ingest, Structure Selector, and Pipeline Status.
3. WHEN a user switches between sections, THE Structure_Onboarding_Panel SHALL preserve the state of each section (search results, input values).

### Requirement 2: RCSB Search Integration

**User Story:** As a researcher, I want to search RCSB by keyword from the onboarding panel, so that I can find protein structures to analyze.

#### Acceptance Criteria

1. WHEN a user enters a search query and submits, THE Structure_Onboarding_Panel SHALL call the RCSB search API and display results with PDB ID, title, resolution, and method.
2. WHEN search results are displayed, THE Structure_Onboarding_Panel SHALL provide an "Ingest & Run" button for each result.
3. IF the RCSB search API returns an error, THEN THE Structure_Onboarding_Panel SHALL display the error message and allow retry.

### Requirement 3: Direct PDB Ingest

**User Story:** As a researcher, I want to ingest a structure by typing its PDB ID directly, so that I can quickly load known structures without searching.

#### Acceptance Criteria

1. WHEN a user types a PDB ID and submits, THE Structure_Onboarding_Panel SHALL call the ingest API and trigger Auto_Pipeline upon success.
2. WHILE ingestion is in progress, THE Structure_Onboarding_Panel SHALL display a loading indicator and disable the submit button.
3. IF ingestion fails, THEN THE Structure_Onboarding_Panel SHALL display the error message and allow retry.

### Requirement 4: Auto-Pipeline After Ingest

**User Story:** As a researcher, I want the full normalize→inference→pipeline to run automatically after ingesting a structure from RCSB, so that I don't have to manually trigger each step.

#### Acceptance Criteria

1. WHEN a structure is successfully ingested (from RCSB search or direct ingest), THE Structure_Onboarding_Panel SHALL automatically call the pipeline run API with all default modules enabled.
2. WHILE the pipeline is running, THE Structure_Onboarding_Panel SHALL display progress (current step and percentage) in the Pipeline Status section.
3. WHEN the pipeline completes successfully, THE Structure_Onboarding_Panel SHALL set the ingested structure as the active structure and trigger a dashboard refresh.
4. IF the pipeline fails, THEN THE Structure_Onboarding_Panel SHALL display the error and provide a retry button.

### Requirement 5: Structure Selector

**User Story:** As a researcher, I want to select from previously ingested structures, so that I can switch between analyses without re-ingesting.

#### Acceptance Criteria

1. WHEN the Structure_Onboarding_Panel loads, THE Structure_Onboarding_Panel SHALL fetch and display all previously ingested structures with their PDB ID, title, and processing status.
2. WHEN a user clicks a structure in the list, THE Structure_Onboarding_Panel SHALL set it as the active structure in Dashboard_Context.
3. THE Structure_Onboarding_Panel SHALL visually indicate which structure is currently active.
4. WHEN a new structure is ingested or pipeline completes, THE Structure_Onboarding_Panel SHALL refresh the structure list.
