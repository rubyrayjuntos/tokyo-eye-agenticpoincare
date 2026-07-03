# Implementation Plan: Structure Onboarding Panel

## Overview

Add a "Structures" tab to the default dashboard layout with RCSB search, direct ingest, structure selection, and auto-pipeline. Reuse existing API client methods and adapt UI patterns from existing panel components.

## Tasks

- [x] 1. Create the `useAutoIngestPipeline` hook
  - Create `visualizer/frontend/src/lib/useAutoIngestPipeline.ts`
  - Implement ingest → pipeline → poll → complete/error flow
  - Use `api.ingest`, `api.runPipeline`, `api.getPipelineStatus` from existing API client
  - Poll every 2s, cleanup on unmount
  - _Requirements: 3.1, 4.1, 4.2, 4.3, 4.4_

- [x] 1.1 Write property test for auto-pipeline trigger
  - **Property 2: Auto-pipeline triggers after successful ingest**
  - **Validates: Requirements 3.1, 4.1**

- [x] 1.2 Write property test for pipeline completion
  - **Property 3: Pipeline completion activates structure and refreshes dashboard**
  - **Validates: Requirements 4.3, 5.4**

- [x] 2. Create the `StructureOnboardingPanel` component
  - Create `visualizer/frontend/src/components/StructureOnboardingPanel.tsx`
  - Implement four-section tabbed layout: RCSB Search, Direct Ingest, Structure Selector, Pipeline Status
  - RCSB Search: query input, search button, results list with "Ingest & Run" per result
  - Direct Ingest: PDB ID input, submit button, uses `useAutoIngestPipeline`
  - Structure Selector: fetch and list structures, click to set active, highlight active
  - Pipeline Status: show current job progress/step/status from the hook
  - Read `activeStructure`, `setActiveStructure`, `triggerRefresh`, `refreshKey` from `useDashboard()`
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 3.2, 3.3, 4.2, 4.4, 5.1, 5.2, 5.3, 5.4_

- [x] 3. Wire into IdeLayout
  - [x] 3.1 Add "Structures" tab to DEFAULT_LAYOUT right-side tabset
    - Add `{ type: "tab", name: "Structures", component: "structure_onboarding" }` to the right tabset children
    - _Requirements: 1.1_
  - [x] 3.2 Add factory case for `structure_onboarding`
    - Import `StructureOnboardingPanel` and add switch case in factory
    - _Requirements: 1.1_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure the build passes (`npx vite build`)
  - Verify the new tab appears in the default layout
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required (comprehensive mode)
- Each task references specific requirements for traceability
- The `useAutoIngestPipeline` hook is the core logic unit; the panel component is mostly UI composition
- Existing `RCSBSearchPanel` and `PipelineControls` components remain available as reference but are not reused directly (they have different layout assumptions)
