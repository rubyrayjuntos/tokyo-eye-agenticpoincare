# Implementation Plan: Agent Viewport Context

## Overview

Enriches the chat agent's context with full dashboard viewport state by: (1) lifting view state from Poincaré/3D viewer components to App.tsx, (2) building a comprehensive context payload in AgentChat, (3) expanding the backend context formatter to handle the new fields.

## Tasks

- [x] 1. Frontend: Lift viewer state to App.tsx
  - [x] 1.1 Add viewport state to App.tsx and pass to AgentChat
    - Add state variables for poincareColorMode, poincareSelectedResidue, mobiusFocus, brushSelection, viewerColorMode, riskThreshold, activePanel
    - Wire callbacks from PoincareScatter, MolecularViewer, ToolPanelSidebar to update these
    - Pass as props to AgentChat
    - _Requirements: 1.1, 1.2, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 5.1_

- [x] 2. Frontend: Enrich AgentChat.buildContext()
  - [x] 2.1 Rewrite buildContext to produce full ViewportState payload
    - Add `useHydration()` hook to AgentChat
    - Assemble poincare section from lifted props
    - Assemble viewer_3d section from lifted props
    - Assemble data_summary from hydration data (counts, persistence, resistance summary)
    - Include active_panel and pipeline status
    - Handle null hydration gracefully
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3_

- [x] 3. Backend: Expand _build_context_block()
  - [x] 3.1 Rewrite _build_context_block to handle enriched payload
    - Format sections: Structure, Poincaré View, 3D Viewer, Data Availability, Active Analysis
    - Omit sections with null/empty data
    - Enforce 2000 character cap with truncation
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 3.2 Write property test for context block formatting
    - **Property 4: Context block formatting with null omission**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [x] 3.3 Write property test for context block size bound
    - **Property 5: Context block size bound**
    - **Validates: Requirements 6.4**

- [x] 4. Backend: Add resistance summary to context
  - [x] 4.1 Compute resistance classification counts in context builder
    - When resistance_summary is present in context, format λ₂, hinge count, and H/M/S distribution
    - _Requirements: 3.4, 4.3_

  - [x] 4.2 Write property test for resistance classification count accuracy
    - **Property 6: Resistance classification count accuracy**
    - **Validates: Requirements 3.4, 4.3**

- [x] 5. Checkpoint - Verify integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required (comprehensive testing from start)
- The frontend changes are primarily wiring — the state already exists in the viewer components, it just needs to be lifted to App.tsx
- `useHydration()` is already available in the component tree via HydrationProvider
- The backend `_build_context_block` is a pure function — easy to test with Hypothesis
- No database changes required
