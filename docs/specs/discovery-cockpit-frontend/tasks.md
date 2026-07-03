# Implementation Plan: Discovery Cockpit Frontend

## Overview

Rebuilds the frontend component tree following the Discovery Cockpit design while preserving backend protocol contracts. Build order: chrome/layout → WebSocket wiring → chat rail → slot viewers → selection sync → directives → onboarding. Existing integration hooks (useViewportSocket, api.ts, useDirectives) are preserved and adapted.

## Tasks

- [x] 1. Project scaffold and Tailwind theme
  - [x] 1.1 Set up new component directory structure under `visualizer/frontend/src/` (replace existing components/, keep lib/)
    - Create: components/layout/, components/viewers/, components/chat/, components/tools/, components/onboard/
    - Preserve: lib/useViewportSocket.ts, lib/api.ts, lib/useDirectives.ts, lib/useOrchestratorPolicy.ts, lib/types.ts, lib/context.ts
    - _Requirements: 8.1, 8.2, 8.3, 8.5_
  - [x] 1.2 Configure Tailwind theme with Discovery Cockpit tokens
    - Extend tailwind.config with colors: bg (#07080C, #0F1117, #161822), teal (#5DDBC2), magenta (#C026D3), slate, text
    - Add font families, spacing scale, border-radius tokens matching prototype
    - _Requirements: 1.2_

- [x] 2. Layout chrome (static, mock state)
  - [x] 2.1 Implement CockpitLayout component with CSS Grid 3-panel layout
    - grid-cols-[320px_1fr_380px], responsive breakpoints at <1024px
    - Slots for left (Poincaré), center (3D), right (chat), top (nav), and collapsible sidebar (tools)
    - _Requirements: 1.1, 1.3, 1.6_
  - [x] 2.2 Implement NavBar component
    - Structure badge (PDB ID, structure_id), phase stepper (6 phases as dots/labels), lifecycle badge, connection indicator
    - Props: structureId, discoveryPhase, hypothesisLifecycle, connectionStatus
    - _Requirements: 1.4_
  - [x] 2.3 Implement ToolDock component (collapsible sidebar)
    - Renders tool list from allowedTools, grays out blockedTools with prerequisite tooltip
    - Opens tool panels as overlays without occluding main viewers
    - Props: allowedTools, blockedTools, activePanel, onPanelSelect
    - _Requirements: 1.5, 5.3_
  - [x] 2.4 Wire App.tsx to render CockpitLayout with mock data, verify layout renders correctly
    - _Requirements: 1.1_

- [x] 3. WebSocket state derivation
  - [x] 3.1 Connect useViewportSocket hook to App.tsx, derive state from state_snapshot messages
    - On state_snapshot: update discoveryPhase, hypothesisLifecycle, plannerPolicy, selectedResidue in App state
    - On phase_transition: update phase indicator and ToolDock allowed/blocked
    - On disconnect: set connectionStatus='disconnected', show indicator in NavBar
    - On reconnect: receive snapshot, restore full state
    - _Requirements: 5.1, 5.2, 5.4, 5.5, 5.6_
  - [x] 3.2 Write property tests for state derivation
    - **Property 1: State derivation from backend**
    - **Property 5: Tool dock phase gating**
    - **Property 6: Reconnection state recovery**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

- [x] 4. Chat rail
  - [x] 4.1 Implement ChatRail component
    - Message input, message list with streaming support, directive annotations
    - buildViewportState() assembles payload from current App state (poincare color mode, selected residue, highlighted residues, viewer_3d state, active panel, pipeline status)
    - Sends viewport_state with each POST /api/agent/chat
    - Shows discovery phase and lifecycle as contextual badges in header
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [x] 4.2 Write property tests for viewport state payload
    - **Property 3: Viewport state payload completeness**
    - **Validates: Requirements 4.1, 8.4**

- [x] 5. Checkpoint - Verify layout + WS + chat working
  - Ensure layout renders, WebSocket connects, state derives correctly, chat sends viewport state. Ask user if questions arise.

- [x] 6. Slot viewers into panels
  - [x] 6.1 Implement PoincarePanel wrapping existing PoincareScatter rendering logic
    - Panel chrome: color mode selector dropdown, detail tooltip on hover, Möbius focus toggle
    - Wire click handler to emit selection through useViewportSocket
    - Wire color mode to respond to set_metric directives
    - Wire highlight state from backend directive/selection
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_
  - [x] 6.2 Implement MolecularPanel wrapping existing MolecularViewer Three.js logic
    - Panel chrome: color mode selector, highlight info bar
    - Wire click handler to emit selection through useViewportSocket
    - Wire camera animation on selection_sync and focus directives
    - Wire highlight state from backend directive/selection
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 7. Selection sync and directives
  - [x] 7.1 Wire cross-viewport selection sync
    - On selection from any source (Poincaré click, 3D click, agent directive): emit through WS
    - On selection_sync received: update both viewers (highlight + camera in 3D, highlight + tooltip in Poincaré)
    - Clear previous selection before applying new one
    - Handle selection_error gracefully (toast notification)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [x] 7.2 Wire directive consumption via useDirectives
    - On viewport_directive(highlight): apply to both viewers
    - On viewport_directive(focus): center 3D camera on residue
    - On viewport_directive(set_metric): change Poincaré color mode
    - On viewport_directive(clear): reset all highlights and selections
    - Show directive announcement in chat before execution (inspect-preview-commit)
    - _Requirements: 2.4, 2.5, 3.2, 3.3, 4.3_
  - [x] 7.3 Write property tests for selection sync and directives
    - **Property 2: Selection sync round-trip**
    - **Property 4: Directive application correctness**
    - **Validates: Requirements 2.4, 2.5, 3.2, 3.3, 6.1, 6.2, 6.3, 6.4**

- [x] 8. Structure onboarding
  - [x] 8.1 Implement StructureOnboard component
    - PDB ID input with validation (4-char alphanumeric)
    - Progress indicator during ingestion (calls POST /api/ingest)
    - Error display on failure (404, timeout)
    - Previously ingested structures list (calls GET /api/structures)
    - On completion: triggers structure load into both viewers
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Final checkpoint - Full integration test
  - Verify end-to-end: ingest structure → viewers populate → click residue → sync works → chat with viewport context → agent directive applied. Ask user if questions arise.

## Notes

- All tasks are required (no optional markers)
- Property tests use fast-check; unit tests use Vitest; e2e uses Playwright
- The lib/ directory is preserved — hooks are adapted, not rewritten
- Existing PoincareScatter.tsx and MolecularViewer.tsx canvas/Three.js logic is reused inside new panel chrome components
- The old component files (AgentChat.tsx, DockedLayout.tsx, etc.) are deprecated once new equivalents are verified working
- Vite config unchanged: port 3000, proxy /api and /ws to agent container :8000
