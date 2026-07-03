# Requirements Document

## Introduction

This feature replaces the current Tokyo Eye frontend with the "Discovery Cockpit" design — a new visual architecture that maintains all existing backend orchestration protocol contracts (WebSocket state push, viewport directives, context enrichment, selection sync) while delivering a redesigned layout optimized for the scientific instrument workflow. The HTML prototypes in `visualizer/frontend-refactor/` define the visual target; the existing `visualizer/frontend/src/lib/` integration hooks (useViewportSocket, useOrchestratorPolicy, useDirectives, viewportMachine) define the contract.

The Discovery Cockpit is not a greenfield build — it reuses the proven backend protocol and integration patterns while replacing the component tree and layout system.

## Glossary

- **Discovery_Cockpit**: The new dashboard layout with Poincaré disc, 3D molecular viewer, chat panel, and tool sidebar arranged for scientific workflow navigation
- **Backend_Protocol**: The existing WebSocket + REST contract: state_snapshot push, viewport_directive emission, selection_sync broadcasting, context enrichment on chat
- **Viewport_Socket**: The WebSocket connection managed by useViewportSocket that receives orchestrator state pushes and directive commands
- **Directive_Consumer**: Frontend logic that receives and applies viewport directives (highlight, focus, set_metric, clear) to visual components
- **State_Derivation**: Frontend rendering derived from backend-pushed discovery phase and hypothesis lifecycle state — no independent state machines
- **Dev_Handoff**: The HTML prototype in frontend-refactor/ that defines visual design, layout, color palette, and component boundaries

## Requirements

### Requirement 1: Layout and Visual Architecture

**User Story:** As a researcher, I want a cohesive dark-themed cockpit layout where Poincaré disc, 3D structure viewer, and chat panel are simultaneously visible, so that I can correlate abstract topological signals with spatial structure while conversing with the agent.

#### Acceptance Criteria

1. THE Discovery_Cockpit SHALL render a 3-panel layout: Poincaré disc (left), 3D molecular viewer (center), and chat panel (right)
2. THE layout SHALL use the dark theme palette from the Dev_Handoff prototype (background #07080C, accent magenta #C026D3, teal #5DDBC2)
3. THE layout SHALL be responsive: panels resize proportionally on viewport change with minimum usable dimensions
4. THE Discovery_Cockpit SHALL include a top navigation bar showing structure identity, discovery phase indicator, and hypothesis lifecycle badge
5. THE Discovery_Cockpit SHALL include a collapsible tool sidebar that displays available tools filtered by current discovery phase
6. WHEN a tool panel is opened from the sidebar, THE Discovery_Cockpit SHALL render the panel content without occluding the main viewers

### Requirement 2: Poincaré Disc Viewer Integration

**User Story:** As a researcher, I want the Poincaré disc viewer to display hyperbolic embeddings with color-coded metrics, residue selection, and Möbius focus.

#### Acceptance Criteria

1. THE Poincare_Viewer SHALL render residue points on the Poincaré disc using coordinates from the embeddings API endpoint
2. THE Poincare_Viewer SHALL support color modes: cone_depth, epistemic_uncertainty, aleatoric_uncertainty, plasticity, allosteric, resistance
3. WHEN a residue is clicked on the Poincaré disc, THE Poincare_Viewer SHALL emit a selection event through the Viewport_Socket for cross-viewport sync
4. WHEN a viewport_directive with action `highlight` is received, THE Poincare_Viewer SHALL visually distinguish the specified residues
5. WHEN a viewport_directive with action `set_metric` is received, THE Poincare_Viewer SHALL switch to the specified color mode
6. THE Poincare_Viewer SHALL support Möbius focus transformation centering the disc on a selected residue
7. THE Poincare_Viewer SHALL support brush selection for selecting multiple residues in a region

### Requirement 3: 3D Molecular Viewer Integration

**User Story:** As a researcher, I want the 3D structure viewer to display the protein with synchronized highlighting and camera control driven by both user interaction and agent directives.

#### Acceptance Criteria

1. THE Molecular_Viewer SHALL render the protein structure using Three.js with cartoon representation
2. WHEN a selection_sync event is received, THE Molecular_Viewer SHALL highlight the selected residue and animate the camera to center on its 3D coordinates
3. WHEN a viewport_directive with action `focus` is received, THE Molecular_Viewer SHALL center the camera on the specified residue
4. THE Molecular_Viewer SHALL support color modes matching the Poincaré viewer: cone_depth, epistemic, aleatoric, plasticity, allosteric, resistance
5. WHEN a residue is clicked in the 3D viewer, THE Molecular_Viewer SHALL emit a selection event through the Viewport_Socket
6. THE Molecular_Viewer SHALL support highlighting multiple residues simultaneously when receiving highlight directives

### Requirement 4: Chat Panel with Context Enrichment

**User Story:** As a researcher, I want the chat panel to send my current viewport state with each message and render agent directives inline with responses.

#### Acceptance Criteria

1. THE Chat_Panel SHALL send the current viewport_state payload with each chat message (poincare color mode, selected residue, highlighted residues, 3D viewer state, active panel, pipeline status)
2. THE Chat_Panel SHALL render agent messages with directive annotations showing what viewport actions the agent performed
3. WHEN the agent's response includes viewport directives, THE Chat_Panel SHALL display a preview annotation before the directive executes (inspect-preview-commit)
4. THE Chat_Panel SHALL display the current discovery phase and hypothesis lifecycle as contextual indicators
5. THE Chat_Panel SHALL support streaming responses from the agent
6. WHEN no structure is loaded, THE Chat_Panel SHALL still function for general queries without viewport context

### Requirement 5: Backend State Derivation

**User Story:** As a system architect, I want the frontend to derive all workflow state from backend WebSocket pushes rather than maintaining independent state machines.

#### Acceptance Criteria

1. WHEN a state_snapshot message is received via WebSocket, THE Discovery_Cockpit SHALL update all displayed phase and lifecycle indicators
2. WHEN a phase_transition message is received, THE Discovery_Cockpit SHALL update the discovery phase indicator and tool sidebar availability
3. THE tool sidebar SHALL filter available tools based on the allowed/blocked lists from the backend-pushed PlannerPolicy
4. WHEN the WebSocket disconnects, THE Discovery_Cockpit SHALL display a connection status indicator and attempt reconnection
5. WHEN the WebSocket reconnects, THE Discovery_Cockpit SHALL receive a full state_snapshot and re-render all state-derived UI
6. THE Discovery_Cockpit SHALL NOT maintain independent discovery phase or hypothesis lifecycle state machines — all state comes from backend push

### Requirement 6: Cross-Viewport Selection Sync

**User Story:** As a researcher, I want clicking a residue in any viewer to simultaneously highlight it in all other viewers with camera centering.

#### Acceptance Criteria

1. WHEN a residue is selected in any viewport, THE Selection_Sync SHALL propagate the canonical ResidueSelection (structure_id, chain_id, residue_number) to all other viewports via the backend Sync_Bus
2. WHEN a selection is propagated to the 3D viewer, THE Molecular_Viewer SHALL highlight the residue and smoothly animate the camera to center on it
3. WHEN a selection is propagated to the Poincaré disc, THE Poincare_Viewer SHALL highlight the corresponding point and show its metrics in a tooltip or detail panel
4. WHEN a new selection is made, THE previous selection SHALL be cleared from all viewports before the new one applies
5. WHEN a selection references a residue that does not exist in the current structure, THE system SHALL display an error state rather than silently failing

### Requirement 7: Structure Loading and Onboarding

**User Story:** As a researcher, I want to load structures by PDB ID and see the full ingestion progress before the cockpit populates.

#### Acceptance Criteria

1. THE Discovery_Cockpit SHALL provide a structure search/input component for submitting PDB IDs
2. WHEN a PDB ID is submitted, THE Discovery_Cockpit SHALL call POST /api/ingest and display progress feedback
3. WHEN ingestion completes, THE Discovery_Cockpit SHALL automatically load the structure data and populate both viewers
4. THE Discovery_Cockpit SHALL display a list of previously ingested structures for quick selection
5. WHEN a structure is loading, THE Discovery_Cockpit SHALL show loading states in both viewers without blocking chat

### Requirement 8: Existing Integration Hook Reuse

**User Story:** As a developer, I want the new frontend to reuse proven integration patterns from the existing codebase where applicable.

#### Acceptance Criteria

1. THE Discovery_Cockpit SHALL reuse the WebSocket protocol logic from the existing useViewportSocket hook pattern
2. THE Discovery_Cockpit SHALL reuse the REST API client pattern from the existing api.ts module
3. THE Discovery_Cockpit SHALL reuse the directive consumption pattern from useDirectives
4. THE Discovery_Cockpit SHALL use the same viewport_state payload shape that the backend context_builder expects
5. THE Discovery_Cockpit SHALL maintain the same Vite dev server configuration (port 3000, proxy to agent on 8000)

</content>
</invoke>