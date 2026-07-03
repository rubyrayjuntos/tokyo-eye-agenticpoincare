# Requirements Document

## Introduction

This feature establishes the backend orchestration layer as the single source of truth for Tokyo Eye's scientific workflow state. It connects the existing frontend state machines to server-side enforcement, enables the agent to both perceive and act through the dashboard, and ensures cross-viewport synchronization when residues are selected from any source (user click, agent directive, or API call).

## Glossary

- **Orchestrator**: The backend module that maintains discovery phase and hypothesis lifecycle state, derives planner policy, and enforces tool gating
- **Discovery_Phase**: The current scientific workflow stage (residue, topology, structure, pocket, screening, report)
- **Hypothesis_Lifecycle**: The epistemic status of the current scientific claim (emergent, framed, testing, supported, contradicted, revised, synthesized)
- **Planner_Policy**: The derived policy object that combines phase, lifecycle, structure scope, and viewport state into allowed/blocked tools and reasoning posture
- **Viewport_Directive**: A JSON command sent from the backend to the frontend via WebSocket that instructs the dashboard to highlight residues, move the camera, change color modes, or clear selections
- **Context_Block**: A formatted text summary of the current dashboard viewport state prepended to the user's message before sending to the LLM
- **Residue_Selection**: A canonical residue identity (structure_id, chain_id, residue_number) that must resolve consistently across all linked viewports
- **Sync_Bus**: The WebSocket channel that propagates selection events to all subscribed viewports

## Requirements

### Requirement 1: Backend State Authority

**User Story:** As a system architect, I want the backend to be the single source of truth for discovery phase and hypothesis lifecycle state, so that all clients derive their state from one canonical source.

#### Acceptance Criteria

1. WHEN a new structure session begins, THE Orchestrator SHALL initialize discovery phase to `residue` and hypothesis lifecycle to `emergent`
2. WHEN a phase transition event occurs (tool completion, user override, hypothesis collapse), THE Orchestrator SHALL update the canonical state and push the new state to all connected clients via WebSocket
3. THE Frontend SHALL derive its displayed phase and lifecycle state from the backend-pushed state rather than maintaining independent state
4. WHEN multiple clients are connected to the same session, THE Orchestrator SHALL ensure all clients see the same phase and lifecycle state
5. IF the WebSocket connection drops and reconnects, THEN THE Orchestrator SHALL send the current full state snapshot to the reconnecting client

### Requirement 2: Context Enrichment

**User Story:** As a scientist, I want the chat agent to be aware of what I'm currently viewing in the dashboard, so that its answers reference my current focus rather than speaking generically.

#### Acceptance Criteria

1. WHEN a user sends a chat message, THE Frontend SHALL include the current viewport state in the request payload (selected residue, color mode, highlighted residues, active panel, Möbius focus state)
2. WHEN the backend receives a chat request with viewport context, THE Orchestrator SHALL format the viewport state into a compact text block and prepend it to the user's message before sending to the LLM
3. THE Context_Block SHALL include: structure identity, selected residue with metrics, Poincaré color mode, 3D viewer color mode, highlighted residue IDs, active panel name, and pipeline completion flags
4. WHEN no viewport context is provided (e.g., API-only clients), THE Orchestrator SHALL proceed without context enrichment rather than failing
5. THE Context_Block SHALL be bounded to a maximum token budget to avoid consuming excessive LLM context

### Requirement 3: Agent Viewport Interaction

**User Story:** As a scientist, I want the agent to be able to highlight residues, move the camera, and change color modes on my dashboard, so that its explanations are spatially grounded in the evidence I can see.

#### Acceptance Criteria

1. WHEN the agent decides to highlight specific residues, THE Orchestrator SHALL emit a Viewport_Directive with action `highlight` containing the residue IDs, color, and style
2. WHEN the agent decides to focus on a specific residue, THE Orchestrator SHALL emit a Viewport_Directive with action `focus` containing the residue ID and camera center flag
3. WHEN the agent decides to change the Poincaré color mode, THE Orchestrator SHALL emit a Viewport_Directive with action `set_metric` containing the target metric name
4. WHEN the agent emits a directive, THE Orchestrator SHALL announce the viewport change in the chat response before executing it (inspect-preview-commit pattern)
5. WHEN a Viewport_Directive is emitted, THE Frontend SHALL apply the directive to all subscribed viewports simultaneously
6. THE Agent SHALL be able to issue a `clear` directive to reset all highlights and camera focus to neutral state

### Requirement 4: Tool Gating Enforcement

**User Story:** As a scientist, I want the system to prevent premature tool usage based on workflow maturity, so that the agent cannot skip ahead to docking or screening before the evidence justifies it.

#### Acceptance Criteria

1. WHEN the agent attempts to invoke a tool, THE Orchestrator SHALL check the tool name against the current Planner_Policy's allowed and blocked tool lists
2. IF a tool is blocked by the current discovery phase, THEN THE Orchestrator SHALL reject the invocation and return an explanation of why the tool is unavailable at this stage
3. WHEN a tool invocation is rejected, THE Orchestrator SHALL suggest which phase advancement or evidence is needed to unlock the tool
4. WHEN the user explicitly requests a blocked tool, THE Orchestrator SHALL either advance the phase with user-override provenance or explain the prerequisite
5. THE Orchestrator SHALL derive the tool policy from the current discovery phase using the same `toolPolicyForPhase` logic already implemented in the frontend

### Requirement 5: Cross-Viewport Residue Selection Sync

**User Story:** As a scientist, I want to click a residue on the Poincaré disc and have it simultaneously highlight on the 3D structure viewer with the camera centering on it, so that I can immediately understand the spatial meaning of abstract topological signals.

#### Acceptance Criteria

1. WHEN a residue is selected in any viewport (Poincaré disc, 3D viewer, residue table, or agent directive), THE Sync_Bus SHALL propagate the selection to all other subscribed viewports
2. WHEN a selection is propagated to the 3D viewer, THE Molecular_Viewer SHALL highlight the residue and animate the camera to center on its 3D coordinates
3. WHEN a selection is propagated to the Poincaré disc, THE Poincare_Viewer SHALL highlight the corresponding point and display its metrics in the detail panel
4. THE Sync_Bus SHALL use a canonical Residue_Selection identity (structure_id, chain_id, residue_number) to prevent mismatches between viewers that use different indexing schemes
5. WHEN a residue is selected that does not exist in the current structure context, THE Sync_Bus SHALL reject the selection and surface an error state rather than silently failing
6. WHEN a new selection is made, THE previous selection SHALL be cleared from all viewports before the new one is applied

### Requirement 6: State-Driven Agent Language

**User Story:** As a scientist, I want the agent to speak with appropriate confidence based on the current hypothesis status, so that I can trust its language reflects the actual evidence strength.

#### Acceptance Criteria

1. WHEN the hypothesis lifecycle is `emergent`, THE Orchestrator SHALL instruct the LLM to use exploratory language (may, could, possible signal, worth testing)
2. WHEN the hypothesis lifecycle is `supported`, THE Orchestrator SHALL instruct the LLM to use bounded confident language (consistent with, supported by current evidence)
3. WHEN the hypothesis lifecycle is `contradicted`, THE Orchestrator SHALL instruct the LLM to explicitly name the contradiction and recommend regression
4. THE Orchestrator SHALL pass the current `allowedSpeechActs` and `blockedSpeechActs` to the LLM system prompt
5. THE Orchestrator SHALL pass the current `reasoningMode` to the LLM system prompt so it shapes response structure appropriately
