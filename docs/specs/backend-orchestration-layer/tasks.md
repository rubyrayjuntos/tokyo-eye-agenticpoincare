# Implementation Plan: Backend Orchestration Layer

## Overview

Build the server-side orchestration layer that maintains canonical state for discovery phase and hypothesis lifecycle, enforces tool gating, enriches LLM context with dashboard state, enables agent viewport interaction, and synchronizes residue selection across all viewers. The backend is the single source of truth; the frontend derives its state from WebSocket-pushed snapshots.

## Tasks

- [x] 1. Create orchestration module with core data models
  - Create `agent/orchestration/__init__.py`
  - Create `agent/orchestration/models.py` with DiscoveryPhase enum, HypothesisLifecycleState enum, DiscoveryPhaseState, HypothesisState, PlannerPolicy, ResidueSelection, SelectionResult Pydantic models
  - Create `agent/orchestration/tool_policy.py` with TOOL_POLICY dict and `tool_policy_for_phase()` function
  - Create `agent/orchestration/reasoning_policy.py` with `reasoning_policy_for_lifecycle_state()` function
  - _Requirements: 1.1, 4.5, 6.1, 6.2, 6.3_

- [x] 2. Implement SessionOrchestrator
  - [x] 2.1 Create `agent/orchestration/orchestrator.py` with SessionOrchestrator class
    - Constructor initializes discovery_phase=residue, hypothesis_lifecycle=emergent
    - `derive_policy()` computes PlannerPolicy from current state
    - `can_invoke_tool(tool_name)` checks against derived policy
    - `transition_discovery(event)` applies phase transitions with guards
    - `transition_hypothesis(event)` applies lifecycle transitions with guards
    - `update_viewport(state)` caches latest viewport state
    - `select_residue(selection)` validates and returns SelectionResult
    - `get_state_snapshot()` returns full serializable state
    - _Requirements: 1.1, 1.2, 4.1, 4.2, 5.4, 5.5_

  - [x] 2.2 Write property tests for SessionOrchestrator
    - **Property 1: Session initialization invariant**
    - **Property 7: Tool gating correctness**
    - **Property 8: Tool policy equivalence**
    - **Validates: Requirements 1.1, 4.1, 4.2, 4.5**

- [x] 3. Implement context block builder
  - [x] 3.1 Create `agent/orchestration/context_builder.py`
    - `build_context_block(orchestrator, viewport_state)` formats state into compact text
    - Include structure identity, selected residue, color modes, pipeline flags
    - Include reasoning mode, allowed/blocked speech acts from hypothesis lifecycle
    - Implement token budget enforcement (truncation strategy for large highlight lists)
    - Handle None viewport state gracefully (return empty context rather than error)
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 6.4, 6.5_

  - [x] 3.2 Write property tests for context block builder
    - **Property 3: Context block completeness**
    - **Property 4: Context block size bound**
    - **Property 12: Lifecycle-to-language-policy mapping**
    - **Validates: Requirements 2.2, 2.3, 2.5, 6.1-6.5**

- [x] 4. Implement tool gating integration
  - [x] 4.1 Create `agent/orchestration/gates.py` with `gated_tool_invoke()` wrapper
    - Check `orchestrator.can_invoke_tool()` before executing handler
    - Return structured error ToolResult if blocked (includes phase name and unlock hint)
    - On successful completion, call `orchestrator.handle_tool_completion()` to fire state transitions
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 4.2 Integrate tool gate into Agent loop (`agent/llm/base.py`)
    - Modify `_execute_tool()` to accept an optional orchestrator parameter
    - If orchestrator present, route through `gated_tool_invoke()` before handler execution
    - _Requirements: 4.1_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement residue selection sync
  - [x] 6.1 Add selection validation to SessionOrchestrator
    - `select_residue(selection)` validates structure_id + chain_id + residue_number against loaded structure
    - Returns SelectionResult with valid=True and the canonical selection, or valid=False with error
    - Clears previous selection on new valid selection
    - _Requirements: 5.4, 5.5, 5.6_

  - [x] 6.2 Extend WebSocket protocol for selection sync
    - Add `selection_sync` message type to `ViewportConnectionManager.send_to_session()`
    - Add `selection_error` message type for invalid selections
    - Handle incoming `viewport_event` with `event_type=selection` from frontend
    - Route selection through orchestrator validation before broadcasting
    - _Requirements: 5.1, 5.4, 5.5_

  - [x] 6.3 Write property tests for selection sync
    - **Property 9: Selection propagation with canonical identity**
    - **Property 10: Invalid selection rejection**
    - **Property 11: Selection replacement**
    - **Validates: Requirements 5.1, 5.4, 5.5, 5.6**

- [x] 7. Implement agent viewport interaction
  - [x] 7.1 Create viewport directive emission in orchestrator
    - Add `emit_directive(directive)` method to SessionOrchestrator
    - Validates directive against current structure context
    - Pushes directive to all connected clients via ViewportConnectionManager
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 7.2 Implement inspect-preview-commit pattern in chat router
    - Modify chat endpoint to prepend directive announcement to response text
    - Ensure message text is returned to client BEFORE directive push fires
    - Agent tools that produce directives include a `message` field explaining the action
    - _Requirements: 3.4, 3.6_

  - [x] 7.3 Write property tests for directive emission
    - **Property 5: Agent directive action correctness**
    - **Property 6: Directive announcement precedes execution**
    - **Validates: Requirements 3.1-3.6**

- [x] 8. Integrate context enrichment into chat endpoint
  - [x] 8.1 Modify `coordinator/routers/chat.py` to use context builder
    - Accept `viewport_state` from ChatRequest (already defined in request model)
    - Pass viewport_state to `build_context_block(orchestrator, viewport_state)`
    - Inject context block into Agent.run() context parameter
    - _Requirements: 2.1, 2.2_

  - [x] 8.2 Extend frontend chat client to send viewport state
    - Modify AgentChat component to gather current viewport state before sending message
    - Include Poincaré state (color_mode, selected_residue, brush_selection)
    - Include 3D viewer state (color_mode, risk_threshold, highlighted_residues)
    - Include active panel and pipeline status
    - _Requirements: 2.1_

- [x] 9. Wire state push over WebSocket
  - [x] 9.1 Add state snapshot push to ViewportConnectionManager
    - After any orchestrator state transition, push `state_snapshot` message to all session clients
    - Include discovery_phase, hypothesis_lifecycle, structure_scope, and derived policy
    - Add `phase_transition` and `hypothesis_transition` messages for targeted UI updates
    - _Requirements: 1.2, 1.4_

  - [x] 9.2 Handle reconnection with full state sync
    - On `viewport_register` message, send current full state snapshot to the registering client
    - Ensure no state is lost during disconnection period
    - _Requirements: 1.5_

  - [x] 9.3 Write stateful property test for state consistency
    - **Property 2: State consistency across connected clients**
    - Use Hypothesis RuleBasedStateMachine to generate event sequences
    - Assert all simulated clients receive identical state after each transition
    - **Validates: Requirements 1.2, 1.4**

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required — property tests are written alongside each component
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- The orchestrator is session-scoped (one per active session, stored in session store)
- Frontend state machines become derived views — they read from backend-pushed state rather than maintaining independent transitions
