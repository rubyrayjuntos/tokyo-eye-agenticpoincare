# Requirements Document

## Introduction

The dashboard chat agent currently receives minimal context about the user's view state when answering questions. This feature enriches the agent's context payload with the full dashboard viewport state — including Poincaré disc settings, 3D molecular viewer state, hydration data summaries, pipeline run status, hypothesis state, and active tool panel information. This allows the agent to provide contextually relevant answers based on what the user is actually looking at.

## Glossary

- **Viewport_State**: The complete set of user-facing view settings across all dashboard panels (color modes, selections, toggles, active panels)
- **Context_Payload**: The JSON object sent from the frontend to the `/api/agent/chat` endpoint alongside each user message
- **Context_Block**: The formatted string representation of the context payload that is prepended to the user message before sending to the LLM
- **Hydration_Summary**: A condensed representation of the hydrated data (counts, availability flags, key metrics) rather than raw data
- **Poincare_State**: The current configuration of the Poincaré disc viewer (color mode, Möbius focus, selected residue, brush selection)
- **Molecular_Viewer_State**: The current configuration of the 3D structure viewer (color mode, threshold, highlighted residues)

## Requirements

### Requirement 1: Frontend Context Gathering

**User Story:** As a scientist, I want the chat agent to be aware of what I'm currently viewing in the dashboard, so that its answers are relevant to my current analysis focus.

#### Acceptance Criteria

1. WHEN a user sends a chat message, THE AgentChat component SHALL include the current Poincaré disc state in the context payload (color mode, Möbius focus enabled, selected residue ID, brush-selected residue IDs)
2. WHEN a user sends a chat message, THE AgentChat component SHALL include the current Molecular Viewer state in the context payload (color mode, risk threshold value, highlighted residue IDs)
3. WHEN a user sends a chat message, THE AgentChat component SHALL include the hydration data summary in the context payload (residue count, source leak count, hypothesis count, top uncertainty residues, resistance data availability, pipeline phase completion flags)
4. WHEN a user sends a chat message, THE AgentChat component SHALL include the active tool panel name in the context payload
5. WHEN hydration data is not yet loaded, THE AgentChat component SHALL send null for data-dependent context fields without error

### Requirement 2: Poincaré Disc Context

**User Story:** As a scientist, I want the agent to know which residue I have selected in the Poincaré disc and what color mode I'm viewing, so that it can reference that residue in its answers.

#### Acceptance Criteria

1. WHEN the Poincaré disc has a selected residue, THE Context_Payload SHALL include that residue's ID, name, chain, epistemic uncertainty, and cone depth
2. WHEN Möbius focus is enabled, THE Context_Payload SHALL indicate the focus residue and that the view is transformed
3. WHEN a brush selection is active, THE Context_Payload SHALL include the list of residue IDs in the selection region
4. THE Context_Payload SHALL include the current Poincaré color mode (cone_depth, uncertainty, or other active mode)

### Requirement 3: Molecular Viewer Context

**User Story:** As a scientist, I want the agent to know which color mode I'm using in the 3D viewer and what threshold I've set, so that it can discuss the visible residue classifications.

#### Acceptance Criteria

1. THE Context_Payload SHALL include the current 3D viewer color mode (spectrum, cone_depth, epistemic, aleatoric, plasticity, allosteric, resistance)
2. WHEN a risk threshold is set, THE Context_Payload SHALL include the threshold value
3. WHEN residues are highlighted in the viewer, THE Context_Payload SHALL include the highlighted residue IDs
4. WHEN resistance color mode is active, THE Context_Payload SHALL include the count of high_sensitivity, moderate, and stable residues visible above threshold

### Requirement 4: Hydration Data Summary Context

**User Story:** As a scientist, I want the agent to know what data is available for the current structure without sending raw data, so that it can suggest relevant analyses.

#### Acceptance Criteria

1. THE Context_Payload SHALL include a persistence status summary indicating which data types are available (embeddings, graph, sites, phase2, phase4, phase5, phase6, resistance)
2. THE Context_Payload SHALL include counts for key data types (residues, source leaks, hypotheses, provenance runs, annotations)
3. WHEN resistance data is available, THE Context_Payload SHALL include the spectral summary (lambda_2, hinge count) and classification distribution
4. WHEN hypotheses exist, THE Context_Payload SHALL include the count and status distribution (proposed, supported, contradicted)

### Requirement 5: Pipeline and Tool Panel Context

**User Story:** As a scientist, I want the agent to know which tool panel I have open and whether the pipeline has been run, so that it can provide targeted guidance.

#### Acceptance Criteria

1. THE Context_Payload SHALL include the name of the currently active tool panel (hypothesis, graph_topology, data_tools, plot_generator, provenance, rcsb_search, or none)
2. THE Context_Payload SHALL include the most recent pipeline run status for the active structure (queued, running, complete, failed, or never_run)
3. WHEN the pipeline is running, THE Context_Payload SHALL include the current step name and progress percentage

### Requirement 6: Backend Context Formatting

**User Story:** As a system developer, I want the backend to format the enriched context payload into a structured prompt section, so that the LLM can parse and use the information effectively.

#### Acceptance Criteria

1. WHEN the backend receives an enriched context payload, THE Context_Block builder SHALL format all provided fields into a structured multi-section text block
2. THE Context_Block SHALL organize information by section (Structure, Poincaré View, 3D Viewer, Data Availability, Active Analysis)
3. WHEN context fields are null or absent, THE Context_Block builder SHALL omit those sections without error
4. THE Context_Block SHALL remain under 2000 characters to avoid excessive token usage
