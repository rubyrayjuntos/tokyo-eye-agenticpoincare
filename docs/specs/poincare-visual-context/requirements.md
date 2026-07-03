# Requirements Document

## Introduction

This feature gives the agent perceptual awareness of the Poincaré disc — both as a visual image (what the user literally sees) and as structured topological data (clusters, neighborhoods, density patterns, angular organization). The goal is to close the perception gap: when a user looks at the disc and says "what's that cluster?", the agent can reason about the same spatial reality. The structured interpretation is always available in context; the visual snapshot is captured on demand.

## Glossary

- **Disc_Topology_Tool**: A backend tool that computes spatial statistics from stored `hyp_projection_2d` VECTOR(2) coordinates — k-NN neighborhoods, clusters, radial density, angular sectors, hub residues
- **Disc_Neighborhood_Tool**: A backend tool that returns the k-nearest residues on the disc to a given residue, with hyperbolic distances and cluster membership
- **Visual_Snapshot**: A PNG image of the Poincaré disc canvas captured by the frontend and sent to the backend for inclusion in LLM multimodal context
- **Disc_Summary**: A compact text block describing the disc's spatial organization (number of clusters, their angular positions, key hub residues, density distribution) included in the LLM context block
- **Angular_Sector**: A named region of the disc described by its angular position (0°-360°) and radial depth (center vs. periphery)
- **Disc_Cluster**: A group of residues that are spatially proximate on the Poincaré disc (identified by HDBSCAN on hyperbolic coordinates)

## Requirements

### Requirement 1: Structured Disc Topology Awareness

**User Story:** As a scientist, I want the agent to always understand the spatial organization of the Poincaré disc, so that it can reason about clustering patterns, neighborhoods, and topological structure without needing a visual image.

#### Acceptance Criteria

1. WHEN a structure is loaded and embeddings are available, THE Disc_Topology_Tool SHALL compute cluster assignments for all residues using HDBSCAN on the stored 2D Poincaré coordinates
2. WHEN the Disc_Topology_Tool is invoked, THE system SHALL return: cluster count, cluster membership for each residue, cluster centroids (as angular sector descriptions), inter-cluster distances, and hub residues (highest disc-degree within each cluster)
3. WHEN a chat message is sent with a loaded structure, THE Orchestrator SHALL include a compact Disc_Summary in the context block describing the current disc organization
4. THE Disc_Summary SHALL include: number of clusters, their approximate angular positions on the disc, the count of residues in each cluster, bridge residues that connect clusters, and any isolated peripheral residues
5. THE Disc_Summary SHALL be bounded to 150 tokens maximum to fit within the existing context block budget

### Requirement 2: Disc Neighborhood Query

**User Story:** As a scientist, I want the agent to know what's near a selected residue on the disc, so that when I select or ask about a residue, the agent can describe its local topological context.

#### Acceptance Criteria

1. WHEN a residue is selected or referenced, THE Disc_Neighborhood_Tool SHALL return the k-nearest residues on the Poincaré disc (default k=8) with their hyperbolic distances
2. WHEN the Disc_Neighborhood_Tool returns results, THE system SHALL include each neighbor's cluster membership, cone_depth, and uncertainty values
3. WHEN the selected residue is a hub (high disc-degree), THE Disc_Neighborhood_Tool SHALL annotate it as such and include the cluster it anchors
4. WHEN the selected residue is peripheral (high radial distance from center), THE Disc_Neighborhood_Tool SHALL note its isolation and nearest cluster

### Requirement 3: On-Demand Visual Snapshot

**User Story:** As a scientist, I want the agent to be able to see exactly what I see on the Poincaré disc when I ask it to look, so that it can interpret visual patterns, colors, and spatial relationships the way I do.

#### Acceptance Criteria

1. WHEN the user or agent requests a visual snapshot, THE Frontend SHALL capture the current Poincaré disc canvas as a PNG image and transmit it to the backend
2. WHEN the backend receives a visual snapshot with a chat message, THE Orchestrator SHALL include the image as a multimodal content block in the LLM request
3. WHEN the visual snapshot is captured, THE image SHALL reflect the current state of the disc including: color mode, highlighted residues, selected residue, edge visibility, and any active filters
4. THE Visual_Snapshot SHALL be captured at a maximum resolution of 512x512 pixels to bound token cost
5. IF the LLM provider does not support vision, THEN THE Orchestrator SHALL skip the image content block and rely on the structured Disc_Summary only

### Requirement 4: Disc Topology Computation

**User Story:** As a system architect, I want the disc topology to be computed from the persisted VECTOR(2) coordinates server-side, so that it is consistent regardless of frontend rendering state.

#### Acceptance Criteria

1. THE Disc_Topology_Tool SHALL compute k-NN edges using true Poincaré distance (not Euclidean L2 on the disc coordinates)
2. THE Disc_Topology_Tool SHALL use HDBSCAN clustering on the hyperbolic distance matrix to identify natural clusters
3. WHEN computing disc topology, THE system SHALL use the same curvature parameter stored with the structure's embeddings
4. THE Disc_Topology_Tool SHALL cache its results per (structure_id, run_id) pair to avoid recomputation on every message
5. WHEN a new GNN run completes for a structure, THE cached disc topology SHALL be invalidated

### Requirement 5: Context Integration

**User Story:** As a scientist, I want the agent's disc awareness to be seamlessly integrated into its responses, so that it can reference spatial patterns naturally without me having to ask it to look.

#### Acceptance Criteria

1. WHEN the structured Disc_Summary is present in context, THE agent SHALL be able to reference cluster positions, neighborhoods, and hub residues in its responses without explicit tool calls
2. WHEN the user asks about a visual pattern (e.g., "what's that cluster?"), THE agent SHALL use both the Disc_Summary and, if available, the visual snapshot to ground its answer
3. WHEN the agent highlights residues via a viewport directive, THE Disc_Summary SHALL inform the agent which cluster or angular sector those residues occupy
4. THE Disc_Summary SHALL be updated whenever the active structure or run changes, ensuring the agent's spatial model stays current
