# Requirements Document

## Introduction

The Graph Topology Compare feature extends Tokyo Eye's analytical capabilities by persisting the molecular contact graph (edges and per-node metrics) and providing agent tools to query, compare, and analyze graph topology across structures. This enables WT/mutant comparison at the structural level — identifying lost H-bonds, gained contacts, altered centrality, and communication bottlenecks.

## Glossary

- **Graph_Builder**: The existing component (`science/dtie/common/graph_builder.py`) that constructs PyG Data objects from residue coordinates.
- **Normalizer**: The governed write path (`data/normalizer/core.py`) through which all data writes must pass.
- **Contact_Graph**: A graph where nodes are residues and edges represent physical proximity (Cα distance < cutoff) or specific interactions (H-bonds, salt bridges, disulfides).
- **Edge_Type**: Classification of a graph edge: `h_bond`, `contact`, `covalent`, `disulfide`, `salt_bridge`.
- **Node_Metrics**: Per-residue graph-theoretic measures: degree, betweenness centrality, clustering coefficient, closeness centrality, eigenvector centrality, bridge status, conductance.
- **Bridge_Residue**: A residue whose removal disconnects the contact graph into two or more components.
- **Edge_Diff**: The set of edges gained, lost, or changed in weight between two structures.
- **Metric_Diff**: The per-residue difference in node metrics between two structures.
- **Graph_Tools**: The set of agent tools exposing graph topology queries to the LLM coordinator.

## Requirements

### Requirement 1: Graph Edge Persistence

**User Story:** As a structural biologist, I want the molecular contact graph edges to be persisted after structure ingestion, so that I can query and compare graph topology later.

#### Acceptance Criteria

1. WHEN a structure is ingested and the contact graph is built, THE Normalizer SHALL persist all edges to `fact_graph_edge` with source_residue_id, target_residue_id, edge_type, distance, and weight.
2. THE Normalizer SHALL enforce idempotent upsert semantics on the natural key (run_id, source_residue_id, target_residue_id, edge_type).
3. WHEN an edge is persisted, THE Normalizer SHALL validate that both source_residue_id and target_residue_id conform to the canonical residue_id format.
4. IF a residue_id fails validation, THEN THE Normalizer SHALL reject the entire batch and log an audit error.

### Requirement 2: Graph Node Metrics Persistence

**User Story:** As a structural biologist, I want per-residue graph metrics computed and stored, so that I can identify structurally important residues.

#### Acceptance Criteria

1. WHEN graph edges are persisted for a structure, THE Normalizer SHALL compute node metrics (degree, betweenness, clustering coefficient, closeness, eigenvector centrality, bridge status, conductance) and persist them to `fact_graph_node_metrics`.
2. THE Normalizer SHALL enforce idempotent upsert semantics on the natural key (run_id, residue_id).
3. THE Normalizer SHALL compute graph metrics using networkx in a background thread via `asyncio.to_thread` to avoid blocking the event loop.

### Requirement 3: Query Graph Metrics

**User Story:** As a researcher using the agent, I want to retrieve per-residue graph metrics for a structure, so that I can identify high-centrality or bridge residues.

#### Acceptance Criteria

1. WHEN a user requests graph metrics for a structure, THE Graph_Tools SHALL return per-residue degree, betweenness, clustering coefficient, closeness, eigenvector centrality, bridge status, and conductance.
2. WHERE a residue_ids filter is provided, THE Graph_Tools SHALL return metrics only for the specified residues.
3. WHERE a metric_type filter is provided, THE Graph_Tools SHALL return only the requested metric columns.

### Requirement 4: Compare Graph Topology

**User Story:** As a researcher, I want to compare the graph topology of two structures (e.g., WT vs mutant), so that I can identify structural changes at the edge and node level.

#### Acceptance Criteria

1. WHEN a user requests a graph comparison between two structures, THE Graph_Tools SHALL return the edge diff: edges gained in structure B, edges lost from structure A, and edges with changed weights.
2. WHEN a user requests a graph comparison, THE Graph_Tools SHALL return the metric diff: per-residue delta for betweenness, degree, clustering coefficient, and other metrics.
3. WHEN a comparison is performed, THE Graph_Tools SHALL match residues across structures using canonical residue_id alignment (same chain + residue_index).

### Requirement 5: H-Bond Network Query

**User Story:** As a researcher, I want to extract the H-bond subgraph for a structure or region, so that I can analyze hydrogen bonding patterns.

#### Acceptance Criteria

1. WHEN a user requests the H-bond network for a structure, THE Graph_Tools SHALL return all edges of type `h_bond`.
2. WHERE a residue_ids filter is provided, THE Graph_Tools SHALL return only H-bond edges involving at least one of the specified residues.

### Requirement 6: Bridge Residue Detection

**User Story:** As a researcher, I want to identify bridge residues in the contact graph, so that I can find allosteric communication bottlenecks.

#### Acceptance Criteria

1. WHEN a user requests bridge residues for a structure, THE Graph_Tools SHALL return all residues marked as bridges (is_bridge = TRUE) from the persisted node metrics.
2. THE Graph_Tools SHALL include the residue's betweenness centrality and degree alongside the bridge status.

### Requirement 7: Shortest Path Query

**User Story:** As a researcher, I want to find the shortest path between two residues in the contact graph, so that I can understand signal propagation routes.

#### Acceptance Criteria

1. WHEN a user requests the shortest path between two residues, THE Graph_Tools SHALL return the ordered list of residue_ids along the shortest path in the contact graph.
2. THE Graph_Tools SHALL return the total path length (sum of edge distances).
3. IF no path exists between the two residues, THEN THE Graph_Tools SHALL return an empty path and indicate the residues are in disconnected components.

### Requirement 8: Database Schema

**User Story:** As a developer, I want a migration that creates the graph topology tables, so that edges and metrics can be stored.

#### Acceptance Criteria

1. THE Migration SHALL create `fact_graph_edge` with columns: edge_id, run_id, structure_id, source_residue_id, target_residue_id, edge_type, distance_angstrom, hyperbolic_distance, weight, metadata, computed_at.
2. THE Migration SHALL create `fact_graph_node_metrics` with columns: metric_id, run_id, structure_id, residue_id, degree, betweenness, clustering_coefficient, closeness, eigenvector_centrality, is_bridge, conductance, computed_at.
3. THE Migration SHALL create a unique index on (run_id, source_residue_id, target_residue_id, edge_type) for `fact_graph_edge`.
4. THE Migration SHALL create a unique index on (run_id, residue_id) for `fact_graph_node_metrics`.
5. THE Migration SHALL add foreign key references to provenance_run, dim_structure, and dim_residue.

### Requirement 9: Normalizer Integration

**User Story:** As a developer, I want a dedicated normalizer path for graph topology data, so that writes follow the governed single-write-path pattern.

#### Acceptance Criteria

1. THE Normalizer SHALL expose a `normalize_graph_topology` method that accepts a validated payload of edges and triggers metric computation.
2. THE Normalizer SHALL create a provenance_run record before writing graph data.
3. THE Normalizer SHALL register all created assets in the governed_asset catalog.
4. THE Normalizer SHALL log audit entries for both success and failure of graph topology writes.

### Requirement 10: Tool Registration

**User Story:** As a developer, I want graph tools registered with the agent, so that the LLM can invoke them during conversations.

#### Acceptance Criteria

1. THE Graph_Tools SHALL be registered in `agent/llm/agents.py` under a `GRAPH_TOOLS` list.
2. WHEN the agent starts, THE Diagnostics module SHALL verify graph tool availability.
