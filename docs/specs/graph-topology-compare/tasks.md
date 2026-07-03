# Implementation Plan: Graph Topology Compare

## Overview

Implements graph topology persistence and agent tools in order: migration → payload models → normalizer path → graph builder extension → agent tools → tool registration. Python throughout, using Hypothesis for property-based tests.

## Tasks

- [x] 1. Create database migration for graph topology tables
  - Create `data/aurora/migrations/031_graph_topology.sql`
  - Define `fact_graph_edge` and `fact_graph_node_metrics` tables
  - Add unique indexes on natural keys and foreign key references
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 2. Add payload models for graph topology
  - [x] 2.1 Add `GraphEdge` and `GraphTopologyPayload` Pydantic models to `science/dtie/common/normalizer_payloads.py`
    - Include edge_type validation against allowlist
    - Include residue_id format validation
    - _Requirements: 1.3, 9.1_

- [x] 3. Implement normalizer path
  - [x] 3.1 Add `normalize_graph_topology` method to `data/normalizer/core.py`
    - Follow existing pattern: provenance → validate → transaction → upsert edges → compute metrics → upsert metrics → register assets → audit
    - Compute metrics via `asyncio.to_thread` using networkx
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 9.1, 9.2, 9.3, 9.4_

  - [x] 3.2 Write property tests for normalizer graph topology path
    - **Property 1: Edge persistence round trip**
    - **Property 2: Idempotent graph topology writes**
    - **Property 3: Invalid residue_id rejection**
    - **Property 4: Node metrics computation correctness**
    - **Property 11: Governance invariants**
    - **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 9.2, 9.3, 9.4**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Extend GraphBuilder to emit edges for persistence
  - [x] 5.1 Add `extract_edges_for_persistence` method to `science/dtie/common/graph_builder.py`
    - Convert ProteinGraph edge_index + edge_attr into list of `GraphEdge` objects
    - Classify edges by type (contact by default, h_bond by distance/angle heuristic)
    - _Requirements: 1.1_

- [x] 6. Implement agent tools
  - [x] 6.1 Create `agent/tools/graph_tools.py` with `get_graph_metrics`
    - Query `fact_graph_node_metrics` with optional residue_ids and metric_type filters
    - Return `ToolResult` with viewport directive highlighting high-betweenness residues
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 6.2 Implement `compare_graphs` tool
    - Join edges from two structures, compute gained/lost/changed edges
    - Compute per-residue metric deltas with canonical residue_id alignment
    - Highlight gained/lost H-bonds in viewer
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 6.3 Implement `get_hbond_network` tool
    - Query `fact_graph_edge` filtered to edge_type='h_bond'
    - Support optional residue_ids filter
    - _Requirements: 5.1, 5.2_

  - [x] 6.4 Implement `find_graph_bridges` tool
    - Query `fact_graph_node_metrics` WHERE is_bridge = TRUE
    - Include betweenness and degree in response
    - _Requirements: 6.1, 6.2_

  - [x] 6.5 Implement `get_shortest_paths` tool
    - Load edges into networkx graph in `asyncio.to_thread`
    - Compute shortest path and total distance
    - Handle disconnected components gracefully
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 6.6 Write property tests for graph tools
    - **Property 5: Graph metrics query filtering**
    - **Property 6: Edge diff correctness**
    - **Property 7: Metric diff with canonical alignment**
    - **Property 8: H-bond network filtering**
    - **Property 9: Bridge detection correctness**
    - **Property 10: Shortest path correctness**
    - **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 6.1, 7.1, 7.2**

- [x] 7. Register tools with the agent
  - [x] 7.1 Add `GRAPH_TOOLS` list to `agent/llm/agents.py` with tool definitions
    - Wire handlers in `create_coordinator`
    - _Requirements: 10.1_

  - [x] 7.2 Add graph tools to `agent/tools/diagnostics.py` startup check
    - _Requirements: 10.2_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Each task references specific requirements for traceability
- Property tests use Hypothesis with minimum 100 iterations
- All writes go through the Normalizer — no direct INSERTs
