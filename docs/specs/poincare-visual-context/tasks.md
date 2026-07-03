# Implementation Plan: Poincaré Visual Context

## Overview

Build the agent's dual-mode perceptual awareness of the Poincaré disc: a structured topology computation (always in context) and an on-demand visual snapshot (multimodal LLM input). The structured path uses stored VECTOR(2) coordinates with true Poincaré distance and HDBSCAN clustering. The visual path captures the frontend canvas and passes it as an image content block.

## Tasks

- [x] 1. Implement Poincaré distance and disc topology core
  - [x] 1.1 Create `agent/tools/disc_topology.py` with core computation
    - Implement `poincare_distance(p, q, c)` using the arcosh formula
    - Implement `compute_pairwise_distances(coordinates, c)` returning a distance matrix
    - Implement `compute_disc_topology(coordinates, curvature_c, min_cluster_size)` using HDBSCAN on the hyperbolic distance matrix
    - Define dataclasses: `DiscCluster`, `DiscTopologyResult`, `DiscNeighborhood`, `NeighborInfo`
    - Compute angular sector (0-360° mapped to "N","NE","E","SE","S","SW","W","NW") for each cluster centroid
    - Identify hub residues (highest k-NN degree within each cluster)
    - Identify bridge residues (connected to residues in 2+ different clusters)
    - Identify peripheral residues (radial distance > 0.85 of disc boundary)
    - Compute radial density profile: core (r<0.3), mid (0.3≤r<0.7), periphery (r≥0.7)
    - _Requirements: 1.1, 1.2, 4.1, 4.2, 4.3_

  - [x] 1.2 Write property tests for Poincaré distance
    - **Property 7: Poincaré distance correctness**
    - **Validates: Requirements 4.1**

  - [x] 1.3 Write property tests for disc topology computation
    - **Property 1: Topology completeness**
    - **Validates: Requirements 1.1, 1.2**

- [x] 2. Implement disc neighborhood query
  - [x] 2.1 Add `compute_disc_neighborhood(target_residue_id, coordinates, topology, curvature_c, k)` to `disc_topology.py`
    - Find k-nearest residues by Poincaré distance
    - Annotate each neighbor with cluster_id, cone_depth, epistemic_uncertainty
    - Set is_hub=True if target has highest disc-degree in its cluster
    - Set is_peripheral=True if target radius > 0.85 of boundary
    - Include nearest cluster reference for peripheral residues
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 2.2 Write property tests for neighborhood query
    - **Property 4: Neighborhood query completeness**
    - **Property 5: Topology annotation correctness**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 3. Implement topology cache and agent tools
  - [x] 3.1 Create `agent/tools/disc_topology_cache.py` with `TopologyCache`
    - LRU cache keyed by (structure_id, run_id) with configurable max_size (default 50)
    - `get(structure_id, run_id)` → `DiscTopologyResult | None`
    - `put(result)` → stores in cache
    - `invalidate(structure_id)` → removes all entries for a structure
    - _Requirements: 4.4, 4.5_

  - [x] 3.2 Create agent tool wrappers in `agent/tools/disc_tools.py`
    - `get_disc_topology(structure_id, run_id)` → fetches coordinates from DB, computes or retrieves cached topology
    - `get_disc_neighborhood(structure_id, residue_id, k)` → returns neighborhood using cached topology
    - Register tools in the agent tool registry
    - _Requirements: 1.1, 2.1_

  - [x] 3.3 Write property test for cache idempotence
    - **Property 9: Topology cache idempotence**
    - **Validates: Requirements 4.4**

- [x] 4. Checkpoint - Ensure topology tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Integrate disc summary into context block
  - [x] 5.1 Create `agent/orchestration/disc_summary.py` with `format_disc_summary(topology)`
    - Format clusters with angular sector, size, hub residue
    - Include bridge and peripheral residues
    - Enforce 150-token (≈600 char) budget with truncation
    - _Requirements: 1.3, 1.4, 1.5_

  - [x] 5.2 Extend context builder to include disc summary
    - Modify `build_context_block()` to accept optional `DiscTopologyResult`
    - Append disc summary section after viewport configuration section
    - Ensure total context block still respects the 500-token budget (disc summary uses its own 150-token sub-budget)
    - _Requirements: 1.3, 5.4_

  - [x] 5.3 Write property tests for disc summary
    - **Property 2: Disc summary content completeness**
    - **Property 3: Disc summary size bound**
    - **Property 10: Summary freshness on structure change**
    - **Validates: Requirements 1.3, 1.4, 1.5, 5.4**

- [x] 6. Implement on-demand visual snapshot
  - [x] 6.1 Add frontend snapshot capture utility
    - Create `visualizer/frontend/src/lib/snapshotCapture.ts`
    - `capturePoincareSnapshot(canvas, maxSize=512)` → base64 PNG string
    - Downscale canvas if larger than maxSize×maxSize
    - _Requirements: 3.1, 3.4_

  - [x] 6.2 Extend chat request model and backend handling
    - Add `poincare_snapshot: str | None` field to `ChatRequest` in `coordinator/routers/chat.py`
    - Validate base64 format and size (reject if > 1MB)
    - Pass snapshot to LLM message builder
    - _Requirements: 3.2_

  - [x] 6.3 Modify LLM message construction for multimodal content
    - Extend `agent/llm/base.py` message builder to accept optional image
    - If image present and provider supports vision: include as image_url content block
    - If provider does not support vision: skip image, no error
    - Add `supports_vision` property to LLM provider interface
    - _Requirements: 3.2, 3.5_

  - [x] 6.4 Write property test for conditional image inclusion
    - **Property 6: Visual snapshot conditional inclusion**
    - **Validates: Requirements 3.2, 3.5**

- [x] 7. Wire snapshot trigger in frontend chat component
  - [x] 7.1 Add snapshot trigger to chat send flow
    - Before sending chat message, check if user message references visual patterns (heuristic: mentions "see", "look", "cluster", "pattern", "show")
    - Or: add a "📷 Show disc" button next to chat input that attaches snapshot
    - Or: agent can request snapshot via WebSocket `request_snapshot` message
    - Include captured base64 in chat request payload
    - _Requirements: 3.1, 3.3_

  - [x] 7.2 Write property test for curvature parameterization
    - **Property 8: Curvature parameterization**
    - **Validates: Requirements 4.3**

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required — property tests are written alongside each component
- Property tests validate the 10 correctness properties from the design document
- The disc topology computation depends on `hdbscan` and `numpy` (both available in science requirements)
- The agent container needs `hdbscan` added to `requirements-agent.txt`
- Frontend snapshot capture requires no new dependencies (Three.js canvas supports `toDataURL` natively)
- The Poincaré distance function must match the curvature convention used by the GNN (stored as `curvature_c` in the embedding response)
