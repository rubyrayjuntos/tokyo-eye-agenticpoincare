# Implementation Plan: Science Container API

## Overview

Add a FastAPI HTTP service to the science container that exposes compute endpoints (GNN, pipeline, cryptic scan, motif analysis, MD validation). Replace the agent's docker-shell dispatch with a typed httpx client. Add the motif persistence migration. This is a short-term solution — the plan is to layer Prefect orchestration on top later.

## Tasks

- [x] 1. Create science API app skeleton
  - Create `science/api/__init__.py`
  - Create `science/api/app.py` with FastAPI app, lifespan (DB pool open/close), and health router
  - Create `science/api/routers/__init__.py`
  - Create `science/api/routers/health.py` with `GET /health` (GPU detection, checkpoint listing, DB connectivity)
  - Update `Dockerfile.science` CMD to run uvicorn on port 8001
  - Update `docker-compose.yml`: remove `entrypoint`/`command` overrides, add healthcheck for science container
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement GNN inference endpoint
  - [x] 2.1 Create `science/api/routers/compute.py` with `POST /compute/gnn`
    - Accept GNNRequest (structure_id, model_version, device, checkpoint_path)
    - Load checkpoint, compute hash, build graph from DB residues, run forward pass
    - Persist embeddings through Normalizer with provenance
    - Return GNNResponse (run_id, node_count, checkpoint_version_hash, duration_ms)
    - Handle 404 when structure has no residues
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.2 Write property test for GNN persistence completeness
    - **Property 2: GNN persistence completeness**
    - **Validates: Requirements 2.2, 2.3**

- [x] 3. Implement full pipeline endpoint
  - [x] 3.1 Add `POST /compute/pipeline` to compute router
    - Accept PipelineRequest (structure_id, source_leak_only, device, checkpoint_path)
    - Run GNN inference → DTIEOrchestrator with all phases
    - Persist all phase outputs through Normalizer
    - Return PipelineResponse (run_id, phases_run, assets_created, duration_ms)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 3.2 Write property test for pipeline phase coverage
    - **Property 3: Pipeline phase coverage**
    - **Validates: Requirements 3.2, 3.3**

- [x] 4. Implement cryptic binding site scan endpoint
  - [x] 4.1 Add `POST /compute/cryptic-scan` to compute router
    - Accept CrypticScanRequest (structure_id, candidate_percentile, cluster_distance, min_cluster_size, max_pockets)
    - Check GNN embeddings exist (422 if not)
    - Run CrypticScanner (seed → pocket → merge)
    - Persist to fact_cryptic_site and fact_binding_site_scan
    - Return CrypticScanResponse (sites_found, sites list, duration_ms)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 4.2 Write property test for cryptic scan precondition enforcement
    - **Property 4: Cryptic scan precondition enforcement**
    - **Validates: Requirements 4.5**

- [x] 5. Implement motif analysis endpoint and migration
  - [x] 5.1 Create migration `data/aurora/migrations/045_hyperbolic_motif.sql`
    - Create fact_hyperbolic_motif table with columns per design
    - Add indexes on structure_id, run_id
    - Add unique constraint on (run_id, cluster_id)
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 5.2 Add `POST /compute/motif-analysis` to compute router
    - Accept MotifAnalysisRequest (structure_id, min_cluster_size, min_samples)
    - Check hyperbolic embeddings exist (422 if not)
    - Compute Poincaré distance matrix, HDBSCAN clustering, identify medoids
    - Persist to fact_hyperbolic_motif through Normalizer
    - Return MotifAnalysisResponse (motif_count, motifs list, duration_ms)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.3 Write property test for motif persistence round-trip
    - **Property 5: Motif persistence round-trip**
    - **Validates: Requirements 5.3, 5.4, 8.1, 8.3**

- [x] 6. Implement MD validation endpoint
  - [x] 6.1 Add `POST /compute/md-validate` to compute router
    - Accept MDValidateRequest (structure_id, site_id, duration_ns, temperature_k, force_field, dry_run)
    - Check OpenMM availability (501 if not installed)
    - If dry_run: validate inputs, return without simulation
    - If real: run steered MD, measure pocket persistence
    - Persist MD results linked to cryptic site and provenance
    - Return MDValidateResponse (pocket_open_fraction, confidence_delta, duration_ms)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 7. Checkpoint - Verify science API starts and responds
  - Build science container, start it, verify `/health` responds
  - Test GNN endpoint with a test structure
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement agent-side ScienceClient
  - [x] 8.1 Create `agent/tools/science_client.py`
    - ScienceClient class with httpx async methods
    - Typed exceptions: ScienceTimeoutError, ScienceComputeError
    - Methods: health(), run_gnn(), run_pipeline(), run_cryptic_scan(), run_motif_analysis(), run_md_validate()
    - Configurable timeouts (600s pipeline, 30s health)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 8.2 Write property test for ScienceClient error typing
    - **Property 6: ScienceClient error typing**
    - **Validates: Requirements 7.3, 7.4**

  - [x] 8.3 Wire ScienceClient into dashboard pipeline dispatch
    - Replace `science_dispatch.run_full_pipeline_via_container` with `ScienceClient.run_pipeline()`
    - Update `_run_pipeline_background` in `dashboard.py` to use ScienceClient
    - Update `check_science_container` to use `ScienceClient.health()`
    - _Requirements: 7.6_

- [x] 9. Wire ScienceClient into agent tools
  - Update `agent/llm/agents.py` `_run_pipeline` to use ScienceClient
  - Update tool handlers that dispatch to science container
  - _Requirements: 7.6_

- [x] 10. Final checkpoint - End-to-end ingest → pipeline → hydrate
  - Ingest a structure via `/api/ingest`
  - Trigger pipeline via `/api/pipeline/run`
  - Verify pipeline completes and DB has embeddings
  - Verify `/api/structures/{id}/hydrate` returns populated data
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- This is a short-term FastAPI solution — Prefect will be layered on later for orchestration, retries, and observability
- All tasks are required including property tests
- The science container needs torch, PyG, geoopt, biotite, and optionally OpenMM
- The agent container only needs httpx (already added to requirements-agent.txt)
- MD validation (task 6) can be deferred if OpenMM isn't in the science image yet
