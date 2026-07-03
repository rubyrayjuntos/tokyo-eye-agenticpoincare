# Requirements Document

## Introduction

The science container currently has no HTTP interface — the agent container dispatches compute jobs by shelling out to `docker compose exec`, which fails from inside a container. This feature adds a lightweight FastAPI service to the science container that exposes GNN inference, the full DTIE pipeline, cryptic binding site scanning, hyperbolic motif analysis, and MD validation as HTTP endpoints. The agent container calls these via httpx over the internal Docker network. All compute results persist to the DB through the Normalizer, honoring the principle that computed data is never ephemeral.

## Glossary

- **Science_API**: The FastAPI service running inside the science container on port 8001 (internal network only)
- **Agent_Container**: The lightweight coordinator that dispatches compute requests to the Science_API via httpx
- **Normalizer**: The governed write path (`data/normalizer/core.py`) that validates, persists, and audits all writes
- **InferenceEngine**: The in-process GNN runner that loads a checkpoint and executes the DTIE pipeline
- **Pipeline_Job**: A compute request tracked by job_id with status polling
- **Motif_Analyzer**: The hyperbolic Poincaré motif discovery module that identifies recurring geometric patterns
- **Cryptic_Scanner**: The binding site detection pipeline (seed → pocket → merge → MD validate)

## Requirements

### Requirement 1: Science Container HTTP Service

**User Story:** As the agent container, I want to call the science container's compute capabilities via HTTP, so that I don't need Docker socket access or shell-out patterns.

#### Acceptance Criteria

1. WHEN the science container starts, THE Science_API SHALL launch a FastAPI application on port 8001 bound to 0.0.0.0
2. THE Science_API SHALL expose a `GET /health` endpoint that returns service status, available checkpoints, and GPU availability
3. THE Science_API SHALL be reachable from the agent container at `http://science:8001` via the Docker network
4. THE Science_API SHALL use the same DATABASE_URL as the agent container to write results through the Normalizer
5. IF the Science_API encounters an unrecoverable error during startup, THEN THE Science_API SHALL log the error and exit with a non-zero status code

### Requirement 2: GNN Inference Endpoint

**User Story:** As a researcher, I want to run GNN inference on a structure via an API call, so that hyperbolic embeddings are computed and persisted without manual Docker commands.

#### Acceptance Criteria

1. THE Science_API SHALL expose a `POST /compute/gnn` endpoint that accepts structure_id, model_version, device, and checkpoint_path
2. WHEN the endpoint is called, THE InferenceEngine SHALL load the model, build the protein graph from DB residues, run forward pass, and persist embeddings through the Normalizer
3. WHEN inference completes, THE Science_API SHALL return run_id, structure_id, node_count, checkpoint_version_hash, and duration_ms
4. THE Science_API SHALL tag all persisted records with the checkpoint_version_hash in provenance metadata
5. IF the structure has no residues in dim_residue, THEN THE Science_API SHALL return HTTP 404 with a descriptive error

### Requirement 3: Full Pipeline Endpoint

**User Story:** As a researcher, I want to trigger the complete DTIE pipeline (GNN → all phases → source-leak → allosteric sites) in one call, so that all computed data is available after a single request.

#### Acceptance Criteria

1. THE Science_API SHALL expose a `POST /compute/pipeline` endpoint that accepts structure_id and optional source_leak_only flag
2. WHEN the endpoint is called, THE InferenceEngine SHALL execute GNN inference followed by all enabled DTIE phases
3. WHEN the pipeline completes, THE Science_API SHALL return run_id, phases_run list, assets_created count, and duration_ms
4. WHEN source_leak_only is true, THE Science_API SHALL execute only GNN inference and source-leak detection phases
5. THE Science_API SHALL persist all phase outputs through the Normalizer with provenance linking to the parent run_id

### Requirement 4: Cryptic Binding Site Scan Endpoint

**User Story:** As a researcher, I want to scan for cryptic binding sites on a structure, so that transient pockets are identified and persisted for downstream analysis.

#### Acceptance Criteria

1. THE Science_API SHALL expose a `POST /compute/cryptic-scan` endpoint that accepts structure_id and scan parameters (candidate_percentile, cluster_distance, min_cluster_size, max_pockets)
2. WHEN the endpoint is called, THE Cryptic_Scanner SHALL execute the full scan pipeline (seed generation → pocket detection → site merging)
3. WHEN the scan completes, THE Science_API SHALL return the discovered sites with residue membership, confidence scores, and druggability metrics
4. THE Science_API SHALL persist all discovered cryptic sites through the Normalizer into fact_cryptic_site and fact_binding_site_scan tables
5. IF no GNN embeddings exist for the structure, THEN THE Science_API SHALL return HTTP 422 indicating pipeline must run first

### Requirement 5: Hyperbolic Motif Analysis Endpoint

**User Story:** As a researcher, I want to discover recurring geometric patterns in hyperbolic embedding space, so that structural motifs are cataloged and available for cross-structure comparison.

#### Acceptance Criteria

1. THE Science_API SHALL expose a `POST /compute/motif-analysis` endpoint that accepts structure_id and clustering parameters (min_cluster_size, min_samples, metric)
2. WHEN the endpoint is called, THE Motif_Analyzer SHALL compute Poincaré distance matrix, run HDBSCAN clustering, identify medoid seeds, and classify motifs
3. WHEN analysis completes, THE Science_API SHALL return discovered motifs with cluster assignments, medoid residues, angular sectors, and radial density profile
4. THE Science_API SHALL persist motif results into a fact_hyperbolic_motif table with provenance
5. IF no hyperbolic embeddings exist for the structure, THEN THE Science_API SHALL return HTTP 422

### Requirement 6: MD Validation Endpoint

**User Story:** As a researcher, I want to validate cryptic site predictions with molecular dynamics, so that transient pocket stability is confirmed computationally.

#### Acceptance Criteria

1. THE Science_API SHALL expose a `POST /compute/md-validate` endpoint that accepts structure_id, site_id, and simulation parameters (duration_ns, temperature_k, force_field)
2. WHEN the endpoint is called with OpenMM available, THE Science_API SHALL run a steered MD simulation targeting the cryptic site and measure pocket persistence
3. WHEN validation completes, THE Science_API SHALL return pocket_open_fraction, rmsd_trajectory, and confidence_delta
4. THE Science_API SHALL persist MD validation results linked to the cryptic site and run provenance
5. IF OpenMM is not installed, THEN THE Science_API SHALL return HTTP 501 with a message indicating the dependency is unavailable
6. THE Science_API SHALL support a dry-run mode that validates inputs without running the simulation

### Requirement 7: Agent-Side HTTP Client

**User Story:** As the agent container, I want a typed async client for the Science_API, so that pipeline dispatch is a simple function call with proper error handling.

#### Acceptance Criteria

1. THE Agent_Container SHALL provide a `ScienceClient` class that wraps httpx calls to `http://science:8001`
2. THE ScienceClient SHALL expose async methods: `run_gnn()`, `run_pipeline()`, `run_cryptic_scan()`, `run_motif_analysis()`, `run_md_validate()`, `health()`
3. WHEN a Science_API call times out, THE ScienceClient SHALL raise a typed ScienceTimeoutError with the endpoint and duration
4. WHEN a Science_API call returns an error response, THE ScienceClient SHALL raise a typed ScienceComputeError with the error details
5. THE ScienceClient SHALL configure a default timeout of 600s for pipeline calls and 30s for health checks
6. THE dashboard `POST /api/pipeline/run` background task SHALL use the ScienceClient instead of Docker shell dispatch

### Requirement 8: Motif Persistence Schema

**User Story:** As a system architect, I want a dedicated table for hyperbolic motif results, so that motif data is queryable and available for dashboard hydration.

#### Acceptance Criteria

1. THE System SHALL provide a migration creating `fact_hyperbolic_motif` with columns: motif_id, structure_id, run_id, cluster_id, residue_ids (TEXT[]), medoid_residue_id, centroid_angle_deg, centroid_radius, motif_size, angular_sector, classification
2. THE migration SHALL include indexes on structure_id and run_id for efficient hydration queries
3. THE migration SHALL include a unique constraint on (run_id, cluster_id) to support idempotent upserts
4. THE fact_hyperbolic_motif table SHALL be included in the hydration endpoint response when motif data exists
