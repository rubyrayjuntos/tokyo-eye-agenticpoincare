# Requirements Document

## Introduction

The DTIE v5 pipeline currently executes inside a Docker container (`science` service), which introduces container drift: the host model checkpoints and code evolve, but the container runs a frozen snapshot. This causes stale high-uncertainty values to persist in the database while local validation shows correct results. This spec defines an in-process pipeline runner (`InferenceEngine`) that executes the full GNN inference and phase pipeline directly in the host Python process, bypassing Docker overhead, enabling debugger attachment, and ensuring the same checkpoint and code used for validation are used for persistence.

## Glossary

- **InferenceEngine**: The in-process wrapper class that loads a checkpoint, runs GNN inference, executes downstream phases, and persists results through the Normalizer with upsert semantics.
- **Container_Drift**: The condition where the Docker container's code/weights diverge from the host's validated state, producing stale or incorrect results.
- **Checkpoint_Version_Hash**: A SHA-256 hash of the checkpoint file bytes, used to tag persisted records so staleness is detectable by querying version.
- **Stale_Row**: A database record produced by an older checkpoint version that no longer reflects the current model's output.
- **Upsert**: An INSERT that, on conflict with a natural key, updates the existing row rather than failing or skipping.
- **Normalizer**: The governed write path (`data/normalizer/core.py`) that validates, persists, and audits all data writes.
- **V5GNNRunner**: The existing async runner class in `science/dtie/v5/gnn/runner.py` that wraps GOSPConeMapper-v5 for inference.
- **DTIEOrchestrator**: The pipeline orchestrator in `science/dtie/v5/orchestrator/pipeline.py` that sequences phases after GNN inference.
- **ProvenanceContext**: Required metadata (run_id, structure_id, model_version, checkpoint_uri, etc.) accompanying every Normalizer write.

## Requirements

### Requirement 1: In-Process GNN Inference

**User Story:** As a developer, I want to run GNN inference directly in my host Python process, so that I use the exact checkpoint and code I just validated without Docker volume mounting uncertainty.

#### Acceptance Criteria

1. WHEN the InferenceEngine is instantiated with a checkpoint path, THE InferenceEngine SHALL load the GOSPConeMapper-v5 model into the host process memory
2. WHEN the InferenceEngine `run` method is called with a PDB ID, THE InferenceEngine SHALL build the protein graph, run GNN inference, and return a GNNInferenceResult
3. WHEN the InferenceEngine loads a checkpoint, THE InferenceEngine SHALL compute and store the Checkpoint_Version_Hash (SHA-256 of the checkpoint file bytes)
4. IF the checkpoint file does not exist at the specified path, THEN THE InferenceEngine SHALL raise a FileNotFoundError with the attempted path

### Requirement 2: Full Pipeline Execution

**User Story:** As a researcher, I want the in-process runner to execute the full phase pipeline (not just GNN), so that all downstream analyses (source-leak, allosteric sites, pharmacophore) run with the same model outputs.

#### Acceptance Criteria

1. WHEN the InferenceEngine `run` method completes GNN inference, THE InferenceEngine SHALL pass the GNNInferenceResult to the DTIEOrchestrator for full phase execution
2. WHEN the pipeline completes, THE InferenceEngine SHALL return a PipelineResult containing all phase outputs and the run_id
3. WHEN the InferenceEngine is configured with `source_leak_only=True`, THE InferenceEngine SHALL execute only GNN inference and source-leak detection phases
4. THE InferenceEngine SHALL accept a `device` parameter (default: "cpu") to control whether inference runs on CPU or GPU

### Requirement 3: Checkpoint-Versioned Persistence

**User Story:** As a developer, I want all persisted results tagged with the checkpoint version hash, so that I can identify and query stale rows produced by older model versions.

#### Acceptance Criteria

1. WHEN the InferenceEngine persists results through the Normalizer, THE InferenceEngine SHALL include the Checkpoint_Version_Hash in the ProvenanceContext metadata
2. WHEN the InferenceEngine persists results, THE InferenceEngine SHALL set `checkpoint_uri` in ProvenanceContext to the absolute path of the checkpoint file
3. WHEN querying results, THE System SHALL support filtering by checkpoint_version_hash to identify which records were produced by which model version
4. THE InferenceEngine SHALL include the checkpoint_version_hash in the returned PipelineResult for caller inspection

### Requirement 4: Forced Upsert on Stale Data

**User Story:** As a developer, I want the in-process runner to overwrite existing results for the same structure, so that stale high-uncertainty values from container drift are replaced with correct values.

#### Acceptance Criteria

1. WHEN the InferenceEngine persists GNN node embeddings for a structure that already has results in the database, THE Normalizer SHALL update the existing rows via upsert keyed on (run_id, residue_id)
2. WHEN the InferenceEngine persists phase outputs for a structure with existing results, THE Normalizer SHALL update existing rows via upsert on their natural keys
3. THE InferenceEngine SHALL generate a new unique run_id for each execution, ensuring provenance distinguishes runs even for the same structure
4. WHEN `force_overwrite=True` is passed to the InferenceEngine, THE InferenceEngine SHALL delete existing results for the target structure before persisting new ones

### Requirement 5: Database Connection Management

**User Story:** As a developer, I want the in-process runner to manage its own database connection using the same DATABASE_URL as the rest of the system, so that it writes to the same governed data layer.

#### Acceptance Criteria

1. THE InferenceEngine SHALL read DATABASE_URL from environment variables (with fallback to the development default)
2. WHEN the InferenceEngine `run` method is called, THE InferenceEngine SHALL open an async database connection, execute the pipeline, commit on success, and close the connection
3. IF the database connection fails, THEN THE InferenceEngine SHALL raise a ConnectionError with the database URL (credentials redacted)
4. THE InferenceEngine SHALL support an optional `db` parameter to accept an externally-managed database connection for testing

### Requirement 6: CLI Interface

**User Story:** As a developer, I want a CLI entrypoint for the in-process runner, so that I can invoke it from the terminal with the same ergonomics as the Docker-based runner.

#### Acceptance Criteria

1. THE InferenceEngine CLI SHALL accept `--structure` (required), `--checkpoint` (default: checkpoints_v5/v5_stage4_11prot.pt), `--device` (default: cpu), and `--force-overwrite` (flag) arguments
2. WHEN the CLI completes successfully, THE InferenceEngine SHALL print a JSON summary containing run_id, structure_id, checkpoint_version_hash, num_nodes, phases_run, and assets_created
3. WHEN the CLI encounters an error, THE InferenceEngine SHALL print a JSON error object with error message and exit with non-zero status
4. THE InferenceEngine CLI SHALL accept `--source-leak-only` to limit execution to GNN + source-leak detection

### Requirement 7: Stale Data Flush Utility

**User Story:** As a developer, I want a utility to flush stale results from the database before switching to the in-process runner, so that old container-produced data doesn't interfere with fresh results.

#### Acceptance Criteria

1. THE System SHALL provide a `flush_stale` function that deletes all results for a given structure_id where checkpoint_version_hash does not match the current checkpoint
2. WHEN `flush_stale` is called, THE System SHALL report the number of rows deleted from each fact table (fact_source_leak, fact_allosteric_site, fact_phase3_persistence, governed_asset)
3. THE System SHALL provide a `--flush-before-run` CLI flag on the InferenceEngine that calls `flush_stale` before executing the pipeline
4. IF `flush_stale` is called without a structure_id, THEN THE System SHALL raise a ValueError requiring explicit structure targeting (no global flush without confirmation)
