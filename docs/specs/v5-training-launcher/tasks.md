# Implementation Plan: V5 Training Launcher

## Overview

Build a training launcher script that patches legacy imports and enables stage-selective execution of the v5 GOSPConeMapper training curriculum from the consolidated monorepo. Implementation uses Python with Hypothesis for property-based testing.

## Tasks

- [x] 1. Create core launcher module with import patching
  - [x] 1.1 Create `tokyo-eye-agenticpoincare/scripts/train_launcher.py` with `patch_legacy_imports()` function
    - Patch `sys.modules['Gnnv5']` → `science.dtie.v5.gnn.model`
    - Patch `sys.modules['train_v4']` → `experiments.training.v4.train_v4`
    - Verify all required symbols exist, raise `ImportError` with descriptive message if missing
    - Handle deprecated nomenclature aliasing
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 Write property test for import alias resolution
    - **Property 1: Import alias resolution**
    - **Validates: Requirements 1.1, 1.3**

- [x] 2. Implement ConvergenceMonitor
  - [x] 2.1 Create `ConvergenceMonitor` class in `train_launcher.py`
    - `record_epoch(metrics)` — stores epoch metrics history
    - `should_abort()` — returns `(True, reason)` if NaN cone_loss or depth_std=0.0 for 3 consecutive epochs
    - `stage_summary()` — returns final cone_loss, depth_std, variance penalty status
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 2.2 Write property test for fail-fast on convergence collapse
    - **Property 4: Fail-fast on convergence collapse**
    - **Validates: Requirements 3.5**

- [-] 3. Implement CheckpointManager
  - [x] 3.1 Create `CheckpointManager` class in `train_launcher.py`
    - `save(model, optimizer, epoch, stage, metrics, config, output_dir)` — saves checkpoint with all required fields
    - `load(path)` — loads checkpoint, raises `FileNotFoundError` if missing
    - Checkpoint includes: `model_state_dict`, `optimizer_state_dict`, `epoch`, `stage`, `metrics`, `training_config`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 3.2 Write property test for checkpoint structure completeness
    - **Property 5: Checkpoint structure completeness**
    - **Validates: Requirements 4.1, 4.2**

  - [x] 3.3 Write property test for checkpoint save/resume round-trip
    - **Property 6: Checkpoint save/resume round-trip**
    - **Validates: Requirements 4.3**

- [x] 4. Implement StageRunner with stage-selective execution
  - [x] 4.1 Create `StageRunner` class in `train_launcher.py`
    - Stage definitions with epochs, freeze flags, and loss coefficients
    - `run(stage, model, proteins, config)` — executes selected stage(s)
    - Integrates `ConvergenceMonitor` for per-epoch tracking
    - Integrates `CheckpointManager` for end-of-stage saves
    - Checkpoint naming: `v5_stage{N}_{count}prot.pt`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 4.2 Write property test for checkpoint naming convention
    - **Property 2: Checkpoint naming convention**
    - **Validates: Requirements 2.5**

  - [x] 4.3 Write property test for epoch metrics completeness
    - **Property 3: Epoch metrics completeness**
    - **Validates: Requirements 3.1, 3.2**

- [x] 5. Implement RCSBDownloader with rate limiting
  - [x] 5.1 Create `RCSBDownloader` class in `train_launcher.py`
    - Wraps existing `_download_pdb` logic with rate limiting (1s delay between requests)
    - Exponential backoff on failure (1s, 2s, 4s, max 3 retries)
    - Logs warning and continues on permanent failure
    - Reports loaded protein count
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.2 Write property test for protein count consistency
    - **Property 7: Protein count consistency**
    - **Validates: Requirements 5.4**

- [x] 6. Implement CLI interface and main entry point
  - [x] 6.1 Add CLI argument parsing to `train_launcher.py`
    - `--device` (default: cpu), `--lr` (default: 5e-4), `--hidden` (default: 128)
    - `--num_layers` (default: 6), `--num_experts` (default: 4)
    - `--output_dir` (default: ./checkpoints_v5), `--pdb_dir` (default: /tmp/dtie_pdb_cache)
    - `--stage` (optional: 1, 2, 3), `--resume` (optional: path)
    - Validate stage value, exit with error on invalid input
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 6.2 Wire main() entry point
    - Call `patch_legacy_imports()`
    - Parse CLI args into `TrainingConfig`
    - Initialize `RCSBDownloader`, load proteins
    - Initialize model, optimizer via `build_optimizer`
    - Handle `--resume` (load checkpoint, restore state)
    - Invoke `StageRunner.run()`
    - Print final convergence summary
    - _Requirements: 1.1, 2.4, 4.3, 5.4_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Integration validation
  - [x] 8.1 Run Stage 1 radial calibration with synthetic data
    - Create a minimal integration test that patches imports, builds a small synthetic protein graph, runs 2 epochs of Stage 1, and verifies the checkpoint is saved with correct structure
    - Verify gradient isolation: radial_head gradients are non-zero, angular_head gradients are zero during Stage 1
    - _Requirements: 2.1, 3.1, 4.1_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks are all required for high-assurance from the start
- Each task references specific requirements for traceability
- The launcher lives in `scripts/train_launcher.py` alongside existing scripts
- Property tests use Hypothesis (minimum 100 iterations per property)
- The training script `experiments/training/v5/train_v5.py` is NOT modified — all adaptation happens in the launcher
