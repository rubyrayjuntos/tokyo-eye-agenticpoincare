# Requirements Document

## Introduction

A training launcher script that resolves legacy import paths from the v5 training script (`experiments/training/v5/train_v5.py`) to the consolidated module locations in the monorepo, enabling local execution of the 3-stage curriculum training for the GOSPConeMapper-v5 model. The launcher must support running individual stages (particularly Stage 1 radial calibration) in isolation to verify convergence before committing to the full 80-epoch curriculum.

## Glossary

- **Launcher**: The adapter script that patches legacy module imports and invokes the v5 training loop from the consolidated repository layout.
- **GOSPConeMapper**: The v5 decoupled radial-angular hyperbolic GNN model defined in `science.dtie.v5.gnn.model`.
- **RadialHead**: The MLP sub-module that predicts per-residue burial depth (scalar). Supervised exclusively by cone loss.
- **AngularHead**: The MLP sub-module that predicts per-residue angular direction (unit vector). Supervised by domain separation and angular diversity losses.
- **Stage_1**: Radial calibration phase — 20 epochs with angular head frozen, cone_coeff=0.30.
- **Stage_2**: Angular structure phase — 30 epochs with radial head frozen, domain_sep dominant.
- **Stage_3**: Joint fine-tuning phase — 30 epochs with all parameters unfrozen, reduced learning rate.
- **Curriculum**: The ordered sequence of Stage_1 → Stage_2 → Stage_3 totaling 80 epochs.
- **Legacy_Imports**: The old module references (`from Gnnv5 import ...`, `from train_v4 import ...`) used in `train_v5.py` that must be redirected to consolidated paths.
- **Convergence_Metric**: The cone_loss MSE and depth_std variance penalty used to evaluate whether the radial head has learned burial depth differentiation.
- **TRAINING_TARGETS**: The set of 11 PDB structures (KRAS WT/G12D, NRAS, BRAF, STAT3, etc.) used as training data.

## Requirements

### Requirement 1: Legacy Import Resolution

**User Story:** As a developer, I want the launcher to resolve legacy import paths to consolidated module locations, so that the existing training script runs without modification in the monorepo.

#### Acceptance Criteria

1. WHEN the Launcher is executed, THE Launcher SHALL patch `sys.modules` to alias `Gnnv5` to `science.dtie.v5.gnn.model`
2. WHEN the Launcher is executed, THE Launcher SHALL make `TRAINING_TARGETS` and `load_protein_graph` available under the `train_v4` namespace
3. WHEN the training script imports `GOSPConeMapper`, `gosp_loss_v5`, `build_optimizer`, or `precompute_clustering` from `Gnnv5`, THE Launcher SHALL resolve these to the corresponding symbols in `science.dtie.v5.gnn.model`
4. IF a required symbol is missing from the consolidated module, THEN THE Launcher SHALL raise an ImportError with a descriptive message identifying the missing symbol and its expected location
5. IF the legacy script imports deprecated nomenclature (e.g., `GOSPConeMapper`), THE Launcher SHALL map it to the current monorepo equivalent within the patched module space

### Requirement 2: Stage-Selective Execution

**User Story:** As a developer, I want to run individual training stages in isolation, so that I can verify radial calibration convergence before committing to the full curriculum.

#### Acceptance Criteria

1. WHEN the Launcher is invoked with `--stage 1`, THE Launcher SHALL execute only Stage_1 (radial calibration, 20 epochs)
2. WHEN the Launcher is invoked with `--stage 2`, THE Launcher SHALL execute only Stage_2 (angular structure, 30 epochs)
3. WHEN the Launcher is invoked with `--stage 3`, THE Launcher SHALL execute only Stage_3 (joint fine-tuning, 30 epochs)
4. WHEN the Launcher is invoked without a `--stage` argument, THE Launcher SHALL execute the full Curriculum (Stage_1 → Stage_2 → Stage_3)
5. WHEN a stage completes, THE Launcher SHALL save a checkpoint file named `v5_stage{N}_{structure_count}prot.pt` in the output directory

### Requirement 3: Convergence Monitoring

**User Story:** As a developer, I want per-epoch loss metrics and gradient norms logged, so that I can evaluate whether the radial head is learning burial depth differentiation.

#### Acceptance Criteria

1. WHEN an epoch completes, THE Launcher SHALL log the cone_loss, depth_std, and total_loss values
2. WHEN an epoch completes, THE Launcher SHALL log gradient norms for radial_head, angular_head, and backbone parameter groups separately
3. WHEN Stage_1 completes, THE Launcher SHALL report the final cone_loss MSE and depth_std values as a convergence summary
4. WHEN the depth_std exceeds 0.20 at Stage_1 completion, THE Launcher SHALL log a message indicating the variance penalty is satisfied
5. IF cone_loss evaluates to NaN OR depth_std collapses to exactly 0.0 for three consecutive epochs, THEN THE Launcher SHALL abort the training run and log a fatal convergence error

### Requirement 4: Checkpoint Management

**User Story:** As a developer, I want training checkpoints saved with metadata, so that I can resume training or deploy a stage-specific checkpoint to the inference runner.

#### Acceptance Criteria

1. WHEN a checkpoint is saved, THE Launcher SHALL include `model_state_dict`, `optimizer_state_dict`, `epoch`, `stage`, and `metrics` in the checkpoint file
2. WHEN a checkpoint is saved, THE Launcher SHALL include a `training_config` field recording hyperparameters (lr, coefficients, frozen heads)
3. WHEN the Launcher is invoked with `--resume <path>`, THE Launcher SHALL load the checkpoint and resume training from the saved epoch and stage
4. IF the checkpoint file does not exist at the resume path, THEN THE Launcher SHALL raise a FileNotFoundError with the attempted path

### Requirement 5: Training Data Loading

**User Story:** As a developer, I want the launcher to download and cache PDB structures from RCSB, so that training can run without pre-staged data.

#### Acceptance Criteria

1. WHEN a PDB structure is not found in the local cache directory, THE Launcher SHALL download it from RCSB
2. WHEN a PDB structure is successfully downloaded, THE Launcher SHALL cache it in the `--pdb_dir` directory for subsequent runs
3. WHEN a PDB download fails, THE Launcher SHALL log a warning and continue with the remaining structures
4. THE Launcher SHALL load all available structures from TRAINING_TARGETS and report the count before training begins
5. WHEN downloading multiple structures from RCSB, THE Launcher SHALL implement a polite rate limit (minimum 1-second delay between requests) and exponential backoff for failed network calls

### Requirement 6: Configuration and CLI

**User Story:** As a developer, I want CLI arguments for all key hyperparameters, so that I can experiment with training configurations without editing code.

#### Acceptance Criteria

1. THE Launcher SHALL accept `--device` (default: cpu), `--lr` (default: 5e-4), `--hidden` (default: 128), `--num_layers` (default: 6), `--num_experts` (default: 4) as CLI arguments
2. THE Launcher SHALL accept `--output_dir` (default: ./checkpoints_v5) and `--pdb_dir` (default: /tmp/dtie_pdb_cache) as CLI arguments
3. THE Launcher SHALL accept `--stage` (optional, values: 1, 2, 3) to select individual stage execution
4. THE Launcher SHALL accept `--resume` (optional, path to checkpoint) for resuming training
5. WHEN an invalid stage number is provided, THE Launcher SHALL exit with an error message listing valid stage values
