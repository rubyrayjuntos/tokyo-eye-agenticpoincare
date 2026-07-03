# Requirements Document

## Introduction

The Cryptic Site Calibration System addresses scientific validation gaps identified in peer review of the Cryptic Binding Site Discovery System (v1). The v1 implementation provides well-organized hypothesis generation infrastructure but lacks calibration against known ground truth, uses arbitrary thresholds without citation, conflates GNN-derived signals with physics-derived signals, and presents self-referential predictions alongside independent validations without distinction. This spec establishes the requirements for closing those gaps: a benchmark dataset, threshold calibration against known positives/negatives, explicit signal provenance labeling, and prediction independence classification.

## Glossary

- **Benchmark_Set**: A curated collection of experimentally confirmed cryptic/allosteric binding sites used to evaluate heuristic accuracy
- **Calibration_Run**: An execution of the full cryptic site discovery pipeline against a known-positive or known-negative site for threshold tuning
- **Ground_Truth_Label**: An experimentally validated classification (true_positive, true_negative, ambiguous) for a site in the benchmark set
- **Signal_Provenance**: The origin classification of a mapping signal: GNN_learned (from neural network inference) or structural_physics (from deterministic geometric computation)
- **Validation_Independence**: Classification of whether a falsifiable prediction is tested by an independent physical method (independent_physics) or by re-inference on the same model (model_self_consistency)
- **Precision**: The fraction of sites classified as a given type that are true positives per the benchmark set
- **ROC_AUC**: Area under the receiver operating characteristic curve, measuring classifier discrimination ability
- **SMD_Pulling_Rate**: The velocity at which a steered molecular dynamics simulation applies force, affecting work values via non-equilibrium bias
- **Jarzynski_Bias**: Systematic overestimation of free energy differences in non-equilibrium pulling simulations due to insufficient sampling

## Requirements

### Requirement 1: Benchmark Dataset for Heuristic Validation

**User Story:** As a computational biologist, I want a curated benchmark set of experimentally confirmed cryptic and allosteric sites, so that I can measure whether the classification heuristic identifies real sites with stated precision.

#### Acceptance Criteria

1. THE Benchmark_Set SHALL contain at minimum 20 experimentally confirmed cryptic binding sites with Ground_Truth_Labels sourced from published literature (PDB + citation)
2. THE Benchmark_Set SHALL contain at minimum 10 known-negative sites (surface pockets, non-cryptic) to measure false-positive rate
3. THE Benchmark_Set SHALL include at least 3 sites from the project's own validated findings (KRAS, SPOP, or equivalent) with independent experimental evidence
4. WHEN the classification heuristic is evaluated against the Benchmark_Set, THE system SHALL report per-site-type precision, recall, and overall ROC-AUC
5. THE system SHALL define a minimum acceptable precision threshold (configurable, default 0.7) below which the heuristic is flagged as unreliable for that site_type
6. WHEN precision for any site_type falls below the minimum threshold, THE system SHALL emit a warning in the ToolResult indicating low classifier confidence

### Requirement 2: MD Threshold Calibration

**User Story:** As a researcher, I want MD success thresholds calibrated against known binding vs. non-binding sites, so that the pass/fail gate reflects physical reality rather than arbitrary numbers.

#### Acceptance Criteria

1. THE system SHALL run at least 3 known true-positive cryptic sites through the SMD_three_phase protocol and record the work values
2. THE system SHALL run at least 3 known true-negative (non-binding) sites through the same protocol and record the work values
3. WHEN calibration data is available, THE system SHALL set the work threshold at a value that separates true positives from true negatives with stated sensitivity and specificity
4. THE system SHALL document the SMD pulling rate, force constant, and simulation parameters used for calibration alongside the threshold
5. THE system SHALL include a calibration_metadata field in DEFAULT_SUCCESS_CRITERIA recording source (calibrated vs. provisional), date, and reference sites used
6. WHEN a threshold is marked as "provisional" (uncalibrated), THE system SHALL include a warning in the MD validation result indicating the threshold lacks empirical basis

### Requirement 3: Signal Provenance Labeling

**User Story:** As a system architect maintaining IP clarity, I want each mapping signal explicitly labeled with its provenance (GNN-learned vs. structural-physics), so that downstream consumers and patent documentation correctly classify the module's dependencies.

#### Acceptance Criteria

1. THE Mapper SHALL annotate each signal in the multi-signal scoring function with a Signal_Provenance label (GNN_learned or structural_physics)
2. THE CrypticBindingSiteSpec SHALL include a signal_provenance_summary field listing which provenance classes were load-bearing in the discovery
3. WHEN all load-bearing signals are GNN_learned, THE system SHALL classify the discovery as "GNN-gated" in the provenance summary
4. WHEN the discovery uses a mix of GNN and physics signals, THE system SHALL classify it as "hybrid-gated"
5. THE system SHALL expose signal provenance in the ToolResult metadata for audit purposes

### Requirement 4: Prediction Independence Classification

**User Story:** As a researcher, I want each falsifiable prediction annotated with its validation independence level, so that I can distinguish genuine physical validation from model self-consistency checks.

#### Acceptance Criteria

1. THE FalsifiablePrediction schema SHALL include a validation_class field with allowed values: "independent_physics", "model_self_consistency", "experimental"
2. WHEN a prediction's test_tool invokes the same GNN that produced the discovery, THE system SHALL classify it as "model_self_consistency"
3. WHEN a prediction's test_tool invokes MD simulation or experimental assay, THE system SHALL classify it as "independent_physics" or "experimental" respectively
4. WHEN the Agent presents falsifiable predictions to the user, THE Agent SHALL group predictions by validation_class and indicate that model_self_consistency predictions carry lower epistemic weight
5. THE system SHALL not count model_self_consistency predictions toward the overall validation confidence of a site

### Requirement 5: Heuristic Versioning and Reproducibility

**User Story:** As a researcher, I want classification heuristic versions tracked alongside results, so that when thresholds change during calibration, I can trace which version produced each historical classification.

#### Acceptance Criteria

1. THE CrypticBindingSiteSpec SHALL include a heuristic_version field recording the version of the classification heuristic that produced the site_type inference
2. WHEN the heuristic thresholds or decision tree structure change, THE system SHALL increment the heuristic_version
3. THE system SHALL persist heuristic_version in the fact_cryptic_site table alongside the spec
4. WHEN comparing results across heuristic versions, THE system SHALL support filtering by heuristic_version in queries

### Requirement 6: Calibration Pipeline Integration

**User Story:** As a developer, I want a repeatable calibration pipeline that runs the benchmark set through the discovery system and reports metrics, so that threshold tuning is systematic rather than manual.

#### Acceptance Criteria

1. THE system SHALL provide a calibration script that runs all benchmark sites through the mapper and reports precision/recall/ROC-AUC per site_type
2. THE system SHALL provide a calibration script that runs positive/negative control sites through each SMD protocol and reports threshold recommendations
3. WHEN calibration completes, THE system SHALL output a structured report (JSON) with recommended thresholds, confidence intervals, and comparison to previous calibration
4. THE calibration pipeline SHALL be runnable via `make calibrate-cryptic` from the project root
5. WHEN calibration metrics degrade compared to previous run, THE system SHALL flag the regression in the report
