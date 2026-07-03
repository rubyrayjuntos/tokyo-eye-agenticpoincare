# Requirements Document

## Introduction

The Resistance Profiler (RP) is a module that automates drug resistance mechanism classification using the DTIE v5 hyperbolic GNN pipeline. It performs virtual mutations on protein structures, runs GNN inference on the perturbed graphs, and classifies the resulting topological signatures into resistance mechanism types (Type I: Steric/Binding Interference vs Type II: Allosteric Uncoupling). This enables high-throughput screening of clinical variants and prediction of resistance mechanisms before they are observed clinically.

The module builds on validated experimental results showing that the hyperbolic GNN produces distinct, quantifiable topological signatures for different resistance mechanisms: localized uncertainty collapse for steric mutations (T315I) vs distributed hub disruption for allosteric mutations (E255K).

## Glossary

- **Resistance_Profiler**: The module that orchestrates virtual mutation scanning and mechanism classification.
- **Virtual_Mutation**: A computational perturbation of the protein graph that simulates an amino acid substitution by modifying node features and edge attributes without altering graph connectivity.
- **Hub_Delta**: The change in epistemic uncertainty at identified propagation hub residues between wild-type and mutant inference runs.
- **Propagation_Radius**: The number of residues beyond the mutation site that show significant topological perturbation (Δeps > threshold).
- **Mechanism_Class**: The classification output — Type_I_Steric, Type_II_Allosteric, or Hybrid.
- **Edge_Perturbation_Factor**: A scaling factor applied to edge attributes around the mutation site to simulate steric or electrostatic changes.
- **Mutation_Spec**: A description of a single amino acid substitution (chain, residue_index, wild_type_aa, mutant_aa).
- **Resistance_Report**: The structured JSON output containing classification, metrics, and affected pathways.
- **Propagation_Hub**: A residue identified by the wild-type pipeline as a source leak or high-coupling pathway target that serves as a reference point for measuring mutation impact.

## Requirements

### Requirement 1: Virtual Mutation Graph Perturbation

**User Story:** As a researcher, I want to apply virtual mutations to a protein graph before GNN inference, so that I can observe how the model's topological outputs change in response to specific amino acid substitutions.

#### Acceptance Criteria

1. WHEN a Mutation_Spec is provided with a valid chain, residue_index, and mutant amino acid, THE Resistance_Profiler SHALL locate the corresponding node in the protein graph and apply feature and edge perturbations
2. WHEN applying a virtual mutation, THE Resistance_Profiler SHALL modify edge attributes for edges incident to the mutated node using an Edge_Perturbation_Factor derived from the physicochemical difference between wild-type and mutant amino acids
3. WHEN applying a virtual mutation, THE Resistance_Profiler SHALL modify the node feature vector at the mutation site to reflect the mutant amino acid properties (hydropathy, volume, charge)
4. THE Resistance_Profiler SHALL preserve graph connectivity (node count and edge count) after applying a virtual mutation
5. IF the specified residue_index does not exist in the target chain, THEN THE Resistance_Profiler SHALL raise a ValueError with the attempted residue and available range

### Requirement 2: Wild-Type Baseline Caching

**User Story:** As a researcher running multiple mutations against the same structure, I want the wild-type inference result cached, so that I avoid redundant computation and can directly compare mutant results to a consistent baseline.

#### Acceptance Criteria

1. WHEN the Resistance_Profiler runs its first mutation for a given structure_id, THE Resistance_Profiler SHALL perform a wild-type GNN inference and cache the result
2. WHEN subsequent mutations are requested for the same structure_id, THE Resistance_Profiler SHALL reuse the cached wild-type result without re-running inference
3. THE Resistance_Profiler SHALL store the wild-type propagation hubs (source leak residues and top coupling pathway targets) as reference points for Hub_Delta computation
4. WHEN a force_refresh parameter is set to True, THE Resistance_Profiler SHALL discard the cached baseline and recompute

### Requirement 3: Resistance Mechanism Classification

**User Story:** As a researcher, I want each mutation automatically classified into a resistance mechanism type, so that I can quickly understand whether a variant disrupts drug binding or allosteric propagation.

#### Acceptance Criteria

1. WHEN a mutant inference result is compared to the wild-type baseline, THE Resistance_Profiler SHALL compute Hub_Delta for each identified propagation hub
2. WHEN Hub_Delta values are all below the allosteric threshold (absolute value < 0.003) AND the mutation site shows significant local perturbation (Δeps < -0.05), THE Resistance_Profiler SHALL classify the mutation as Type_I_Steric
3. WHEN any Hub_Delta exceeds the allosteric threshold (absolute value >= 0.003) AND the Propagation_Radius exceeds 5 residues, THE Resistance_Profiler SHALL classify the mutation as Type_II_Allosteric
4. WHEN a mutation shows both localized steric signature AND partial hub disruption, THE Resistance_Profiler SHALL classify it as Hybrid with a steric_score and allosteric_score
5. THE Resistance_Profiler SHALL compute a confidence_score between 0.0 and 1.0 based on the magnitude of the distinguishing metrics relative to the classification thresholds

### Requirement 4: Batch Mutation Scanning

**User Story:** As a researcher, I want to scan a list of mutations against a single structure in one invocation, so that I can efficiently profile an entire variant library.

#### Acceptance Criteria

1. WHEN a list of Mutation_Specs is provided, THE Resistance_Profiler SHALL run the wild-type baseline once and then apply each mutation independently
2. WHEN processing a batch, THE Resistance_Profiler SHALL return a list of Resistance_Reports in the same order as the input mutations
3. THE Resistance_Profiler SHALL report progress during batch processing (mutations completed / total)
4. IF any single mutation fails (e.g., invalid residue), THEN THE Resistance_Profiler SHALL record the error in that mutation's report and continue processing remaining mutations

### Requirement 5: Structured Output

**User Story:** As a researcher, I want the profiling results in a structured JSON format, so that I can integrate them into downstream drug design workflows and databases.

#### Acceptance Criteria

1. THE Resistance_Profiler SHALL output a Resistance_Report containing: variant name, mechanism_class, confidence_score, site_uncertainty_delta, hub_propagation_deltas, affected_pathways, and structural_impact description
2. WHEN a batch scan completes, THE Resistance_Profiler SHALL output a summary containing: total_variants, type_i_count, type_ii_count, hybrid_count, and the full list of individual reports
3. THE Resistance_Profiler SHALL include the wild-type baseline metrics (source leaks, top pathways, spectral gap) in the output for reference
4. THE Resistance_Profiler SHALL serialize all output as JSON with numpy values converted to native Python types

### Requirement 6: Amino Acid Property Lookup

**User Story:** As a researcher, I want the mutation perturbation to be physicochemically grounded, so that the virtual mutation reflects real biochemical differences between amino acids.

#### Acceptance Criteria

1. THE Resistance_Profiler SHALL maintain a lookup table mapping each of the 20 standard amino acids to physicochemical properties (van der Waals volume, hydropathy index, charge at pH 7)
2. WHEN computing the Edge_Perturbation_Factor, THE Resistance_Profiler SHALL derive it from the ratio of mutant volume to wild-type volume (steric component) combined with the charge difference (electrostatic component)
3. WHEN the volume ratio exceeds 1.2, THE Resistance_Profiler SHALL apply edge expansion (factor > 1.0) to simulate steric clash
4. WHEN the charge difference is non-zero, THE Resistance_Profiler SHALL apply additional edge expansion to edges connecting to oppositely-charged neighbors (simulating electrostatic repulsion from charge reversal)

### Requirement 7: CLI Interface

**User Story:** As a researcher, I want a command-line interface for the Resistance Profiler, so that I can run scans from the terminal with the same ergonomics as the pipeline runner.

#### Acceptance Criteria

1. THE Resistance_Profiler CLI SHALL accept --structure (required), --mutations (required, comma-separated list of CHAIN:RESIDUE:MUTANT_AA), and --checkpoint (optional) arguments
2. WHEN the CLI completes successfully, THE Resistance_Profiler SHALL print the JSON batch report to stdout
3. WHEN the CLI encounters a fatal error, THE Resistance_Profiler SHALL print a JSON error object and exit with non-zero status
4. THE Resistance_Profiler CLI SHALL accept --hub-residues (optional, comma-separated) to specify custom propagation hubs instead of auto-detecting from source leaks
