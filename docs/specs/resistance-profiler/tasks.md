# Implementation Plan: Resistance Profiler

## Overview

Implement the Resistance Profiler as a new module at `science/dtie/v5/resistance/` with four layers: data models, mutation operator, mechanism classifier, and orchestrator. Python with Hypothesis for property-based testing. Outputs are persisted through the Normalizer with `run_type="in_silico_mutation"` provenance.

## Tasks

- [x] 1. Create data models and amino acid property table
  - [x] 1.1 Create `science/dtie/v5/resistance/__init__.py` and `models.py`
    - Define MutationSpec, HubMetrics, ResistanceReport, BatchReport dataclasses
    - Define AMINO_ACID_PROPERTIES lookup table (20 standard amino acids with volume, hydropathy, charge)
    - Define MutantStructureId convention: `{structure_id}_mut_{chain}{residue}{mutant_aa}` (e.g., `1iep_mut_A315I`)
    - Define configurable classification thresholds with version tag
    - _Requirements: 5.1, 6.1_

  - [x] 1.2 Write property test for perturbation factor formula
    - **Property 2: Perturbation factor formula correctness**
    - **Validates: Requirements 6.2, 6.3, 6.4**

- [x] 2. Implement Mutation Operator
  - [x] 2.1 Create `science/dtie/v5/resistance/operator.py`
    - Implement `MutationOperator.compute_perturbation_factor(wt_aa, mut_aa)`
    - Implement `MutationOperator.apply_perturbation(pyg_data, node_idx, mutation)`
    - Implement `MutationOperator.build_mutant_graph(structure_id, mutation)`
    - Validate residue exists in graph, raise ValueError if not
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 6.2, 6.3, 6.4_

  - [x] 2.2 Write property test for graph connectivity invariant
    - **Property 1: Graph connectivity invariant**
    - **Validates: Requirements 1.4**

  - [x] 2.3 Write property test for edge perturbation application
    - **Property 3: Edge perturbation application**
    - **Validates: Requirements 1.2**

  - [x] 2.4 Write property test for invalid residue error handling
    - **Property 9: Invalid residue error handling**
    - **Validates: Requirements 1.5**

- [x] 3. Implement Orchestrator with Baseline Caching
  - [x] 3.1 Create `science/dtie/v5/resistance/profiler.py`
    - Implement `ResistanceProfiler.__init__` with runner, builder, operator, classifier
    - Implement `ResistanceProfiler._ensure_baseline(structure_id)` with caching
    - Baseline runs full pipeline (reuses Phase 4 spectral + source leak detection for hub auto-detection)
    - Implement `ResistanceProfiler._auto_detect_hubs(wt_result, structure_id)` using source leaks + Phase 4 pathway targets
    - Implement `ResistanceProfiler.profile_mutation(structure_id, mutation, hub_residues)`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1_

  - [x] 3.2 Implement provenance for in-silico mutations
    - WT baseline run recorded as `run_type="resistance_baseline"` in provenance_run
    - Mutant runs recorded as `run_type="in_silico_mutation"` with parent_run_id pointing to baseline
    - Mutant structure_id uses convention `{base}_mut_{chain}{res}{aa}` (not written to dim_structure — virtual only)
    - ResistanceReport persisted to a new `fact_resistance_profile` table via Normalizer
    - _Requirements: 2.1, 5.1, 5.3_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Mechanism Classifier
  - [x] 5.1 Create `science/dtie/v5/resistance/classifier.py`
    - Implement `MechanismClassifier.classify(wt_nodes, mut_nodes, mutation, hub_residues)`
    - Implement classification decision tree (Type I / Type II / Hybrid)
    - Implement `MechanismClassifier.compute_confidence(site_delta, max_hub_delta, propagation_radius)`
    - Thresholds configurable via ClassifierConfig with version string for reproducibility
    - Relationship to Phase 4: classifier consumes Phase 4 pathway data from baseline to identify coupling targets; it does NOT re-run spectral analysis on the mutant (the perturbation is at the GNN level, not the graph Laplacian level)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.2 Write property test for classification decision tree
    - **Property 4: Classification decision tree correctness**
    - **Validates: Requirements 3.2, 3.3, 3.4**

  - [x] 5.3 Write property test for confidence score bounds
    - **Property 5: Confidence score bounds**
    - **Validates: Requirements 3.5**

- [x] 6. Implement Batch Scanning and Output
  - [x] 6.1 Implement batch scanning in profiler.py
    - Implement `ResistanceProfiler.profile_batch(structure_id, mutations, hub_residues, on_progress)`
    - Single WT baseline, independent mutation loops
    - Error isolation per mutation (failed mutations get error in report, batch continues)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 6.2 Implement structured JSON output
    - BatchReport serialization with numpy→native conversion
    - Include WT baseline metrics (source leaks, spectral gap, top pathways) in output
    - Per-residue ResistanceContribution output (Δeps per residue for downstream dashboard use)
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.3 Write property test for batch output order preservation
    - **Property 6: Batch output order preservation**
    - **Validates: Requirements 4.2**

  - [x] 6.4 Write property test for report completeness and serializability
    - **Property 7: Report completeness and serializability**
    - **Validates: Requirements 5.1, 5.4**

  - [x] 6.5 Write property test for batch summary count invariant
    - **Property 8: Batch summary count invariant**
    - **Validates: Requirements 5.2**

- [x] 7. Implement CLI interface
  - [x] 7.1 Create `scripts/resistance_profiler.py` with argparse CLI
    - Arguments: --structure (required), --mutations (required), --checkpoint, --hub-residues, --thresholds-version
    - JSON output on success, JSON error on failure with non-zero exit
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 7.2 Write property test for CLI argument parsing
    - **Property 10: CLI argument parsing**
    - **Validates: Requirements 7.1**

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The Resistance Profiler composes existing V5GNNRunner and GraphBuilder — no reimplementation
- Property tests use Hypothesis with synthetic GNNNodeOutput lists (no real inference for classifier tests)
- Integration tests require real DB + checkpoint (1IEP structure)
- Classification thresholds are empirically derived from T315I/E255K experiments and versioned for reproducibility
- Relationship to Phase 4: The profiler CONSUMES Phase 4 outputs (pathway targets, spectral gap) from the WT baseline run to identify hubs. It does NOT re-run Phase 4 on mutant graphs — the mutation effect is measured at the GNN embedding level (Δeps, Δdepth), not at the graph Laplacian level.
- Provenance: In-silico mutations are recorded with `run_type="in_silico_mutation"` and linked to their baseline via parent_run_id. Mutant "structures" are virtual (not in dim_structure) — they exist only as perturbed graphs in the inference pipeline.
- Dashboard integration: Per-residue ResistanceContribution outputs enable future visualization of WT→mutant differential maps in the frontend.
