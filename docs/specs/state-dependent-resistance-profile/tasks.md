# Implementation Plan: State-Dependent Resistance Profile (SDRP)

## Overview

Implement the SDRP module as a new set of files under `science/dtie/v5/resistance/` that wraps the existing `ResistanceProfiler`. Python with Hypothesis for property-based testing. The module computes multi-structure ensemble profiles with SSS scoring, stability metrics, mechanism shift detection, and categorization.

## Tasks

- [x] 1. Create SDRP data models
  - [x] 1.1 Create `science/dtie/v5/resistance/sdrp_models.py`
    - Define BindingContext, StructureEntry, KinaseStateSet dataclasses
    - Define StateProfileEntry with raw metrics (mechanism_class, stability_score, confidence_score, site_uncertainty_delta, max_hub_delta, propagation_radius)
    - Define MechanismShift with structural_basis extension point
    - Define EnsembleProfile with state_profile dict, sss_score, category, clinical_relevance, mechanism_shifts, error_structures
    - Define JSON serialization helpers (ensemble_profile_to_dict) with numpy→native conversion
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2. Implement SSS computation
  - [x] 2.1 Create `science/dtie/v5/resistance/sdrp_sss.py`
    - Define CLASS_SPACE = ["Type_I_Steric", "Type_II_Allosteric", "Hybrid", "Neutral"]
    - Define SSS_NORMALIZATION_FACTOR = log₂(4) = 2.0
    - Implement `mechanism_to_probability_vector(mechanism_class, confidence_score)` — confidence as weight on classified category, remainder uniform
    - Implement `compute_sss(state_entries)` — JSD of probability vectors, normalized by 2.0, skip None entries
    - Use scipy.spatial.distance.jensenshannon or manual implementation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 2.2 Write property test for SSS equals JSD of probability vectors
    - **Property 4: SSS equals JSD of probability vectors**
    - **Validates: Requirements 3.1, 3.2**

  - [x] 2.3 Write property test for SSS bounds
    - **Property 5: SSS bounds**
    - **Validates: Requirements 3.5**

  - [x] 2.4 Write property test for SSS monotonicity with divergence
    - **Property 6: SSS monotonicity with divergence**
    - **Validates: Requirements 3.3, 3.4**

- [x] 3. Implement stability scorer
  - [x] 3.1 Create `science/dtie/v5/resistance/sdrp_stability.py`
    - Implement `compute_stability_score(site_delta, max_hub_delta, propagation_radius, mechanism_class, config)` — margin to nearest decision boundary
    - Type_I_Steric: min(site above threshold, hub below threshold)
    - Type_II_Allosteric: min(hub above threshold, radius above threshold)
    - Hybrid/Neutral: distance to nearest clear-class boundary
    - Clamp to [0.0, 1.0]
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 3.2 Write property test for stability score bounds and monotonicity
    - **Property 7: Stability score bounds and monotonicity**
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [x] 4. Implement categorizer and shift detector
  - [x] 4.1 Create `science/dtie/v5/resistance/sdrp_categorizer.py`
    - Define SSS_SWITCH_THRESHOLD = 0.5, SSS_STATIC_THRESHOLD = 0.3
    - Implement `categorize(sss_score)` → "Conformational_Switch" | "Static_Disruptor" | "Intermediate"
    - Implement `generate_clinical_relevance(category, variant, state_profile, mechanism_shifts)` → non-empty summary string
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 4.2 Create `science/dtie/v5/resistance/sdrp_shifts.py`
    - Implement `detect_mechanism_shifts(state_profile)` → list of MechanismShift for all unique pairs with different mechanism_class
    - Annotate each shift with delta_site_uncertainty and delta_max_hub
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 4.3 Write property test for categorization threshold correctness
    - **Property 8: Categorization threshold correctness**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

  - [x] 4.4 Write property test for mechanism shift detection completeness
    - **Property 9: Mechanism shift detection completeness**
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement SDRP Engine orchestrator
  - [x] 6.1 Create `science/dtie/v5/resistance/sdrp_engine.py`
    - Implement `SDRPEngine.__init__` wrapping existing ResistanceProfiler
    - Implement `SDRPEngine.profile_variant(variant, structures, hub_residues)`:
      - Run ResistanceProfiler.profile_mutation per structure independently
      - Collect per-structure ResistanceReports with error isolation
      - Compute StateProfileEntry per successful structure (including stability_score)
      - Compute SSS from state entries
      - Detect mechanism shifts
      - Categorize and generate clinical_relevance
      - Assemble EnsembleProfile
    - Handle KinaseStateSet (fill None for missing canonical states)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

  - [x] 6.2 Implement batch profiling in SDRPEngine
    - Implement `SDRPEngine.profile_batch(variants, structures, hub_residues, on_progress)`
    - Compute each structure's WT baseline once (shared across variants via ResistanceProfiler cache)
    - Return EnsembleProfiles in same order as input variants
    - Progress callback with (completed, total)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 6.3 Write property test for ensemble output count matches input structures
    - **Property 1: Ensemble output count matches input structures**
    - **Validates: Requirements 1.1, 1.3, 1.4**

  - [x] 6.4 Write property test for Ensemble_Profile completeness
    - **Property 2: Ensemble_Profile completeness**
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x] 6.5 Write property test for JSON serializability round-trip
    - **Property 3: JSON serializability round-trip**
    - **Validates: Requirements 2.4**

  - [x] 6.6 Write property test for batch output order preservation
    - **Property 10: Batch output order preservation**
    - **Validates: Requirements 7.1, 7.4**

- [x] 7. Implement persistence layer
  - [x] 7.1 Create database migration `data/aurora/migrations/037_ensemble_resistance_profile.sql`
    - Create fact_ensemble_resistance_profile table with UNIQUE(variant, structure_ids) for upsert
    - Add indexes on variant, category, sss_score
    - _Requirements: 8.1, 8.3_

  - [x] 7.2 Implement persistence in SDRPEngine
    - Implement `SDRPEngine._persist_ensemble_profile(profile)` with upsert semantics
    - Record provenance linking ensemble to all per-structure run_ids
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 7.3 Write property test for persistence upsert idempotence
    - **Property 11: Persistence upsert idempotence**
    - **Validates: Requirements 8.3**

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All property tests are required (comprehensive testing from start)
- The SDRP Engine wraps the existing ResistanceProfiler — no changes to single-structure classification logic
- Property tests use Hypothesis with synthetic StateProfileEntry lists (no real inference needed for SSS/stability/categorization tests)
- Integration tests require real DB + checkpoint + multiple PDB structures (1IEP, 2HYY, 3CS9)
- SSS normalization uses fixed denominator log₂(4) = 2.0 for cross-study comparability
- Probability vectors use Option A semantics: confidence as weight on winner, remainder uniform across other classes
- None entries from sparse KinaseStateSet are skipped in SSS computation
- The existing `tests/benchmarks/ensemble_runner.py` (consensus-based) is superseded by this module but not deleted — it serves as a migration reference
- Stability score uses margin-to-nearest-boundary semantics (how much noise before the call flips)
- MechanismShift is observational (WHAT changed), with structural_basis as future extension for causal analysis

