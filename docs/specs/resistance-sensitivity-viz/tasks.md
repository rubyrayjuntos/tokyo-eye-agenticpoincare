# Implementation Plan: Resistance Sensitivity Visualization

## Overview

Adds a "Resistance" color mode to the 3D Molecular Viewer by extending the hydration endpoint with Phase 4 per-residue sensitivity scores, adding TypeScript types, and wiring the new color mode + tooltip enrichment into the existing MolecularViewer component.

## Tasks

- [x] 1. Backend: Compute per-residue resistance sensitivity score
  - [x] 1.1 Add `_compute_resistance_sensitivity()` function to dashboard router
    - Query `fact_resistance_pathway` for the structure
    - Query `fact_resistance_spectral` for hinge_residues
    - Aggregate coupling_strength per residue, apply hinge bonus (1.5×)
    - Normalize scores to [0, 1], classify as high_sensitivity/moderate/stable
    - _Requirements: 1.1, 1.3_

  - [x] 1.2 Write property test for score monotonicity
    - **Property 2: Resistance score monotonicity with coupling count**
    - **Validates: Requirements 2.1, 3.1**

- [x] 2. Backend: Extend hydration endpoint response
  - [x] 2.1 Add `resistance_data` field to hydration response
    - Call `_compute_resistance_sensitivity()` when Phase 4 data exists
    - Return null when no Phase 4 data available
    - Include both per-residue scores and spectral summary
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 2.2 Write property test for resistance data completeness
    - **Property 1: Resistance data completeness and shape**
    - **Validates: Requirements 1.1, 1.3**

- [x] 3. Frontend: Add TypeScript types and HydrationProvider extension
  - [x] 3.1 Add `ResistanceResidue` and `ResistanceData` interfaces to `types.ts`
    - Add `resistance_data: ResistanceData | null` to `HydrationResponse`
    - _Requirements: 4.1_

  - [x] 3.2 Expose `resistanceData` through HydrationProvider context
    - Add accessor: `resistanceData: hydration?.resistance_data ?? null`
    - _Requirements: 4.2_

- [x] 4. Frontend: Add "Resistance" color mode to MolecularViewer
  - [x] 4.1 Add "resistance" to StructureColorMode type and dropdown
    - Add `resistanceToHex()` function (white → yellow → orange → deep red)
    - Implement coloring logic consuming `resistanceData` from hydration
    - Add glow spheres for high_sensitivity residues
    - Show threshold slider in resistance mode
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 4.2 Enrich tooltip with resistance data
    - Show sensitivity_score + classification when resistance data available
    - Show "⚠ High Sensitivity" warning for high_sensitivity residues
    - Show hinge indicator when `is_hinge` is true
    - Gracefully omit section when no resistance data
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks are all required (comprehensive testing from start)
- Phase 4 data comes from `fact_resistance_pathway` and `fact_resistance_spectral` tables
- Per-residue score = sum(coupling_strength) for all pathways involving the residue, with 1.5× hinge bonus
- The hydration endpoint already queries Phase 4 data — we extend it with the per-residue aggregation
- Frontend follows the same pattern as plasticity/allosteric modes (colorMode switch + threshold slider)
