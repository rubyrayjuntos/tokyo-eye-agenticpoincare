# Implementation Plan: Data Inspector

## Overview

Surfaces all computed DTIE pipeline data through a new Data Inspector panel with per-residue tables, pharmacophore pockets, drug candidates, new 3D viewer color modes, and comprehensive export. Built incrementally: merge logic first, then panel UI, then 3D viewer enhancements, then export.

## Tasks

- [x] 1. Frontend: Residue data merge utility
  - [x] 1.1 Implement mergeResidueData pure function
  - [x] 1.2 Write property test for residue merge completeness
  - [x] 1.3 Implement sort and filter utilities
  - [x] 1.4 Write property tests for sort and filter correctness

- [x] 2. Frontend: Data Inspector panel — Overview and Residues tabs
  - [x] 2.1 Create DataInspectorPanel component with tab structure
  - [x] 2.2 Implement Overview tab
  - [x] 2.3 Implement Residues tab

- [x] 3. Frontend: Pockets and Candidates tabs
  - [x] 3.1 Implement Pockets tab (Phase 5)
  - [x] 3.2 Implement Candidates tab (Phase 6)
  - [x] 3.3 Write property test for drug candidate counts accuracy

- [x] 4. Frontend: 3D Viewer pocket color modes
  - [x] 4.1 Add "pockets" color mode to MolecularViewer
  - [x] 4.2 Add "drug_candidates" color mode to MolecularViewer

- [x] 5. Backend: Enhanced export endpoint
  - [x] 5.1 Create comprehensive export endpoint
  - [x] 5.2 Write property test for export completeness

- [x] 6. Frontend: Agent context integration
  - [x] 6.1 Extend buildContext with data inspector metadata
  - [x] 6.2 Write property test for data inspector context payload

- [x] 7. Checkpoint — Verify integration

- [x] 8. Frontend: Phase counts accuracy test
  - [x] 8.1 Write property test for phase counts accuracy

- [x] 9. Final checkpoint

## Notes

- All tasks are required (comprehensive testing from start)
- The HydrationProvider already fetches Phase 5/6 data — it's just not rendered. No new API calls needed for the panel itself.
- The merge function is the critical piece — pure function, fully testable with Hypothesis/fast-check
- New color modes in MolecularViewer follow the exact pattern of existing modes (plasticity, allosteric, resistance)
- Export enhancement extends the existing export_structure_data tool to include all phases
- Drug therapy / docking data display is scoped to what Phase 5/6 already compute. If the pharmacophore pipeline itself needs changes, that's a separate spec.
