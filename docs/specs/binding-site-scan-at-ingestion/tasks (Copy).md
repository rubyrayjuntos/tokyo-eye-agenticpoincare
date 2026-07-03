# Implementation Plan: Binding Site Scan at Ingestion

## Overview

Build the full-structure binding site scan system bottom-up: seed generator → pocket detector adapter → site merger → scan phase orchestrator → pipeline integration → query interface → real MD calibration → on-demand MD validation.

## Tasks

- [x] 1. Implement Seed Generator with DBSCAN clustering
  - [x] 1.1 Create `agent/tools/cryptic/seed_generator.py`
    - Implement `filter_qualifying_residues(nodes, uncertainty_threshold, cone_depth_threshold)` → list of qualifying residues
    - Implement `cluster_residues_dbscan(qualifying, ca_coords, eps_angstrom, min_cluster_size)` → list of clusters
    - Implement `generate_seeds_from_gnn(...)` composing filter + cluster + score + cap at max_clusters
    - Use scikit-learn DBSCAN for spatial clustering
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 Write property tests for seed generator (Properties 1, 2, 3)
    - **Property 1: Qualifying Residue Filter Correctness**
    - **Property 2: Minimum Cluster Size Enforcement**
    - **Property 3: Output Cap and Score Ordering**
    - **Validates: Requirements 1.1, 1.3, 1.4, 1.5**

- [x] 2. Implement Surface Pocket Detector Adapter
  - [x] 2.1 Create `agent/tools/cryptic/pocket_detector.py`
    - Implement `detect_surface_pockets(structure_id, db)` that dispatches fpocket via science container
    - Parse fpocket results into `GeometryPocket` dataclass (centroid, volume, druggability, residue_ids)
    - Handle fpocket unavailability gracefully (return empty list with warning)
    - _Requirements: 2.1, 2.4_

- [x] 3. Implement Site Merger
  - [x] 3.1 Create `agent/tools/cryptic/site_merger.py`
    - Implement `compute_druggability_score(composite_gnn_score, fpocket_druggability, volume, site_type)` → float [0, 1]
    - Implement `merge_candidates(gnn_candidates, geometry_pockets, overlap_threshold)` → list[UnifiedCandidate]
    - Merging logic: spatial overlap detection (centroid distance < threshold), discovery_method assignment
    - Implement `assign_ranks(candidates)` → candidates with site_rank 1..N by druggability desc
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3_

  - [ ]* 3.2 Write property tests for site merger (Properties 4, 5, 6, 7)
    - **Property 4: Merge Spatial Overlap Detection**
    - **Property 5: Discovery Method Annotation Completeness**
    - **Property 6: Druggability Score Bounds**
    - **Property 7: Site Rank Ordering Invariant**
    - **Validates: Requirements 2.3, 2.4, 2.5, 3.2, 3.3, 5.1**

- [x] 4. Implement Scan Phase Orchestrator
  - [x] 4.1 Create `agent/tools/cryptic/scan_phase.py`
    - Implement `run_full_structure_scan(structure_id, db, ...)` that composes seed_generator + pocket_detector + site_merger
    - Fetch GNN node outputs from fact_gnn_node_embedding
    - Fetch Cα coordinates from dim_atom
    - Run mapper classification on each cluster (re-use existing mapper logic)
    - Persist all candidates to fact_cryptic_site via normalizer
    - Persist scan metadata to fact_binding_site_scan
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 4.2 Write property test for scan provenance (Property 8)
    - **Property 8: Scan Provenance Completeness**
    - **Validates: Requirements 4.4**

- [x] 5. Checkpoint - Verify scan phase components
  - Run property tests P1-P8 and confirm they pass
  - Verify seed generator produces correct clusters from synthetic data
  - Verify merger correctly handles overlap/non-overlap cases
  - Ask the user if questions arise

- [x] 6. Database schema extensions
  - [x] 6.1 Create migration `data/aurora/migrations/044_binding_site_scan.sql`
    - Add druggability_score, site_rank, discovery_method, scan_run_id, volume_angstrom3, fpocket_druggability columns to fact_cryptic_site
    - Create fact_binding_site_scan table for scan metadata
    - Add index on (structure_id, site_rank) for fast ranked queries
    - _Requirements: 3.4, 4.4_

- [x] 7. Implement Query Interface
  - [x] 7.1 Add `query_binding_sites()` to `agent/tools/cryptic/tool.py`
    - SELECT from fact_cryptic_site WHERE structure_id, with optional filters
    - Support filtering by site_type, min_druggability_score, md_validation_status
    - Return ToolResult with ranked site list or "pipeline not run" message
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 7.2 Write property tests for query interface (Properties 9, 10)
    - **Property 9: Query Result Field Completeness**
    - **Property 10: Query Filter Correctness**
    - **Validates: Requirements 5.3, 5.5**

- [x] 8. Integrate Scan Phase into Pipeline
  - [x] 8.1 Wire scan phase into the pipeline orchestrator
    - Add scan phase call after GNN + graph topology in `_run_pipeline_background()` or equivalent
    - Ensure scan is triggered by `run_full_pipeline_via_container` completion
    - Handle re-scan: delete old results for structure before inserting new
    - _Requirements: 4.1, 4.5_

- [x] 9. Checkpoint - End-to-end scan verification
  - Verify full pipeline produces ranked binding sites for a test structure
  - Verify query_binding_sites returns pre-computed results instantly
  - Ask the user if questions arise

- [x] 10. Implement Real SMD Runner (replace stub)
  - [x] 10.1 Create `science/dtie/cryptic/smd_runner_real.py`
    - Implement real OpenMM steered MD using the GPU science container
    - Accept --spec-json and --protocol arguments (same interface as stub)
    - Output structured JSON with actual work_kcal_mol, strain_delta, duration_ms
    - Support all 5 protocols: SMD_three_phase, SMD_stent_stabilization, SMD_lid_restraint, SMD_clamp_stabilization, SMD_strain_relief
    - _Requirements: 6.1_

  - [x] 10.2 Update science_dispatch to use real SMD runner
    - Switch smd_runner module path from stub to real implementation
    - Ensure timeout handling works for GPU jobs (default 3600s per site)
    - _Requirements: 6.1_

- [x] 11. Implement Real MD Calibration Pipeline
  - [x] 11.1 Create `scripts/calibrate_cryptic_real_md.py`
    - Implement `run_real_md_calibration(benchmark_path, protocol, db)` dispatching real SMD per benchmark site
    - Implement `calibrate_all_protocols(benchmark_path, db)` running all protocols
    - Implement `apply_calibrated_thresholds(results, output_path)` writing calibrated values
    - Collect work values from completed MD jobs, apply Youden's J, output calibrated thresholds
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 11.2 Write property test for calibrated threshold structure (Property 11)
    - **Property 11: Calibrated Threshold Structure**
    - **Validates: Requirements 6.5**

  - [x] 11.3 Add `make calibrate-cryptic-real` target to Makefile
    - Run `PYTHONPATH=. python scripts/calibrate_cryptic_real_md.py --benchmark data/calibration/benchmark_cryptic_sites.json --output data/calibration/calibrated_thresholds.json`
    - _Requirements: 6.4 (via Makefile)_

- [x] 12. Run Heuristic v1 Calibration
  - [x] 12.1 Extend `scripts/calibrate_cryptic.py` to run heuristic evaluation against benchmark
    - Map each benchmark site through the mapper's _infer_site_type
    - Compare predicted site_type against ground_truth_type
    - Compute precision/recall/F1 per site_type
    - Output updated precision_by_site_type and flagged_types
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 13. Implement On-Demand MD Validation
  - [x] 13.1 Add `validate_site_md()` and `validate_top_sites_md()` to tool.py
    - Dispatch real SMD for a specific site_id
    - Update fact_cryptic_site md_validation_status: pending → running → passed/failed
    - Support batch validation of top N sites by rank
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 13.2 Write property test for MD status transitions (Property 12)
    - **Property 12: MD Validation Status Transitions**
    - **Validates: Requirements 8.2**

- [x] 14. Final checkpoint - Full system verification
  - Run all 12 property tests and confirm they pass
  - Run `make calibrate-cryptic-real` against benchmark dataset (at least 1 site per protocol)
  - Verify full pipeline produces ranked sites for a test structure
  - Verify query returns instant results
  - Verify on-demand MD updates status correctly
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The real SMD runner (task 10) requires OpenMM installed in the science container with GPU access
- Calibration runs (task 11) are expensive (hours of GPU time) — start with a small subset of benchmark sites
- The scan phase (task 4) re-uses the existing mapper classification logic from `agent/tools/cryptic/mapper.py`
- fpocket must be installed in the science container (it already is per v3 phase5 code)
- Property tests use Hypothesis library (100+ iterations each)
- Existing calibration spec properties (P3 Youden's J, P2 precision flagging, P10 confidence exclusion) continue to apply

