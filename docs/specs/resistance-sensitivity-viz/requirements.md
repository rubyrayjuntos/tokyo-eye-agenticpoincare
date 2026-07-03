# Requirements Document

## Introduction

The DTIE pipeline Phase 4 (Resistance Mapping) produces per-residue perturbation sensitivity data that quantifies how much each residue's local graph topology changes under simulated mutations. This data is persisted in the database but not yet surfaced in the dashboard. This feature adds a "Resistance" color mode to the 3D Molecular Viewer that colors residues by their perturbation factor, enriches the hover tooltip with resistance classification data, and extends the hydration endpoint to deliver Phase 4 results to the frontend.

## Glossary

- **Perturbation_Factor**: A per-residue scalar from Phase 4 measuring how sensitive the local contact graph is to edge perturbation at that position (higher = more disruptive mutation site)
- **Resistance_Classification**: A categorical label assigned by the resistance classifier (e.g., "high_sensitivity", "moderate", "stable") based on perturbation thresholds
- **Hydration_Endpoint**: The `/api/hydrate/{structure_id}` endpoint that delivers all per-structure data to the frontend in a single request
- **Phase4_Result**: The output of `run_phase4_resistance` containing per-residue perturbation factors, affected edge counts, and resistance classifications

## Requirements

### Requirement 1: Hydration Endpoint Extension

**User Story:** As a frontend developer, I want the hydration endpoint to include Phase 4 resistance data per residue, so that the viewer can color by resistance sensitivity without additional API calls.

#### Acceptance Criteria

1. WHEN the hydration endpoint is called for a structure with Phase 4 results, THE System SHALL include a `resistance_data` field in the response containing per-residue perturbation factors
2. WHEN Phase 4 results do not exist for a structure, THE System SHALL return `resistance_data` as null without error
3. THE `resistance_data` field SHALL contain an array of objects with `residue_id`, `perturbation_factor`, `affected_edges`, and `classification` fields

### Requirement 2: Resistance Color Mode

**User Story:** As a scientist, I want to color the 3D protein structure by resistance sensitivity, so that I can immediately see which residues are most vulnerable to resistance mutations.

#### Acceptance Criteria

1. WHEN the user selects "Resistance" from the color mode dropdown, THE Molecular_Viewer SHALL color each residue by its perturbation factor using a sequential warm colormap (white → yellow → orange → deep red)
2. WHEN resistance data is not available, THE Molecular_Viewer SHALL display the mode as disabled or fall back to spectrum coloring with a notification
3. WHEN in Resistance mode, THE Molecular_Viewer SHALL add glow spheres to residues classified as "high_sensitivity"

### Requirement 3: Tooltip Enrichment with Phase 4 Data

**User Story:** As a scientist, I want the hover tooltip to show resistance-relevant data when available, so that I can assess mutation risk without switching views.

#### Acceptance Criteria

1. WHEN hovering a residue that has Phase 4 data, THE Tooltip SHALL display the perturbation factor value and its classification label
2. WHEN the residue is classified as high sensitivity, THE Tooltip SHALL display a warning indicator and a brief interpretation
3. WHEN Phase 4 data is not available for the hovered residue, THE Tooltip SHALL omit the resistance section without error

### Requirement 4: Frontend Type Definitions

**User Story:** As a frontend developer, I want TypeScript types for resistance data, so that the integration is type-safe.

#### Acceptance Criteria

1. THE Frontend SHALL define a `ResistanceData` interface matching the hydration endpoint response shape
2. THE HydrationProvider SHALL expose resistance data through its context value
3. THE MolecularViewer SHALL consume resistance data from the hydration context
