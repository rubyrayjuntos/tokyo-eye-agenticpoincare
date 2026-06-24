/**
 * buildViewportState — assembles the viewport_state payload sent with each chat message.
 *
 * This is a pure function that takes the current cockpit state and produces the
 * ViewportState payload matching the backend context_builder's expected shape.
 *
 * Requirements: 4.1, 8.4
 */

import type {
  ViewportState,
  SelectedResidueInfo,
  PoincareColorMode,
  StructureColorModeType,
  ActivePanelName,
} from "./types";

export interface BuildViewportStateInputs {
  structureId: string | null;
  structureTitle: string | null;

  poincareColorMode: PoincareColorMode;
  mobiusFocusEnabled: boolean;
  mobiusFocusResidue: string | null;
  selectedResidue: SelectedResidueInfo | null;
  brushSelectedIds: string[];

  viewerColorMode: StructureColorModeType;
  riskThreshold: number;
  highlightedResidueIds: string[];

  residueCount: number;
  sourceLeakCount: number;
  hypothesisCount: number;
  provenanceRunCount: number;
  annotationCount: number;

  activePanel: ActivePanelName;

  pipelineStatus: string;
  pipelineCurrentStep: string | null;
  pipelineProgress: number | null;
}

/**
 * Builds the ViewportState payload from current cockpit state.
 * All required fields are always present (nullable fields may be null).
 */
export function buildViewportState(inputs: BuildViewportStateInputs): ViewportState {
  return {
    structure_id: inputs.structureId,
    structure_title: inputs.structureTitle,

    poincare: {
      color_mode: inputs.poincareColorMode,
      mobius_focus_enabled: inputs.mobiusFocusEnabled,
      mobius_focus_residue: inputs.mobiusFocusResidue,
      selected_residue: inputs.selectedResidue,
      brush_selected_ids: inputs.brushSelectedIds,
    },

    viewer_3d: {
      color_mode: inputs.viewerColorMode,
      risk_threshold: inputs.riskThreshold,
      highlighted_residue_ids: inputs.highlightedResidueIds,
    },

    data_summary: {
      residue_count: inputs.residueCount,
      source_leak_count: inputs.sourceLeakCount,
      hypothesis_count: inputs.hypothesisCount,
      hypothesis_status_distribution: null,
      provenance_run_count: inputs.provenanceRunCount,
      annotation_count: inputs.annotationCount,
      top_uncertainty_residues: [],
      persistence_status: {},
      resistance_summary: null,
    },

    active_panel: inputs.activePanel,

    pipeline: {
      status: inputs.pipelineStatus,
      current_step: inputs.pipelineCurrentStep,
      progress: inputs.pipelineProgress,
    },
  };
}
