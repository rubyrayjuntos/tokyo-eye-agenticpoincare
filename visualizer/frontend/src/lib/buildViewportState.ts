/**
 * buildViewportState — thin adapter for tests and legacy callers.
 * Production paths should use selectAgentViewportContext directly.
 */

import type {
  ViewportState,
  SelectedResidueInfo,
  PoincareColorMode,
  StructureColorModeType,
  ActivePanelName,
} from "./types";
import type { ViewportContext } from "./viewportMachine";
import { selectAgentViewportContext } from "./selectAgentViewportContext";

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

  /** When true, emit a synthetic data_summary from count fields (property tests). */
  simulateHydration?: boolean;
}

function inputsToViewportContext(inputs: BuildViewportStateInputs): ViewportContext {
  return {
    highlightedResidues: inputs.highlightedResidueIds,
    brushSelectedIds: inputs.brushSelectedIds,
    selectedResidue: inputs.selectedResidue,
    userSelectedResidue: inputs.selectedResidue?.residue_id ?? null,
    mobiusFocusEnabled: inputs.mobiusFocusEnabled,
    currentDirective: null,
    activeMetric: "cone_depth",
    isRadarActive: false,
    poincareColorMode: inputs.poincareColorMode,
    viewerColorMode: inputs.viewerColorMode,
    riskThreshold: inputs.riskThreshold,
    selectedPocketId: null,
    activePanel: inputs.activePanel,
    sidebarOpen: true,
    activeEditorTab: "structure",
    bottomPanelOpen: true,
    activeBottomPanel: "summary",
    layoutModelJSON: null,
  };
}

export function buildViewportState(inputs: BuildViewportStateInputs): ViewportState {
  const viewport = inputsToViewportContext(inputs);
  const simulateHydration = inputs.simulateHydration ?? true;

  const state = selectAgentViewportContext({
    activeStructure: inputs.structureId
      ? {
          structure_id: inputs.structureId,
          pdb_id: inputs.structureId,
          title: inputs.structureTitle ?? inputs.structureId,
          has_embeddings: inputs.residueCount > 0,
          last_run_id: inputs.pipelineStatus === "complete" ? "test-run" : null,
        } as import("./types").Structure
      : null,
    viewport,
    hydrationSlice: simulateHydration
      ? {
          hydration: { structure_id: inputs.structureId ?? "" },
          embeddings:
            inputs.residueCount > 0
              ? {
                  structure_id: inputs.structureId ?? "test",
                  curvature: -1,
                  residues: Array.from({ length: inputs.residueCount }, (_, i) => ({
                    residue_id: `A:${i + 1}`,
                    x: 0,
                    y: 0,
                    z: 0,
                    epistemic_uncertainty: 0.1,
                    aleatoric_uncertainty: 0.1,
                    cone_depth: 0.5,
                    plasticity: 0,
                    hyperbolic_radius: 0.1,
                    hyperbolic_angle: 0,
                  })),
                }
              : null,
          sourceLeaks:
            inputs.sourceLeakCount > 0
              ? { leaks: Array.from({ length: inputs.sourceLeakCount }, () => ({} as never)) }
              : null,
          resistanceData: null,
          hypotheses: null,
          provenanceRuns: null,
          annotations: null,
          pharmacophorePockets: null,
          drugCandidates: null,
          persistenceStatus: { embeddings_persisted: true },
        }
      : {
          hydration: null,
          embeddings: null,
          sourceLeaks: null,
          resistanceData: null,
          hypotheses: null,
          provenanceRuns: null,
          annotations: null,
          pharmacophorePockets: null,
          drugCandidates: null,
          persistenceStatus: null,
        },
    structureScope: null,
    compareState: null,
  });

  if (!simulateHydration) {
    return state;
  }

  return {
    ...state,
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
    pipeline: {
      status: inputs.pipelineStatus,
      current_step: inputs.pipelineCurrentStep,
      progress: inputs.pipelineProgress,
    },
  };
}