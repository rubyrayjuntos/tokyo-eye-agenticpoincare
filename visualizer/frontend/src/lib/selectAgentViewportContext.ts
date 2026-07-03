/**
 * selectAgentViewportContext — pure selector for the agent chat viewport_state payload.
 *
 * Single source of truth: derives ViewportState from machine context + hydration +
 * structure scope. Used by ChatRail and any shell that talks to the agent.
 */

import type { ViewportContext } from "./viewportMachine";
import type { StructureScopeContext } from "./plannerPolicySelector";
import type {
  ActivePanelName,
  CompareState,
  HydrationResponse,
  PersistenceStatus,
  Structure,
  ViewportState,
} from "./types";
import type {
  Annotation,
  DrugCandidateData,
  EmbeddingData,
  Hypothesis,
  PharmacophoreData,
  ProvenanceRun,
  ResistanceData,
  SourceLeakData,
} from "./types";

export interface AgentViewportHydrationSlice {
  hydration: HydrationResponse | null;
  embeddings: EmbeddingData | null;
  sourceLeaks: SourceLeakData | null;
  resistanceData: ResistanceData | null;
  hypotheses: Hypothesis[] | null;
  provenanceRuns: ProvenanceRun[] | null;
  annotations: Annotation[] | null;
  pharmacophorePockets: PharmacophoreData | null;
  drugCandidates: DrugCandidateData | null;
  persistenceStatus: PersistenceStatus | null;
}

export interface AgentViewportContextInput {
  activeStructure: Structure | null;
  viewport: ViewportContext;
  hydrationSlice: AgentViewportHydrationSlice;
  structureScope: StructureScopeContext | null;
  compareState?: CompareState | null;
}

function buildDataInspectorContext(slice: AgentViewportHydrationSlice) {
  const {
    embeddings,
    sourceLeaks,
    resistanceData,
    pharmacophorePockets,
    drugCandidates,
  } = slice;

  const phasesComputed: string[] = [];
  const phaseCounts: Record<string, number> = {};

  if (embeddings?.residues?.length) {
    phasesComputed.push("embeddings");
    phaseCounts.embeddings = embeddings.residues.length;
  }
  if (sourceLeaks?.leaks?.length) {
    phasesComputed.push("source_leaks");
    phaseCounts.source_leaks = sourceLeaks.leaks.length;
  }
  if (resistanceData?.residues?.length) {
    phasesComputed.push("resistance");
    phaseCounts.resistance = resistanceData.residues.length;
  }
  if (pharmacophorePockets?.pockets?.length) {
    phasesComputed.push("phase5");
    phaseCounts.phase5 = pharmacophorePockets.pockets.length;
  }
  if (drugCandidates?.candidates?.length) {
    phasesComputed.push("phase6");
    phaseCounts.phase6 = drugCandidates.candidates.length;
  }

  const topDruggabilityPocket = pharmacophorePockets?.pockets?.length
    ? Math.max(...pharmacophorePockets.pockets.map((p) => p.druggability_score))
    : null;

  const topDrugCandidateScore = drugCandidates?.candidates?.length
    ? Math.max(...drugCandidates.candidates.map((c) => c.combined_druggability))
    : null;

  let admetPassRate: number | null = null;
  if (drugCandidates?.candidates?.length) {
    const passed = drugCandidates.candidates.filter((c) => c.admet_pass).length;
    admetPassRate = passed / drugCandidates.candidates.length;
  }

  return {
    phases_computed: phasesComputed,
    phase_counts: phaseCounts,
    top_druggability_pocket: topDruggabilityPocket,
    top_drug_candidate_score: topDrugCandidateScore,
    admet_pass_rate: admetPassRate,
  };
}

function derivePipelineStatus(
  activeStructure: Structure | null,
  structureScope: StructureScopeContext | null,
): ViewportState["pipeline"] {
  if (!activeStructure) {
    return { status: "never_run", current_step: null, progress: null };
  }
  if (structureScope) {
    if (structureScope.pipelineReady) {
      return { status: "complete", current_step: null, progress: null };
    }
    if (structureScope.inferenceReady || structureScope.ingestionReady) {
      return { status: "running", current_step: "onboarding", progress: null };
    }
  }
  if (activeStructure.last_run_id || activeStructure.has_embeddings) {
    return { status: "complete", current_step: null, progress: null };
  }
  return { status: "never_run", current_step: null, progress: null };
}

function buildDataSummary(
  slice: AgentViewportHydrationSlice,
): ViewportState["data_summary"] {
  const {
    hydration,
    embeddings,
    sourceLeaks,
    resistanceData,
    hypotheses,
    provenanceRuns,
    annotations,
    persistenceStatus,
  } = slice;

  if (!hydration) {
    return null;
  }

  const topUncertaintyResidues = embeddings?.residues
    ? [...embeddings.residues]
        .sort((a, b) => b.epistemic_uncertainty - a.epistemic_uncertainty)
        .slice(0, 5)
        .map((r) => ({
          residue_id: r.residue_id,
          epistemic_uncertainty: r.epistemic_uncertainty,
        }))
    : [];

  let hypothesisStatusDistribution: Record<string, number> | null = null;
  if (hypotheses && hypotheses.length > 0) {
    hypothesisStatusDistribution = {};
    for (const h of hypotheses) {
      hypothesisStatusDistribution[h.status] =
        (hypothesisStatusDistribution[h.status] || 0) + 1;
    }
  }

  let resistanceSummary: NonNullable<ViewportState["data_summary"]>["resistance_summary"] =
    null;
  if (resistanceData) {
    resistanceSummary = {
      lambda_2: resistanceData.spectral.lambda_2,
      hinge_count: resistanceData.spectral.hinge_count,
      high_sensitivity_count: resistanceData.residues.filter(
        (r) => r.classification === "high_sensitivity",
      ).length,
      moderate_count: resistanceData.residues.filter(
        (r) => r.classification === "moderate",
      ).length,
      stable_count: resistanceData.residues.filter(
        (r) => r.classification === "stable",
      ).length,
    };
  }

  const ps = persistenceStatus;
  const persistenceStatusMap: Record<string, boolean> = ps
    ? {
        embeddings: Boolean(ps.embeddings_persisted),
        graph: Boolean(ps.graph_persisted),
        sites: Boolean(ps.sites_persisted),
        binding_scan: Boolean(ps.binding_scan_persisted),
        phase2: Boolean(ps.phase2_persisted),
        phase4: Boolean(ps.phase4_persisted),
        phase5: Boolean(ps.phase5_persisted),
        phase6: Boolean(ps.phase6_persisted),
        resistance: resistanceData != null,
      }
    : {};

  return {
    residue_count: embeddings?.residues?.length ?? 0,
    source_leak_count: sourceLeaks?.leaks?.length ?? 0,
    hypothesis_count: hypotheses?.length ?? 0,
    hypothesis_status_distribution: hypothesisStatusDistribution,
    provenance_run_count: provenanceRuns?.length ?? 0,
    annotation_count: annotations?.length ?? 0,
    top_uncertainty_residues: topUncertaintyResidues,
    persistence_status: persistenceStatusMap,
    resistance_summary: resistanceSummary,
  };
}

export function selectAgentViewportContext(
  input: AgentViewportContextInput,
): ViewportState {
  const {
    activeStructure,
    viewport,
    hydrationSlice,
    structureScope,
    compareState,
  } = input;

  const activePanel = viewport.activePanel as ActivePanelName;

  return {
    structure_id: activeStructure?.structure_id ?? null,
    structure_title: activeStructure?.title ?? null,
    is_radar_active: viewport.isRadarActive,
    selected_pocket_id: viewport.selectedPocketId,

    poincare: {
      color_mode: viewport.poincareColorMode,
      mobius_focus_enabled: viewport.mobiusFocusEnabled,
      mobius_focus_residue:
        viewport.mobiusFocusEnabled && viewport.selectedResidue
          ? viewport.selectedResidue.residue_id
          : null,
      selected_residue: viewport.selectedResidue,
      brush_selected_ids: viewport.brushSelectedIds,
    },

    viewer_3d: {
      color_mode: viewport.viewerColorMode,
      risk_threshold: viewport.riskThreshold,
      highlighted_residue_ids: viewport.highlightedResidues,
    },

    data_summary: buildDataSummary(hydrationSlice),

    active_panel: activePanel,

    pipeline: derivePipelineStatus(activeStructure, structureScope),

    compare:
      compareState?.active && compareState.secondaryStructure
        ? {
            secondary_structure_id: compareState.secondaryStructure.structure_id,
            top_movers: compareState.displacements
              ? compareState.displacements.displacements
                  .slice(0, 10)
                  .map((d) => ({
                    residue_id: d.residue_id,
                    displacement: d.displacement,
                  }))
              : [],
            edge_diff: compareState.graphDiff
              ? {
                  gained: compareState.graphDiff.edge_diff.gained_count,
                  lost: compareState.graphDiff.edge_diff.lost_count,
                  changed: compareState.graphDiff.edge_diff.changed_count,
                }
              : { gained: 0, lost: 0, changed: 0 },
          }
        : null,

    data_inspector:
      activePanel === "data_inspector"
        ? buildDataInspectorContext(hydrationSlice)
        : null,
  };
}