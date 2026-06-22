import { createContext, useContext, useEffect, useState, useCallback } from "react";
import type { ReactNode } from "react";
import { api } from "../lib/api";
import { useDashboard } from "../lib/context";
import type {
  HydrationResponse,
  EmbeddingData,
  GraphMetricsData,
  AllostericSitesData,
  SourceLeakData,
  ResistanceData,
  Hypothesis,
  ProvenanceRun,
  Annotation,
  PharmacophoreData,
  DrugCandidateData,
} from "../lib/types";

export interface LoadingState {
  embeddings: boolean;
  graph_metrics: boolean;
  allosteric_sites: boolean;
  source_leaks: boolean;
  hypotheses: boolean;
  provenance_runs: boolean;
  annotations: boolean;
}

export interface HydrationContextValue {
  /** Full hydration response (null before first load) */
  hydration: HydrationResponse | null;
  /** Per-panel loading indicators */
  loading: LoadingState;
  /** Whether the overall hydration fetch is in progress */
  isHydrating: boolean;
  /** Error from the hydration fetch, if any */
  error: string | null;
  /** Force a re-hydration of the current structure */
  refresh: () => void;
  /** Explicit multi-structure hydration support for Therapeutic Compiler (pathway samples etc.).
   * Keyed by structure_id (lowercase). Populated from therapeuticCompilerState + active.
   */
  multiHydrations: Record<string, HydrationResponse | null>;
  /** Refresh all (active + multi) */
  refreshAll: () => void;
  /** Convenience accessors */
  embeddings: EmbeddingData | null;
  graphMetrics: GraphMetricsData | null;
  allostericSites: AllostericSitesData | null;
  sourceLeaks: SourceLeakData | null;
  resistanceData: ResistanceData | null;
  hypotheses: Hypothesis[] | null;
  provenanceRuns: ProvenanceRun[] | null;
  annotations: Annotation[] | null;
  pharmacophorePockets: PharmacophoreData | null;
  drugCandidates: DrugCandidateData | null;
  phase4_resistance: ResistanceData | null;
  bufferingAtlas: Record<string, unknown> | null;
  persistenceStatus: HydrationResponse["persistence_status"] | null;
}

const defaultLoading: LoadingState = {
  embeddings: false,
  graph_metrics: false,
  allosteric_sites: false,
  source_leaks: false,
  hypotheses: false,
  provenance_runs: false,
  annotations: false,
};

const HydrationContext = createContext<HydrationContextValue>({
  hydration: null,
  loading: defaultLoading,
  isHydrating: false,
  error: null,
  refresh: () => {},
  multiHydrations: {},
  refreshAll: () => {},
  embeddings: null,
  graphMetrics: null,
  allostericSites: null,
  sourceLeaks: null,
  resistanceData: null,
  hypotheses: null,
  provenanceRuns: null,
  annotations: null,
  pharmacophorePockets: null,
  drugCandidates: null,
  phase4_resistance: null,
  bufferingAtlas: null,
  persistenceStatus: null,
});

export function useHydration() {
  return useContext(HydrationContext);
}

interface HydrationProviderProps {
  children: ReactNode;
}

export function HydrationProvider({ children }: HydrationProviderProps) {
  const { activeStructure, refreshKey } = useDashboard();

  const [hydration, setHydration] = useState<HydrationResponse | null>(null);
  const [loading, setLoading] = useState<LoadingState>(defaultLoading);
  const [isHydrating, setIsHydrating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Multi-structure hydration state for Therapeutic Compiler (explicit support)
  const [multiHydrations, setMultiHydrations] = useState<Record<string, HydrationResponse | null>>({});

  const fetchHydration = useCallback(async (structureId: string) => {
    setIsHydrating(true);
    setError(null);
    // Set all panels to loading
    setLoading({
      embeddings: true,
      graph_metrics: true,
      allosteric_sites: true,
      source_leaks: true,
      hypotheses: true,
      provenance_runs: true,
      annotations: true,
    });

    try {
      const data = await api.hydrate(structureId);
      setHydration(data);
      setError(null);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Hydration failed";
      setError(msg);
      setHydration(null);
    } finally {
      setIsHydrating(false);
      setLoading(defaultLoading);
    }
  }, []);

  // Fetch a single additional structure for multi support
  const fetchMultiHydration = useCallback(async (structureId: string) => {
    try {
      const data = await api.hydrate(structureId);
      setMultiHydrations(prev => ({ ...prev, [structureId]: data }));
    } catch (e) {
      setMultiHydrations(prev => ({ ...prev, [structureId]: null }));
    }
  }, []);

  // Fetch hydration when active structure changes or refreshKey increments
  useEffect(() => {
    if (activeStructure) {
      fetchHydration(activeStructure.structure_id);
    } else {
      setHydration(null);
      setError(null);
      setLoading(defaultLoading);
    }
  }, [activeStructure, refreshKey, fetchHydration]);

  // Explicit multi-structure: when therapeuticCompilerState changes, hydrate the structures it references
  const { therapeuticCompilerState } = useDashboard();
  useEffect(() => {
    if (!therapeuticCompilerState) return;
    const structs = new Set<string>();
    if (therapeuticCompilerState.pathwayStructureMap) {
      Object.values(therapeuticCompilerState.pathwayStructureMap).forEach((s: any) => s && structs.add(String(s).toLowerCase()));
    }
    if (therapeuticCompilerState.atlas?.nodes) {
      therapeuticCompilerState.atlas.nodes.forEach((n: any) => {
        const sid = n.metadata?.structure_id || n.id?.split('_').pop();
        if (sid) structs.add(String(sid).toLowerCase());
      });
    }
    // Also include active if present
    if (activeStructure) structs.add(activeStructure.structure_id.toLowerCase());

    Array.from(structs).forEach(sid => {
      if (sid !== activeStructure?.structure_id?.toLowerCase() && !multiHydrations[sid]) {
        fetchMultiHydration(sid);
      }
    });
  }, [therapeuticCompilerState, activeStructure, multiHydrations, fetchMultiHydration]);

  const refresh = useCallback(() => {
    if (activeStructure) {
      fetchHydration(activeStructure.structure_id);
    }
  }, [activeStructure, fetchHydration]);

  const refreshAll = useCallback(() => {
    refresh();
    Object.keys(multiHydrations).forEach(sid => fetchMultiHydration(sid));
  }, [refresh, multiHydrations, fetchMultiHydration]);

  const value: HydrationContextValue = {
    hydration,
    loading,
    isHydrating,
    error,
    refresh,
    embeddings: hydration?.embeddings ?? null,
    graphMetrics: hydration?.graph_metrics ?? null,
    allostericSites: hydration?.allosteric_sites ?? null,
    sourceLeaks: hydration?.source_leaks
      ? {
          structure_id: hydration.source_leaks.structure_id ?? activeStructure?.structure_id ?? "",
          leaks: hydration.source_leaks.source_leaks ?? hydration.source_leaks.leaks ?? [],
        }
      : null,
    resistanceData: hydration?.resistance_data ?? null,
    hypotheses: hydration?.hypotheses ?? null,
    provenanceRuns: hydration?.provenance_runs ?? null,
    annotations: hydration?.annotations ?? null,
    // Map the governed phase5/phase6 (with our backend enrichment for connected_allosteric_locks,
    // connecting_pathways, connecting_coupling_sum etc. per pocket, plus B-chain aware residue_ids)
    // into the shape expected by GraphTopologyPanel, MolecularViewer radar, and DataInspector.
    pharmacophorePockets: hydration?.phase5_pharmacophore
      ? {
          structure_id: hydration.phase5_pharmacophore.structure_id,
          pockets: hydration.phase5_pharmacophore.pharmacophores || hydration.phase5_pharmacophore.pockets || [],
          count: hydration.phase5_pharmacophore.count || (hydration.phase5_pharmacophore.pharmacophores || []).length,
        }
      : (hydration?.pharmacophore_pockets ?? null),
    drugCandidates: hydration?.phase6_drug_candidates
      ? {
          structure_id: hydration.phase6_drug_candidates.structure_id,
          candidates: hydration.phase6_drug_candidates.candidates || hydration.phase6_drug_candidates.drug_candidates || [],
          count: hydration.phase6_drug_candidates.count || 0,
          admet_passed_count: hydration.phase6_drug_candidates.admet_passed_count ?? 0,
          state_selective_count: hydration.phase6_drug_candidates.state_selective_count ?? 0,
        }
      : (hydration?.drug_candidates ?? null),
    phase4_resistance: hydration?.phase4_resistance ?? null,
    bufferingAtlas: hydration?.buffering_atlas ?? null,  // live (X, Y) from Phase 7 / governed facts for the current structure
    persistenceStatus: hydration?.persistence_status ?? null,
    multiHydrations,
    refreshAll,
  };

  return (
    <HydrationContext.Provider value={value}>
      {children}
    </HydrationContext.Provider>
  );
}
