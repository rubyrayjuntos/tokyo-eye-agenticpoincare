import { createContext, useContext, useEffect, useState, useCallback, useMemo } from "react";
import type { ReactNode } from "react";
import { useMachine } from "@xstate/react";

import { useDashboard } from "../lib/context";
import { buildHydrationView } from "../lib/hydrationView";
import { loadHydrationBundle } from "../lib/hydrationLoader";
import {
  hydrationMachine,
  hydrationMachineStatus,
} from "../lib/hydrationMachine";
import type {
  HydrationResponse,
  GraphMetricsData,
  AllostericSitesData,
  SourceLeakData,
  ResistanceData,
  Hypothesis,
  ProvenanceRun,
  Annotation,
  PharmacophoreData,
  DrugCandidateData,
  StructureAnalysisSnapshot,
  ArtifactAvailabilityEntry,
  HydrateMeta,
  PersistenceStatus,
  EmbeddingData,
} from "../lib/types";
import { useOptionalWorkbenchBus } from "../workbench/useOptionalWorkbenchBus";

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
  hydration: HydrationResponse | null;
  loading: LoadingState;
  isHydrating: boolean;
  error: string | null;
  refresh: () => void;
  multiHydrations: Record<string, HydrationResponse | null>;
  refreshAll: () => void;
  structureSnapshot: StructureAnalysisSnapshot | null;
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
  persistenceStatus: PersistenceStatus | null;
  artifactAvailability: Record<string, ArtifactAvailabilityEntry> | null;
  hydrateMeta: HydrateMeta | null;
  hydrateSignalsInferred: boolean;
  hydrationStatus: "idle" | "loading" | "loaded" | "failed";
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
  structureSnapshot: null,
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
  artifactAvailability: null,
  hydrateMeta: null,
  hydrateSignalsInferred: false,
  hydrationStatus: "idle",
});

export function useHydration(): HydrationContextValue & {
  hydrationBusStatus: "idle" | "loading" | "loaded" | "failed" | null;
} {
  const hydration = useContext(HydrationContext);
  const bus = useOptionalWorkbenchBus();
  const [busPayload, setBusPayload] = useState<
    import("../workbench/eventRegistry").EventRegistry["data:hydration_updated"] | null
  >(null);

  useEffect(() => {
    if (!bus) {
      setBusPayload(null);
      return;
    }

    const cached = bus.getLatestState("data:hydration_updated");
    if (cached) {
      setBusPayload(cached);
    }

    return bus.subscribe("data:hydration_updated", (payload) => {
      setBusPayload(payload);
    });
  }, [bus]);

  const busMatchesActive =
    busPayload?.structureId != null &&
    hydration.hydration?.structure_id != null &&
    busPayload.structureId.toLowerCase() ===
      hydration.hydration.structure_id.toLowerCase();

  if (!busMatchesActive || !busPayload) {
    return { ...hydration, hydrationBusStatus: busPayload?.status ?? null };
  }

  return {
    ...hydration,
    isHydrating: busPayload.status === "loading",
    error: busPayload.error ?? hydration.error,
    hydrateSignalsInferred: busPayload.signalsInferred,
    hydrationBusStatus: busPayload.status,
  };
}

interface HydrationProviderProps {
  children: ReactNode;
}

export function HydrationProvider({ children }: HydrationProviderProps) {
  const { activeStructure, refreshKey } = useDashboard();
  const workbenchBus = useOptionalWorkbenchBus();
  const [hydrationState, sendHydration] = useMachine(hydrationMachine);
  const machineContext = hydrationState.context;
  const hydrationStatus = hydrationMachineStatus(hydrationState.value);

  const [multiHydrations, setMultiHydrations] = useState<Record<string, HydrationResponse | null>>({});

  const runHydration = useCallback(
    async (structureId: string) => {
      sendHydration({ type: "HYDRATION_REQUESTED", structureId });
      try {
        const result = await loadHydrationBundle(structureId);
        sendHydration({ type: "HYDRATION_SUCCEEDED", result });
      } catch (err: unknown) {
        const msg =
          err && typeof err === "object" && "message" in err
            ? (err as { message: string }).message
            : "Hydration failed";
        sendHydration({ type: "HYDRATION_FAILED", error: msg });
      }
    },
    [sendHydration],
  );

  const fetchMultiHydration = useCallback(async (structureId: string) => {
    try {
      const result = await loadHydrationBundle(structureId);
      setMultiHydrations((prev) => ({ ...prev, [structureId]: result.hydration }));
    } catch {
      setMultiHydrations((prev) => ({ ...prev, [structureId]: null }));
    }
  }, []);

  useEffect(() => {
    if (activeStructure) {
      void runHydration(activeStructure.structure_id);
    } else {
      sendHydration({ type: "HYDRATION_CLEARED" });
    }
  }, [activeStructure, refreshKey, runHydration, sendHydration]);

  const { therapeuticCompilerState } = useDashboard();
  useEffect(() => {
    if (!therapeuticCompilerState) return;
    const structs = new Set<string>();
    if (therapeuticCompilerState.pathwayStructureMap) {
      Object.values(therapeuticCompilerState.pathwayStructureMap).forEach((s: unknown) =>
        s && structs.add(String(s).toLowerCase()),
      );
    }
    if (therapeuticCompilerState.atlas?.nodes) {
      therapeuticCompilerState.atlas.nodes.forEach((n: { metadata?: { structure_id?: string }; id?: string }) => {
        const sid = n.metadata?.structure_id || n.id?.split("_").pop();
        if (sid) structs.add(String(sid).toLowerCase());
      });
    }
    if (activeStructure) structs.add(activeStructure.structure_id.toLowerCase());

    Array.from(structs).forEach((sid) => {
      if (sid !== activeStructure?.structure_id?.toLowerCase() && !multiHydrations[sid]) {
        void fetchMultiHydration(sid);
      }
    });
  }, [therapeuticCompilerState, activeStructure, multiHydrations, fetchMultiHydration]);

  const refresh = useCallback(() => {
    if (activeStructure) {
      void runHydration(activeStructure.structure_id);
    }
  }, [activeStructure, runHydration]);

  const refreshAll = useCallback(() => {
    refresh();
    Object.keys(multiHydrations).forEach((sid) => void fetchMultiHydration(sid));
  }, [refresh, multiHydrations, fetchMultiHydration]);

  const hydration = machineContext.hydration;
  const hydrationView = buildHydrationView(hydration, activeStructure?.structure_id);

  const loading = useMemo<LoadingState>(() => {
    if (hydrationStatus !== "loading") return defaultLoading;
    return {
      embeddings: true,
      graph_metrics: true,
      allosteric_sites: true,
      source_leaks: true,
      hypotheses: true,
      provenance_runs: true,
      annotations: true,
    };
  }, [hydrationStatus]);

  useEffect(() => {
    if (!workbenchBus) return;

    workbenchBus.publish("data:hydration_updated", {
      structureId: machineContext.structureId,
      status: hydrationStatus,
      degraded: Boolean(machineContext.hydrateMeta?.degraded),
      residueCount: hydrationView.embeddings?.residues?.length ?? 0,
      error: machineContext.error,
      signalsInferred: machineContext.signalsInferred,
      contractVersion: machineContext.hydrateMeta?.contract_version ?? null,
    });

    if (hydrationStatus === "loaded" && machineContext.structureId) {
      workbenchBus.publish("data:hydration_ready", {
        structureId: machineContext.structureId,
        residueCount: hydrationView.embeddings?.residues?.length ?? 0,
        degraded: Boolean(machineContext.hydrateMeta?.degraded),
      });
    }
  }, [
    workbenchBus,
    hydrationStatus,
    machineContext.structureId,
    machineContext.error,
    machineContext.hydrateMeta,
    machineContext.signalsInferred,
    hydrationView.embeddings?.residues?.length,
  ]);

  const value: HydrationContextValue = {
    hydration,
    loading,
    isHydrating: hydrationStatus === "loading",
    error: machineContext.error,
    refresh,
    structureSnapshot: hydrationView.structureSnapshot,
    embeddings: hydrationView.embeddings,
    graphMetrics: hydrationView.graphMetrics,
    allostericSites: hydrationView.allostericSites,
    sourceLeaks: hydrationView.sourceLeaks,
    resistanceData: hydrationView.resistanceData,
    hypotheses: hydrationView.hypotheses,
    provenanceRuns: hydrationView.provenanceRuns,
    annotations: (hydration?.annotations as Annotation[] | null | undefined) ?? null,
    pharmacophorePockets: hydrationView.pharmacophorePockets,
    drugCandidates: hydrationView.drugCandidates,
    phase4_resistance: hydrationView.phase4Resistance,
    bufferingAtlas: (hydration?.buffering_atlas as Record<string, unknown> | null) ?? null,
    persistenceStatus: hydrationView.persistenceStatus,
    artifactAvailability: machineContext.artifactAvailability,
    hydrateMeta: machineContext.hydrateMeta,
    hydrateSignalsInferred: machineContext.signalsInferred,
    multiHydrations,
    refreshAll,
    hydrationStatus,
  };

  return (
    <HydrationContext.Provider value={value}>
      {children}
    </HydrationContext.Provider>
  );
}