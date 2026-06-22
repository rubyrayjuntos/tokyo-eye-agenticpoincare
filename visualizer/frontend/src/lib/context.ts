import { createContext, useContext } from "react";
import type { AgentChatResponse, Structure, ViewportDirective, CompareState, PoincareColorMode, StructureColorModeType, ActivePanelName } from "./types";
import type { DiscoveryPhaseContext, DiscoveryPhaseEvent } from "./discoveryPhaseMachine";
import type { HypothesisLifecycleContext, HypothesisLifecycleEvent } from "./hypothesisLifecycleMachine";
import type { PlannerPolicy, SessionMode, StructureScopeContext } from "./plannerPolicySelector";

export interface DashboardContextValue {
  activeStructure: Structure | null;
  setActiveStructure: (s: Structure | null) => void;
  chatOpen: boolean;
  setChatOpen: (open: boolean) => void;
  highlightedResidues: string[];
  setHighlightedResidues: (residues: string[]) => void;
  refreshKey: number;
  triggerRefresh: () => void;
  emitDirective: (directive: ViewportDirective) => void;
  currentDirective: ViewportDirective | null;
  compareState: CompareState;
  enterCompareMode: (secondary: Structure) => void;
  exitCompareMode: () => void;
  selectedPocketId: number | null;
  setSelectedPocketId: (id: number | null) => void;
  isRadarActive: boolean;
  setIsRadarActive: (active: boolean) => void;
  therapeuticCompilerState: any | null;
  setTherapeuticCompilerState: (state: any | null) => void;
  collapseSimulationState: {
    fraction: number;
    goal: string;
    active: boolean;
  };
  setCollapseSimulationState: (state: { fraction: number; goal: string; active: boolean }) => void;
  latestAgentTelemetry: AgentChatResponse["telemetry"] | null;
  setLatestAgentTelemetry: (telemetry: AgentChatResponse["telemetry"] | null) => void;
  agentSessionId: string | null;
  setAgentSessionId: (sessionId: string | null) => void;

  discoveryContext: DiscoveryPhaseContext | null;
  sendDiscovery: (event: DiscoveryPhaseEvent) => void;
  hypothesisContext: HypothesisLifecycleContext | null;
  sendHypothesis: (event: HypothesisLifecycleEvent) => void;
  plannerPolicy: PlannerPolicy | null;
  sessionMode: SessionMode;
  setSessionMode: (mode: SessionMode) => void;
  structureScope: StructureScopeContext | null;
  setStructureScope: ((scope: StructureScopeContext) => void) | null;

  // Lifted unified XState values for component layout overrides
  poincareColorMode: PoincareColorMode;
  viewerColorMode: StructureColorModeType;
  riskThreshold: number;
  activePanel: ActivePanelName;
  layoutModelJSON: any | null;
  userSelectedResidue: string | null;
  sendViewport: (event: any) => void;
}

export const DashboardContext = createContext<DashboardContextValue>({
  activeStructure: null,
  setActiveStructure: () => {},
  chatOpen: false,
  setChatOpen: () => {},
  highlightedResidues: [],
  setHighlightedResidues: () => {},
  refreshKey: 0,
  triggerRefresh: () => {},
  emitDirective: () => {},
  currentDirective: null,
  compareState: {
    active: false,
    secondaryStructure: null,
    displacements: null,
    graphDiff: null,
    loading: false,
    error: null,
  },
  enterCompareMode: () => {},
  exitCompareMode: () => {},
  selectedPocketId: null,
  setSelectedPocketId: () => {},
  isRadarActive: false,
  setIsRadarActive: () => {},
  therapeuticCompilerState: null,
  setTherapeuticCompilerState: () => {},
  collapseSimulationState: { fraction: 0.0, goal: "Exploit isolated targets post-fragmentation", active: false },
  setCollapseSimulationState: () => {},
  latestAgentTelemetry: null,
  setLatestAgentTelemetry: () => {},
  agentSessionId: null,
  setAgentSessionId: () => {},
  discoveryContext: null,
  sendDiscovery: () => {},
  hypothesisContext: null,
  sendHypothesis: () => {},
  plannerPolicy: null,
  sessionMode: 'scientific_rigor',
  setSessionMode: () => {},
  structureScope: null,
  setStructureScope: null,
  poincareColorMode: "cone_depth",
  viewerColorMode: "spectrum",
  riskThreshold: 0,
  activePanel: null,
  layoutModelJSON: null,
  userSelectedResidue: null,
  sendViewport: () => {},
});

export function useDashboard() {
  return useContext(DashboardContext);
}
