import { useState, useCallback } from "react";
import NavBar from "./components/NavBar";
import KPIBar from "./components/KPIBar";
import AgentTelemetryPanel from "./components/AgentTelemetryPanel";
import ResultsTable from "./components/ResultsTable";
import VizGrid from "./components/VizGrid";
import AgentChat from "./components/AgentChat";
import ToolPanelSidebar from "./components/ToolPanelSidebar";
import VisualizationToolbar from "./components/controls/VisualizationToolbar";
import CompareIndicatorBar from "./components/CompareIndicatorBar";
import { HydrationProvider } from "./context/HydrationProvider";
import { DashboardContext } from "./lib/context";
import { api } from "./lib/api";
import { useViewportSocket } from "./lib/useViewportSocket";
import { useActor } from "@xstate/react";
import { viewportMachine } from "./lib/viewportMachine";
import { useOrchestratorPolicy } from "./lib/useOrchestratorPolicy";
import { useLegacyViewportBridge } from "./lib/useLegacyViewportBridge";
import type {
  Structure,
  CompareState,
  AgentChatResponse,
} from "./lib/types";

export default function App() {
  const [activeStructure, setActiveStructure] = useState<Structure | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  const [compareState, setCompareState] = useState<CompareState>({
    active: false,
    secondaryStructure: null,
    displacements: null,
    graphDiff: null,
    loading: false,
    error: null,
  });

  const [selectedPocketId, setSelectedPocketId] = useState<number | null>(null);
  const [therapeuticCompilerState, setTherapeuticCompilerState] = useState<any | null>(null);
  const [collapseSimulationState, setCollapseSimulationState] = useState({
    fraction: 0.0,
    goal: "Exploit isolated targets post-fragmentation",
    active: false,
  });
  const [latestAgentTelemetry, setLatestAgentTelemetry] = useState<AgentChatResponse["telemetry"] | null>(null);
  const [agentSessionId, setAgentSessionId] = useState<string | null>(null);

  const [viewportState, sendViewport] = useActor(viewportMachine);
  const orchestrator = useOrchestratorPolicy(viewportState.context);
  const viewport = useLegacyViewportBridge(viewportState, sendViewport);

  const enterCompareMode = useCallback(
    (secondary: Structure) => {
      if (!activeStructure) return;
      if (!activeStructure.has_embeddings || !secondary.has_embeddings) return;

      setCompareState({
        active: true,
        secondaryStructure: secondary,
        displacements: null,
        graphDiff: null,
        loading: true,
        error: null,
      });

      const primaryId = activeStructure.structure_id;
      const secondaryId = secondary.structure_id;

      Promise.all([
        api.compareEmbeddings(primaryId, secondaryId).catch(() => null),
        api.compareGraphsDirect(primaryId, secondaryId).catch(() => null),
      ]).then(([displacements, graphDiff]) => {
        setCompareState((prev) => ({
          ...prev,
          displacements: displacements ?? null,
          graphDiff: graphDiff ?? null,
          loading: false,
          error:
            !displacements && !graphDiff
              ? "Failed to load comparison data"
              : null,
        }));
      });
    },
    [activeStructure],
  );

  const exitCompareMode = useCallback(() => {
    setCompareState({
      active: false,
      secondaryStructure: null,
      displacements: null,
      graphDiff: null,
      loading: false,
      error: null,
    });
  }, []);

  const triggerRefresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
  }, []);

  useViewportSocket({
    onDirective: viewport.emitDirective,
    enabled: true,
    sessionId: agentSessionId,
  });

  return (
    <DashboardContext.Provider
      value={{
        activeStructure,
        setActiveStructure,
        chatOpen,
        setChatOpen,
        highlightedResidues: viewport.highlightedResidues,
        setHighlightedResidues: viewport.setHighlightedResidues,
        refreshKey,
        triggerRefresh,
        emitDirective: viewport.emitDirective,
        currentDirective: viewport.currentDirective,
        compareState,
        enterCompareMode,
        exitCompareMode,
        selectedPocketId,
        setSelectedPocketId,
        isRadarActive: viewport.isRadarActive,
        setIsRadarActive: viewport.setIsRadarActive,
        therapeuticCompilerState,
        setTherapeuticCompilerState,
        collapseSimulationState,
        setCollapseSimulationState,
        latestAgentTelemetry,
        setLatestAgentTelemetry,
        agentSessionId,
        setAgentSessionId,
        discoveryContext: orchestrator.discoveryContext,
        sendDiscovery: orchestrator.sendDiscovery,
        hypothesisContext: orchestrator.hypothesisContext,
        sendHypothesis: orchestrator.sendHypothesis,
        plannerPolicy: orchestrator.plannerPolicy,
        sessionMode: orchestrator.sessionMode,
        setSessionMode: orchestrator.setSessionMode,
        structureScope: orchestrator.structureScope,
        setStructureScope: orchestrator.setStructureScope,
        poincareColorMode: viewport.poincareColorMode,
        viewerColorMode: viewport.viewerColorMode,
        riskThreshold: viewport.riskThreshold,
        activePanel: viewport.activePanel,
        sidebarOpen: viewportState.context.sidebarOpen,
        activeEditorTab: viewportState.context.activeEditorTab,
        bottomPanelOpen: viewportState.context.bottomPanelOpen,
        activeBottomPanel: viewportState.context.activeBottomPanel,
        layoutModelJSON: null,
        userSelectedResidue: viewportState.context.userSelectedResidue,
        sendViewport,
      }}
    >
      <HydrationProvider>
        <div className="w-full h-screen flex flex-col bg-zinc-950 text-zinc-100 font-sans overflow-hidden">
          <NavBar />
          <KPIBar />
          <div className="flex flex-1 min-h-0">
            <AgentTelemetryPanel />
            <main className="flex-1 flex flex-col gap-2 p-3 min-h-0 overflow-hidden">
              <VisualizationToolbar
                selectedResidues={viewport.highlightedResidues}
                onDirective={viewport.emitDirective}
              />
              <CompareIndicatorBar />
              <VizGrid
                onPoincareColorModeChange={viewport.setPoincareColorMode}
                onPoincareSelectedResidueChange={viewport.setPoincareSelectedResidue}
                onMobiusFocusChange={viewport.setMobiusFocus}
                onBrushSelectionChange={viewport.setBrushSelection}
                onViewerColorModeChange={viewport.setViewerColorMode}
                onRiskThresholdChange={viewport.setRiskThreshold}
              />
              <div className="shrink-0 max-h-[25%] overflow-y-auto">
                <ResultsTable />
              </div>
            </main>
            <ToolPanelSidebar onActivePanelChange={viewport.setActivePanel} />
            <AgentChat
              poincareColorMode={viewport.poincareColorMode}
              poincareSelectedResidue={viewport.poincareSelectedResidue}
              mobiusFocus={viewport.mobiusFocus}
              brushSelection={viewport.brushSelection}
              viewerColorMode={viewport.viewerColorMode}
              riskThreshold={viewport.riskThreshold}
              activePanel={viewport.activePanel}
            />
          </div>
        </div>
      </HydrationProvider>
    </DashboardContext.Provider>
  );
}