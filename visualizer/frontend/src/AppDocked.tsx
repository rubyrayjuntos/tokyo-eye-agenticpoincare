import { useState, useCallback } from "react";
import NavBar from "./components/NavBar";
import KPIBar from "./components/KPIBar";
import AgentTelemetryPanel from "./components/AgentTelemetryPanel";
import ResultsTable from "./components/ResultsTable";
import AgentChat from "./components/AgentChat";
import ToolPanelSidebar from "./components/ToolPanelSidebar";
import VisualizationToolbar from "./components/controls/VisualizationToolbar";
import CompareIndicatorBar from "./components/CompareIndicatorBar";
import PoincareScatter from "./components/PoincareScatter";
import MolecularViewer from "./components/MolecularViewer";
import { DockedLayout } from "./components/DockedLayout";
import { HydrationProvider } from "./context/HydrationProvider";
import { DashboardContext } from "./lib/context";
import { useViewportSocket } from "./lib/useViewportSocket";
import { useActor } from "@xstate/react";
import { viewportMachine } from "./lib/viewportMachine";
import { useOrchestratorPolicy } from "./lib/useOrchestratorPolicy";
import type {
  Structure,
  ViewportDirective,
  PoincareColorMode,
  SelectedResidueInfo,
  StructureColorModeType,
  ActivePanelName,
  CompareState,
  AgentChatResponse,
} from "./lib/types";

export default function AppDocked() {
  const [activeStructure, setActiveStructure] = useState<Structure | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [highlightedResidues, setHighlightedResidues] = useState<string[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [currentDirective, setCurrentDirective] = useState<ViewportDirective | null>(null);

  const [poincareColorMode, setPoincareColorMode] = useState<PoincareColorMode>("cone_depth");
  const [poincareSelectedResidue, setPoincareSelectedResidue] = useState<SelectedResidueInfo | null>(null);
  const [mobiusFocus, setMobiusFocus] = useState(false);
  const [brushSelection, setBrushSelection] = useState<string[]>([]);
  const [viewerColorMode, setViewerColorMode] = useState<StructureColorModeType>("spectrum");
  const [riskThreshold, setRiskThreshold] = useState(0);
  const [activePanel, setActivePanel] = useState<ActivePanelName>(null);

  const [compareState, setCompareState] = useState<CompareState>({
    active: false,
    secondaryStructure: null,
    displacements: null,
    graphDiff: null,
    loading: false,
    error: null,
  });

  const [selectedPocketId, setSelectedPocketId] = useState<number | null>(null);
  const [isRadarActive, setIsRadarActive] = useState<boolean>(false);
  const [therapeuticCompilerState, setTherapeuticCompilerState] = useState<any | null>(null);
  const [collapseSimulationState, setCollapseSimulationState] = useState({
    fraction: 0.0,
    goal: "Exploit isolated targets post-fragmentation",
    active: false,
  });
  const [latestAgentTelemetry, setLatestAgentTelemetry] = useState<AgentChatResponse["telemetry"] | null>(null);
  const [agentSessionId, setAgentSessionId] = useState<string | null>(null);

  // XState machines
  const [viewportState, sendViewport] = useActor(viewportMachine);
  const orchestrator = useOrchestratorPolicy(viewportState.context);

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
    },
    [activeStructure]
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
    setRefreshKey((prev) => prev + 1);
  }, []);

  const emitDirective = useCallback(
    (directive: ViewportDirective) => {
      const normalizedDirective = directive;

      setCurrentDirective(normalizedDirective);
      if (normalizedDirective.action === "highlight" || normalizedDirective.action === "focus") {
        const residues = normalizedDirective.highlight_groups?.flatMap((g) => g.residue_ids) ?? [];
        if (residues.length > 0) setHighlightedResidues(residues);
      } else if (normalizedDirective.action === "clear") {
        setHighlightedResidues([]);
        setIsRadarActive(false);
      }
    },
    []
  );

  useViewportSocket({ onDirective: emitDirective, enabled: true, sessionId: agentSessionId });

  return (
    <DashboardContext.Provider
      value={{
        activeStructure, setActiveStructure,
        chatOpen, setChatOpen,
        highlightedResidues, setHighlightedResidues,
        refreshKey, triggerRefresh,
        emitDirective, currentDirective,
        compareState, enterCompareMode, exitCompareMode,
        selectedPocketId, setSelectedPocketId,
        isRadarActive, setIsRadarActive,
        therapeuticCompilerState, setTherapeuticCompilerState,
        collapseSimulationState, setCollapseSimulationState,
        latestAgentTelemetry, setLatestAgentTelemetry,
        agentSessionId, setAgentSessionId,
        discoveryContext: orchestrator.discoveryContext,
        sendDiscovery: orchestrator.sendDiscovery,
        hypothesisContext: orchestrator.hypothesisContext,
        sendHypothesis: orchestrator.sendHypothesis,
        plannerPolicy: orchestrator.plannerPolicy,
        sessionMode: orchestrator.sessionMode,
        setSessionMode: orchestrator.setSessionMode,
        structureScope: orchestrator.structureScope,
        setStructureScope: orchestrator.setStructureScope,
        poincareColorMode,
        viewerColorMode,
        riskThreshold,
        activePanel,
        sidebarOpen: viewportState.context.sidebarOpen,
        activeEditorTab: viewportState.context.activeEditorTab,
        bottomPanelOpen: viewportState.context.bottomPanelOpen,
        activeBottomPanel: viewportState.context.activeBottomPanel,
        layoutModelJSON: null,
        userSelectedResidue: null,
        sendViewport,
      }}
    >
      <HydrationProvider>
        <div className="flex flex-col w-screen h-screen bg-[#0a0e27]">
          <NavBar />
          <KPIBar />
          <div className="flex flex-col gap-2 px-3 py-2 bg-[#050810]/50 border-b border-cyan-900/20">
            <VisualizationToolbar selectedResidues={highlightedResidues} onDirective={emitDirective} />
            <CompareIndicatorBar />
          </div>
          <DockedLayout
            poincare={
              <div className="w-full h-full flex items-center justify-center p-4">
                <PoincareScatter
                  onColorModeChange={setPoincareColorMode}
                  onSelectedResidueChange={setPoincareSelectedResidue}
                  onMobiusFocusChange={setMobiusFocus}
                  onBrushSelectionChange={setBrushSelection}
                />
              </div>
            }
            molecularViewer={
              <div className="w-full h-full flex items-center justify-center p-4">
                <MolecularViewer onColorModeChange={setViewerColorMode} onRiskThresholdChange={setRiskThreshold} />
              </div>
            }
            agentChat={
              <AgentChat
                poincareColorMode={poincareColorMode}
                poincareSelectedResidue={poincareSelectedResidue}
                mobiusFocus={mobiusFocus}
                brushSelection={brushSelection}
                viewerColorMode={viewerColorMode}
                riskThreshold={riskThreshold}
                activePanel={activePanel}
              />
            }
            sidebarLeft={
              <div className="flex flex-col gap-4">
                <AgentTelemetryPanel />
                <ToolPanelSidebar onActivePanelChange={setActivePanel} />
              </div>
            }
          />
          <div className="shrink-0 max-h-[15%] overflow-y-auto border-t border-cyan-900/20 p-3 bg-[#050810]/50">
            <ResultsTable />
          </div>
        </div>
      </HydrationProvider>
    </DashboardContext.Provider>
  );
}
