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

export default function App() {
  const [activeStructure, setActiveStructure] = useState<Structure | null>(
    null
  );
  const [chatOpen, setChatOpen] = useState(false);
  const [highlightedResidues, setHighlightedResidues] = useState<string[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [currentDirective, setCurrentDirective] =
    useState<ViewportDirective | null>(null);

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
    setRefreshKey((k) => k + 1);
  }, []);

  const emitDirective = useCallback(
    (directive: ViewportDirective) => {
      const normalizedDirective =
        directive.action === "focus" &&
        (!directive.highlight_groups || directive.highlight_groups.length === 0) &&
        directive.focus_residues?.length
          ? {
              ...directive,
              highlight_groups: [
                {
                  residue_ids: directive.focus_residues,
                  color: "#4ecdc4",
                  style: "glow" as const,
                  label: "Focus",
                },
              ],
            }
          : directive;

      setCurrentDirective(normalizedDirective);
      if (
        normalizedDirective.action === "highlight" ||
        normalizedDirective.action === "focus"
      ) {
        const residues =
          normalizedDirective.highlight_groups?.flatMap((g) => g.residue_ids) ??
          [];
        if (residues.length > 0) {
          setHighlightedResidues(residues);
        }
      } else if (normalizedDirective.action === "clear") {
        setHighlightedResidues([]);
        setIsRadarActive(false);
      }
    },
    []
  );

  useViewportSocket({
    onDirective: emitDirective,
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
        highlightedResidues,
        setHighlightedResidues,
        refreshKey,
        triggerRefresh,
        emitDirective,
        currentDirective,
        compareState,
        enterCompareMode,
        exitCompareMode,
        selectedPocketId,
        setSelectedPocketId,
        isRadarActive,
        setIsRadarActive,
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
        poincareColorMode,
        viewerColorMode,
        riskThreshold,
        activePanel,
        layoutModelJSON: null,
        userSelectedResidue: null,
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
                selectedResidues={highlightedResidues}
                onDirective={emitDirective}
              />
              <CompareIndicatorBar />
              <VizGrid
                onPoincareColorModeChange={setPoincareColorMode}
                onPoincareSelectedResidueChange={setPoincareSelectedResidue}
                onMobiusFocusChange={setMobiusFocus}
                onBrushSelectionChange={setBrushSelection}
                onViewerColorModeChange={setViewerColorMode}
                onRiskThresholdChange={setRiskThreshold}
              />
              <div className="shrink-0 max-h-[25%] overflow-y-auto">
                <ResultsTable />
              </div>
            </main>
            <ToolPanelSidebar onActivePanelChange={setActivePanel} />
            <AgentChat
              poincareColorMode={poincareColorMode}
              poincareSelectedResidue={poincareSelectedResidue}
              mobiusFocus={mobiusFocus}
              brushSelection={brushSelection}
              viewerColorMode={viewerColorMode}
              riskThreshold={riskThreshold}
              activePanel={activePanel}
            />
          </div>
        </div>
      </HydrationProvider>
    </DashboardContext.Provider>
  );
}
