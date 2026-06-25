import { useCallback, useRef, useState, type ReactNode } from "react";
import {
  ArrowLeftRight,
  BarChart3,
  BookOpenText,
  Command,
  Database,
  GitBranch,
  Lightbulb,
  Search,
  TableProperties,
} from "lucide-react";
import { useActor } from "@xstate/react";

import { CockpitNavBar, IdeShellLayout, ToolDock } from "./components/layout";
import type { IdeShellLayoutModel, IdeShellPane } from "./components/layout";
import { ChatRail } from "./components/chat";
import { PoincarePanel, MolecularPanel } from "./components/viewers";
import type { MolecularColorMode, PanelPoincareColorMode } from "./components/viewers";
import { ToolPanelHost } from "./components/tools/ToolPanelHost";
import CompareIndicatorBar from "./components/CompareIndicatorBar";
import VisualizationToolbar from "./components/controls/VisualizationToolbar";
import { BottomPanelHost } from "./components/workbench/BottomPanelHost";
import {
  PanelBrokerProvider,
  usePanelBroker,
} from "./components/workbench/panelBroker";
import {
  EDITOR_TAB_REGISTRY,
  BOTTOM_PANEL_REGISTRY,
  getBottomPanelDefinition,
  getToolPanelDefinition,
  TOOL_PANEL_REGISTRY,
} from "./components/workbench/panelRegistry";
import { DashboardContext } from "./lib/context";
import { HydrationProvider, useHydration } from "./context/HydrationProvider";
import { api } from "./lib/api";
import { buildViewportState } from "./lib/buildViewportState";
import { useDirectives } from "./lib/useDirectives";
import { useOrchestratorPolicy } from "./lib/useOrchestratorPolicy";
import { useSelectionSync } from "./lib/useSelectionSync";
import { useViewportSocket } from "./lib/useViewportSocket";
import type { PhaseTransitionPayload, StateSnapshotPayload } from "./lib/useViewportSocket";
import { viewportMachine } from "./lib/viewportMachine";
import type {
  ActivePanelName,
  AgentChatResponse,
  BottomPanelId,
  CompareState,
  EditorTabId,
  PoincareColorMode,
  SelectedResidueInfo,
  Structure,
  StructureColorModeType,
  ViewportDirective,
  ViewportState,
} from "./lib/types";
import type { DiscoveryPhase } from "./lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "./lib/hypothesisLifecycleMachine";

const TOOL_ICONS: Record<string, ReactNode> = {
  briefing: <BookOpenText className="h-4 w-4" />,
  control_console: <Command className="h-4 w-4" />,
  rcsb_search: <Search className="h-4 w-4" />,
  graph_topology: <GitBranch className="h-4 w-4" />,
  hypothesis: <Lightbulb className="h-4 w-4" />,
  data_tools: <Database className="h-4 w-4" />,
  provenance: <GitBranch className="h-4 w-4" />,
  plot_generator: <BarChart3 className="h-4 w-4" />,
  compare: <ArrowLeftRight className="h-4 w-4" />,
  data_inspector: <TableProperties className="h-4 w-4" />,
};

const BOOTSTRAP_STRUCTURE: Structure = {
  structure_id: "4uj1",
  pdb_id: "4uj1",
  title: "KRAS G12C GDP-bound structure",
  resolution: 1.768,
  method: "X-RAY DIFFRACTION",
  source: "rcsb",
  chains: ["A", "B"],
  residue_count: 730,
  has_embeddings: true,
  last_run_id: null,
  ingested_at: "2026-06-24T20:04:48.560743+00:00",
};

function formatPanelLabel(panel: ActivePanelName) {
  return panel ? panel.replace(/_/g, " ") : "inspector";
}

interface CockpitShellProps {
  activeStructure: Structure | null;
  activePanel: ActivePanelName;
  highlightedResidues: string[];
  setHighlightedResidues: (residues: string[]) => void;
  agentSessionId: string | null;
  setAgentSessionId: (sessionId: string | null) => void;
  poincareColorMode: PanelPoincareColorMode;
  setPoincareColorMode: (mode: PanelPoincareColorMode) => void;
  mobiusFocusEnabled: boolean;
  setMobiusFocusEnabled: (enabled: boolean) => void;
  selectedResidue: SelectedResidueInfo | null;
  setSelectedResidue: (residue: SelectedResidueInfo | null) => void;
  brushSelection: string[];
  setBrushSelection: (residues: string[]) => void;
  molecularColorMode: MolecularColorMode;
  setMolecularColorMode: (mode: MolecularColorMode) => void;
  emitDirective: (directive: ViewportDirective) => void;
  triggerRefresh: () => void;
  orchestrator: ReturnType<typeof useOrchestratorPolicy>;
  viewportState: { context: any };
  sendViewport: (event: any) => void;
  latestToast: string | null;
  connectionStatus: "connected" | "disconnected" | "reconnecting";
  sidebarOpen: boolean;
  activeEditorTab: EditorTabId;
  bottomPanelOpen: boolean;
  activeBottomPanel: BottomPanelId;
}

function CockpitShell({
  activeStructure,
  activePanel,
  highlightedResidues,
  setHighlightedResidues,
  setAgentSessionId,
  poincareColorMode,
  setPoincareColorMode,
  mobiusFocusEnabled,
  setMobiusFocusEnabled,
  selectedResidue,
  setSelectedResidue,
  brushSelection,
  setBrushSelection,
  molecularColorMode,
  setMolecularColorMode,
  emitDirective,
  orchestrator,
  viewportState,
  sendViewport,
  latestToast,
  connectionStatus,
  sidebarOpen,
  activeEditorTab,
  bottomPanelOpen,
  activeBottomPanel,
}: CockpitShellProps) {
  const {
    hydration,
    embeddings,
    sourceLeaks,
    hypotheses,
    provenanceRuns,
    annotations,
  } = useHydration();
  const panelBroker = usePanelBroker();

  const handlePanelSelect = useCallback(
    (panel: string | null) => {
      const nextPanel = panel as ActivePanelName;
      panelBroker.emitPort(TOOL_PANEL_REGISTRY.rcsb_search.manifest, "open-panel", {
        panelId: nextPanel,
        open: nextPanel !== null,
      });
    },
    [panelBroker],
  );

  const handleEditorTabSelect = useCallback(
    (tab: EditorTabId) => {
      panelBroker.emitPort(BOTTOM_PANEL_REGISTRY.results.manifest, "open-editor", {
        tab,
      });
    },
    [panelBroker],
  );

  const handleBottomPanelSelect = useCallback(
    (panel: BottomPanelId) => {
      panelBroker.emitPort(TOOL_PANEL_REGISTRY.provenance.manifest, "open-bottom-panel", {
        panelId: panel,
        open: true,
      });
    },
    [panelBroker],
  );

  const toggleSidebar = useCallback(() => {
    panelBroker.emitPort(TOOL_PANEL_REGISTRY.data_inspector.manifest, "toggle", {
      target: "sidebar",
      value: !sidebarOpen,
    });
  }, [panelBroker, sidebarOpen]);

  const toggleBottomPanel = useCallback(() => {
    panelBroker.emitPort(BOTTOM_PANEL_REGISTRY.telemetry.manifest, "toggle", {
      target: "bottom-panel",
      value: !bottomPanelOpen,
    });
  }, [bottomPanelOpen, panelBroker]);

  const mergeLayoutModel = useCallback(
    (partial: IdeShellLayoutModel) => {
      sendViewport({
        type: "SET_LAYOUT_MODEL",
        modelJSON: {
          ...(viewportState.context.layoutModelJSON ?? {}),
          ...partial,
        },
      });
    },
    [sendViewport, viewportState.context.layoutModelJSON],
  );

  const handleLayoutModelChange = useCallback(
    (partial: IdeShellLayoutModel) => {
      panelBroker.emitPort(TOOL_PANEL_REGISTRY.data_tools.manifest, "layout", {
        partialLayout: partial,
      });
    },
    [panelBroker],
  );

  const viewportStateBuilder = useCallback((): ViewportState => {
    return buildViewportState({
      structureId: activeStructure?.structure_id ?? null,
      structureTitle: activeStructure?.title ?? null,
      poincareColorMode: viewportState.context.poincareColorMode,
      mobiusFocusEnabled,
      mobiusFocusResidue: selectedResidue?.residue_id ?? null,
      selectedResidue,
      brushSelectedIds: brushSelection,
      viewerColorMode: viewportState.context.viewerColorMode,
      riskThreshold: viewportState.context.riskThreshold,
      highlightedResidueIds: viewportState.context.highlightedResidues,
      residueCount: embeddings?.residues?.length ?? 0,
      sourceLeakCount: sourceLeaks?.leaks?.length ?? 0,
      hypothesisCount: hypotheses?.length ?? 0,
      provenanceRunCount: provenanceRuns?.length ?? 0,
      annotationCount: annotations?.length ?? 0,
      activePanel,
      pipelineStatus: activeStructure
        ? hydration?.persistence_status?.embeddings_persisted
          ? "complete"
          : "running"
        : "never_run",
      pipelineCurrentStep: null,
      pipelineProgress: null,
    });
  }, [
    activeStructure,
    activePanel,
    annotations,
    brushSelection,
    embeddings,
    hydration,
    hypotheses,
    mobiusFocusEnabled,
    provenanceRuns,
    selectedResidue,
    sourceLeaks,
    viewportState.context,
  ]);

  const handleResidueClick = useCallback(
    (residueId: string) => {
      setHighlightedResidues([residueId]);
    },
    [setHighlightedResidues],
  );

  const sidebarDefinition = sidebarOpen ? getToolPanelDefinition(activePanel) : null;

  const sidebarPane: IdeShellPane | null = sidebarDefinition
    ? {
        id: activePanel!,
        title: sidebarDefinition.title,
        accent: "slate",
        content: <ToolPanelHost activePanel={activePanel} />,
      }
    : null;

  const editorPane: IdeShellPane =
    activeEditorTab === "agent"
      ? {
          id: "agent",
          title: "Agent Session",
          accent: "slate",
          content: (
            <ChatRail
              structureId={activeStructure?.structure_id ?? null}
              pdbId={activeStructure?.pdb_id ?? null}
              discoveryPhase={orchestrator.discoveryContext.phase}
              hypothesisLifecycle={orchestrator.hypothesisContext.stateLabel}
              viewportStateBuilder={viewportStateBuilder}
              onDirective={emitDirective}
              onSessionId={setAgentSessionId}
            />
          ),
        }
      : activeEditorTab === "manifold"
        ? {
            id: "manifold-editor",
            title: "Poincaré Workspace",
            accent: "magenta",
            content: (
              <PoincarePanel
                structureId={activeStructure?.structure_id ?? null}
                colorMode={poincareColorMode}
                selectedResidue={selectedResidue}
                highlightedResidues={viewportState.context.highlightedResidues}
                mobiusFocusEnabled={mobiusFocusEnabled}
                onResidueClick={handleResidueClick}
                onColorModeChange={setPoincareColorMode}
                onBrushSelect={setBrushSelection}
                onMobiusFocusToggle={setMobiusFocusEnabled}
              />
            ),
          }
        : {
            id: "structure-editor",
            title: "3D Structure",
            accent: "teal",
            content: (
              <MolecularPanel
                structureId={activeStructure?.structure_id ?? null}
                colorMode={molecularColorMode}
                selectedResidue={selectedResidue}
                highlightedResidues={viewportState.context.highlightedResidues}
                onResidueClick={handleResidueClick}
                onColorModeChange={setMolecularColorMode}
              />
            ),
          };

  const bottomPanelDefinition = getBottomPanelDefinition(activeBottomPanel);

  return (
    <>
      <IdeShellLayout
        titleBar={
          <div className="flex h-8 items-center justify-between border-b border-slate bg-[#0b0d12] px-3">
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-error/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-warning/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-success/80" />
              </div>
              <span className="ml-2 text-[10px] font-display uppercase tracking-[0.28em] text-text-secondary">
                Tokyo Eye Workspace
              </span>
            </div>
            <div className="flex items-center gap-3 text-[10px] text-text-muted">
              <span>mode:{orchestrator.sessionMode}</span>
              <span>phase:{orchestrator.discoveryContext.phase}</span>
              <span>{activeStructure?.pdb_id?.toUpperCase() ?? "no-structure"}</span>
            </div>
          </div>
        }
        header={
          <CockpitNavBar
            structureId={activeStructure?.structure_id ?? null}
            pdbId={activeStructure?.pdb_id ?? null}
            discoveryPhase={orchestrator.discoveryContext.phase}
            hypothesisLifecycle={orchestrator.hypothesisContext.stateLabel}
            connectionStatus={connectionStatus}
          />
        }
        activityBar={
          <ToolDock
            pinnedTools={["briefing", "control_console"]}
            allowedTools={orchestrator.plannerPolicy.allowedTools}
            blockedTools={orchestrator.plannerPolicy.blockedTools}
            activePanel={activePanel}
            onPanelSelect={handlePanelSelect}
            mode="activity-bar"
            toolIcons={TOOL_ICONS}
          />
        }
        workspaceBar={
          <div className="flex flex-col gap-2">
            <VisualizationToolbar
              selectedResidues={highlightedResidues}
              onDirective={emitDirective}
            />
            <CompareIndicatorBar />
          </div>
        }
        editorTabs={
          <div className="flex items-center gap-1 overflow-x-auto py-1.5">
            {Object.values(EDITOR_TAB_REGISTRY).map((tab) => {
              const active = tab.id === activeEditorTab;
              return (
                <button
                  key={tab.id}
                  onClick={() => handleEditorTabSelect(tab.id)}
                  className={`rounded-t-[var(--radius-badge)] border border-b-0 px-3 py-1 text-[10px] uppercase tracking-[0.18em] transition-colors ${
                    active
                      ? "border-teal-dim/50 bg-bg-elevated text-teal"
                      : "border-slate-light bg-bg text-text-secondary hover:text-text-primary"
                  }`}
                >
                  {tab.title}
                </button>
              );
            })}
          </div>
        }
        sidebarPane={sidebarPane}
        leftPane={{
          id: "poincare-navigator",
          title: "Manifold Navigator",
          accent: "magenta",
          content: (
            <PoincarePanel
              structureId={activeStructure?.structure_id ?? null}
              colorMode={poincareColorMode}
              selectedResidue={selectedResidue}
              highlightedResidues={viewportState.context.highlightedResidues}
              mobiusFocusEnabled={mobiusFocusEnabled}
              onResidueClick={handleResidueClick}
              onColorModeChange={setPoincareColorMode}
              onBrushSelect={setBrushSelection}
              onMobiusFocusToggle={setMobiusFocusEnabled}
            />
          ),
        }}
        centerPane={editorPane}
        rightPane={null}
        bottomPane={
          bottomPanelOpen
            ? {
                id: bottomPanelDefinition.id,
                title: bottomPanelDefinition.title,
                accent: "slate",
                content: (
                  <BottomPanelHost
                    activePanel={activeBottomPanel}
                    onSelectPanel={handleBottomPanelSelect}
                  />
                ),
              }
            : null
        }
        statusBar={
          <div className="flex h-6 items-center justify-between border-t border-teal-dim/30 bg-teal-dim/20 px-3 text-[10px] text-text-secondary">
            <div className="flex items-center gap-3">
              <span>{connectionStatus}</span>
              <span>{activeStructure?.structure_id ?? "no structure loaded"}</span>
              <span>{hydration?.structure_snapshot?.provenance.latest_run_ids_by_pipeline?.embeddings ? "snapshot live" : "legacy/snapshot mixed"}</span>
            </div>
            <div className="flex items-center gap-3">
              <button onClick={toggleSidebar} className="hover:text-text-primary">
                sidebar:{sidebarOpen ? "open" : "closed"}
              </button>
              <button onClick={toggleBottomPanel} className="hover:text-text-primary">
                panel:{bottomPanelOpen ? activeBottomPanel : "closed"}
              </button>
              <span>editor:{activeEditorTab}</span>
              <span>panel:{formatPanelLabel(activePanel)}</span>
              <span>residues:{embeddings?.residues?.length ?? 0}</span>
              <span>leaks:{sourceLeaks?.leaks?.length ?? 0}</span>
            </div>
          </div>
        }
        onLayoutModelChange={handleLayoutModelChange}
      />

      {latestToast ? (
        <div className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2 rounded-[var(--radius-card)] border border-slate-light bg-bg-elevated px-4 py-2 text-xs text-text-secondary shadow-lg animate-[fadeIn_0.2s_ease-out]">
          {latestToast}
        </div>
      ) : null}
    </>
  );
}

export default function AppCockpit() {
  const [activeStructure, setActiveStructure] = useState<Structure | null>(
    BOOTSTRAP_STRUCTURE,
  );
  const [chatOpen, setChatOpen] = useState(false);
  const [highlightedResidues, setHighlightedResidues] = useState<string[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [currentDirective, setCurrentDirective] = useState<ViewportDirective | null>(null);
  const [agentSessionId, setAgentSessionId] = useState<string | null>(null);
  const [poincareColorMode, setPoincareColorMode] = useState<PanelPoincareColorMode>("cone_depth");
  const [poincareSelectedResidue, setPoincareSelectedResidue] = useState<SelectedResidueInfo | null>(null);
  const [mobiusFocus, setMobiusFocus] = useState(false);
  const [brushSelection, setBrushSelection] = useState<string[]>([]);
  const [viewerColorMode, setViewerColorMode] = useState<StructureColorModeType>("spectrum");
  const [riskThreshold, setRiskThreshold] = useState(0);
  const [compareState, setCompareState] = useState<CompareState>({
    active: false,
    secondaryStructure: null,
    displacements: null,
    graphDiff: null,
    loading: false,
    error: null,
  });
  const [selectedPocketId, setSelectedPocketId] = useState<number | null>(null);
  const [isRadarActive, setIsRadarActive] = useState(false);
  const [therapeuticCompilerState, setTherapeuticCompilerState] = useState<any | null>(null);
  const [collapseSimulationState, setCollapseSimulationState] = useState({
    fraction: 0.0,
    goal: "Exploit isolated targets post-fragmentation",
    active: false,
  });
  const [latestAgentTelemetry, setLatestAgentTelemetry] = useState<AgentChatResponse["telemetry"] | null>(null);
  const [viewportState, sendViewport] = useActor(viewportMachine);
  const activePanel = viewportState.context.activePanel;
  const sidebarOpen = viewportState.context.sidebarOpen;
  const activeEditorTab = viewportState.context.activeEditorTab;
  const bottomPanelOpen = viewportState.context.bottomPanelOpen;
  const activeBottomPanel = viewportState.context.activeBottomPanel;
  const orchestrator = useOrchestratorPolicy(viewportState.context);
  const directiveState = useDirectives();

  const triggerRefresh = useCallback(() => {
    setRefreshKey((current) => current + 1);
  }, []);

  const emitDirective = useCallback(
    (directive: ViewportDirective) => {
      sendViewport({ type: "DIRECTIVE_RECEIVED", directive });
      directiveState.applyDirective(directive);
      setCurrentDirective(directive);

      if (directive.action === "highlight" || directive.action === "focus") {
        const residues = directive.highlight_groups?.flatMap((group) => group.residue_ids) ?? [];
        if (residues.length > 0) {
          setHighlightedResidues(residues);
        }
      } else if (directive.action === "clear") {
        setHighlightedResidues([]);
        setIsRadarActive(false);
      }
    },
    [directiveState, sendViewport],
  );

  const selectionSync = useSelectionSync({
    sendEvent: (event) => {
      wsSendEventRef.current?.(event);
    },
    onSelectionApplied: (residueIds) => {
      sendViewport({ type: "USER_SELECT", residues: residueIds });
      setHighlightedResidues(residueIds);
    },
    onSelectionCleared: () => {
      sendViewport({ type: "CLEAR" });
      setHighlightedResidues([]);
    },
  });

  const handleStateSnapshot = useCallback(
    (snapshot: StateSnapshotPayload) => {
      const phase = snapshot.discovery_phase as DiscoveryPhase;
      if (phase && phase !== orchestrator.discoveryContext.phase) {
        orchestrator.sendDiscovery({
          type: "USER_SET_PHASE",
          phase,
          rationale: "backend_push",
        });
      }

      const lifecycle = snapshot.hypothesis_lifecycle as HypothesisLifecycleState;
      if (lifecycle && lifecycle !== orchestrator.hypothesisContext.stateLabel) {
        if (lifecycle === "framed") {
          orchestrator.sendHypothesis({
            type: "START_HYPOTHESIS",
            hypothesisText: "",
            source: "agent",
          });
        } else if (lifecycle === "supported") {
          orchestrator.sendHypothesis({ type: "MARK_SUPPORTED" });
        } else if (lifecycle === "synthesized") {
          orchestrator.sendHypothesis({ type: "MARK_SYNTHESIZED" });
        }
      }

      if (snapshot.selected_residue) {
        const residueIds = [
          `${snapshot.selected_residue.chain_id}:${snapshot.selected_residue.residue_number}`,
        ];
        sendViewport({ type: "USER_SELECT", residues: residueIds });
        setHighlightedResidues(residueIds);
      }
    },
    [orchestrator, sendViewport],
  );

  const handlePhaseTransition = useCallback(
    (payload: PhaseTransitionPayload) => {
      const phase = payload.phase as DiscoveryPhase;
      if (phase) {
        orchestrator.sendDiscovery({
          type: "USER_SET_PHASE",
          phase,
          rationale: `backend:${payload.source}`,
        });
      }
    },
    [orchestrator],
  );

  const { connected, reconnecting, sendEvent } = useViewportSocket({
    onDirective: emitDirective,
    onStateSnapshot: handleStateSnapshot,
    onPhaseTransition: handlePhaseTransition,
    onSelectionSync: selectionSync.handleSelectionSync,
    onSelectionError: selectionSync.handleSelectionError,
    enabled: true,
    sessionId: agentSessionId,
  });

  const wsSendEventRef = useRef(sendEvent);
  wsSendEventRef.current = sendEvent;

  const connectionStatus: "connected" | "disconnected" | "reconnecting" =
    connected ? "connected" : reconnecting ? "reconnecting" : "disconnected";

  const enterCompareMode = useCallback(
    (secondary: Structure) => {
      if (!activeStructure || !activeStructure.has_embeddings || !secondary.has_embeddings) {
        return;
      }

      setCompareState({
        active: true,
        secondaryStructure: secondary,
        displacements: null,
        graphDiff: null,
        loading: true,
        error: null,
      });

      Promise.all([
        api.compareEmbeddings(activeStructure.structure_id, secondary.structure_id).catch(() => null),
        api.compareGraphsDirect(activeStructure.structure_id, secondary.structure_id).catch(() => null),
      ]).then(([displacements, graphDiff]) => {
        setCompareState((previous) => ({
          ...previous,
          displacements: displacements ?? null,
          graphDiff: graphDiff ?? null,
          loading: false,
          error: !displacements && !graphDiff ? "Failed to load comparison data" : null,
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

  const latestToast = selectionSync.toastMessage || directiveState.lastMessage;

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
        poincareColorMode: poincareColorMode as PoincareColorMode,
        viewerColorMode,
        riskThreshold,
        activePanel,
        sidebarOpen,
        activeEditorTab,
        bottomPanelOpen,
        activeBottomPanel,
        layoutModelJSON: viewportState.context.layoutModelJSON,
        userSelectedResidue: viewportState.context.userSelectedResidue,
        sendViewport,
      }}
    >
      <HydrationProvider>
        <PanelBrokerProvider
          sendViewport={sendViewport}
          emitDirective={emitDirective}
          mergeLayoutModel={(partialLayout) => {
            sendViewport({
              type: "SET_LAYOUT_MODEL",
              modelJSON: {
                ...(viewportState.context.layoutModelJSON ?? {}),
                ...partialLayout,
              },
            });
          }}
        >
          <CockpitShell
            activeStructure={activeStructure}
            activePanel={activePanel}
            highlightedResidues={highlightedResidues}
            setHighlightedResidues={setHighlightedResidues}
            agentSessionId={agentSessionId}
            setAgentSessionId={setAgentSessionId}
            poincareColorMode={poincareColorMode}
            setPoincareColorMode={setPoincareColorMode}
            mobiusFocusEnabled={mobiusFocus}
            setMobiusFocusEnabled={setMobiusFocus}
            selectedResidue={poincareSelectedResidue}
            setSelectedResidue={setPoincareSelectedResidue}
            brushSelection={brushSelection}
            setBrushSelection={setBrushSelection}
            molecularColorMode={viewerColorMode as MolecularColorMode}
            setMolecularColorMode={(mode) => setViewerColorMode(mode as StructureColorModeType)}
            emitDirective={emitDirective}
            triggerRefresh={triggerRefresh}
            orchestrator={orchestrator}
            viewportState={viewportState}
            sendViewport={sendViewport}
            latestToast={latestToast}
            connectionStatus={connectionStatus}
            sidebarOpen={sidebarOpen}
            activeEditorTab={activeEditorTab}
            bottomPanelOpen={bottomPanelOpen}
            activeBottomPanel={activeBottomPanel}
          />
        </PanelBrokerProvider>
      </HydrationProvider>
    </DashboardContext.Provider>
  );
}
