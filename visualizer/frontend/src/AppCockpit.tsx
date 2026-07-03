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
import { StructurePickerModal } from "./components/structure/StructurePickerModal";
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
import { selectAgentViewportContext } from "./lib/selectAgentViewportContext";
import { selectedResidueFromId } from "./lib/residueId";
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
import {
  WorkbenchProvider,
  useWorkbench,
  WorkbenchCockpitLayout,
  applyBackendSnapshot,
  snapshotFingerprint,
  useWorkbenchPolicy,
  useStructureScopeSync,
  discoveryPhaseToGroup,
} from "./workbench";
import type { WorkbenchBus } from "./workbench/WorkbenchBus";

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
  title: "Structure 4UJ1",
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
  compareState: CompareState;
  agentSessionId: string;
  setAgentSessionId: (sessionId: string) => void;
  onDiscoveryPhaseFromBackend?: (phase: DiscoveryPhase) => void;
  emitDirective: (directive: ViewportDirective) => void;
  orchestrator: ReturnType<typeof useOrchestratorPolicy>;
  viewportState: { context: import("./lib/viewportMachine").ViewportContext };
  sendViewport: (event: any) => void;
  latestToast: string | null;
  connectionStatus: "connected" | "disconnected" | "reconnecting";
  workbenchBus?: WorkbenchBus;
  useWorkbenchLayout?: boolean;
  onOpenStructurePicker: () => void;
}

function CockpitShell({
  activeStructure,
  compareState,
  agentSessionId,
  setAgentSessionId,
  onDiscoveryPhaseFromBackend,
  emitDirective,
  orchestrator,
  viewportState,
  sendViewport,
  latestToast,
  connectionStatus,
  workbenchBus,
  useWorkbenchLayout = false,
  onOpenStructurePicker,
}: CockpitShellProps) {
  const vp = viewportState.context;
  const activePanel = vp.activePanel;
  const sidebarOpen = vp.sidebarOpen;
  const activeEditorTab = vp.activeEditorTab;
  const bottomPanelOpen = vp.bottomPanelOpen;
  const activeBottomPanel = vp.activeBottomPanel;
  const highlightedResidues = vp.highlightedResidues;
  const selectedResidue = vp.selectedResidue;
  const brushSelection = vp.brushSelectedIds;
  const mobiusFocusEnabled = vp.mobiusFocusEnabled;
  const poincareColorMode = vp.poincareColorMode as PanelPoincareColorMode;
  const molecularColorMode = vp.viewerColorMode as MolecularColorMode;

  const {
    hydration,
    embeddings,
    sourceLeaks,
    resistanceData,
    hypotheses,
    provenanceRuns,
    annotations,
    pharmacophorePockets,
    drugCandidates,
    persistenceStatus,
  } = useHydration();
  const panelBroker = usePanelBroker();

  const setPoincareColorMode = useCallback(
    (mode: PanelPoincareColorMode) => {
      sendViewport({ type: "SET_POINCARE_COLOR_MODE", mode });
    },
    [sendViewport],
  );

  const setMolecularColorMode = useCallback(
    (mode: MolecularColorMode) => {
      sendViewport({ type: "SET_VIEWER_COLOR_MODE", mode });
    },
    [sendViewport],
  );

  const setBrushSelection = useCallback(
    (residues: string[]) => {
      sendViewport({ type: "SET_BRUSH_SELECTION", residues });
    },
    [sendViewport],
  );

  const setMobiusFocusEnabled = useCallback(
    (enabled: boolean) => {
      sendViewport({ type: "SET_MOBIUS_FOCUS", enabled });
    },
    [sendViewport],
  );

  useStructureScopeSync({
    activeStructure,
    hydration,
    orchestrator,
    workbenchBus,
    enabled: Boolean(workbenchBus),
  });

  const handlePanelSelect = useCallback(
    (panel: string | null) => {
      if (panel === "rcsb_search") {
        onOpenStructurePicker();
        return;
      }
      const nextPanel = panel as ActivePanelName;
      panelBroker.emitPort(TOOL_PANEL_REGISTRY.rcsb_search.manifest, "open-panel", {
        panelId: nextPanel,
        open: nextPanel !== null,
      });
    },
    [onOpenStructurePicker, panelBroker],
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
    return selectAgentViewportContext({
      activeStructure,
      viewport: vp,
      hydrationSlice: {
        hydration,
        embeddings,
        sourceLeaks,
        resistanceData,
        hypotheses,
        provenanceRuns,
        annotations,
        pharmacophorePockets,
        drugCandidates,
        persistenceStatus,
      },
      structureScope: orchestrator.structureScope,
      compareState,
    });
  }, [
    activeStructure,
    annotations,
    compareState,
    drugCandidates,
    embeddings,
    hydration,
    hypotheses,
    orchestrator.structureScope,
    persistenceStatus,
    pharmacophorePockets,
    provenanceRuns,
    resistanceData,
    sourceLeaks,
    vp,
  ]);

  const handleResidueClick = useCallback(
    (residueId: string) => {
      const structureId = activeStructure?.structure_id ?? null;
      const residues = embeddings?.residues ?? [];
      const selection = selectedResidueFromId(residueId, structureId, residues);
      const canonicalId = selection.residue_id;

      sendViewport({
        type: "SET_SELECTION",
        highlightedResidues: [canonicalId],
        brushSelectedIds: [canonicalId],
        selectedResidue: selection,
      });
    },
    [activeStructure?.structure_id, embeddings?.residues, sendViewport],
  );

  const handleViewportColorModeChange = useCallback(
    (mode: PanelPoincareColorMode) => {
      sendViewport({ type: "SET_POINCARE_COLOR_MODE", mode });
      sendViewport({ type: "SET_VIEWER_COLOR_MODE", mode });
    },
    [sendViewport],
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
              sessionId={agentSessionId}
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

  if (useWorkbenchLayout) {
    return (
      <>
        <WorkbenchCockpitLayout
          activeStructureId={activeStructure?.structure_id ?? null}
          pdbId={activeStructure?.pdb_id ?? null}
          discoveryPhase={orchestrator.discoveryContext.phase}
          hypothesisLifecycle={orchestrator.hypothesisContext.stateLabel}
          highlightedResidues={highlightedResidues}
          poincareColorMode={poincareColorMode}
          molecularColorMode={molecularColorMode}
          selectedResidue={selectedResidue}
          mobiusFocusEnabled={mobiusFocusEnabled}
          viewportStateBuilder={viewportStateBuilder}
          onResidueClick={handleResidueClick}
          onColorModeChange={handleViewportColorModeChange}
          onBrushSelect={setBrushSelection}
          onMobiusFocusToggle={setMobiusFocusEnabled}
          onMolecularColorModeChange={setMolecularColorMode}
          onDirective={emitDirective}
          onSessionId={setAgentSessionId}
          onDiscoveryPhase={onDiscoveryPhaseFromBackend ?? ((phase) =>
            orchestrator.sendDiscovery({ type: "USER_SET_PHASE", phase })
          )}
          sessionId={agentSessionId}
          connectionStatus={connectionStatus}
          onPanelSelect={handlePanelSelect}
          activePanel={activePanel}
          allowedTools={orchestrator.plannerPolicy.allowedTools}
          blockedTools={orchestrator.plannerPolicy.blockedTools}
          toolIcons={TOOL_ICONS}
          onOpenStructurePicker={onOpenStructurePicker}
        />
        {latestToast ? (
          <div className="fixed bottom-4 right-4 z-50 rounded-lg border border-teal-dim/40 bg-bg-elevated px-4 py-2 text-sm text-teal shadow-lg">
            {latestToast}
          </div>
        ) : null}
      </>
    );
  }

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
              <span>
                {(hydration?.structure_snapshot as import("./lib/types").StructureAnalysisSnapshot | null)
                  ?.provenance.latest_run_ids_by_pipeline?.embeddings
                  ? "snapshot live"
                  : "legacy/snapshot mixed"}
              </span>
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
  const useLegacyShell = import.meta.env.VITE_USE_LEGACY_SHELL === "true";
  if (useLegacyShell) {
    return <AppCockpitCore />;
  }
  return (
    <WorkbenchProvider>
      <AppCockpitWithWorkbench />
    </WorkbenchProvider>
  );
}

function AppCockpitWithWorkbench() {
  const { bus } = useWorkbench();
  return <AppCockpitCore workbenchBus={bus} useWorkbenchLayout />;
}

function AppCockpitCore({
  workbenchBus,
  useWorkbenchLayout = false,
}: {
  workbenchBus?: WorkbenchBus;
  useWorkbenchLayout?: boolean;
} = {}) {
  const [activeStructure, setActiveStructure] = useState<Structure | null>(
    BOOTSTRAP_STRUCTURE,
  );
  const [chatOpen, setChatOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [currentDirective, setCurrentDirective] = useState<ViewportDirective | null>(null);
  const [agentSessionId, setAgentSessionId] = useState<string>(() => crypto.randomUUID());
  const lastSnapshotFingerprintRef = useRef<string>("");

  const handleAgentSessionId = useCallback((sessionId: string) => {
    setAgentSessionId((current) => (current === sessionId ? current : sessionId));
  }, []);

  const [compareState, setCompareState] = useState<CompareState>({
    active: false,
    secondaryStructure: null,
    displacements: null,
    graphDiff: null,
    loading: false,
    error: null,
  });

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
  const orchestrator = useOrchestratorPolicy(viewportState.context, {
    authority: workbenchBus ? "backend" : "local",
  });
  const activeOrchestrator = useWorkbenchPolicy(orchestrator, workbenchBus);
  const directiveState = useDirectives();

  const handleDiscoveryPhaseFromBackend = useCallback(
    (phase: DiscoveryPhase) => {
      if (phase === orchestrator.discoveryContext.phase) return;
      orchestrator.sendDiscovery({
        type: "USER_SET_PHASE",
        phase,
        rationale: "backend_sync",
      });
    },
    [orchestrator],
  );

  const triggerRefresh = useCallback(() => {
    setRefreshKey((current) => current + 1);
  }, []);

  const [structurePickerOpen, setStructurePickerOpen] = useState(false);

  const handleStructureLoaded = useCallback(
    (structure: Structure) => {
      setActiveStructure(structure);
      sendViewport({ type: "CLEAR" });
      triggerRefresh();
      setStructurePickerOpen(false);
    },
    [sendViewport, triggerRefresh],
  );

  const emitDirective = useCallback(
    (directive: ViewportDirective) => {
      sendViewport({ type: "DIRECTIVE_RECEIVED", directive });
      directiveState.applyDirective(directive);
      setCurrentDirective(directive);

      if (directive.action === "clear") {
        sendViewport({ type: "TOGGLE_RADAR", active: false });
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
    },
    onSelectionCleared: () => {
      sendViewport({ type: "CLEAR" });
    },
  });

  const handleStateSnapshot = useCallback(
    (snapshot: StateSnapshotPayload) => {
      const fingerprint = snapshotFingerprint(snapshot);
      if (fingerprint === lastSnapshotFingerprintRef.current) {
        return;
      }
      lastSnapshotFingerprintRef.current = fingerprint;

      if (workbenchBus) {
        applyBackendSnapshot(workbenchBus, snapshot, handleDiscoveryPhaseFromBackend);
      } else {
        const phase = snapshot.discovery_phase as DiscoveryPhase;
        if (phase && phase !== orchestrator.discoveryContext.phase) {
          orchestrator.sendDiscovery({
            type: "USER_SET_PHASE",
            phase,
            rationale: "backend_push",
          });
        }
      }

      const lifecycle = snapshot.hypothesis_lifecycle as HypothesisLifecycleState;
      orchestrator.applyHypothesisFromBackend(lifecycle);

      if (snapshot.selected_residue) {
        const residueIds = [
          `${snapshot.selected_residue.chain_id}:${snapshot.selected_residue.residue_number}`,
        ];
        sendViewport({ type: "USER_SELECT", residues: residueIds });
      }
    },
    [handleDiscoveryPhaseFromBackend, orchestrator, sendViewport, workbenchBus],
  );

  const handlePhaseTransition = useCallback(
    (payload: PhaseTransitionPayload) => {
      const phase = payload.phase as DiscoveryPhase;
      if (!phase) return;

      if (workbenchBus) {
        workbenchBus.publish("system:phase_transition", {
          group: discoveryPhaseToGroup(phase),
          discoveryPhase: phase,
          timestamp: new Date().toISOString(),
        });
      }

      handleDiscoveryPhaseFromBackend(phase);
    },
    [handleDiscoveryPhaseFromBackend, workbenchBus],
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
        highlightedResidues: viewportState.context.highlightedResidues,
        setHighlightedResidues: (residues: string[]) => {
          sendViewport({ type: "USER_SELECT", residues });
        },
        refreshKey,
        triggerRefresh,
        emitDirective,
        currentDirective,
        compareState,
        enterCompareMode,
        exitCompareMode,
        selectedPocketId: viewportState.context.selectedPocketId,
        setSelectedPocketId: (id: number | null) => {
          sendViewport({ type: "SET_SELECTED_POCKET", pocketId: id });
        },
        isRadarActive: viewportState.context.isRadarActive,
        setIsRadarActive: (active: boolean) => {
          sendViewport({ type: "TOGGLE_RADAR", active });
        },
        therapeuticCompilerState,
        setTherapeuticCompilerState,
        collapseSimulationState,
        setCollapseSimulationState,
        latestAgentTelemetry,
        setLatestAgentTelemetry,
        agentSessionId,
        setAgentSessionId: handleAgentSessionId,
        discoveryContext: activeOrchestrator.discoveryContext,
        sendDiscovery: activeOrchestrator.sendDiscovery,
        hypothesisContext: activeOrchestrator.hypothesisContext,
        sendHypothesis: activeOrchestrator.sendHypothesis,
        plannerPolicy: activeOrchestrator.plannerPolicy,
        sessionMode: activeOrchestrator.sessionMode,
        setSessionMode: activeOrchestrator.setSessionMode,
        structureScope: activeOrchestrator.structureScope,
        setStructureScope: activeOrchestrator.setStructureScope,
        poincareColorMode: viewportState.context.poincareColorMode,
        viewerColorMode: viewportState.context.viewerColorMode,
        riskThreshold: viewportState.context.riskThreshold,
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
          workbenchBus={workbenchBus}
          activePhaseGroup={discoveryPhaseToGroup(
            activeOrchestrator.discoveryContext.phase,
          )}
        >
          <CockpitShell
            activeStructure={activeStructure}
            compareState={compareState}
            agentSessionId={agentSessionId}
            setAgentSessionId={handleAgentSessionId}
            onDiscoveryPhaseFromBackend={handleDiscoveryPhaseFromBackend}
            useWorkbenchLayout={useWorkbenchLayout}
            emitDirective={emitDirective}
            orchestrator={activeOrchestrator}
            viewportState={viewportState}
            sendViewport={sendViewport}
            latestToast={latestToast}
            connectionStatus={connectionStatus}
            workbenchBus={workbenchBus}
            onOpenStructurePicker={() => setStructurePickerOpen(true)}
          />
          <StructurePickerModal
            open={structurePickerOpen}
            onClose={() => setStructurePickerOpen(false)}
            onStructureLoaded={handleStructureLoaded}
            initialTab="rcsb"
          />
        </PanelBrokerProvider>
      </HydrationProvider>
    </DashboardContext.Provider>
  );
}
