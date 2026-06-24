import { useState, useCallback, useRef } from "react";
import { CockpitLayout, CockpitNavBar, ToolDock } from "./components/layout";
import { ChatRail } from "./components/chat";
import { PoincarePanel, MolecularPanel } from "./components/viewers";
import { StructureOnboard } from "./components/onboard";
import type { PanelPoincareColorMode } from "./components/viewers";
import type { MolecularColorMode } from "./components/viewers";
import { useViewportSocket } from "./lib/useViewportSocket";
import type { StateSnapshotPayload, PhaseTransitionPayload } from "./lib/useViewportSocket";
import { buildViewportState } from "./lib/buildViewportState";
import { useSelectionSync } from "./lib/useSelectionSync";
import { useDirectives } from "./lib/useDirectives";
import { useActor } from "@xstate/react";
import { viewportMachine } from "./lib/viewportMachine";
import { useOrchestratorPolicy } from "./lib/useOrchestratorPolicy";
import type {
  Structure,
  ViewportDirective,
  ViewportState,
  SelectedResidueInfo,
} from "./lib/types";
import type { DiscoveryPhase } from "./lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "./lib/hypothesisLifecycleMachine";

/**
 * AppCockpit — Discovery Cockpit root wired to real state machines
 *
 * All phase/lifecycle/tool state is derived from:
 *   - viewportMachine (XState) for viewport context
 *   - useOrchestratorPolicy for discovery phase, hypothesis lifecycle, planner policy
 *   - useViewportSocket for backend WS push (directives, state_snapshot)
 *
 * Requirements: 1.1, 5.1, 5.2, 5.3, 5.6
 */

export default function AppCockpit() {
  const [activeStructure, setActiveStructure] = useState<Structure | null>(null);
  const [activePanel, setActivePanel] = useState<string | null>(null);
  const [agentSessionId, setAgentSessionId] = useState<string | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(true);

  // Viewer panel state
  const [poincareColorMode, setPoincareColorMode] = useState<PanelPoincareColorMode>("cone_depth");
  const [mobiusFocusEnabled, setMobiusFocusEnabled] = useState(false);
  const [selectedResidue, setSelectedResidue] = useState<SelectedResidueInfo | null>(null);
  const [brushSelection, setBrushSelection] = useState<string[]>([]);
  const [molecularColorMode, setMolecularColorMode] = useState<MolecularColorMode>("spectrum");

  // XState viewport machine — single source of viewport truth
  const [viewportState, sendViewport] = useActor(viewportMachine);

  // Orchestrator policy — derives discovery phase, hypothesis lifecycle, tool gating
  const orchestrator = useOrchestratorPolicy(viewportState.context);

  // Directive handler — feeds into viewport machine + useDirectives
  const directiveState = useDirectives();

  const emitDirective = useCallback(
    (directive: ViewportDirective) => {
      sendViewport({ type: "DIRECTIVE_RECEIVED", directive });
      directiveState.applyDirective(directive);

      // Handle set_metric: update Poincaré color mode
      if (directive.action === "set_metric" && directive.metric) {
        setPoincareColorMode(directive.metric as PanelPoincareColorMode);
      }
    },
    [sendViewport, directiveState],
  );

  // Selection sync — cross-viewport selection propagation via backend
  const selectionSync = useSelectionSync({
    sendEvent: (event) => {
      // Will be set after useViewportSocket initializes — use ref pattern below
      wsSendEventRef.current?.(event);
    },
    onSelectionApplied: (residueIds) => {
      sendViewport({ type: "USER_SELECT", residues: residueIds });
    },
    onSelectionCleared: () => {
      sendViewport({ type: "CLEAR" });
    },
  });

  // State snapshot handler — derives phase/lifecycle from backend push
  const handleStateSnapshot = useCallback(
    (snapshot: StateSnapshotPayload) => {
      // Update discovery phase from backend
      const phase = snapshot.discovery_phase as DiscoveryPhase;
      if (phase && phase !== orchestrator.discoveryContext.phase) {
        orchestrator.sendDiscovery({ type: "USER_SET_PHASE", phase, rationale: "backend_push" });
      }

      // Update hypothesis lifecycle from backend
      const lifecycle = snapshot.hypothesis_lifecycle as HypothesisLifecycleState;
      if (lifecycle && lifecycle !== orchestrator.hypothesisContext.stateLabel) {
        // The hypothesis machine events map lifecycle states to transitions
        if (lifecycle === "framed") {
          orchestrator.sendHypothesis({ type: "START_HYPOTHESIS", hypothesisText: "", source: "agent" });
        } else if (lifecycle === "supported") {
          orchestrator.sendHypothesis({ type: "MARK_SUPPORTED" });
        } else if (lifecycle === "synthesized") {
          orchestrator.sendHypothesis({ type: "MARK_SYNTHESIZED" });
        }
      }

      // Update selected residue if present
      if (snapshot.selected_residue) {
        const residueIds = [`${snapshot.selected_residue.chain_id}:${snapshot.selected_residue.residue_number}`];
        sendViewport({ type: "USER_SELECT", residues: residueIds });
      }
    },
    [orchestrator, sendViewport],
  );

  // Phase transition handler — updates discovery phase from backend event
  const handlePhaseTransition = useCallback(
    (payload: PhaseTransitionPayload) => {
      const phase = payload.phase as DiscoveryPhase;
      if (phase) {
        orchestrator.sendDiscovery({ type: "USER_SET_PHASE", phase, rationale: `backend:${payload.source}` });
      }
    },
    [orchestrator],
  );

  // WebSocket connection — receives backend state push + directives + selection sync
  const { connected, reconnecting, sendEvent } = useViewportSocket({
    onDirective: emitDirective,
    onStateSnapshot: handleStateSnapshot,
    onPhaseTransition: handlePhaseTransition,
    onSelectionSync: selectionSync.handleSelectionSync,
    onSelectionError: selectionSync.handleSelectionError,
    enabled: true,
    sessionId: agentSessionId,
  });

  // Wire the sendEvent ref for selectionSync (avoids circular dep in hook init)
  const wsSendEventRef = useRef(sendEvent);
  wsSendEventRef.current = sendEvent;

  // Derive connection status from the hook's state
  const connectionStatus: "connected" | "disconnected" | "reconnecting" =
    connected ? "connected" : reconnecting ? "reconnecting" : "disconnected";

  // Derived state from orchestrator policy (backend-driven via machines)
  const discoveryPhase = orchestrator.discoveryContext.phase;
  const hypothesisLifecycle = orchestrator.hypothesisContext.stateLabel;
  const allowedTools = orchestrator.plannerPolicy.allowedTools;
  const blockedTools = orchestrator.plannerPolicy.blockedTools;

  // Viewport state builder — assembles payload for chat context enrichment
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
      residueCount: 0,
      sourceLeakCount: 0,
      hypothesisCount: 0,
      provenanceRunCount: 0,
      annotationCount: 0,
      activePanel: activePanel as any,
      pipelineStatus: activeStructure ? "complete" : "never_run",
      pipelineCurrentStep: null,
      pipelineProgress: null,
    });
  }, [activeStructure, viewportState.context, activePanel, mobiusFocusEnabled, selectedResidue, brushSelection]);

  // Residue click handler — emits selection through WS for cross-viewport sync
  const handleResidueClick = useCallback(
    (residueId: string) => {
      const structureId = activeStructure?.structure_id ?? "";
      selectionSync.emitSelection(residueId, structureId);
    },
    [activeStructure, selectionSync],
  );

  return (
    <>
      <CockpitLayout
      navBar={
        <CockpitNavBar
          structureId={activeStructure?.structure_id ?? null}
          pdbId={activeStructure?.pdb_id ?? null}
          discoveryPhase={discoveryPhase}
          hypothesisLifecycle={hypothesisLifecycle}
          connectionStatus={connectionStatus}
          onStructureSearch={() => setShowOnboarding(true)}
        />
      }
      toolDock={
        <ToolDock
          allowedTools={allowedTools}
          blockedTools={blockedTools}
          activePanel={activePanel}
          onPanelSelect={setActivePanel}
        />
      }
      leftPanel={
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
      }
      centerPanel={
        <MolecularPanel
          structureId={activeStructure?.structure_id ?? null}
          colorMode={molecularColorMode}
          selectedResidue={selectedResidue}
          highlightedResidues={viewportState.context.highlightedResidues}
          onResidueClick={handleResidueClick}
          onColorModeChange={setMolecularColorMode}
        />
      }
      rightPanel={
        <ChatRail
          structureId={activeStructure?.structure_id ?? null}
          pdbId={activeStructure?.pdb_id ?? null}
          discoveryPhase={discoveryPhase}
          hypothesisLifecycle={hypothesisLifecycle}
          viewportStateBuilder={viewportStateBuilder}
          onDirective={emitDirective}
          onSessionId={setAgentSessionId}
        />
      }
    />

      {/* Toast notification for selection errors / directive announcements */}
      {(selectionSync.toastMessage || directiveState.lastMessage) && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-[var(--radius-card)] bg-bg-elevated border border-slate-light shadow-lg text-xs text-text-secondary animate-[fadeIn_0.2s_ease-out]">
          {selectionSync.toastMessage || directiveState.lastMessage}
        </div>
      )}

      {/* Structure onboarding overlay */}
      {showOnboarding && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-bg/80 backdrop-blur-sm">
          <div className="w-96 max-h-[80vh] bg-bg-surface border border-slate rounded-[var(--radius-panel)] shadow-2xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate">
              <span className="text-sm font-display tracking-wider text-teal uppercase">Load Structure</span>
              {activeStructure && (
                <button
                  onClick={() => setShowOnboarding(false)}
                  className="text-xs text-text-muted hover:text-text-primary"
                >
                  Close
                </button>
              )}
            </div>
            <StructureOnboard
              onStructureLoaded={(structure) => {
                setActiveStructure(structure);
                setShowOnboarding(false);
              }}
            />
          </div>
        </div>
      )}
    </>
  );
}
