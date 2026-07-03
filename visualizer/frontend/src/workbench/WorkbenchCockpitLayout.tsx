import { useCallback, useEffect, useRef, useState } from "react";

import { CockpitNavBar } from "../components/layout";
import { ToolDock } from "../components/layout";
import { api } from "../lib/api";
import type { DiscoveryPhase } from "../lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "../lib/hypothesisLifecycleMachine";
import type { ViewportDirective, ViewportState } from "../lib/types";
import type { MolecularColorMode, PanelPoincareColorMode } from "../components/viewers";
import type { SelectedResidueInfo } from "../lib/types";
import { PhaseGroupTabs, resolveActivePhaseGroup } from "./PhaseGroupTabs";
import { WorkbenchCanvas } from "./WorkbenchCanvas";
import { useLayoutEngineAdapter, useWorkbenchBus } from "./WorkbenchProvider";
import { useOrchestrationSync } from "./useOrchestrationSync";
import type { WorkbenchPhaseGroup } from "./phaseGroups";

export interface WorkbenchCockpitLayoutProps {
  activeStructureId: string | null;
  pdbId: string | null;
  discoveryPhase: DiscoveryPhase;
  hypothesisLifecycle: HypothesisLifecycleState;
  highlightedResidues: string[];
  poincareColorMode: PanelPoincareColorMode;
  molecularColorMode: MolecularColorMode;
  selectedResidue: SelectedResidueInfo | null;
  mobiusFocusEnabled: boolean;
  viewportStateBuilder: () => ViewportState;
  onResidueClick: (residueId: string) => void;
  onColorModeChange: (mode: PanelPoincareColorMode) => void;
  onBrushSelect: (residueIds: string[]) => void;
  onMobiusFocusToggle: (enabled: boolean) => void;
  onMolecularColorModeChange: (mode: MolecularColorMode) => void;
  onDirective?: (directive: ViewportDirective) => void;
  onSessionId?: (sessionId: string) => void;
  onDiscoveryPhase: (phase: DiscoveryPhase) => void;
  sessionId: string;
  connectionStatus: "connected" | "disconnected" | "reconnecting";
  onPanelSelect: (panel: string | null) => void;
  activePanel: string | null;
  allowedTools: string[];
  blockedTools: string[];
  toolIcons: Record<string, React.ReactNode>;
  onOpenStructurePicker: () => void;
}

export function WorkbenchCockpitLayout(props: WorkbenchCockpitLayoutProps) {
  const bus = useWorkbenchBus();
  const layoutAdapter = useLayoutEngineAdapter();
  const [activeGroup, setActiveGroup] = useState<WorkbenchPhaseGroup>(() =>
    resolveActivePhaseGroup(props.discoveryPhase),
  );
  const layoutLoadedForSessionRef = useRef<string | null>(null);

  const { setPhaseGroup, setDiscoveryPhase } = useOrchestrationSync({
    sessionId: props.sessionId,
    onDiscoveryPhase: props.onDiscoveryPhase,
  });

  useEffect(() => {
    setActiveGroup(resolveActivePhaseGroup(props.discoveryPhase));
  }, [props.discoveryPhase]);

  const handleSelectGroup = useCallback(
    async (group: WorkbenchPhaseGroup) => {
      setActiveGroup(group);
      await setPhaseGroup(group);
      layoutAdapter.syncRequirements(group);
    },
    [layoutAdapter, setPhaseGroup],
  );

  const handleSelectDiscoveryPhase = useCallback(
    async (phase: DiscoveryPhase) => {
      setActiveGroup(resolveActivePhaseGroup(phase));
      await setDiscoveryPhase(phase);
      layoutAdapter.syncRequirements(resolveActivePhaseGroup(phase));
    },
    [layoutAdapter, setDiscoveryPhase],
  );

  useEffect(() => {
    if (!props.sessionId) return;
    return layoutAdapter.schedulePersist(props.sessionId, async (payload) => {
      await api.putWorkspaceLayout(props.sessionId, {
        layout_json: payload.layout_json,
        active_phase_group: payload.active_phase_group,
      });
    });
  }, [layoutAdapter, props.sessionId]);

  useEffect(() => {
    if (!props.sessionId) return;
    if (layoutLoadedForSessionRef.current === props.sessionId) return;

    let cancelled = false;
    void api
      .getWorkspaceLayout(props.sessionId)
      .then((saved) => {
        if (cancelled) return;
        layoutLoadedForSessionRef.current = props.sessionId;
        const group =
          (saved.active_phase_group as WorkbenchPhaseGroup) ||
          resolveActivePhaseGroup(props.discoveryPhase);
        if (saved.layout_json && Object.keys(saved.layout_json).length > 0) {
          layoutAdapter.loadLayoutFromSnapshot(saved.layout_json as any);
        } else {
          layoutAdapter.applyDefaultLayout(group);
        }
        setActiveGroup(group);
      })
      .catch(() => {
        if (cancelled) return;
        layoutLoadedForSessionRef.current = props.sessionId;
        layoutAdapter.applyDefaultLayout(
          resolveActivePhaseGroup(props.discoveryPhase),
        );
      });

    return () => {
      cancelled = true;
    };
  }, [layoutAdapter, props.discoveryPhase, props.sessionId]);

  useEffect(() => {
    const unsub = bus.subscribe("system:warning", (warning) => {
      console.warn("[WorkbenchBus]", warning.message);
    });
    return unsub;
  }, [bus]);

  return (
    <div className="flex h-screen flex-col bg-ide-bg text-zinc-200">
      <CockpitNavBar
        structureId={props.activeStructureId}
        pdbId={props.pdbId}
        discoveryPhase={props.discoveryPhase}
        hypothesisLifecycle={props.hypothesisLifecycle}
        connectionStatus={props.connectionStatus}
        onStructureSearch={props.onOpenStructurePicker}
      />
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="shrink-0 border-b border-[#1e1930] px-4 py-2">
          <PhaseGroupTabs
            activeGroup={activeGroup}
            activeDiscoveryPhase={props.discoveryPhase}
            onSelectGroup={handleSelectGroup}
            onSelectDiscoveryPhase={handleSelectDiscoveryPhase}
          />
        </div>
        <div className="flex min-h-0 flex-1">
        <ToolDock
          allowedTools={props.allowedTools}
          blockedTools={props.blockedTools}
          activePanel={props.activePanel}
          onPanelSelect={props.onPanelSelect}
          mode="activity-bar"
          toolIcons={props.toolIcons}
        />
        <main className="min-w-0 flex-1 p-2">
          <WorkbenchCanvas
            structureId={props.activeStructureId}
            pdbId={props.pdbId}
            discoveryPhase={props.discoveryPhase}
            hypothesisLifecycle={props.hypothesisLifecycle}
            highlightedResidues={props.highlightedResidues}
            poincareColorMode={props.poincareColorMode}
            molecularColorMode={props.molecularColorMode}
            selectedResidue={props.selectedResidue}
            mobiusFocusEnabled={props.mobiusFocusEnabled}
            viewportStateBuilder={props.viewportStateBuilder}
            onResidueClick={props.onResidueClick}
            onColorModeChange={props.onColorModeChange}
            onBrushSelect={props.onBrushSelect}
            onMobiusFocusToggle={props.onMobiusFocusToggle}
            onMolecularColorModeChange={props.onMolecularColorModeChange}
            onDirective={props.onDirective}
            onSessionId={props.onSessionId}
            sessionId={props.sessionId}
            onOpenStructurePicker={props.onOpenStructurePicker}
          />
        </main>
        </div>
      </div>
    </div>
  );
}
