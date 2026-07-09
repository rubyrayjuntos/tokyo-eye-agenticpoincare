import { useEffect, useMemo } from "react";
import { DockviewReact, type DockviewReadyEvent } from "dockview";
import "dockview/dist/styles/dockview.css";

import { ChatRail } from "../components/chat";
import AgentTelemetryPanel from "../components/AgentTelemetryPanel";
import { BriefingDockPanel, FindingsDockPanel } from "../components/cockpit";
import ControlConsoleDockPanel from "../components/cockpit/ControlConsoleDockPanel";
import ModelLifecycleDockPanel from "../components/cockpit/ModelLifecycleDockPanel";
import MlflowDockPanel from "../components/cockpit/MlflowDockPanel";
import { MolecularPanel, PoincarePanel, TripleViewportPanel } from "../components/viewers";
import type { MolecularColorMode, PanelPoincareColorMode } from "../components/viewers";
import type {
  AgentChatResponse,
  SelectedResidueInfo,
  ViewportDirective,
  ViewportState,
} from "../lib/types";
import type { DiscoveryPhase } from "../lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "../lib/hypothesisLifecycleMachine";
import { useLayoutEngineAdapter } from "./WorkbenchProvider";
import {
  useWorkbenchCanvasProps,
  WorkbenchCanvasPropsProvider,
} from "./WorkbenchCanvasPropsContext";

export interface WorkbenchCanvasProps {
  structureId: string | null;
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
  onAgentResponse?: (response: AgentChatResponse) => void;
  sessionId: string;
  onOpenStructurePicker: () => void;
}

function TripleViewportPanelDock() {
  const props = useWorkbenchCanvasProps();
  return (
    <TripleViewportPanel
      structureId={props.structureId}
      pdbId={props.pdbId}
      colorMode={props.poincareColorMode}
      selectedResidue={props.selectedResidue}
      highlightedResidues={props.highlightedResidues}
      mobiusFocusEnabled={props.mobiusFocusEnabled}
      onResidueClick={props.onResidueClick}
      onColorModeChange={props.onColorModeChange}
      onBrushSelect={props.onBrushSelect}
      onMobiusFocusToggle={props.onMobiusFocusToggle}
      onOpenStructurePicker={props.onOpenStructurePicker}
    />
  );
}

function PoincarePanelDock() {
  const props = useWorkbenchCanvasProps();
  return (
    <PoincarePanel
      structureId={props.structureId}
      colorMode={props.poincareColorMode}
      selectedResidue={props.selectedResidue}
      highlightedResidues={props.highlightedResidues}
      mobiusFocusEnabled={props.mobiusFocusEnabled}
      onResidueClick={props.onResidueClick}
      onColorModeChange={props.onColorModeChange}
      onBrushSelect={props.onBrushSelect}
      onMobiusFocusToggle={props.onMobiusFocusToggle}
    />
  );
}

function MolecularPanelDock() {
  const props = useWorkbenchCanvasProps();
  return (
    <MolecularPanel
      structureId={props.structureId}
      highlightedResidues={props.highlightedResidues}
      colorMode={props.molecularColorMode}
      selectedResidue={props.selectedResidue}
      onResidueClick={props.onResidueClick}
      onColorModeChange={props.onMolecularColorModeChange}
    />
  );
}

function ChatPanelDock() {
  const props = useWorkbenchCanvasProps();
  return (
    <ChatRail
      structureId={props.structureId}
      pdbId={props.pdbId}
      discoveryPhase={props.discoveryPhase}
      hypothesisLifecycle={props.hypothesisLifecycle}
      viewportStateBuilder={props.viewportStateBuilder}
      sessionId={props.sessionId}
      onDirective={props.onDirective}
      onSessionId={props.onSessionId}
    />
  );
}

const DOCK_COMPONENTS = {
  "briefing-panel": BriefingDockPanel,
  "control-console-panel": ControlConsoleDockPanel,
  "findings-dock-panel": FindingsDockPanel,
  "triple-viewport-panel": TripleViewportPanelDock,
  "poincare-panel": PoincarePanelDock,
  "molecular-panel": MolecularPanelDock,
  "chat-panel": ChatPanelDock,
  "telemetry-panel": function TelemetryPanelDock() {
    return (
      <div className="h-full overflow-auto p-2">
        <AgentTelemetryPanel />
      </div>
    );
  },
  "model-lifecycle-panel": function ModelLifecyclePanelDock() {
    const props = useWorkbenchCanvasProps();
    return <ModelLifecycleDockPanel structureId={props.structureId} />;
  },
  "mlflow-panel": MlflowDockPanel,
};

export function WorkbenchCanvas(props: WorkbenchCanvasProps) {
  const layoutAdapter = useLayoutEngineAdapter();

  const components = useMemo(() => DOCK_COMPONENTS, []);

  const handleReady = (event: DockviewReadyEvent) => {
    layoutAdapter.bindNativeEngine(event.api);
  };

  useEffect(() => {
    return () => layoutAdapter.unbindNativeEngine();
  }, [layoutAdapter]);

  return (
    <WorkbenchCanvasPropsProvider value={props}>
      <div className="h-full min-h-[480px] w-full overflow-hidden rounded-xl border border-[#1e1930] bg-ide-bg">
        <DockviewReact
          onReady={handleReady}
          components={components}
          className="dockview-theme-dark h-full"
        />
      </div>
    </WorkbenchCanvasPropsProvider>
  );
}
