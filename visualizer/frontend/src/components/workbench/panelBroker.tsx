import {
  createContext,
  useContext,
  useMemo,
  type PropsWithChildren,
} from "react";

import type { ViewportDirective } from "../../lib/types";
import type {
  DirectivePortPayload,
  LayoutPortPayload,
  MetricPortPayload,
  OpenBottomPanelPortPayload,
  OpenEditorPortPayload,
  OpenPanelPortPayload,
  PanelContributionId,
  PanelManifest,
  PanelPortPayload,
  SelectionPortPayload,
  TogglePortPayload,
  WorkbenchTrigger,
} from "./panelContract";
import { validatePanelPortPayload } from "./panelContract";
import type { WorkbenchBus } from "../../workbench/WorkbenchBus";
import type { WorkbenchPhaseGroup } from "../../workbench/phaseGroups";

function triggerToTopic(type: WorkbenchTrigger): string | null {
  switch (type) {
    case "selection.changed":
      return "ui:selection_changed";
    case "directive.emitted":
      return "data:directive_emitted";
    case "layout.changed":
      return "layout:changed";
    case "panel.activated":
      return "ui:panel_activated";
    default:
      return null;
  }
}

export interface PanelTriggerEvent {
  type: WorkbenchTrigger;
  sourcePanelId: PanelContributionId | "shell";
  payload: PanelPortPayload | Record<string, unknown>;
}

export interface PanelBroker {
  emitPort: <TId extends PanelContributionId>(
    manifest: PanelManifest<TId>,
    portName: string,
    payload: unknown,
  ) => void;
  publishTrigger: (event: PanelTriggerEvent) => void;
}

interface PanelBrokerDependencies {
  sendViewport: (event: any) => void;
  emitDirective: (directive: ViewportDirective) => void;
  mergeLayoutModel: (partialLayout: Record<string, unknown>) => void;
  workbenchBus?: WorkbenchBus;
  activePhaseGroup?: WorkbenchPhaseGroup;
}

function createPanelBroker({
  sendViewport,
  emitDirective,
  mergeLayoutModel,
  workbenchBus,
  activePhaseGroup = "exploration",
}: PanelBrokerDependencies): PanelBroker {
  const publishTrigger = (event: PanelTriggerEvent) => {
    if (!workbenchBus) return;
    const topic = triggerToTopic(event.type);
    if (!topic) return;

    if (topic === "ui:selection_changed") {
      const payload = event.payload as SelectionPortPayload;
      workbenchBus.publish("ui:selection_changed", {
        residueIds: payload.residueIds,
        sourcePanelId: event.sourcePanelId,
      });
      return;
    }

    if (topic === "data:directive_emitted") {
      const payload = event.payload as DirectivePortPayload | MetricPortPayload;
      const directive: Record<string, unknown> =
        "directive" in payload
          ? (payload.directive as unknown as Record<string, unknown>)
          : { metric: (payload as MetricPortPayload).metric };
      workbenchBus.publish("data:directive_emitted", { directive });
      return;
    }

    if (topic === "layout:changed") {
      const payload = event.payload as LayoutPortPayload;
      workbenchBus.publish("layout:changed", {
        workspaceId: "tokyo-eye-default",
        layout: payload.partialLayout,
        activePhaseGroup,
      });
      return;
    }

    if (topic === "ui:panel_activated") {
      const payload = event.payload as OpenPanelPortPayload;
      workbenchBus.publish("ui:panel_activated", {
        panelId: payload.panelId ?? "unknown",
        open: payload.open,
      });
    }
  };

  return {
    emitPort(manifest, portName, payload) {
      const port = manifest.ports.find((candidate) => candidate.name === portName);
      if (!port) {
        throw new Error(`Unknown port "${portName}" for panel "${manifest.id}"`);
      }
      if (!validatePanelPortPayload(port, payload)) {
        throw new Error(`Invalid payload for port "${portName}" on panel "${manifest.id}"`);
      }

      switch (port.kind) {
        case "toggle": {
          const togglePayload = payload as TogglePortPayload;
          sendViewport({
            type:
              togglePayload.target === "sidebar"
                ? "SET_SIDEBAR_OPEN"
                : "SET_BOTTOM_PANEL_OPEN",
            open: togglePayload.value,
          });
          publishTrigger({
            type: "panel.visibility.changed",
            sourcePanelId: manifest.id,
            payload: togglePayload,
          });
          return;
        }
        case "selection": {
          const selectionPayload = payload as SelectionPortPayload;
          if (selectionPayload.residueIds.length > 0) {
            sendViewport({ type: "USER_SELECT", residues: selectionPayload.residueIds });
          } else {
            sendViewport({ type: "CLEAR" });
          }
          publishTrigger({
            type: "selection.changed",
            sourcePanelId: manifest.id,
            payload: selectionPayload,
          });
          return;
        }
        case "metric": {
          const metricPayload = payload as MetricPortPayload;
          emitDirective({
            action: "set_metric",
            metric: metricPayload.metric,
          });
          publishTrigger({
            type: "directive.emitted",
            sourcePanelId: manifest.id,
            payload: metricPayload,
          });
          return;
        }
        case "directive": {
          const directivePayload = payload as DirectivePortPayload;
          emitDirective(directivePayload.directive);
          publishTrigger({
            type: "directive.emitted",
            sourcePanelId: manifest.id,
            payload: directivePayload,
          });
          return;
        }
        case "open-panel": {
          const openPanelPayload = payload as OpenPanelPortPayload;
          sendViewport({ type: "SET_ACTIVE_PANEL", panel: openPanelPayload.panelId });
          sendViewport({
            type: "SET_SIDEBAR_OPEN",
            open: openPanelPayload.open ?? openPanelPayload.panelId !== null,
          });
          publishTrigger({
            type: "panel.activated",
            sourcePanelId: manifest.id,
            payload: openPanelPayload,
          });
          return;
        }
        case "open-editor": {
          const openEditorPayload = payload as OpenEditorPortPayload;
          sendViewport({ type: "SET_ACTIVE_EDITOR_TAB", tab: openEditorPayload.tab });
          publishTrigger({
            type: "editor.activated",
            sourcePanelId: manifest.id,
            payload: openEditorPayload,
          });
          return;
        }
        case "open-bottom-panel": {
          const openBottomPanelPayload = payload as OpenBottomPanelPortPayload;
          sendViewport({
            type: "SET_ACTIVE_BOTTOM_PANEL",
            panel: openBottomPanelPayload.panelId,
          });
          sendViewport({
            type: "SET_BOTTOM_PANEL_OPEN",
            open: openBottomPanelPayload.open ?? true,
          });
          publishTrigger({
            type: "bottom-panel.activated",
            sourcePanelId: manifest.id,
            payload: openBottomPanelPayload,
          });
          return;
        }
        case "layout": {
          const layoutPayload = payload as LayoutPortPayload;
          mergeLayoutModel(layoutPayload.partialLayout);
          publishTrigger({
            type: "layout.changed",
            sourcePanelId: manifest.id,
            payload: layoutPayload,
          });
          return;
        }
      }
    },
    publishTrigger,
  };
}

const PanelBrokerContext = createContext<PanelBroker | null>(null);

interface PanelBrokerProviderProps extends PropsWithChildren {
  sendViewport: (event: any) => void;
  emitDirective: (directive: ViewportDirective) => void;
  mergeLayoutModel: (partialLayout: Record<string, unknown>) => void;
  workbenchBus?: WorkbenchBus;
  activePhaseGroup?: WorkbenchPhaseGroup;
}

export function PanelBrokerProvider({
  children,
  sendViewport,
  emitDirective,
  mergeLayoutModel,
  workbenchBus,
  activePhaseGroup,
}: PanelBrokerProviderProps) {
  const broker = useMemo(
    () =>
      createPanelBroker({
        sendViewport,
        emitDirective,
        mergeLayoutModel,
        workbenchBus,
        activePhaseGroup,
      }),
    [activePhaseGroup, emitDirective, mergeLayoutModel, sendViewport, workbenchBus],
  );

  return (
    <PanelBrokerContext.Provider value={broker}>
      {children}
    </PanelBrokerContext.Provider>
  );
}

export function usePanelBroker() {
  const broker = useContext(PanelBrokerContext);
  if (!broker) {
    throw new Error("usePanelBroker must be used inside PanelBrokerProvider");
  }
  return broker;
}

export { createPanelBroker };
