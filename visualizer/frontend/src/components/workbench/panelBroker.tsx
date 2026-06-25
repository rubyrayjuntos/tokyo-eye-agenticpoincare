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
}

function createPanelBroker({
  sendViewport,
  emitDirective,
  mergeLayoutModel,
}: PanelBrokerDependencies): PanelBroker {
  const publishTrigger = (_event: PanelTriggerEvent) => {
    // The broker intentionally keeps trigger publication lightweight for now.
    // Future panel subscriptions can layer on top of this stable event shape.
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
}

export function PanelBrokerProvider({
  children,
  sendViewport,
  emitDirective,
  mergeLayoutModel,
}: PanelBrokerProviderProps) {
  const broker = useMemo(
    () =>
      createPanelBroker({
        sendViewport,
        emitDirective,
        mergeLayoutModel,
      }),
    [emitDirective, mergeLayoutModel, sendViewport],
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
