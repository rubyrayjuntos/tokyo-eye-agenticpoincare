import type {
  ActivePanelName,
  BottomPanelId,
  EditorTabId,
  ToolPanelId,
  ViewportDirective,
} from "../../lib/types";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type WorkbenchRegion = "activity-sidebar" | "editor" | "bottom-panel";
export type PanelContributionId = ToolPanelId | BottomPanelId | EditorTabId;

export type PanelPortKind =
  | "toggle"
  | "selection"
  | "metric"
  | "directive"
  | "open-panel"
  | "open-editor"
  | "open-bottom-panel"
  | "layout";

export type PanelPortDirection = "input" | "output" | "bidirectional";

export type WorkbenchTrigger =
  | "panel.activated"
  | "panel.visibility.changed"
  | "editor.activated"
  | "bottom-panel.activated"
  | "selection.changed"
  | "directive.emitted"
  | "layout.changed";

export interface TogglePortPayload {
  target: "sidebar" | "bottom-panel";
  value: boolean;
}

export interface SelectionPortPayload {
  residueIds: string[];
}

export interface MetricPortPayload {
  metric: string;
}

export interface DirectivePortPayload {
  directive: ViewportDirective;
}

export interface OpenPanelPortPayload {
  panelId: ActivePanelName;
  open?: boolean;
}

export interface OpenEditorPortPayload {
  tab: EditorTabId;
}

export interface OpenBottomPanelPortPayload {
  panelId: BottomPanelId;
  open?: boolean;
}

export interface LayoutPortPayload {
  partialLayout: Record<string, JsonValue>;
}

export type PanelPortPayload =
  | TogglePortPayload
  | SelectionPortPayload
  | MetricPortPayload
  | DirectivePortPayload
  | OpenPanelPortPayload
  | OpenEditorPortPayload
  | OpenBottomPanelPortPayload
  | LayoutPortPayload;

export interface PanelPortDefinition {
  name: string;
  kind: PanelPortKind;
  direction: PanelPortDirection;
  description?: string;
}

export interface PanelManifest<TId extends string> {
  id: TId;
  title: string;
  region: WorkbenchRegion;
  ports: PanelPortDefinition[];
  subscribesTo?: WorkbenchTrigger[];
  emits?: WorkbenchTrigger[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isJsonValue(value: unknown): value is JsonValue {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return true;
  }

  if (Array.isArray(value)) {
    return value.every(isJsonValue);
  }

  if (isRecord(value)) {
    return Object.values(value).every(isJsonValue);
  }

  return false;
}

function isViewportDirectivePayload(value: unknown): value is DirectivePortPayload {
  return (
    isRecord(value) &&
    "directive" in value &&
    isRecord(value.directive) &&
    typeof value.directive.action === "string"
  );
}

export function validatePanelPortPayload(
  port: Pick<PanelPortDefinition, "kind">,
  payload: unknown,
): payload is PanelPortPayload {
  switch (port.kind) {
    case "toggle":
      return (
        isRecord(payload) &&
        (payload.target === "sidebar" || payload.target === "bottom-panel") &&
        typeof payload.value === "boolean"
      );
    case "selection":
      return (
        isRecord(payload) &&
        Array.isArray(payload.residueIds) &&
        payload.residueIds.every((value) => typeof value === "string")
      );
    case "metric":
      return isRecord(payload) && typeof payload.metric === "string";
    case "directive":
      return isViewportDirectivePayload(payload);
    case "open-panel":
      return (
        isRecord(payload) &&
        ("panelId" in payload) &&
        (payload.panelId === null || typeof payload.panelId === "string") &&
        (payload.open === undefined || typeof payload.open === "boolean")
      );
    case "open-editor":
      return (
        isRecord(payload) &&
        (payload.tab === "manifold" || payload.tab === "structure" || payload.tab === "agent")
      );
    case "open-bottom-panel":
      return (
        isRecord(payload) &&
        (payload.panelId === "summary" ||
          payload.panelId === "hypotheses" ||
          payload.panelId === "pockets" ||
          payload.panelId === "fragments" ||
          payload.panelId === "motifs" ||
          payload.panelId === "results" ||
          payload.panelId === "telemetry") &&
        (payload.open === undefined || typeof payload.open === "boolean")
      );
    case "layout":
      return (
        isRecord(payload) &&
        "partialLayout" in payload &&
        isRecord(payload.partialLayout) &&
        isJsonValue(payload.partialLayout)
      );
    default:
      return false;
  }
}
