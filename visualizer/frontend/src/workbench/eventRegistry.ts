import type { DiscoveryPhase } from "../lib/discoveryPhaseMachine";
import type { WorkbenchPhaseGroup } from "./phaseGroups";

export type WorkbenchPanelComponentType =
  | "briefing-panel"
  | "triple-viewport-panel"
  | "findings-dock-panel"
  | "poincare-panel"
  | "molecular-panel"
  | "chat-panel"
  | "telemetry-panel";

export interface OrchestrationSnapshotPayload {
  session_id?: string;
  discovery_phase: string;
  hypothesis_lifecycle: string;
  structure_scope: Record<string, unknown>;
  selected_residue: Record<string, unknown> | null;
  policy: {
    allowed_tools?: string[];
    blocked_tools?: string[];
    preferred_tools?: string[];
    discovery_phase?: string;
    hypothesis_state?: string;
    reasoning_mode?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface EventRegistry {
  "system:orchestration_snapshot": OrchestrationSnapshotPayload;
  "system:phase_transition": {
    group: WorkbenchPhaseGroup;
    discoveryPhase: DiscoveryPhase;
    timestamp: string;
  };
  "system:warning": { message: string; timestamp?: string };
  "broker:log_added": {
    id: string;
    topic: string;
    payload: unknown;
    timestamp: string;
    type: "system" | "interaction" | "tool_call" | "warning";
  };
  "ui:selection_changed": { residueIds: string[]; sourcePanelId?: string };
  "ui:panel_activated": { panelId: string; open?: boolean };
  "ui:interaction": {
    sourcePanelId: string;
    interactionType: string;
    payload: unknown;
  };
  "data:directive_emitted": { directive: Record<string, unknown> };
  "data:structure_scope_updated": {
    primaryStructureId: string | null;
    structureIds: string[];
    activeChainIds: string[];
    ingestionReady: boolean;
    inferenceReady: boolean;
    pipelineReady: boolean;
  };
  "data:hydration_ready": {
    structureId: string;
    residueCount: number;
    degraded: boolean;
  };
  "data:hydration_updated": {
    structureId: string | null;
    status: "idle" | "loading" | "loaded" | "failed";
    degraded: boolean;
    residueCount: number;
    error: string | null;
    signalsInferred: boolean;
    contractVersion: string | null;
  };
  "layout:spawn_panel": {
    id: string;
    componentType: WorkbenchPanelComponentType;
    title: string;
  };
  "layout:sync_requirements": { requiredPanels: WorkbenchPanelComponentType[] };
  "layout:changed": {
    workspaceId: string;
    layout: Record<string, unknown>;
    activePhaseGroup: WorkbenchPhaseGroup;
  };
  "tool:call": { tool: string; args?: Record<string, unknown> };
}

export type WorkbenchTopic = keyof EventRegistry;

export type SubscriptionCallback<T extends WorkbenchTopic = WorkbenchTopic> = (
  payload: EventRegistry[T],
) => void;

export type MiddlewareNext = () => void;

export type WorkbenchMiddleware = <T extends WorkbenchTopic>(
  topic: T,
  payload: EventRegistry[T],
  next: MiddlewareNext,
) => void;
