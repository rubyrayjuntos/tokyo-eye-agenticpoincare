import type { DockviewApi, SerializedDockview } from "dockview";

import type { WorkbenchBus } from "./WorkbenchBus";
import type { WorkbenchPanelComponentType } from "./eventRegistry";
import type { WorkbenchPhaseGroup } from "./phaseGroups";
import { PHASE_GROUP_CONFIG } from "./phaseGroups";

const DEFAULT_PANEL_TITLES: Record<WorkbenchPanelComponentType, string> = {
  "briefing-panel": "Briefing",
  "control-console-panel": "Control Console",
  "triple-viewport-panel": "Discovery Viewports",
  "findings-dock-panel": "Findings",
  "poincare-panel": "Poincaré Disc",
  "molecular-panel": "3D Structure",
  "chat-panel": "Agent Chat",
  "telemetry-panel": "Telemetry",
  "model-lifecycle-panel": "Model Lifecycle",
  "mlflow-panel": "MLflow",
};

const COCKPIT_CORE_PANELS: WorkbenchPanelComponentType[] = [
  "briefing-panel",
  "triple-viewport-panel",
  "findings-dock-panel",
  "chat-panel",
];

type PendingLayout =
  | { kind: "snapshot"; snapshot: SerializedDockview }
  | { kind: "default"; group: WorkbenchPhaseGroup };

function panelComponentType(panelId: string): WorkbenchPanelComponentType | null {
  const match = panelId.match(/^panel_(.+?)(?:_\d+)?$/);
  return (match?.[1] as WorkbenchPanelComponentType | undefined) ?? null;
}

export class LayoutEngineAdapter {
  private dockviewApi: DockviewApi | null = null;
  private isSynchronizing = false;
  private persistTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingLayout: PendingLayout | null = null;
  private defaultLayoutGroup: WorkbenchPhaseGroup | null = null;

  constructor(
    private readonly bus: WorkbenchBus,
    private readonly workspaceId = "tokyo-eye-default",
  ) {
    this.setupBrokerListeners();
  }

  bindNativeEngine(api: DockviewApi): void {
    this.dockviewApi = api;
    this.setupNativeListeners();
    this.flushPendingLayout();
  }

  unbindNativeEngine(): void {
    this.dockviewApi = null;
  }

  syncRequirements(group: WorkbenchPhaseGroup): void {
    const required = PHASE_GROUP_CONFIG[group].requiredPanels;
    this.bus.publish("layout:sync_requirements", { requiredPanels: required });
  }

  loadLayoutFromSnapshot(snapshot: SerializedDockview): void {
    if (!this.dockviewApi) {
      this.pendingLayout = { kind: "snapshot", snapshot };
      return;
    }
    this.isSynchronizing = true;
    try {
      this.dockviewApi.fromJSON(snapshot);
      this.defaultLayoutGroup = null;
    } catch (error) {
      console.error("[LayoutEngineAdapter] Failed to load layout:", error);
    } finally {
      this.isSynchronizing = false;
    }
  }

  applyDefaultLayout(group: WorkbenchPhaseGroup): void {
    if (!this.dockviewApi) {
      this.pendingLayout = { kind: "default", group };
      return;
    }
    if (this.defaultLayoutGroup === group && this.dockviewApi.panels.length > 0) {
      return;
    }

    this.isSynchronizing = true;
    try {
      this.dockviewApi.panels.forEach((panel) => panel.api.close());
      const required = PHASE_GROUP_CONFIG[group].requiredPanels;
      this.applyCockpitLayout(required);
      this.defaultLayoutGroup = group;
    } finally {
      this.isSynchronizing = false;
    }
  }

  private applyCockpitLayout(required: WorkbenchPanelComponentType[]): void {
    if (!this.dockviewApi) return;

    const ordered = [
      ...COCKPIT_CORE_PANELS.filter((p) => required.includes(p)),
      ...required.filter((p) => !COCKPIT_CORE_PANELS.includes(p)),
    ];

    let centerId: string | null = null;
    let leftId: string | null = null;

    ordered.forEach((componentType) => {
      const id = `panel_${componentType}`;
      if (componentType === "briefing-panel") {
        this.dockviewApi!.addPanel({
          id,
          component: componentType,
          title: DEFAULT_PANEL_TITLES[componentType],
        });
        leftId = id;
        return;
      }

      if (componentType === "triple-viewport-panel") {
        this.dockviewApi!.addPanel({
          id,
          component: componentType,
          title: DEFAULT_PANEL_TITLES[componentType],
          position: leftId
            ? { referencePanel: leftId, direction: "right" }
            : undefined,
        });
        centerId = id;
        return;
      }

      if (componentType === "chat-panel" && centerId) {
        this.dockviewApi!.addPanel({
          id,
          component: componentType,
          title: DEFAULT_PANEL_TITLES[componentType],
          position: { referencePanel: centerId, direction: "right" },
        });
        return;
      }

      if (componentType === "findings-dock-panel" && centerId) {
        this.dockviewApi!.addPanel({
          id,
          component: componentType,
          title: DEFAULT_PANEL_TITLES[componentType],
          position: { referencePanel: centerId, direction: "below" },
        });
        return;
      }

      const reference = centerId ?? leftId ?? this.dockviewApi!.panels[0]?.id;
      this.dockviewApi!.addPanel({
        id,
        component: componentType,
        title: DEFAULT_PANEL_TITLES[componentType],
        position: reference
          ? { referencePanel: reference, direction: "below" }
          : undefined,
      });
    });
  }

  private flushPendingLayout(): void {
    if (!this.dockviewApi || !this.pendingLayout) return;
    const pending = this.pendingLayout;
    this.pendingLayout = null;
    if (pending.kind === "snapshot") {
      this.loadLayoutFromSnapshot(pending.snapshot);
    } else {
      this.applyDefaultLayout(pending.group);
    }
  }

  private setupBrokerListeners(): void {
    this.bus.subscribe("layout:spawn_panel", (request) => {
      if (!this.dockviewApi) return;
      const existing = this.dockviewApi.getPanel(request.id);
      if (existing) {
        existing.api.setActive();
        return;
      }
      this.dockviewApi.addPanel({
        id: request.id,
        component: request.componentType,
        title: request.title,
        position: {
          referencePanel: this.dockviewApi.panels[0]?.id,
          direction: "right",
        },
      });
    });

    this.bus.subscribe("layout:sync_requirements", (requirements) => {
      if (!this.dockviewApi || this.isSynchronizing) return;
      const openTypes = new Set(
        this.dockviewApi.panels
          .map((panel) => panelComponentType(panel.id))
          .filter((value): value is WorkbenchPanelComponentType => value != null),
      );

      const missing = requirements.requiredPanels.filter((componentType) => !openTypes.has(componentType));
      if (missing.length === 0) return;

      this.isSynchronizing = true;
      try {
        const referencePanel = this.dockviewApi.panels[0]?.id;
        missing.forEach((componentType, index) => {
          this.dockviewApi!.addPanel({
            id: `panel_${componentType}_${Date.now()}_${index}`,
            component: componentType,
            title: DEFAULT_PANEL_TITLES[componentType],
            position: referencePanel
              ? { referencePanel, direction: "right" }
              : undefined,
          });
        });
      } finally {
        this.isSynchronizing = false;
      }
    });
  }

  private setupNativeListeners(): void {
    if (!this.dockviewApi) return;

    this.dockviewApi.onDidLayoutChange(() => {
      if (this.isSynchronizing || !this.dockviewApi) return;
      const layout = this.dockviewApi.toJSON();
      const phaseGroup =
        this.bus.getLatestState("system:phase_transition")?.group ?? "exploration";

      this.bus.publish("layout:changed", {
        workspaceId: this.workspaceId,
        layout: layout as unknown as Record<string, unknown>,
        activePhaseGroup: phaseGroup,
      });
    });
  }

  schedulePersist(
    sessionId: string,
    persist: (payload: {
      layout_json: Record<string, unknown>;
      active_phase_group: WorkbenchPhaseGroup;
    }) => Promise<void>,
  ): () => void {
    return this.bus.subscribe("layout:changed", (payload) => {
      if (this.persistTimer) clearTimeout(this.persistTimer);
      this.persistTimer = setTimeout(() => {
        void persist({
          layout_json: payload.layout,
          active_phase_group: payload.activePhaseGroup,
        }).catch((error) => {
          console.error("[LayoutEngineAdapter] Layout persist failed:", error);
        });
      }, 800);
    });
  }
}
