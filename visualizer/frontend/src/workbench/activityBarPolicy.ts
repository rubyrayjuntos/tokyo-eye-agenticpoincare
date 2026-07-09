/**
 * Always-visible activity-bar panels for the developer control-center workbench.
 * Separate from discoveryPhaseMachine toolPolicy (agent tools by phase).
 */
export const PINNED_ACTIVITY_PANELS = [
  "briefing",
  "control_console",
  "model_lifecycle",
] as const;

export type PinnedActivityPanel = (typeof PINNED_ACTIVITY_PANELS)[number];

/** Dockview component + title for pinned UI panels that open in the canvas. */
export const PINNED_PANEL_DOCK_SPAWN: Partial<
  Record<
    PinnedActivityPanel,
    {
      id: string;
      componentType:
        | "briefing-panel"
        | "control-console-panel"
        | "model-lifecycle-panel";
      title: string;
    }
  >
> = {
  briefing: {
    id: "panel_briefing-panel",
    componentType: "briefing-panel",
    title: "Briefing",
  },
  control_console: {
    id: "panel_control-console-panel",
    componentType: "control-console-panel",
    title: "Control Console",
  },
  model_lifecycle: {
    id: "panel_model-lifecycle-panel",
    componentType: "model-lifecycle-panel",
    title: "Model Lifecycle",
  },
};
