import { setup, assign } from "xstate";
import type { ViewportDirective, PoincareColorMode, StructureColorModeType, ActivePanelName } from "./types";

export interface ViewportContext {
  highlightedResidues: string[];
  userSelectedResidue: string | null;
  currentDirective: ViewportDirective | null;
  activeMetric: string;
  isRadarActive: boolean;
  poincareColorMode: PoincareColorMode;
  viewerColorMode: StructureColorModeType;
  riskThreshold: number;
  selectedPocketId: number | null;
  activePanel: ActivePanelName;
  layoutModelJSON: Record<string, any> | null;
}

export type ViewportEvent =
  | { type: "DIRECTIVE_RECEIVED"; directive: ViewportDirective }
  | { type: "USER_SELECT"; residues: string[] }
  | { type: "CLEAR" }
  | { type: "TOGGLE_RADAR"; active: boolean }
  | { type: "SET_POINCARE_COLOR_MODE"; mode: PoincareColorMode }
  | { type: "SET_VIEWER_COLOR_MODE"; mode: StructureColorModeType }
  | { type: "SET_RISK_THRESHOLD"; threshold: number }
  | { type: "SET_SELECTED_POCKET"; pocketId: number | null }
  | { type: "SET_ACTIVE_PANEL"; panel: ActivePanelName }
  | { type: "SET_LAYOUT_MODEL"; modelJSON: Record<string, any> };

export const viewportMachine = setup({
  types: {} as {
    context: ViewportContext;
    events: ViewportEvent;
  },
  guards: {
    isHighlightAction: ({ event }) =>
      event.type === "DIRECTIVE_RECEIVED" && event.directive.action === "highlight",
    isFocusAction: ({ event }) =>
      event.type === "DIRECTIVE_RECEIVED" && event.directive.action === "focus",
    isMetricAction: ({ event }) =>
      event.type === "DIRECTIVE_RECEIVED" && event.directive.action === "set_metric",
    isClearAction: ({ event }) =>
      event.type === "DIRECTIVE_RECEIVED" && event.directive.action === "clear",
  },
  actions: {
    applyDirective: assign({
      currentDirective: ({ event, context }) => {
        if (event.type !== "DIRECTIVE_RECEIVED") return context.currentDirective;
        return event.directive;
      },
      highlightedResidues: ({ event, context }) => {
        if (event.type !== "DIRECTIVE_RECEIVED") return context.highlightedResidues;
        const d = event.directive;
        
        // Normalize focus_residues → highlight_groups without mutating the event
        if (d.action === "focus" && d.focus_residues?.length && !d.highlight_groups?.length) {
          const normalized = [{
            residue_ids: d.focus_residues,
            color: "#00ffff",
            style: "glow" as const,
            label: "Focus"
          }];
          return normalized.flatMap((g) => g.residue_ids);
        }
        return d.highlight_groups?.flatMap((g) => g.residue_ids) || context.highlightedResidues;
      },
      poincareColorMode: ({ event, context }) => {
        if (event.type === "DIRECTIVE_RECEIVED" && event.directive.action === "set_color_mode" && event.directive.component_id === "poincare-viewer" && event.directive.color_mode) {
           return event.directive.color_mode as PoincareColorMode;
        }
        return context.poincareColorMode;
      },
      viewerColorMode: ({ event, context }) => {
        if (event.type === "DIRECTIVE_RECEIVED" && event.directive.action === "set_color_mode" && event.directive.component_id === "molecular-viewer" && event.directive.color_mode) {
           return event.directive.color_mode as StructureColorModeType;
        }
        return context.viewerColorMode;
      }
    }),
    applyMetric: assign({
      currentDirective: ({ event, context }) => 
        event.type === "DIRECTIVE_RECEIVED" ? event.directive : context.currentDirective,
      activeMetric: ({ event, context }) => 
        (event.type === "DIRECTIVE_RECEIVED" && event.directive.metric) ? event.directive.metric : context.activeMetric,
    }),
    applyUserSelection: assign({
      highlightedResidues: ({ event, context }) => 
        event.type === "USER_SELECT" ? event.residues : context.highlightedResidues,
      userSelectedResidue: ({ event, context }) =>
        event.type === "USER_SELECT" && event.residues.length === 1
          ? event.residues[0]
          : (event.type === "USER_SELECT" ? null : context.userSelectedResidue),
    }),
    clearState: assign({
      currentDirective: () => null,
      highlightedResidues: () => [] as string[],
      userSelectedResidue: () => null as string | null,
      isRadarActive: () => false,
    }),
    applyRadarToggle: assign({
      isRadarActive: ({ event, context }) => 
        event.type === "TOGGLE_RADAR" ? event.active : context.isRadarActive,
    }),
    applyUIPrefs: assign({
      poincareColorMode: ({ event, context }) => 
        event.type === "SET_POINCARE_COLOR_MODE" ? event.mode : context.poincareColorMode,
      viewerColorMode: ({ event, context }) => 
        event.type === "SET_VIEWER_COLOR_MODE" ? event.mode : context.viewerColorMode,
      riskThreshold: ({ event, context }) => 
        event.type === "SET_RISK_THRESHOLD" ? event.threshold : context.riskThreshold,
      selectedPocketId: ({ event, context }) => 
        event.type === "SET_SELECTED_POCKET" ? event.pocketId : context.selectedPocketId,
      activePanel: ({ event, context }) => 
        event.type === "SET_ACTIVE_PANEL" ? event.panel : context.activePanel,
      layoutModelJSON: ({ event, context }) => 
        event.type === "SET_LAYOUT_MODEL" ? event.modelJSON : context.layoutModelJSON,
    }),
  },
}).createMachine({
  id: "viewport",
  initial: "idle",
  context: {
    highlightedResidues: [],
    userSelectedResidue: null,
    currentDirective: null,
    activeMetric: "cone_depth",
    isRadarActive: false,
    poincareColorMode: "cone_depth",
    viewerColorMode: "spectrum",
    riskThreshold: 0,
    selectedPocketId: null,
    activePanel: null,
    layoutModelJSON: null,
  },
  states: {
    idle: {
      on: {
        DIRECTIVE_RECEIVED: [
          { target: "highlighted", guard: "isHighlightAction", actions: "applyDirective" },
          { target: "focused", guard: "isFocusAction", actions: "applyDirective" },
          { target: "idle", guard: "isMetricAction", actions: "applyMetric" },
          { target: "idle", guard: "isClearAction", actions: "clearState" },
          { target: "idle", actions: "applyDirective" }, // Default layout/property action catch-all
        ],
        USER_SELECT: { target: "highlighted", actions: "applyUserSelection" },
        TOGGLE_RADAR: { actions: "applyRadarToggle" },
        SET_POINCARE_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_VIEWER_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_RISK_THRESHOLD: { actions: "applyUIPrefs" },
        SET_SELECTED_POCKET: { actions: "applyUIPrefs" },
        SET_ACTIVE_PANEL: { actions: "applyUIPrefs" },
        SET_LAYOUT_MODEL: { actions: "applyUIPrefs" },
      },
    },
    highlighted: {
      on: {
        DIRECTIVE_RECEIVED: [
          { target: "highlighted", guard: "isHighlightAction", actions: "applyDirective" },
          { target: "focused", guard: "isFocusAction", actions: "applyDirective" },
          { target: "highlighted", guard: "isMetricAction", actions: "applyMetric" },
          { target: "idle", guard: "isClearAction", actions: "clearState" },
          { target: "highlighted", actions: "applyDirective" },
        ],
        USER_SELECT: { target: "highlighted", actions: "applyUserSelection" },
        CLEAR: { target: "idle", actions: "clearState" },
        TOGGLE_RADAR: { actions: "applyRadarToggle" },
        SET_POINCARE_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_VIEWER_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_RISK_THRESHOLD: { actions: "applyUIPrefs" },
        SET_SELECTED_POCKET: { actions: "applyUIPrefs" },
        SET_ACTIVE_PANEL: { actions: "applyUIPrefs" },
        SET_LAYOUT_MODEL: { actions: "applyUIPrefs" },
      },
    },
    focused: {
      on: {
        DIRECTIVE_RECEIVED: [
          { target: "highlighted", guard: "isHighlightAction", actions: "applyDirective" },
          { target: "focused", guard: "isFocusAction", actions: "applyDirective" },
          { target: "focused", guard: "isMetricAction", actions: "applyMetric" },
          { target: "idle", guard: "isClearAction", actions: "clearState" },
          { target: "focused", actions: "applyDirective" },
        ],
        USER_SELECT: { target: "highlighted", actions: "applyUserSelection" },
        CLEAR: { target: "idle", actions: "clearState" },
        TOGGLE_RADAR: { actions: "applyRadarToggle" },
        SET_POINCARE_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_VIEWER_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_RISK_THRESHOLD: { actions: "applyUIPrefs" },
        SET_SELECTED_POCKET: { actions: "applyUIPrefs" },
        SET_ACTIVE_PANEL: { actions: "applyUIPrefs" },
        SET_LAYOUT_MODEL: { actions: "applyUIPrefs" },
      },
    },
  },
});
