import { setup, assign } from "xstate";
import type {
  ViewportDirective,
  PoincareColorMode,
  StructureColorModeType,
  ActivePanelName,
  BottomPanelId,
  EditorTabId,
  SelectedResidueInfo,
} from "./types";

export interface ViewportContext {
  highlightedResidues: string[];
  brushSelectedIds: string[];
  selectedResidue: SelectedResidueInfo | null;
  userSelectedResidue: string | null;
  mobiusFocusEnabled: boolean;
  currentDirective: ViewportDirective | null;
  activeMetric: string;
  isRadarActive: boolean;
  poincareColorMode: PoincareColorMode;
  viewerColorMode: StructureColorModeType;
  riskThreshold: number;
  selectedPocketId: number | null;
  activePanel: ActivePanelName;
  sidebarOpen: boolean;
  activeEditorTab: EditorTabId;
  bottomPanelOpen: boolean;
  activeBottomPanel: BottomPanelId;
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
  | { type: "SET_SIDEBAR_OPEN"; open: boolean }
  | { type: "SET_ACTIVE_EDITOR_TAB"; tab: EditorTabId }
  | { type: "SET_BOTTOM_PANEL_OPEN"; open: boolean }
  | { type: "SET_ACTIVE_BOTTOM_PANEL"; panel: BottomPanelId }
  | { type: "SET_LAYOUT_MODEL"; modelJSON: Record<string, any> }
  | { type: "SET_MOBIUS_FOCUS"; enabled: boolean }
  | { type: "SET_BRUSH_SELECTION"; residues: string[] }
  | { type: "SET_SELECTED_RESIDUE"; residue: SelectedResidueInfo | null }
  | {
      type: "SET_SELECTION";
      highlightedResidues: string[];
      brushSelectedIds: string[];
      selectedResidue: SelectedResidueInfo | null;
    };

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
      brushSelectedIds: ({ event, context }) =>
        event.type === "USER_SELECT" ? event.residues : context.brushSelectedIds,
      userSelectedResidue: ({ event, context }) =>
        event.type === "USER_SELECT" && event.residues.length === 1
          ? event.residues[0]
          : event.type === "USER_SELECT"
            ? null
            : context.userSelectedResidue,
      selectedResidue: ({ event, context }) =>
        event.type === "USER_SELECT" && event.residues.length === 1
          ? context.selectedResidue?.residue_id === event.residues[0]
            ? context.selectedResidue
            : {
                residue_id: event.residues[0],
                residue_name: null,
                chain_label: null,
                epistemic_uncertainty: null,
                cone_depth: null,
              }
          : event.type === "USER_SELECT"
            ? null
            : context.selectedResidue,
    }),
    applySelection: assign({
      highlightedResidues: ({ event }) =>
        event.type === "SET_SELECTION" ? event.highlightedResidues : [],
      brushSelectedIds: ({ event }) =>
        event.type === "SET_SELECTION" ? event.brushSelectedIds : [],
      selectedResidue: ({ event }) =>
        event.type === "SET_SELECTION" ? event.selectedResidue : null,
      userSelectedResidue: ({ event }) => {
        if (event.type !== "SET_SELECTION") return null;
        const ids = event.highlightedResidues;
        return ids.length === 1 ? ids[0] : null;
      },
    }),
    applyBrushSelection: assign({
      brushSelectedIds: ({ event, context }) =>
        event.type === "SET_BRUSH_SELECTION" ? event.residues : context.brushSelectedIds,
    }),
    applySelectedResidue: assign({
      selectedResidue: ({ event, context }) =>
        event.type === "SET_SELECTED_RESIDUE" ? event.residue : context.selectedResidue,
      userSelectedResidue: ({ event, context }) =>
        event.type === "SET_SELECTED_RESIDUE"
          ? event.residue?.residue_id ?? null
          : context.userSelectedResidue,
      highlightedResidues: ({ event, context }) =>
        event.type === "SET_SELECTED_RESIDUE" && event.residue
          ? [event.residue.residue_id]
          : context.highlightedResidues,
      brushSelectedIds: ({ event, context }) =>
        event.type === "SET_SELECTED_RESIDUE" && event.residue
          ? [event.residue.residue_id]
          : context.brushSelectedIds,
    }),
    applyMobiusFocus: assign({
      mobiusFocusEnabled: ({ event, context }) =>
        event.type === "SET_MOBIUS_FOCUS" ? event.enabled : context.mobiusFocusEnabled,
    }),
    clearState: assign({
      currentDirective: () => null,
      highlightedResidues: () => [] as string[],
      brushSelectedIds: () => [] as string[],
      selectedResidue: () => null as SelectedResidueInfo | null,
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
      sidebarOpen: ({ event, context }) =>
        event.type === "SET_SIDEBAR_OPEN" ? event.open : context.sidebarOpen,
      activeEditorTab: ({ event, context }) =>
        event.type === "SET_ACTIVE_EDITOR_TAB" ? event.tab : context.activeEditorTab,
      bottomPanelOpen: ({ event, context }) =>
        event.type === "SET_BOTTOM_PANEL_OPEN" ? event.open : context.bottomPanelOpen,
      activeBottomPanel: ({ event, context }) =>
        event.type === "SET_ACTIVE_BOTTOM_PANEL" ? event.panel : context.activeBottomPanel,
      layoutModelJSON: ({ event, context }) => 
        event.type === "SET_LAYOUT_MODEL" ? event.modelJSON : context.layoutModelJSON,
    }),
  },
}).createMachine({
  id: "viewport",
  initial: "idle",
  context: {
    highlightedResidues: [],
    brushSelectedIds: [],
    selectedResidue: null,
    userSelectedResidue: null,
    mobiusFocusEnabled: false,
    currentDirective: null,
    activeMetric: "cone_depth",
    isRadarActive: false,
    poincareColorMode: "cone_depth",
    viewerColorMode: "spectrum",
    riskThreshold: 0,
    selectedPocketId: null,
    activePanel: "briefing",
    sidebarOpen: true,
    activeEditorTab: "structure",
    bottomPanelOpen: true,
    activeBottomPanel: "summary",
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
        SET_SELECTION: { target: "highlighted", actions: "applySelection" },
        SET_BRUSH_SELECTION: { actions: "applyBrushSelection" },
        SET_SELECTED_RESIDUE: { target: "highlighted", actions: "applySelectedResidue" },
        SET_MOBIUS_FOCUS: { actions: "applyMobiusFocus" },
        TOGGLE_RADAR: { actions: "applyRadarToggle" },
        SET_POINCARE_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_VIEWER_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_RISK_THRESHOLD: { actions: "applyUIPrefs" },
        SET_SELECTED_POCKET: { actions: "applyUIPrefs" },
        SET_ACTIVE_PANEL: { actions: "applyUIPrefs" },
        SET_SIDEBAR_OPEN: { actions: "applyUIPrefs" },
        SET_ACTIVE_EDITOR_TAB: { actions: "applyUIPrefs" },
        SET_BOTTOM_PANEL_OPEN: { actions: "applyUIPrefs" },
        SET_ACTIVE_BOTTOM_PANEL: { actions: "applyUIPrefs" },
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
        SET_SELECTION: { target: "highlighted", actions: "applySelection" },
        SET_BRUSH_SELECTION: { actions: "applyBrushSelection" },
        SET_SELECTED_RESIDUE: { target: "highlighted", actions: "applySelectedResidue" },
        SET_MOBIUS_FOCUS: { actions: "applyMobiusFocus" },
        CLEAR: { target: "idle", actions: "clearState" },
        TOGGLE_RADAR: { actions: "applyRadarToggle" },
        SET_POINCARE_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_VIEWER_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_RISK_THRESHOLD: { actions: "applyUIPrefs" },
        SET_SELECTED_POCKET: { actions: "applyUIPrefs" },
        SET_ACTIVE_PANEL: { actions: "applyUIPrefs" },
        SET_SIDEBAR_OPEN: { actions: "applyUIPrefs" },
        SET_ACTIVE_EDITOR_TAB: { actions: "applyUIPrefs" },
        SET_BOTTOM_PANEL_OPEN: { actions: "applyUIPrefs" },
        SET_ACTIVE_BOTTOM_PANEL: { actions: "applyUIPrefs" },
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
        SET_SELECTION: { target: "highlighted", actions: "applySelection" },
        SET_BRUSH_SELECTION: { actions: "applyBrushSelection" },
        SET_SELECTED_RESIDUE: { target: "highlighted", actions: "applySelectedResidue" },
        SET_MOBIUS_FOCUS: { actions: "applyMobiusFocus" },
        CLEAR: { target: "idle", actions: "clearState" },
        TOGGLE_RADAR: { actions: "applyRadarToggle" },
        SET_POINCARE_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_VIEWER_COLOR_MODE: { actions: "applyUIPrefs" },
        SET_RISK_THRESHOLD: { actions: "applyUIPrefs" },
        SET_SELECTED_POCKET: { actions: "applyUIPrefs" },
        SET_ACTIVE_PANEL: { actions: "applyUIPrefs" },
        SET_SIDEBAR_OPEN: { actions: "applyUIPrefs" },
        SET_ACTIVE_EDITOR_TAB: { actions: "applyUIPrefs" },
        SET_BOTTOM_PANEL_OPEN: { actions: "applyUIPrefs" },
        SET_ACTIVE_BOTTOM_PANEL: { actions: "applyUIPrefs" },
        SET_LAYOUT_MODEL: { actions: "applyUIPrefs" },
      },
    },
  },
});
