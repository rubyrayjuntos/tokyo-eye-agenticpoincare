import { useCallback, useReducer } from "react";
import type { ViewportDirective } from "./types";

export type ColorMetric = "cone_depth" | "epistemic" | "aleatoric" | "centrality";

interface DirectiveState {
  highlights: Set<string>;
  focusTargets: string[];
  activeMetric: ColorMetric;
  annotations: Map<string, string>;
  lastMessage: string | null;
}

type DirectiveAction =
  | { type: "highlight"; ids: string[] }
  | { type: "focus"; ids: string[] }
  | { type: "set_metric"; metric: ColorMetric }
  | { type: "annotate"; residue: string; text: string }
  | { type: "clear" }
  | { type: "set_message"; msg: string | null };

function reducer(state: DirectiveState, action: DirectiveAction): DirectiveState {
  switch (action.type) {
    case "highlight":
      return { ...state, highlights: new Set(action.ids) };
    case "focus":
      return { ...state, focusTargets: action.ids };
    case "set_metric":
      return { ...state, activeMetric: action.metric };
    case "annotate": {
      const next = new Map(state.annotations);
      next.set(action.residue, action.text);
      return { ...state, annotations: next };
    }
    case "clear":
      return { ...state, highlights: new Set(), focusTargets: [], annotations: new Map(), lastMessage: null };
    case "set_message":
      return { ...state, lastMessage: action.msg };
    default:
      return state;
  }
}

export function useDirectives(initialMetric: ColorMetric = "cone_depth") {
  const [state, dispatch] = useReducer(reducer, {
    highlights: new Set<string>(),
    focusTargets: [],
    activeMetric: initialMetric,
    annotations: new Map<string, string>(),
    lastMessage: null,
  });

  const applyDirective = useCallback((directive: ViewportDirective) => {
    switch (directive.action) {
      case "highlight":
        dispatch({ type: "highlight", ids: directive.residue_ids ?? [] });
        if (directive.message) dispatch({ type: "set_message", msg: directive.message });
        break;
      case "focus":
        dispatch({ type: "focus", ids: directive.residue_ids ?? [] });
        if (directive.focus_residue) dispatch({ type: "focus", ids: [directive.focus_residue] });
        break;
      case "set_metric":
      case "set_color_mode":
        if (directive.metric) dispatch({ type: "set_metric", metric: directive.metric as ColorMetric });
        if (directive.color_mode) dispatch({ type: "set_metric", metric: directive.color_mode as ColorMetric });
        break;
      case "annotate":
        if (directive.residue_ids?.length && directive.annotation) {
          directive.residue_ids.forEach(id =>
            dispatch({ type: "annotate", residue: id, text: directive.annotation! })
          );
        }
        break;
      case "clear":
        dispatch({ type: "clear" });
        break;
    }
    if (directive.message) dispatch({ type: "set_message", msg: directive.message });
  }, []);

  const clearHighlights = useCallback(() => dispatch({ type: "clear" }), []);

  return {
    highlights: state.highlights,
    focusTargets: state.focusTargets,
    activeMetric: state.activeMetric,
    annotations: state.annotations,
    lastMessage: state.lastMessage,
    applyDirective,
    clearHighlights,
  };
}

// Utility helpers consumed by visualizer components

export function getHighlightColor(nodeId: string, highlights: Set<string>): string | null {
  return highlights.has(nodeId) ? "#fbbf24" : null;
}

export function shouldPulse(nodeId: string, focusTargets: string[]): boolean {
  return focusTargets.includes(nodeId);
}
