/**
 * Hook for managing viewport directive state — highlights, focus, annotations
 * pushed by the agent via WebSocket.
 */

import { useState, useCallback } from "react";
import type { ViewportDirective } from "./types";

export interface HighlightEntry {
  residueIds: string[];
  color: string;
  style: "glow" | "pulse" | "outline" | "color";
  label?: string;
}

export interface AnnotationEntry {
  residueId: string;
  text: string;
}

export interface FocusRecenterRequest {
  residueIds: string[];
}

export function useDirectives(initialMetric = "cone_depth") {
  const [highlights, setHighlights] = useState<HighlightEntry[]>([]);
  const [focusTargets, setFocusTargets] = useState<string[]>([]);
  const [focusRecenter, setFocusRecenter] = useState<FocusRecenterRequest | null>(null);
  const [activeMetric, setActiveMetric] = useState(initialMetric);
  const [annotations, setAnnotations] = useState<AnnotationEntry[]>([]);
  const [lastMessage, setLastMessage] = useState<string | null>(null);

  const applyDirective = useCallback((directive: ViewportDirective) => {
    switch (directive.action) {
      case "highlight":
        if (directive.highlight_groups) {
          setHighlights(
            directive.highlight_groups.map((g) => ({
              residueIds: g.residue_ids,
              color: g.color,
              style: g.style,
              label: g.label,
            }))
          );
        }
        break;

      case "set_metric":
        if (directive.metric) {
          setActiveMetric(directive.metric);
        }
        break;

      case "focus":
        if (directive.focus_residues) {
          setFocusTargets(directive.focus_residues);
          // Signal that we need to recenter the disc on these residues
          setFocusRecenter({ residueIds: directive.focus_residues });
        }
        break;

      case "clear":
        setHighlights([]);
        setFocusTargets([]);
        setFocusRecenter(null);
        setAnnotations([]);
        break;

      case "annotate":
        if (directive.message && directive.focus_residues?.length) {
          const newAnnotations = directive.focus_residues.map((rid) => ({
            residueId: rid,
            text: directive.message!,
          }));
          setAnnotations((prev) => [...prev, ...newAnnotations]);
        }
        break;
    }

    if (directive.message) {
      setLastMessage(directive.message);
      setTimeout(() => setLastMessage(null), 5000);
    }
  }, []);

  const clearHighlights = useCallback(() => {
    setHighlights([]);
    setFocusTargets([]);
    setFocusRecenter(null);
    setAnnotations([]);
    setLastMessage(null);
  }, []);

  /** Consume and clear the pending focus-recenter request */
  const consumeFocusRecenter = useCallback(() => {
    setFocusRecenter(null);
  }, []);

  return {
    highlights,
    focusTargets,
    focusRecenter,
    activeMetric,
    annotations,
    lastMessage,
    applyDirective,
    clearHighlights,
    consumeFocusRecenter,
  };
}

/**
 * Get the highlight color for a node, if it's in any highlight group.
 */
export function getHighlightColor(
  nodeId: string,
  highlights: HighlightEntry[]
): string | null {
  for (const group of highlights) {
    if (group.residueIds.includes(nodeId)) {
      return group.color;
    }
  }
  return null;
}

/**
 * Check if a node should pulse (based on highlight style).
 */
export function shouldPulse(
  nodeId: string,
  highlights: HighlightEntry[]
): boolean {
  for (const group of highlights) {
    if (group.style === "pulse" && group.residueIds.includes(nodeId)) {
      return true;
    }
  }
  return false;
}
