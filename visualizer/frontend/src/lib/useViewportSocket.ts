/**
 * WebSocket hook for the Viewport Protocol.
 *
 * Connects to /ws/viewport on the backend and:
 * - Registers the frontend viewport on connect
 * - Receives ViewportDirectives pushed by the agent
 * - Receives state_snapshot and phase_transition messages for state derivation
 * - Sends selection/hover events back to the backend
 * - Auto-reconnects on disconnect with exponential backoff
 */

import { useEffect, useRef, useCallback, useState } from "react";
import type { ViewportDirective } from "./types";

// In development, use relative WebSocket path so Vite proxy forwards to backend.
// In production (or when VITE_WS_URL is set), use the explicit URL.
const DEFAULT_WS_BASE =
  typeof window !== "undefined" && import.meta.env.DEV
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`
    : typeof window !== "undefined"
      ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000`
      : "ws://localhost:8000";

const WS_BASE = import.meta.env.VITE_WS_URL || DEFAULT_WS_BASE;

// ---------------------------------------------------------------------------
// Server-pushed message types (from backend orchestrator)
// ---------------------------------------------------------------------------

export interface StateSnapshotPayload {
  session_id: string;
  discovery_phase: string;
  hypothesis_lifecycle: string;
  structure_scope: Record<string, unknown>;
  selected_residue: {
    structure_id: string;
    chain_id: string;
    residue_number: number;
    source: string;
  } | null;
  policy: {
    allowed_tools: string[];
    blocked_tools: string[];
    preferred_tools: string[];
    discovery_phase: string;
    hypothesis_state: string;
    reasoning_mode: string;
    session_mode: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface PhaseTransitionPayload {
  phase: string;
  source: string;
  snapshot: StateSnapshotPayload;
}

export interface HypothesisTransitionPayload {
  state: string;
  snapshot: StateSnapshotPayload;
}

export interface SelectionSyncPayload {
  residue_selection: {
    structure_id: string;
    chain_id: string;
    residue_number: number;
    source: string;
  };
}

export interface SelectionErrorPayload {
  error: string;
}

// ---------------------------------------------------------------------------
// Hook options
// ---------------------------------------------------------------------------

interface UseViewportSocketOptions {
  /** Called when a directive is received from the backend */
  onDirective: (directive: ViewportDirective) => void;
  /** Called when a state_snapshot is received (initial connect or reconnect) */
  onStateSnapshot?: (snapshot: StateSnapshotPayload) => void;
  /** Called when a phase_transition is received */
  onPhaseTransition?: (payload: PhaseTransitionPayload) => void;
  /** Called when a hypothesis_transition is received */
  onHypothesisTransition?: (payload: HypothesisTransitionPayload) => void;
  /** Called when a selection_sync is received */
  onSelectionSync?: (payload: SelectionSyncPayload) => void;
  /** Called when a selection_error is received */
  onSelectionError?: (payload: SelectionErrorPayload) => void;
  /** Whether the socket should be active */
  enabled?: boolean;
  /** Shared session id used by chat and viewport websocket */
  sessionId?: string | null;
}

interface ViewportSocketState {
  connected: boolean;
  reconnecting: boolean;
  error: string | null;
}

export function useViewportSocket({
  onDirective,
  onStateSnapshot,
  onPhaseTransition,
  onHypothesisTransition,
  onSelectionSync,
  onSelectionError,
  enabled = true,
  sessionId = null,
}: UseViewportSocketOptions): ViewportSocketState & {
  sendEvent: (event: ViewportEvent) => void;
} {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempts = useRef(0);
  const [state, setState] = useState<ViewportSocketState>({
    connected: false,
    reconnecting: false,
    error: null,
  });

  // Stable refs for all callbacks so reconnects don't stale-close over them
  const onDirectiveRef = useRef(onDirective);
  onDirectiveRef.current = onDirective;
  const onStateSnapshotRef = useRef(onStateSnapshot);
  onStateSnapshotRef.current = onStateSnapshot;
  const onPhaseTransitionRef = useRef(onPhaseTransition);
  onPhaseTransitionRef.current = onPhaseTransition;
  const onHypothesisTransitionRef = useRef(onHypothesisTransition);
  onHypothesisTransitionRef.current = onHypothesisTransition;
  const onSelectionSyncRef = useRef(onSelectionSync);
  onSelectionSyncRef.current = onSelectionSync;
  const onSelectionErrorRef = useRef(onSelectionError);
  onSelectionErrorRef.current = onSelectionError;

  const connect = useCallback(() => {
    if (!enabled) return;

    try {
      // In dev, REQUIRE_AUTH=false so no token needed
      const url = new URL(`${WS_BASE}/ws/viewport`);
      if (sessionId) {
        url.searchParams.set("session_id", sessionId);
      }
      const ws = new WebSocket(url.toString());
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttempts.current = 0;
        setState({ connected: true, reconnecting: false, error: null });

        // Register our viewport
        ws.send(
          JSON.stringify({
            type: "viewport_register",
            viewport_id: "dashboard-main",
            viewer_type: "poincare_disc",
            capabilities: ["highlight", "set_metric", "focus", "clear"],
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const msgType = data.type;

          // Route based on message type
          switch (msgType) {
            case "state_snapshot":
              onStateSnapshotRef.current?.(data.payload ?? data);
              break;

            case "phase_transition":
              onPhaseTransitionRef.current?.(data.payload ?? data);
              // Phase transitions include a snapshot — also fire snapshot handler
              if (data.payload?.snapshot || data.snapshot) {
                onStateSnapshotRef.current?.(data.payload?.snapshot ?? data.snapshot);
              }
              break;

            case "hypothesis_transition":
              onHypothesisTransitionRef.current?.(data.payload ?? data);
              // Hypothesis transitions also include a snapshot
              if (data.payload?.snapshot || data.snapshot) {
                onStateSnapshotRef.current?.(data.payload?.snapshot ?? data.snapshot);
              }
              break;

            case "selection_sync":
              onSelectionSyncRef.current?.(data.payload ?? data);
              break;

            case "selection_error":
              onSelectionErrorRef.current?.(data.payload ?? data);
              break;

            case "semantic_command": {
              // Directive pushed from agent — extract and dispatch
              const directive = data as ViewportDirective;
              if (
                directive.action &&
                ["highlight", "set_metric", "focus", "clear", "annotate", "compare_runs"].includes(
                  directive.action
                )
              ) {
                onDirectiveRef.current(directive);
              }
              break;
            }

            default: {
              // Legacy format: check if it looks like a directive directly
              const directive = data.directive ?? data;
              if (
                directive.action &&
                ["highlight", "set_metric", "focus", "clear", "annotate", "compare_runs"].includes(
                  directive.action
                )
              ) {
                onDirectiveRef.current(directive as ViewportDirective);
              }
              // register_ack, pong, etc. are informational — ignore
              break;
            }
          }
        } catch {
          // Non-JSON message, ignore
        }
      };

      ws.onclose = () => {
        setState({ connected: false, reconnecting: true, error: null });
        wsRef.current = null;
        scheduleReconnect();
      };

      ws.onerror = () => {
        setState((s) => ({ ...s, error: "WebSocket connection failed" }));
      };
    } catch (err) {
      setState({ connected: false, reconnecting: false, error: "Failed to create WebSocket" });
    }
  }, [enabled, sessionId]);

  const scheduleReconnect = useCallback(() => {
    if (!enabled) return;
    const delay = Math.min(1000 * 2 ** reconnectAttempts.current, 30000);
    reconnectAttempts.current += 1;
    reconnectTimer.current = setTimeout(connect, delay);
  }, [connect, enabled]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    if (enabled) {
      connect();
    }
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.close(1000);
        wsRef.current = null;
      }
    };
  }, [connect, enabled]);

  const sendEvent = useCallback((event: ViewportEvent) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(event));
    }
  }, []);

  return { ...state, sendEvent };
}

// --- Event types sent FROM frontend TO backend ---

export interface ViewportSelectionEvent {
  type: "viewport_event";
  viewport_id: string;
  event_type: "selection";
  payload: { residue_ids: string[] };
}

export interface ViewportHoverEvent {
  type: "viewport_event";
  viewport_id: string;
  event_type: "hover";
  payload: { residue_id: string | null };
}

export interface ViewportMetricChangeEvent {
  type: "viewport_event";
  viewport_id: string;
  event_type: "metric_change";
  payload: { metric: string };
}

export type ViewportEvent =
  | ViewportSelectionEvent
  | ViewportHoverEvent
  | ViewportMetricChangeEvent;
