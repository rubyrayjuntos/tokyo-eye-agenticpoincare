/**
 * WebSocket hook for the Viewport Protocol.
 *
 * Connects to /ws/viewport on the backend and:
 * - Registers the frontend viewport on connect
 * - Receives ViewportDirectives pushed by the agent
 * - Sends selection/hover events back to the backend
 * - Auto-reconnects on disconnect with exponential backoff
 */

import { useEffect, useRef, useCallback, useState } from "react";
import type { ViewportDirective } from "./types";

const DEFAULT_WS_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000`
    : "ws://localhost:8000";

const WS_BASE = import.meta.env.VITE_WS_URL || DEFAULT_WS_BASE;

interface UseViewportSocketOptions {
  /** Called when a directive is received from the backend */
  onDirective: (directive: ViewportDirective) => void;
  /** Whether the socket should be active */
  enabled?: boolean;
  /** Shared session id used by chat and viewport websocket */
  sessionId?: string | null;
}

interface ViewportSocketState {
  connected: boolean;
  error: string | null;
}

export function useViewportSocket({
  onDirective,
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
    error: null,
  });

  // Stable ref for the callback so reconnects don't stale-close over it
  const onDirectiveRef = useRef(onDirective);
  onDirectiveRef.current = onDirective;

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
        setState({ connected: true, error: null });

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
        } catch {
          // Non-JSON message, ignore
        }
      };

      ws.onclose = () => {
        setState({ connected: false, error: null });
        wsRef.current = null;
        scheduleReconnect();
      };

      ws.onerror = () => {
        setState((s) => ({ ...s, error: "WebSocket connection failed" }));
      };
    } catch (err) {
      setState({ connected: false, error: "Failed to create WebSocket" });
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
