/**
 * Agent client — handles REST calls and WebSocket viewport protocol for the
 * Poincaré disc visualizer talking to the FastAPI agent backend.
 */

import type { ViewportDirective } from "./types";

const DEFAULT_API_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://localhost:8000";

const API_BASE = import.meta.env.VITE_API_URL || DEFAULT_API_BASE;

const DEFAULT_WS_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000`
    : "ws://localhost:8000";

const WS_BASE = import.meta.env.VITE_WS_URL || DEFAULT_WS_BASE;

// Re-export the ViewportDirective type for convenience
export type { ViewportDirective } from "./types";

/** Viewport state payload sent with chat messages (Requirements 2.1) */
export interface ChatViewportState {
  structure_id?: string | null;
  current_metric?: string;
  curvature?: number;
  poincare_color_mode?: string | null;
  viewer_3d_color_mode?: string | null;
  selected_residue?: string | null;
  highlighted_residues?: string[];
  active_panel?: string | null;
  pipeline_flags?: Record<string, boolean>;
  risk_threshold?: number | null;
  brush_selection?: string[];
}

/** Chat response from the backend */
export interface ChatResponse {
  response: string;
  session_id: string;
  viewport_directives?: ViewportDirective[] | null;
  tool_results?: Record<string, unknown>[] | null;
}

/**
 * Send a chat message to the agent with full viewport state context.
 * Requirements: 2.1 — frontend includes viewport state in request payload.
 * Requirements: 3.1 — optional poincare_snapshot attached when relevant.
 */
export async function sendChat(
  message: string,
  sessionId: string,
  viewportState: ChatViewportState,
  poincareSnapshot?: string | null,
): Promise<ChatResponse> {
  const body: Record<string, unknown> = {
    message,
    session_id: sessionId,
  };

  // Include visual snapshot if provided (Requirements 3.1, 3.2)
  if (poincareSnapshot) {
    body.poincare_snapshot = poincareSnapshot;
  }

  // Include the core viewport_state for backward compatibility
  if (viewportState.structure_id || viewportState.current_metric || viewportState.curvature) {
    body.viewport_state = {
      structure_id: viewportState.structure_id ?? null,
      current_metric: viewportState.current_metric ?? "cone_depth",
      curvature: viewportState.curvature ?? null,
    };
  }

  // Include extended viewport fields for context enrichment
  if (viewportState.poincare_color_mode) {
    body.poincare_color_mode = viewportState.poincare_color_mode;
  }
  if (viewportState.viewer_3d_color_mode) {
    body.viewer_3d_color_mode = viewportState.viewer_3d_color_mode;
  }
  if (viewportState.selected_residue) {
    body.selected_residue = viewportState.selected_residue;
  }
  if (viewportState.highlighted_residues && viewportState.highlighted_residues.length > 0) {
    body.highlighted_residues = viewportState.highlighted_residues;
  }
  if (viewportState.active_panel) {
    body.active_panel = viewportState.active_panel;
  }
  if (viewportState.pipeline_flags && Object.keys(viewportState.pipeline_flags).length > 0) {
    body.pipeline_flags = viewportState.pipeline_flags;
  }
  if (viewportState.risk_threshold != null) {
    body.risk_threshold = viewportState.risk_threshold;
  }
  if (viewportState.brush_selection && viewportState.brush_selection.length > 0) {
    body.brush_selection = viewportState.brush_selection;
  }

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Chat request failed (${res.status}): ${text}`);
  }

  return res.json();
}

/** Residue data returned by the poincare endpoint */
export interface PoincareResidue {
  residue_id: string;
  chain_label: string;
  residue_name: string;
  x: number;
  y: number;
  cone_depth: number;
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
}

export interface PoincareDataResponse {
  structure_id: string;
  pdb_id: string;
  curvature_c: number | null;
  residues: PoincareResidue[];
}

/**
 * Check if the agent backend is reachable.
 */
export async function checkAgent(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/kpis`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Fetch Poincaré disc data for a given PDB ID from the agent backend.
 */
export async function fetchPoincareData(pdbId: string): Promise<PoincareDataResponse> {
  const res = await fetch(`${API_BASE}/api/structures/${pdbId}/embeddings`);
  if (!res.ok) throw new Error(`Failed to fetch poincare data: ${res.status}`);
  const data = await res.json();

  // Normalize backend response shape
  return {
    structure_id: data.structure_id ?? pdbId,
    pdb_id: pdbId,
    curvature_c: data.curvature ?? null,
    residues: (data.residues ?? []).map((r: Record<string, unknown>) => ({
      residue_id: String(r.residue_id ?? ""),
      chain_label: String(r.chain_label ?? ""),
      residue_name: String(r.residue_name ?? ""),
      x: Number(r.x ?? 0),
      y: Number(r.y ?? 0),
      cone_depth: Number(r.cone_depth ?? 0),
      epistemic_uncertainty: Number(r.epistemic_uncertainty ?? 0),
      aleatoric_uncertainty: Number(r.aleatoric_uncertainty ?? 0),
    })),
  };
}

// ---------------------------------------------------------------------------
// WebSocket viewport client (imperative class — used by App.tsx directly)
// ---------------------------------------------------------------------------

export interface ViewportClientOptions {
  sessionId: string;
  onDirective: (directive: ViewportDirective) => void;
  onConnection: (connected: boolean) => void;
  onSnapshotRequest?: () => void;
}

export class ViewportClient {
  private ws: WebSocket | null = null;
  private options: ViewportClientOptions;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private attempts = 0;

  constructor(options: ViewportClientOptions) {
    this.options = options;
  }

  connect(): void {
    try {
      const url = new URL(`${WS_BASE}/ws/viewport`);
      url.searchParams.set("session_id", this.options.sessionId);
      const ws = new WebSocket(url.toString());
      this.ws = ws;

      ws.onopen = () => {
        this.attempts = 0;
        this.options.onConnection(true);
        ws.send(JSON.stringify({
          type: "viewport_register",
          viewport_id: "dashboard-main",
          viewer_type: "poincare_disc",
          capabilities: ["highlight", "set_metric", "focus", "clear"],
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Handle snapshot request from agent (Requirements 3.3)
          if (data.type === "request_snapshot") {
            this.options.onSnapshotRequest?.();
            return;
          }
          const directive = data.directive ?? data;
          if (directive.action) {
            this.options.onDirective(directive as ViewportDirective);
          }
        } catch { /* non-JSON, ignore */ }
      };

      ws.onclose = () => {
        this.options.onConnection(false);
        this.ws = null;
        this.scheduleReconnect();
      };

      ws.onerror = () => {
        this.options.onConnection(false);
      };
    } catch {
      this.options.onConnection(false);
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.ws) {
      this.ws.close(1000);
      this.ws = null;
    }
  }

  sendHover(nodeId: string | null): void {
    this.send({
      type: "viewport_event",
      viewport_id: "dashboard-main",
      event_type: "hover",
      payload: { residue_id: nodeId },
    });
  }

  sendSelection(residueIds: string[]): void {
    this.send({
      type: "viewport_event",
      viewport_id: "dashboard-main",
      event_type: "selection",
      payload: { residue_ids: residueIds },
    });
  }

  private send(msg: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  private scheduleReconnect(): void {
    const delay = Math.min(1000 * 2 ** this.attempts, 30000);
    this.attempts += 1;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }
}
