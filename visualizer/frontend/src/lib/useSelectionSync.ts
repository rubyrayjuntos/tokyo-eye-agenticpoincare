/**
 * useSelectionSync — Cross-viewport selection synchronization hook
 *
 * Handles:
 * - Emitting selection events to backend via WebSocket
 * - Receiving selection_sync from backend and updating all viewports
 * - Clearing previous selection before applying new one
 * - Handling selection_error gracefully (toast notification)
 *
 * Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
 */

import { useState, useCallback, useRef } from "react";
import type { ViewportEvent } from "./useViewportSocket";
import type { SelectionSyncPayload, SelectionErrorPayload } from "./useViewportSocket";

export interface ResidueSelection {
  structureId: string;
  chainId: string;
  residueNumber: number;
  source: "user" | "agent" | "sync";
}

export interface SelectionSyncState {
  currentSelection: ResidueSelection | null;
  selectionError: string | null;
  /** Auto-clears after timeout */
  toastMessage: string | null;
}

export interface UseSelectionSyncOptions {
  /** Function to emit viewport events via WS */
  sendEvent: (event: ViewportEvent) => void;
  /** Callback to update highlight state in viewport machine */
  onSelectionApplied: (residueIds: string[]) => void;
  /** Callback to clear highlights (before applying new selection) */
  onSelectionCleared: () => void;
}

export function useSelectionSync({
  sendEvent,
  onSelectionApplied,
  onSelectionCleared,
}: UseSelectionSyncOptions) {
  const [state, setState] = useState<SelectionSyncState>({
    currentSelection: null,
    selectionError: null,
    toastMessage: null,
  });

  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Show a toast message that auto-dismisses
  const showToast = useCallback((message: string) => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setState((s) => ({ ...s, toastMessage: message }));
    toastTimer.current = setTimeout(() => {
      setState((s) => ({ ...s, toastMessage: null }));
    }, 4000);
  }, []);

  /**
   * Emit a selection from any viewport (user click in Poincaré or 3D viewer).
   * Sends the selection to backend via WS for sync broadcasting.
   */
  const emitSelection = useCallback(
    (residueId: string, structureId: string) => {
      // Parse residueId format: "A:123" or just "123"
      let chainId = "A";
      let residueNumber = 0;
      if (residueId.includes(":")) {
        const parts = residueId.split(":");
        chainId = parts[0];
        residueNumber = parseInt(parts[1], 10);
      } else {
        residueNumber = parseInt(residueId, 10) || 0;
      }

      // Clear previous selection before applying new one (Req 6.4)
      onSelectionCleared();

      // Update local state
      const selection: ResidueSelection = {
        structureId,
        chainId,
        residueNumber,
        source: "user",
      };
      setState((s) => ({ ...s, currentSelection: selection, selectionError: null }));

      // Emit to backend via WS
      sendEvent({
        type: "viewport_event",
        viewport_id: "dashboard-main",
        event_type: "selection",
        payload: { residue_ids: [residueId] },
      });

      // Apply locally (optimistic update so the UI responds immediately)
      onSelectionApplied([residueId]);
    },
    [sendEvent, onSelectionApplied, onSelectionCleared],
  );

  /**
   * Handle selection_sync received from backend.
   * Updates all viewports to highlight the same residue.
   */
  const handleSelectionSync = useCallback(
    (payload: SelectionSyncPayload) => {
      const rs = payload.residue_selection;
      if (!rs) return;

      // Clear previous before applying (Req 6.4)
      onSelectionCleared();

      const selection: ResidueSelection = {
        structureId: rs.structure_id,
        chainId: rs.chain_id,
        residueNumber: rs.residue_number,
        source: rs.source as "user" | "agent" | "sync",
      };

      setState((s) => ({ ...s, currentSelection: selection, selectionError: null }));

      // Apply highlight to all viewports
      const residueId = `${rs.chain_id}:${rs.residue_number}`;
      onSelectionApplied([residueId]);
    },
    [onSelectionApplied, onSelectionCleared],
  );

  /**
   * Handle selection_error from backend (Req 6.5).
   * Display error state rather than silently failing.
   */
  const handleSelectionError = useCallback(
    (payload: SelectionErrorPayload) => {
      setState((s) => ({
        ...s,
        selectionError: payload.error,
      }));
      showToast(`Selection error: ${payload.error}`);
    },
    [showToast],
  );

  /** Clear the current selection (user action or directive) */
  const clearSelection = useCallback(() => {
    onSelectionCleared();
    setState({ currentSelection: null, selectionError: null, toastMessage: null });
  }, [onSelectionCleared]);

  return {
    ...state,
    emitSelection,
    handleSelectionSync,
    handleSelectionError,
    clearSelection,
  };
}
