import { useEffect, useRef } from "react";

import { api } from "../lib/api";
import type { StateSnapshotPayload } from "../lib/useViewportSocket";
import type { DiscoveryPhase } from "../lib/discoveryPhaseMachine";
import { useWorkbenchBus } from "./WorkbenchProvider";
import type { WorkbenchBus } from "./WorkbenchBus";
import {
  discoveryPhaseToGroup,
  groupToDefaultDiscoveryPhase,
  type WorkbenchPhaseGroup,
} from "./phaseGroups";

interface UseOrchestrationSyncOptions {
  sessionId: string | null;
  onDiscoveryPhase?: (phase: DiscoveryPhase) => void;
}

export function snapshotFingerprint(snapshot: StateSnapshotPayload): string {
  return JSON.stringify({
    session_id: snapshot.session_id,
    discovery_phase: snapshot.discovery_phase,
    hypothesis_lifecycle: snapshot.hypothesis_lifecycle,
    selected_residue: snapshot.selected_residue,
  });
}

export function useOrchestrationSync({
  sessionId,
  onDiscoveryPhase,
}: UseOrchestrationSyncOptions): {
  setPhaseGroup: (group: WorkbenchPhaseGroup) => Promise<void>;
  setDiscoveryPhase: (phase: DiscoveryPhase) => Promise<void>;
} {
  const bus = useWorkbenchBus();
  const onDiscoveryPhaseRef = useRef(onDiscoveryPhase);
  onDiscoveryPhaseRef.current = onDiscoveryPhase;
  const lastFetchedSessionRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    if (lastFetchedSessionRef.current === sessionId) return;
    lastFetchedSessionRef.current = sessionId;

    let cancelled = false;
    void api
      .getOrchestrationSnapshot(sessionId)
      .then((snapshot) => {
        if (cancelled) return;
        applyBackendSnapshot(
          bus,
          snapshot as StateSnapshotPayload,
          (phase) => onDiscoveryPhaseRef.current?.(phase),
        );
      })
      .catch((error) => {
        if (cancelled) return;
        console.warn("[useOrchestrationSync] Failed to load orchestration snapshot:", error);
        lastFetchedSessionRef.current = null;
      });

    return () => {
      cancelled = true;
    };
  }, [bus, sessionId]);

  const setDiscoveryPhase = async (phase: DiscoveryPhase) => {
    bus.publish("system:phase_transition", {
      group: discoveryPhaseToGroup(phase),
      discoveryPhase: phase,
      timestamp: new Date().toISOString(),
    });

    if (sessionId) {
      try {
        await api.postOrchestrationEvent(sessionId, {
          type: "USER_SET_PHASE",
          phase,
        });
      } catch (error) {
        console.warn("[useOrchestrationSync] Failed to post discovery phase:", error);
      }
    }

    onDiscoveryPhaseRef.current?.(phase);
  };

  const setPhaseGroup = async (group: WorkbenchPhaseGroup) => {
    await setDiscoveryPhase(groupToDefaultDiscoveryPhase(group));
  };

  return { setPhaseGroup, setDiscoveryPhase };
}

export function applyBackendSnapshot(
  bus: WorkbenchBus,
  snapshot: StateSnapshotPayload,
  onDiscoveryPhase?: (phase: DiscoveryPhase) => void,
): boolean {
  const fingerprint = snapshotFingerprint(snapshot);
  const lastApplied = bus.getLatestState("system:orchestration_snapshot");
  if (lastApplied && snapshotFingerprint(lastApplied as StateSnapshotPayload) === fingerprint) {
    return false;
  }

  bus.setOrchestrationSnapshot(snapshot);
  bus.publish("system:orchestration_snapshot", snapshot);
  if (snapshot.discovery_phase) {
    const phase = snapshot.discovery_phase as DiscoveryPhase;
    onDiscoveryPhase?.(phase);
    bus.publish("system:phase_transition", {
      group: discoveryPhaseToGroup(phase),
      discoveryPhase: phase,
      timestamp: new Date().toISOString(),
    });
  }
  return true;
}
