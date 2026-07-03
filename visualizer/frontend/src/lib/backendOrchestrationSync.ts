import type { DiscoveryPhaseEvent } from "./discoveryPhaseMachine";
import type { HypothesisLifecycleEvent } from "./hypothesisLifecycleMachine";
import type { HypothesisLifecycleState } from "./hypothesisLifecycleMachine";

export type OrchestrationAuthority = "local" | "backend";

export type SendHypothesis = (event: HypothesisLifecycleEvent) => void;

const BACKEND_DISCOVERY_RATIONALES = new Set(["backend_sync", "backend_push"]);

export function isBackendAuthorizedDiscoveryEvent(event: DiscoveryPhaseEvent): boolean {
  if (event.type !== "USER_SET_PHASE") {
    return true;
  }
  const rationale = event.rationale ?? "";
  return (
    BACKEND_DISCOVERY_RATIONALES.has(rationale) || rationale.startsWith("backend:")
  );
}

export function guardDiscoveryEvent(
  event: DiscoveryPhaseEvent,
  authority: OrchestrationAuthority,
): DiscoveryPhaseEvent | null {
  if (authority === "local" || isBackendAuthorizedDiscoveryEvent(event)) {
    return event;
  }
  return null;
}

export function hypothesisEventsFromBackendLifecycle(
  lifecycle: HypothesisLifecycleState,
): HypothesisLifecycleEvent[] {
  switch (lifecycle) {
    case "framed":
      return [{ type: "START_HYPOTHESIS", hypothesisText: "", source: "agent" }];
    case "supported":
      return [{ type: "MARK_SUPPORTED" }];
    case "synthesized":
      return [{ type: "MARK_SYNTHESIZED" }];
    default:
      return [];
  }
}

export function guardHypothesisEvent(
  event: HypothesisLifecycleEvent,
  authority: OrchestrationAuthority,
): HypothesisLifecycleEvent | null {
  if (authority === "local") {
    return event;
  }
  return null;
}

/** Apply hypothesis lifecycle transitions from backend snapshots (workbench authority). */
export function applyHypothesisLifecycleFromBackend(
  sendHypothesis: SendHypothesis,
  currentLabel: HypothesisLifecycleState,
  targetLifecycle: HypothesisLifecycleState,
): void {
  if (!targetLifecycle || targetLifecycle === currentLabel) {
    return;
  }

  for (const event of hypothesisEventsFromBackendLifecycle(targetLifecycle)) {
    sendHypothesis(event);
  }
}