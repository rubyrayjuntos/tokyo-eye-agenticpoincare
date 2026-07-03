import { useCallback } from "react";

import { api } from "./api";
import { useDashboard } from "./context";
import type { DiscoveryPhase } from "./discoveryPhaseMachine";
import { useOptionalWorkbenchBus } from "../workbench/useOptionalWorkbenchBus";

/**
 * User-initiated discovery phase changes.
 * In workbench (backend authority) mode, routes through the orchestration API.
 */
export function useDiscoveryPhaseActions() {
  const { sendDiscovery, agentSessionId } = useDashboard();
  const workbenchBus = useOptionalWorkbenchBus();

  const requestDiscoveryPhase = useCallback(
    async (phase: DiscoveryPhase, rationale = "user") => {
      if (workbenchBus && agentSessionId) {
        await api.postOrchestrationEvent(agentSessionId, {
          type: "USER_SET_PHASE",
          phase,
        });
        return;
      }

      sendDiscovery({ type: "USER_SET_PHASE", phase, rationale });
    },
    [agentSessionId, sendDiscovery, workbenchBus],
  );

  return { requestDiscoveryPhase, isBackendAuthority: Boolean(workbenchBus) };
}