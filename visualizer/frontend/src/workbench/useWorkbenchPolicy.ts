import { useEffect, useMemo, useState } from "react";

import type { DiscoveryPhase } from "../lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "../lib/hypothesisLifecycleMachine";
import type { PlannerPolicy, SessionMode } from "../lib/plannerPolicySelector";
import type { OrchestratorPolicyResult } from "../lib/useOrchestratorPolicy";
import type { WorkbenchBus } from "./WorkbenchBus";

function mergeBackendPolicy(
  local: PlannerPolicy,
  workbenchBus: WorkbenchBus,
): PlannerPolicy {
  const snapshot = workbenchBus.getLatestState("system:orchestration_snapshot");
  const policy = snapshot?.policy;
  if (!snapshot || !policy) {
    return local;
  }

  return {
    ...local,
    discoveryPhase: (snapshot.discovery_phase as DiscoveryPhase) ?? local.discoveryPhase,
    hypothesisState:
      (snapshot.hypothesis_lifecycle as HypothesisLifecycleState) ??
      local.hypothesisState,
    allowedTools: (policy.allowed_tools as string[] | undefined) ?? local.allowedTools,
    blockedTools: (policy.blocked_tools as string[] | undefined) ?? local.blockedTools,
    preferredTools:
      (policy.preferred_tools as string[] | undefined) ?? local.preferredTools,
    sessionMode: (policy.session_mode as SessionMode | undefined) ?? local.sessionMode,
    reasoningMode:
      (policy.reasoning_mode as PlannerPolicy["reasoningMode"] | undefined) ??
      local.reasoningMode,
    primaryStructureId:
      (snapshot.structure_scope as { structure_id?: string } | undefined)?.structure_id ??
      local.primaryStructureId,
  };
}

export function useWorkbenchPolicy(
  orchestrator: OrchestratorPolicyResult,
  workbenchBus?: WorkbenchBus,
): OrchestratorPolicyResult {
  const [policyVersion, setPolicyVersion] = useState(0);

  useEffect(() => {
    if (!workbenchBus) return;

    const bump = () => setPolicyVersion((value) => value + 1);
    const unsubSnapshot = workbenchBus.subscribe("system:orchestration_snapshot", bump);
    const unsubPhase = workbenchBus.subscribe("system:phase_transition", bump);
    const unsubScope = workbenchBus.subscribe("data:structure_scope_updated", bump);

    return () => {
      unsubSnapshot();
      unsubPhase();
      unsubScope();
    };
  }, [workbenchBus]);

  const plannerPolicy = useMemo(() => {
    if (!workbenchBus) {
      return orchestrator.plannerPolicy;
    }
    return mergeBackendPolicy(orchestrator.plannerPolicy, workbenchBus);
  }, [orchestrator.plannerPolicy, policyVersion, workbenchBus]);

  return useMemo(
    () => ({
      ...orchestrator,
      plannerPolicy,
    }),
    [orchestrator, plannerPolicy],
  );
}
