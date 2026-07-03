/**
 * useOrchestratorPolicy — React hook that binds all XState machines
 * (discovery phase, hypothesis lifecycle, structure scope) and derives a combined
 * PlannerPolicy for the orchestrator.
 */

import { useCallback, useMemo } from 'react';
import { useMachine } from '@xstate/react';
import { discoveryPhaseMachine } from './discoveryPhaseMachine';
import { hypothesisLifecycleMachine } from './hypothesisLifecycleMachine';
import { structureScopeMachine } from './structureScopeMachine';
import type { StructureScopeEvent } from './structureScopeMachine';
import {
  plannerPolicySelector,
  type PlannerPolicy,
  type SessionMode,
  type StructureScopeContext,
} from './plannerPolicySelector';
import type { DiscoveryPhase, DiscoveryPhaseEvent } from './discoveryPhaseMachine';
import type { HypothesisLifecycleEvent } from './hypothesisLifecycleMachine';
import type { ViewportContext } from './viewportMachine';
import {
  applyHypothesisLifecycleFromBackend,
  guardDiscoveryEvent,
  guardHypothesisEvent,
  type OrchestrationAuthority,
} from './backendOrchestrationSync';
import type { HypothesisLifecycleState } from './hypothesisLifecycleMachine';

export interface UseOrchestratorPolicyOptions {
  authority?: OrchestrationAuthority;
}

export interface OrchestratorPolicyResult {
  authority: OrchestrationAuthority;
  plannerPolicy: PlannerPolicy;
  discoveryContext: ReturnType<typeof discoveryPhaseMachine.getInitialSnapshot>['context'];
  hypothesisContext: ReturnType<typeof hypothesisLifecycleMachine.getInitialSnapshot>['context'];
  structureScope: StructureScopeContext;
  setStructureScope: (scope: StructureScopeContext) => void;
  sendStructureScope: (event: StructureScopeEvent) => void;
  sessionMode: SessionMode;
  setSessionMode: (mode: SessionMode) => void;
  sendDiscovery: (event: DiscoveryPhaseEvent) => void;
  setDiscoveryPhase: (phase: DiscoveryPhase, rationale?: string) => void;
  sendHypothesis: (event: HypothesisLifecycleEvent) => void;
  startHypothesis: (text: string, source: 'user' | 'agent') => void;
  addEvidence: (ref?: string) => void;
  addContradiction: (reason: string, ref?: string) => void;
  reviseHypothesis: (text: string) => void;
  markSupported: () => void;
  markSynthesized: () => void;
  clearHypothesis: () => void;
  /** Apply hypothesis lifecycle from backend snapshot (bypasses user guard). */
  applyHypothesisFromBackend: (lifecycle: HypothesisLifecycleState) => void;
}

export function useOrchestratorPolicy(
  viewportContext: ViewportContext,
  options: UseOrchestratorPolicyOptions = {},
): OrchestratorPolicyResult {
  const authority = options.authority ?? 'local';
  const [discoveryState, sendDiscoveryRaw] = useMachine(discoveryPhaseMachine);
  const [hypothesisState, sendHypothesisRaw] = useMachine(hypothesisLifecycleMachine);
  const [structureScopeState, sendStructureScope] = useMachine(structureScopeMachine);

  const sendDiscovery = useCallback(
    (event: DiscoveryPhaseEvent) => {
      const guarded = guardDiscoveryEvent(event, authority);
      if (guarded) {
        sendDiscoveryRaw(guarded);
      }
    },
    [authority, sendDiscoveryRaw],
  );

  const sendHypothesis = useCallback(
    (event: HypothesisLifecycleEvent) => {
      const guarded = guardHypothesisEvent(event, authority);
      if (guarded) {
        sendHypothesisRaw(guarded);
      }
    },
    [authority, sendHypothesisRaw],
  );

  const structureScope = structureScopeState.context;

  const plannerPolicy = useMemo(
    () =>
      plannerPolicySelector({
        structureScope,
        viewport: viewportContext,
        discovery: discoveryState.context,
        hypothesis: hypothesisState.context,
      }),
    [structureScope, viewportContext, discoveryState.context, hypothesisState.context],
  );

  const sessionMode = structureScope.sessionMode;
  const setSessionMode = useCallback(
    (mode: SessionMode) => sendStructureScope({ type: 'SET_SESSION_MODE', mode }),
    [sendStructureScope],
  );

  const setStructureScope = useCallback(
    (scope: StructureScopeContext) =>
      sendStructureScope({ type: 'SYNC_FROM_WORKSPACE', scope }),
    [sendStructureScope],
  );

  const setDiscoveryPhase = useCallback(
    (phase: DiscoveryPhase, rationale?: string) =>
      sendDiscovery({ type: 'USER_SET_PHASE', phase, rationale }),
    [sendDiscovery],
  );

  const startHypothesis = useCallback(
    (text: string, source: 'user' | 'agent') =>
      sendHypothesis({ type: 'START_HYPOTHESIS', hypothesisText: text, source }),
    [sendHypothesis],
  );

  const addEvidence = useCallback(
    (ref?: string) => sendHypothesis({ type: 'ADD_SUPPORTING_EVIDENCE', ref }),
    [sendHypothesis],
  );

  const addContradiction = useCallback(
    (reason: string, ref?: string) =>
      sendHypothesis({ type: 'ADD_CONTRADICTION', reason, ref }),
    [sendHypothesis],
  );

  const reviseHypothesis = useCallback(
    (text: string) => sendHypothesis({ type: 'REVISE_HYPOTHESIS', hypothesisText: text }),
    [sendHypothesis],
  );

  const markSupported = useCallback(
    () => sendHypothesis({ type: 'MARK_SUPPORTED' }),
    [sendHypothesis],
  );

  const markSynthesized = useCallback(
    () => sendHypothesis({ type: 'MARK_SYNTHESIZED' }),
    [sendHypothesis],
  );

  const clearHypothesis = useCallback(
    () => sendHypothesis({ type: 'CLEAR_HYPOTHESIS' }),
    [sendHypothesis],
  );

  const applyHypothesisFromBackend = useCallback(
    (lifecycle: HypothesisLifecycleState) => {
      applyHypothesisLifecycleFromBackend(
        sendHypothesisRaw,
        hypothesisState.context.stateLabel,
        lifecycle,
      );
    },
    [sendHypothesisRaw, hypothesisState.context.stateLabel],
  );

  return {
    authority,
    plannerPolicy,
    discoveryContext: discoveryState.context,
    hypothesisContext: hypothesisState.context,
    structureScope,
    setStructureScope,
    sendStructureScope,
    sessionMode,
    setSessionMode,
    sendDiscovery,
    setDiscoveryPhase,
    sendHypothesis,
    startHypothesis,
    addEvidence,
    addContradiction,
    reviseHypothesis,
    markSupported,
    markSynthesized,
    clearHypothesis,
    applyHypothesisFromBackend,
  };
}