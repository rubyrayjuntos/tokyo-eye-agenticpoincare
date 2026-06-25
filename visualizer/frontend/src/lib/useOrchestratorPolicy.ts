/**
 * useOrchestratorPolicy — React hook that binds all XState machines
 * (discovery phase, hypothesis lifecycle) and derives a combined
 * PlannerPolicy for the orchestrator.
 *
 * Consumers get:
 * - machine context snapshots (discovery, hypothesis)
 * - planner policy (merged from all machines + session mode)
 * - strongly-typed dispatch helpers
 */

import { useCallback, useMemo, useState } from 'react';
import { useMachine } from '@xstate/react';
import { discoveryPhaseMachine } from './discoveryPhaseMachine';
import { hypothesisLifecycleMachine } from './hypothesisLifecycleMachine';
import {
  plannerPolicySelector,
  createDefaultStructureScope,
  type PlannerPolicy,
  type SessionMode,
  type StructureScopeContext,
} from './plannerPolicySelector';
import type { DiscoveryPhase, DiscoveryPhaseEvent } from './discoveryPhaseMachine';
import type { HypothesisLifecycleEvent } from './hypothesisLifecycleMachine';
import type { ViewportContext } from './viewportMachine';

// ---------------------------------------------------------------------------
// Hook return type
// ---------------------------------------------------------------------------

export interface OrchestratorPolicyResult {
  // Derived policy — the single object the orchestrator consumes
  plannerPolicy: PlannerPolicy;

  // Machine state snapshots for direct UI inspection
  discoveryContext: ReturnType<typeof discoveryPhaseMachine.getInitialSnapshot>['context'];
  hypothesisContext: ReturnType<typeof hypothesisLifecycleMachine.getInitialSnapshot>['context'];

  // Structure scope (for display and mutation)
  structureScope: StructureScopeContext;
  setStructureScope: (scope: StructureScopeContext) => void;
  sessionMode: SessionMode;
  setSessionMode: (mode: SessionMode) => void;

  // Discovery phase dispatch
  sendDiscovery: (event: DiscoveryPhaseEvent) => void;
  setDiscoveryPhase: (phase: DiscoveryPhase, rationale?: string) => void;

  // Hypothesis lifecycle dispatch
  sendHypothesis: (event: HypothesisLifecycleEvent) => void;
  startHypothesis: (text: string, source: 'user' | 'agent') => void;
  addEvidence: (ref?: string) => void;
  addContradiction: (reason: string, ref?: string) => void;
  reviseHypothesis: (text: string) => void;
  markSupported: () => void;
  markSynthesized: () => void;
  clearHypothesis: () => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useOrchestratorPolicy(
  viewportContext: ViewportContext,
): OrchestratorPolicyResult {
  // --- machines ---
  const [discoveryState, sendDiscovery] = useMachine(discoveryPhaseMachine);
  const [hypothesisState, sendHypothesis] = useMachine(hypothesisLifecycleMachine);

  // --- structure scope (local state — promote to machine if lifecycle grows) ---
  const [structureScope, setStructureScope] = useState<StructureScopeContext>(
    createDefaultStructureScope,
  );

  // --- derive planner policy ---
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

  // --- session mode shortcut ---
  const sessionMode = structureScope.sessionMode;
  const setSessionMode = useCallback(
    (mode: SessionMode) =>
      setStructureScope(prev => ({ ...prev, sessionMode: mode })),
    [],
  );

  // --- discovery helpers ---
  const setDiscoveryPhase = useCallback(
    (phase: DiscoveryPhase, rationale?: string) =>
      sendDiscovery({ type: 'USER_SET_PHASE', phase, rationale }),
    [sendDiscovery],
  );

  // --- hypothesis helpers ---
  const startHypothesis = useCallback(
    (text: string, source: 'user' | 'agent') =>
      sendHypothesis({ type: 'START_HYPOTHESIS', hypothesisText: text, source }),
    [sendHypothesis],
  );

  const addEvidence = useCallback(
    (ref?: string) =>
      sendHypothesis({ type: 'ADD_SUPPORTING_EVIDENCE', ref }),
    [sendHypothesis],
  );

  const addContradiction = useCallback(
    (reason: string, ref?: string) =>
      sendHypothesis({ type: 'ADD_CONTRADICTION', reason, ref }),
    [sendHypothesis],
  );

  const reviseHypothesis = useCallback(
    (text: string) =>
      sendHypothesis({ type: 'REVISE_HYPOTHESIS', hypothesisText: text }),
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

  return {
    plannerPolicy,
    discoveryContext: discoveryState.context,
    hypothesisContext: hypothesisState.context,
    structureScope,
    setStructureScope,
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
  };
}
