import { setup, assign } from 'xstate';
import { reasoningPolicyForLifecycleState } from './reasoningPolicy';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type HypothesisLifecycleState =
  | 'emergent'
  | 'framed'
  | 'testing'
  | 'supported'
  | 'contradicted'
  | 'revised'
  | 'synthesized';

export interface HypothesisLifecycleContext {
  stateLabel: HypothesisLifecycleState;

  // current claim
  hypothesisId: string | null;
  hypothesisText: string | null;
  priorHypothesisText: string | null;

  // provenance
  source: 'user' | 'agent' | 'system' | null;
  lastActor: 'user' | 'agent' | 'system' | null;
  lastTransitionAt: number | null;

  // evidence bookkeeping
  supportingEvidenceCount: number;
  contradictionCount: number;
  unresolvedContradictions: string[];
  evidenceRefs: string[];

  // reasoning posture (derived on entry)
  confidenceBand: 'low' | 'medium' | 'high';
  recommendedReasoningMode:
    | 'explore'
    | 'frame'
    | 'test'
    | 'prioritize'
    | 'regress'
    | 'revise'
    | 'synthesize';

  // orchestration control (derived on entry)
  allowedSpeechActs: string[];
  blockedSpeechActs: string[];
  canEscalateTools: boolean;
  shouldAskClarifyingQuestion: boolean;

  // reporting
  summaryReady: boolean;
}

export type HypothesisLifecycleEvent =
  | { type: 'START_HYPOTHESIS'; hypothesisText: string; source: 'user' | 'agent' }
  | { type: 'FRAME_HYPOTHESIS'; hypothesisText?: string }
  | { type: 'ADD_SUPPORTING_EVIDENCE'; ref?: string; note?: string }
  | { type: 'ADD_CONTRADICTION'; reason: string; ref?: string }
  | { type: 'BEGIN_TESTING' }
  | { type: 'REVISE_HYPOTHESIS'; hypothesisText: string }
  | { type: 'MARK_SUPPORTED' }
  | { type: 'MARK_SYNTHESIZED' }
  | { type: 'CLEAR_HYPOTHESIS' }
  | { type: 'RESET_LIFECYCLE' }
  | {
      type: 'SYNC_FROM_DISCOVERY_PHASE';
      phase: 'residue' | 'topology' | 'structure' | 'pocket' | 'screening' | 'report';
    };

// ---------------------------------------------------------------------------
// Machine
// ---------------------------------------------------------------------------

export const hypothesisLifecycleMachine = setup({
  types: {} as {
    context: HypothesisLifecycleContext;
    events: HypothesisLifecycleEvent;
  },
  guards: {
    hasHypothesisText: ({ context }) => !!context.hypothesisText,

    hasSupport: ({ context }) => context.supportingEvidenceCount > 0,

    hasStrongSupport: ({ context }) =>
      context.supportingEvidenceCount >= 2 && context.unresolvedContradictions.length === 0,

    hasContradictions: ({ context }) => context.unresolvedContradictions.length > 0,

    canSynthesize: ({ context }) =>
      context.supportingEvidenceCount > 0 && context.unresolvedContradictions.length === 0,

    canRevise: ({ event }) =>
      event.type === 'REVISE_HYPOTHESIS' && !!event.hypothesisText,
  },
  actions: {
    startHypothesis: assign(({ event }) => {
      if (event.type !== 'START_HYPOTHESIS') return {};
      return {
        hypothesisId: crypto.randomUUID(),
        hypothesisText: event.hypothesisText,
        priorHypothesisText: null,
        source: event.source,
        lastActor: event.source,
        lastTransitionAt: Date.now(),
        confidenceBand: 'low' as const,
      };
    }),

    frameHypothesis: assign(({ event, context }) => ({
      hypothesisText:
        event.type === 'FRAME_HYPOTHESIS' && event.hypothesisText
          ? event.hypothesisText
          : context.hypothesisText,
      lastActor: 'agent' as const,
      lastTransitionAt: Date.now(),
    })),

    addSupportingEvidence: assign(({ event, context }) => {
      if (event.type !== 'ADD_SUPPORTING_EVIDENCE') return {};
      return {
        supportingEvidenceCount: context.supportingEvidenceCount + 1,
        evidenceRefs: event.ref ? [...context.evidenceRefs, event.ref] : context.evidenceRefs,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    addContradiction: assign(({ event, context }) => {
      if (event.type !== 'ADD_CONTRADICTION') return {};
      return {
        contradictionCount: context.contradictionCount + 1,
        unresolvedContradictions: [...context.unresolvedContradictions, event.reason],
        evidenceRefs: event.ref ? [...context.evidenceRefs, event.ref] : context.evidenceRefs,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    reviseHypothesis: assign(({ event, context }) => {
      if (event.type !== 'REVISE_HYPOTHESIS') return {};
      return {
        priorHypothesisText: context.hypothesisText,
        hypothesisText: event.hypothesisText,
        unresolvedContradictions: [] as string[],
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
        confidenceBand: 'low' as const,
      };
    }),

    setLowConfidence: assign({ confidenceBand: 'low' as const }),
    setMediumConfidence: assign({ confidenceBand: 'medium' as const }),
    setHighConfidence: assign({ confidenceBand: 'high' as const }),

    deriveReasoningPolicy: assign(({ context }) => {
      // Use context.stateLabel since entry actions set it before this runs
      return reasoningPolicyForLifecycleState(context.stateLabel);
    }),

    clearHypothesis: assign(() => ({
      stateLabel: 'emergent' as const,
      hypothesisId: null,
      hypothesisText: null,
      priorHypothesisText: null,
      source: null,
      lastActor: 'system' as const,
      lastTransitionAt: Date.now(),
      supportingEvidenceCount: 0,
      contradictionCount: 0,
      unresolvedContradictions: [] as string[],
      evidenceRefs: [] as string[],
      confidenceBand: 'low' as const,
      recommendedReasoningMode: 'explore' as const,
      allowedSpeechActs: [] as string[],
      blockedSpeechActs: [] as string[],
      canEscalateTools: false,
      shouldAskClarifyingQuestion: false,
      summaryReady: false,
    })),
  },
}).createMachine({
  id: 'hypothesisLifecycle',
  initial: 'emergent',
  context: {
    stateLabel: 'emergent',
    hypothesisId: null,
    hypothesisText: null,
    priorHypothesisText: null,
    source: null,
    lastActor: null,
    lastTransitionAt: null,
    supportingEvidenceCount: 0,
    contradictionCount: 0,
    unresolvedContradictions: [],
    evidenceRefs: [],
    confidenceBand: 'low',
    recommendedReasoningMode: 'explore',
    allowedSpeechActs: [],
    blockedSpeechActs: [],
    canEscalateTools: false,
    shouldAskClarifyingQuestion: false,
    summaryReady: false,
  },
  on: {
    CLEAR_HYPOTHESIS: {
      target: '.emergent',
      actions: ['clearHypothesis', 'deriveReasoningPolicy'],
    },
    RESET_LIFECYCLE: {
      target: '.emergent',
      actions: ['clearHypothesis', 'deriveReasoningPolicy'],
    },
  },
  states: {
    emergent: {
      entry: [
        assign({ stateLabel: () => 'emergent' as const }),
        'setLowConfidence',
        'deriveReasoningPolicy',
      ],
      on: {
        START_HYPOTHESIS: {
          target: 'framed',
          actions: 'startHypothesis',
        },
        FRAME_HYPOTHESIS: {
          target: 'framed',
          actions: 'frameHypothesis',
        },
      },
    },

    framed: {
      entry: [
        assign({ stateLabel: () => 'framed' as const }),
        'setLowConfidence',
        'deriveReasoningPolicy',
      ],
      on: {
        BEGIN_TESTING: {
          target: 'testing',
        },
        ADD_CONTRADICTION: {
          target: 'contradicted',
          actions: 'addContradiction',
        },
        REVISE_HYPOTHESIS: {
          target: 'revised',
          guard: 'canRevise',
          actions: 'reviseHypothesis',
        },
      },
    },

    testing: {
      entry: [
        assign({ stateLabel: () => 'testing' as const }),
        'setMediumConfidence',
        'deriveReasoningPolicy',
      ],
      on: {
        ADD_SUPPORTING_EVIDENCE: {
          target: 'testing',
          actions: 'addSupportingEvidence',
        },
        ADD_CONTRADICTION: {
          target: 'contradicted',
          actions: 'addContradiction',
        },
        MARK_SUPPORTED: {
          target: 'supported',
          guard: 'hasStrongSupport',
        },
        REVISE_HYPOTHESIS: {
          target: 'revised',
          guard: 'canRevise',
          actions: 'reviseHypothesis',
        },
      },
    },

    supported: {
      entry: [
        assign({ stateLabel: () => 'supported' as const }),
        'setHighConfidence',
        'deriveReasoningPolicy',
      ],
      on: {
        ADD_CONTRADICTION: {
          target: 'contradicted',
          actions: 'addContradiction',
        },
        MARK_SYNTHESIZED: {
          target: 'synthesized',
          guard: 'canSynthesize',
        },
        REVISE_HYPOTHESIS: {
          target: 'revised',
          guard: 'canRevise',
          actions: 'reviseHypothesis',
        },
      },
    },

    contradicted: {
      entry: [
        assign({ stateLabel: () => 'contradicted' as const }),
        'setLowConfidence',
        'deriveReasoningPolicy',
      ],
      on: {
        REVISE_HYPOTHESIS: {
          target: 'revised',
          guard: 'canRevise',
          actions: 'reviseHypothesis',
        },
        FRAME_HYPOTHESIS: {
          target: 'framed',
          actions: 'frameHypothesis',
        },
      },
    },

    revised: {
      entry: [
        assign({ stateLabel: () => 'revised' as const }),
        'setLowConfidence',
        'deriveReasoningPolicy',
      ],
      on: {
        BEGIN_TESTING: {
          target: 'testing',
        },
        ADD_CONTRADICTION: {
          target: 'contradicted',
          actions: 'addContradiction',
        },
      },
    },

    synthesized: {
      entry: [
        assign({ stateLabel: () => 'synthesized' as const }),
        'setHighConfidence',
        'deriveReasoningPolicy',
      ],
      on: {
        ADD_CONTRADICTION: {
          target: 'contradicted',
          actions: 'addContradiction',
        },
        REVISE_HYPOTHESIS: {
          target: 'revised',
          guard: 'canRevise',
          actions: 'reviseHypothesis',
        },
      },
    },
  },
});
