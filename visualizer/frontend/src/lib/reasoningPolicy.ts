import type { HypothesisLifecycleState } from './hypothesisLifecycleMachine';

export interface ReasoningPolicy {
  recommendedReasoningMode:
    | 'explore'
    | 'frame'
    | 'test'
    | 'prioritize'
    | 'regress'
    | 'revise'
    | 'synthesize';
  allowedSpeechActs: string[];
  blockedSpeechActs: string[];
  canEscalateTools: boolean;
  shouldAskClarifyingQuestion: boolean;
  summaryReady: boolean;
}

export const reasoningPolicyForLifecycleState = (
  state: HypothesisLifecycleState,
): ReasoningPolicy => {
  switch (state) {
    case 'emergent':
      return {
        recommendedReasoningMode: 'explore',
        allowedSpeechActs: ['observe', 'suggest', 'ask_clarifying_question'],
        blockedSpeechActs: ['conclude', 'prioritize_candidates', 'finalize_report'],
        canEscalateTools: false,
        shouldAskClarifyingQuestion: true,
        summaryReady: false,
      };
    case 'framed':
      return {
        recommendedReasoningMode: 'frame',
        allowedSpeechActs: ['state_hypothesis', 'define_tests', 'ask_confirmation'],
        blockedSpeechActs: ['conclude', 'finalize_report'],
        canEscalateTools: false,
        shouldAskClarifyingQuestion: false,
        summaryReady: false,
      };
    case 'testing':
      return {
        recommendedReasoningMode: 'test',
        allowedSpeechActs: ['compare', 'validate', 'falsify', 'request_evidence'],
        blockedSpeechActs: ['finalize_report'],
        canEscalateTools: true,
        shouldAskClarifyingQuestion: false,
        summaryReady: false,
      };
    case 'supported':
      return {
        recommendedReasoningMode: 'prioritize',
        allowedSpeechActs: ['recommend_next_step', 'prioritize_candidates', 'summarize_support'],
        blockedSpeechActs: ['overclaim_certainty'],
        canEscalateTools: true,
        shouldAskClarifyingQuestion: false,
        summaryReady: false,
      };
    case 'contradicted':
      return {
        recommendedReasoningMode: 'regress',
        allowedSpeechActs: ['explain_failure', 'name_contradiction', 'propose_regression'],
        blockedSpeechActs: ['prioritize_candidates', 'finalize_report'],
        canEscalateTools: false,
        shouldAskClarifyingQuestion: true,
        summaryReady: false,
      };
    case 'revised':
      return {
        recommendedReasoningMode: 'revise',
        allowedSpeechActs: ['state_revision', 'compare_old_vs_new', 'define_new_tests'],
        blockedSpeechActs: ['finalize_report'],
        canEscalateTools: false,
        shouldAskClarifyingQuestion: false,
        summaryReady: false,
      };
    case 'synthesized':
      return {
        recommendedReasoningMode: 'synthesize',
        allowedSpeechActs: ['summarize', 'export', 'recommend_next_actions'],
        blockedSpeechActs: ['continue_unbounded_exploration'],
        canEscalateTools: false,
        shouldAskClarifyingQuestion: false,
        summaryReady: true,
      };
    default:
      const _exhaustiveCheck: never = state;
      throw new Error(`Unhandled lifecycle state: ${_exhaustiveCheck}`);
  }
};
