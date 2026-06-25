import type { DiscoveryPhase, DiscoveryPhaseContext } from './discoveryPhaseMachine';
import type { HypothesisLifecycleState, HypothesisLifecycleContext } from './hypothesisLifecycleMachine';
import type { ViewportContext } from './viewportMachine';

// ---------------------------------------------------------------------------
// Session Mode
// ---------------------------------------------------------------------------

export type SessionMode = 'scientific_rigor' | 'free_flow';

// ---------------------------------------------------------------------------
// Structure Scope
// ---------------------------------------------------------------------------

export type StructureScopeType =
  | 'single_structure'
  | 'ensemble'
  | 'ppi'
  | 'proteome';

export interface StructureScopeContext {
  scopeId: string | null;
  scopeType: StructureScopeType;

  structureIds: string[];
  primaryStructureId: string | null;
  activeChainIds: string[];

  ingestionReady: boolean;
  inferenceReady: boolean;
  pipelineReady: boolean;

  provenanceLabel: string | null;
  lastLoadedAt: number | null;
  lastActor: 'user' | 'agent' | 'system' | null;

  requiresReset: boolean;
  resetReason: string | null;

  sessionMode: SessionMode;
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

export interface PlannerPolicyInputs {
  structureScope: StructureScopeContext;
  viewport: ViewportContext;
  discovery: DiscoveryPhaseContext;
  hypothesis: HypothesisLifecycleContext;
}

// ---------------------------------------------------------------------------
// Output — the single policy object the orchestrator consumes
// ---------------------------------------------------------------------------

export interface PlannerPolicy {
  // scope identity
  sessionMode: SessionMode;
  structureScopeType: StructureScopeType;
  scopeId: string | null;
  primaryStructureId: string | null;

  // workflow + epistemic position
  discoveryPhase: DiscoveryPhase;
  hypothesisState: HypothesisLifecycleState;
  reasoningMode:
    | 'explore'
    | 'frame'
    | 'test'
    | 'prioritize'
    | 'regress'
    | 'revise'
    | 'synthesize';

  // tool gating
  allowedTools: string[];
  blockedTools: string[];
  preferredTools: string[];

  // speech control
  allowedSpeechActs: string[];
  blockedSpeechActs: string[];
  speechStyle:
    | 'exploratory'
    | 'analytical'
    | 'comparative'
    | 'cautious'
    | 'directive'
    | 'synthesis';

  // orchestration flags
  shouldAskClarifyingQuestion: boolean;
  shouldAdvancePhase: boolean;
  shouldRegressPhase: boolean;
  shouldResetForNewScope: boolean;

  // targeting
  targetResidues: string[];
  targetPocketIndex: number | null;
  targetClusterId: string | null;

  // reporting
  summaryReady: boolean;
  reportReady: boolean;
  reportType: 'none' | 'provisional' | 'validated';

  // diagnostics
  rationale: string[];
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function speechStyleForPolicy(
  discovery: DiscoveryPhase,
  hypothesis: HypothesisLifecycleState,
): PlannerPolicy['speechStyle'] {
  if (hypothesis === 'contradicted') return 'cautious';
  if (hypothesis === 'synthesized' && discovery === 'report') return 'synthesis';
  if (hypothesis === 'emergent') return 'exploratory';
  if (hypothesis === 'testing') return 'comparative';
  if (hypothesis === 'supported') return 'analytical';
  return 'directive';
}

/** Escalation tools that hypothesis contradiction should suppress. */
const ESCALATION_TOOLS = [
  'screen_fragments',
  'run_docking_surrogate',
  'generate_report',
] as const;

function baselinePolicy(): PlannerPolicy {
  return {
    sessionMode: 'scientific_rigor',
    structureScopeType: 'single_structure',
    scopeId: null,
    primaryStructureId: null,
    discoveryPhase: 'residue',
    hypothesisState: 'emergent',
    reasoningMode: 'explore',
    allowedTools: [],
    blockedTools: [],
    preferredTools: [],
    allowedSpeechActs: [],
    blockedSpeechActs: [],
    speechStyle: 'exploratory',
    shouldAskClarifyingQuestion: false,
    shouldAdvancePhase: false,
    shouldRegressPhase: false,
    shouldResetForNewScope: false,
    targetResidues: [],
    targetPocketIndex: null,
    targetClusterId: null,
    summaryReady: false,
    reportReady: false,
    reportType: 'none',
    rationale: [],
    warnings: [],
  };
}

// ---------------------------------------------------------------------------
// Selector — the main export
// ---------------------------------------------------------------------------

export function plannerPolicySelector(inputs: PlannerPolicyInputs): PlannerPolicy {
  const { structureScope, viewport, discovery, hypothesis } = inputs;
  const policy = baselinePolicy();

  // --- identity ---
  policy.sessionMode = structureScope.sessionMode;
  policy.structureScopeType = structureScope.scopeType;
  policy.scopeId = structureScope.scopeId;
  policy.primaryStructureId = structureScope.primaryStructureId;
  policy.discoveryPhase = discovery.phase;
  policy.hypothesisState = hypothesis.stateLabel;
  policy.reasoningMode = hypothesis.recommendedReasoningMode;

  // ===================================================================
  // 1. Structure scope validity — overrides everything
  // ===================================================================
  if (structureScope.requiresReset) {
    policy.shouldResetForNewScope = true;
    policy.allowedTools = ['load_structure_scope', 'run_ingestion', 'run_inference'];
    policy.blockedTools = ['extract_pockets', 'map_pharmacophore_features', 'screen_fragments', 'run_docking_surrogate', 'generate_report'];
    policy.reasoningMode = 'explore';
    policy.speechStyle = 'cautious';
    policy.rationale.push('A new structure scope was loaded; downstream state is stale until reinitialized.');
    return policy;
  }

  // ===================================================================
  // 2. Pipeline readiness — blocks analytical tools
  // ===================================================================
  if (!structureScope.ingestionReady || !structureScope.inferenceReady) {
    policy.allowedTools = ['run_ingestion', 'run_inference', 'inspect_structure_metadata'];
    policy.blockedTools = ['extract_pockets', 'screen_fragments', 'run_docking_surrogate', 'generate_report'];
    policy.shouldAskClarifyingQuestion = false;
    policy.speechStyle = 'cautious';
    policy.rationale.push('Core structure processing is not complete.');
    return policy;
  }

  // ===================================================================
  // 3. Discovery phase tool gating
  // ===================================================================
  policy.allowedTools = [...discovery.allowedTools];
  policy.blockedTools = [...discovery.blockedTools];

  // ===================================================================
  // 4. Hypothesis lifecycle modulation
  // ===================================================================
  policy.allowedSpeechActs = [...hypothesis.allowedSpeechActs];
  policy.blockedSpeechActs = [...hypothesis.blockedSpeechActs];
  policy.shouldAskClarifyingQuestion = hypothesis.shouldAskClarifyingQuestion;

  // Contradiction suppresses escalation tools even if discovery phase allows them
  if (hypothesis.stateLabel === 'contradicted') {
    policy.shouldRegressPhase = true;
    for (const tool of ESCALATION_TOOLS) {
      if (!policy.blockedTools.includes(tool)) {
        policy.blockedTools.push(tool);
      }
    }
    policy.warnings.push('Hypothesis is contradicted; escalation tools are suppressed.');
  }

  // Emergent: restrict to inspection tools
  if (!hypothesis.canEscalateTools) {
    policy.allowedTools = policy.allowedTools.filter(
      t => !ESCALATION_TOOLS.includes(t as typeof ESCALATION_TOOLS[number]),
    );
  }

  // ===================================================================
  // 5. Viewport target refinement
  // ===================================================================
  policy.targetResidues = viewport.highlightedResidues ?? [];
  policy.targetPocketIndex = discovery.selectedPocketIndex ?? null;
  policy.targetClusterId = discovery.selectedClusterId ?? null;

  // ===================================================================
  // 6. Speech style + reporting
  // ===================================================================
  policy.speechStyle = speechStyleForPolicy(discovery.phase, hypothesis.stateLabel);
  policy.summaryReady = hypothesis.summaryReady;

  // Report readiness requires both workflow maturity AND epistemic maturity
  policy.reportReady =
    discovery.phase === 'report' &&
    hypothesis.stateLabel === 'synthesized' &&
    structureScope.pipelineReady;
  policy.reportType = policy.reportReady ? 'validated' : 'none';

  // ===================================================================
  // 7. Session mode modulation
  // ===================================================================
  if (structureScope.sessionMode === 'scientific_rigor') {
    // Tighten thresholds
    policy.shouldAskClarifyingQuestion =
      policy.shouldAskClarifyingQuestion ||
      (discovery.phase !== 'residue' && hypothesis.stateLabel !== 'supported');

    if (hypothesis.stateLabel === 'contradicted') {
      policy.shouldRegressPhase = true;
    }

    if (discovery.phase === 'report' && !policy.reportReady) {
      policy.reportType = 'none';
    }
  } else {
    // free_flow — loosen thresholds
    policy.shouldAskClarifyingQuestion = false;

    if (hypothesis.stateLabel === 'contradicted') {
      policy.warnings.push('Current branch is contradicted; continuing in exploratory mode.');
      policy.shouldRegressPhase = false;
    }

    if (discovery.phase === 'report' && structureScope.pipelineReady) {
      policy.reportReady = true;
      policy.reportType = hypothesis.stateLabel === 'synthesized' ? 'validated' : 'provisional';
    }
  }

  policy.rationale.push(
    `Phase: ${discovery.phase}, Lifecycle: ${hypothesis.stateLabel}, Mode: ${structureScope.sessionMode}`,
  );

  return policy;
}

// ---------------------------------------------------------------------------
// Default structure scope (for initialization)
// ---------------------------------------------------------------------------

export function createDefaultStructureScope(): StructureScopeContext {
  return {
    scopeId: null,
    scopeType: 'single_structure',
    structureIds: [],
    primaryStructureId: null,
    activeChainIds: [],
    ingestionReady: false,
    inferenceReady: false,
    pipelineReady: false,
    provenanceLabel: null,
    lastLoadedAt: null,
    lastActor: null,
    requiresReset: false,
    resetReason: null,
    sessionMode: 'scientific_rigor',
  };
}
