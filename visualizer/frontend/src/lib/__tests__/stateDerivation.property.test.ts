/**
 * Property-based tests for WebSocket state derivation.
 *
 * Feature: discovery-cockpit-frontend, Property 1: State derivation from backend
 * Feature: discovery-cockpit-frontend, Property 5: Tool dock phase gating
 * Feature: discovery-cockpit-frontend, Property 6: Reconnection state recovery
 *
 * Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
 */
import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import { toolPolicyForPhase } from "../toolPolicy";
import type { DiscoveryPhase } from "../discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "../hypothesisLifecycleMachine";
import {
  plannerPolicySelector,
  createDefaultStructureScope,
  type PlannerPolicyInputs,
  type StructureScopeContext,
} from "../plannerPolicySelector";
import type { ViewportContext } from "../viewportMachine";

// --- Arbitraries ---

const DISCOVERY_PHASES: DiscoveryPhase[] = [
  "residue", "topology", "structure", "pocket", "screening", "report",
];

const HYPOTHESIS_STATES: HypothesisLifecycleState[] = [
  "emergent", "framed", "testing", "supported", "contradicted", "revised", "synthesized",
];

const arbDiscoveryPhase: fc.Arbitrary<DiscoveryPhase> =
  fc.constantFrom(...DISCOVERY_PHASES);

const arbHypothesisState: fc.Arbitrary<HypothesisLifecycleState> =
  fc.constantFrom(...HYPOTHESIS_STATES);

const arbViewportContext: fc.Arbitrary<ViewportContext> = fc.record({
  highlightedResidues: fc.array(fc.string({ minLength: 3, maxLength: 10 }), { maxLength: 5 }),
  userSelectedResidue: fc.option(fc.string({ minLength: 3, maxLength: 10 }), { nil: null }),
  currentDirective: fc.constant(null),
  activeMetric: fc.constantFrom("cone_depth", "epistemic", "aleatoric"),
  isRadarActive: fc.boolean(),
  poincareColorMode: fc.constantFrom("cone_depth" as const, "uncertainty" as const),
  viewerColorMode: fc.constantFrom(
    "spectrum" as const, "cone_depth" as const, "epistemic" as const,
    "aleatoric" as const, "plasticity" as const, "allosteric" as const, "resistance" as const,
  ),
  riskThreshold: fc.float({ min: 0, max: 1, noNaN: true }),
  selectedPocketId: fc.option(fc.nat({ max: 10 }), { nil: null }),
  activePanel: fc.option(
    fc.constantFrom(
      "rcsb_search" as const, "graph_topology" as const, "hypothesis" as const,
      "data_tools" as const, "provenance" as const, "plot_generator" as const,
      "compare" as const, "data_inspector" as const,
    ),
    { nil: null },
  ),
  layoutModelJSON: fc.constant(null),
});

/** Generates a structureScope with ingestion/inference ready (the common case for testing tool gating) */
const arbStructureScopeReady: fc.Arbitrary<StructureScopeContext> = fc.record({
  scopeId: fc.option(fc.uuid(), { nil: null }),
  scopeType: fc.constant("single_structure" as const),
  structureIds: fc.array(fc.uuid(), { minLength: 1, maxLength: 3 }),
  primaryStructureId: fc.option(fc.uuid(), { nil: null }),
  activeChainIds: fc.array(fc.constantFrom("A", "B", "C", "D"), { maxLength: 4 }),
  ingestionReady: fc.constant(true),
  inferenceReady: fc.constant(true),
  pipelineReady: fc.boolean(),
  provenanceLabel: fc.option(fc.string({ maxLength: 20 }), { nil: null }),
  lastLoadedAt: fc.option(fc.nat(), { nil: null }),
  lastActor: fc.option(fc.constantFrom("user" as const, "agent" as const, "system" as const), { nil: null }),
  requiresReset: fc.constant(false),
  resetReason: fc.constant(null),
  sessionMode: fc.constantFrom("scientific_rigor" as const, "free_flow" as const),
});

// Helper to build a discovery context shape matching what the machine produces
function buildDiscoveryContext(phase: DiscoveryPhase) {
  const toolPolicy = toolPolicyForPhase(phase);
  return {
    phase,
    phaseSource: "user" as const,
    lastActor: null,
    lastTransitionAt: null,
    activeStructureId: null,
    selectedResidues: [],
    selectedClusterId: null,
    selectedPocketIndex: null,
    hasResidueSelection: false,
    hasTopologySelection: false,
    hasStructureMapping: false,
    hasPocketExtraction: false,
    hasPharmacophoreMap: false,
    hasScreeningResults: false,
    hasDockingResults: false,
    hasReportDraft: false,
    allowedTools: toolPolicy.allowed,
    blockedTools: toolPolicy.blocked,
  };
}

// Helper to build a hypothesis context shape
function buildHypothesisContext(stateLabel: HypothesisLifecycleState) {
  return {
    stateLabel,
    hypothesisId: null,
    hypothesisText: null,
    priorHypothesisText: null,
    source: null,
    lastActor: null,
    lastTransitionAt: null,
    supportingEvidenceCount: 0,
    contradictingEvidenceCount: 0,
    revisionCount: 0,
    latestEvidenceRef: null,
    latestContradictionReason: null,
    summaryReady: stateLabel === "synthesized",
    canEscalateTools: ["supported", "synthesized", "testing"].includes(stateLabel),
    shouldAskClarifyingQuestion: ["emergent", "framed"].includes(stateLabel),
    recommendedReasoningMode: "explore" as const,
    allowedSpeechActs: [],
    blockedSpeechActs: [],
  };
}

// --- Property 1: State derivation from backend ---

describe("Feature: discovery-cockpit-frontend, Property 1: State derivation from backend", () => {
  it("For any state_snapshot, derived policy discoveryPhase matches the snapshot phase", () => {
    fc.assert(
      fc.property(
        arbDiscoveryPhase,
        arbHypothesisState,
        arbViewportContext,
        arbStructureScopeReady,
        (phase, lifecycleState, viewport, scope) => {
          const discovery = buildDiscoveryContext(phase);
          const hypothesis = buildHypothesisContext(lifecycleState);

          const inputs: PlannerPolicyInputs = {
            structureScope: scope,
            viewport,
            discovery: discovery as any,
            hypothesis: hypothesis as any,
          };

          const policy = plannerPolicySelector(inputs);

          // The derived policy phase MUST match the input phase
          expect(policy.discoveryPhase).toBe(phase);
          // The derived hypothesis state MUST match the input lifecycle
          expect(policy.hypothesisState).toBe(lifecycleState);
        },
      ),
      { numRuns: 200 },
    );
  });
});

// --- Property 5: Tool dock phase gating ---

describe("Feature: discovery-cockpit-frontend, Property 5: Tool dock phase gating", () => {
  it("For any phase, the policy allowedTools is a subset of the phase's tool policy allowed list (when scope is ready)", () => {
    fc.assert(
      fc.property(
        arbDiscoveryPhase,
        arbHypothesisState,
        arbViewportContext,
        arbStructureScopeReady,
        (phase, lifecycleState, viewport, scope) => {
          const discovery = buildDiscoveryContext(phase);
          const hypothesis = buildHypothesisContext(lifecycleState);

          const inputs: PlannerPolicyInputs = {
            structureScope: scope,
            viewport,
            discovery: discovery as any,
            hypothesis: hypothesis as any,
          };

          const policy = plannerPolicySelector(inputs);
          const phasePolicy = toolPolicyForPhase(phase);

          // Every allowed tool in the derived policy must come from the phase's allowed list
          for (const tool of policy.allowedTools) {
            expect(phasePolicy.allowed).toContain(tool);
          }

          // Every blocked tool from the phase MUST appear in the policy blocked list
          // (unless hypothesis lifecycle removed it — but phase blocks are always additive)
          for (const tool of phasePolicy.blocked) {
            expect(policy.blockedTools).toContain(tool);
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  it("No tool appears in both allowedTools and blockedTools simultaneously", () => {
    fc.assert(
      fc.property(
        arbDiscoveryPhase,
        arbHypothesisState,
        arbViewportContext,
        arbStructureScopeReady,
        (phase, lifecycleState, viewport, scope) => {
          const discovery = buildDiscoveryContext(phase);
          const hypothesis = buildHypothesisContext(lifecycleState);

          const inputs: PlannerPolicyInputs = {
            structureScope: scope,
            viewport,
            discovery: discovery as any,
            hypothesis: hypothesis as any,
          };

          const policy = plannerPolicySelector(inputs);

          const overlap = policy.allowedTools.filter((t) =>
            policy.blockedTools.includes(t),
          );
          expect(overlap).toHaveLength(0);
        },
      ),
      { numRuns: 200 },
    );
  });
});

// --- Property 6: Reconnection state recovery ---

describe("Feature: discovery-cockpit-frontend, Property 6: Reconnection state recovery", () => {
  it("For any state_snapshot received after reconnection, deriving policy twice from same snapshot produces identical results (idempotence)", () => {
    fc.assert(
      fc.property(
        arbDiscoveryPhase,
        arbHypothesisState,
        arbViewportContext,
        arbStructureScopeReady,
        (phase, lifecycleState, viewport, scope) => {
          const discovery = buildDiscoveryContext(phase);
          const hypothesis = buildHypothesisContext(lifecycleState);

          const inputs: PlannerPolicyInputs = {
            structureScope: scope,
            viewport,
            discovery: discovery as any,
            hypothesis: hypothesis as any,
          };

          // Simulate: reconnection delivers a snapshot, we derive policy from it
          const policy1 = plannerPolicySelector(inputs);
          // Simulate: same snapshot applied again (idempotence on reconnect)
          const policy2 = plannerPolicySelector(inputs);

          // Must produce identical results
          expect(policy1.discoveryPhase).toBe(policy2.discoveryPhase);
          expect(policy1.hypothesisState).toBe(policy2.hypothesisState);
          expect(policy1.allowedTools).toEqual(policy2.allowedTools);
          expect(policy1.blockedTools).toEqual(policy2.blockedTools);
          expect(policy1.reasoningMode).toBe(policy2.reasoningMode);
          expect(policy1.speechStyle).toBe(policy2.speechStyle);
        },
      ),
      { numRuns: 200 },
    );
  });
});
