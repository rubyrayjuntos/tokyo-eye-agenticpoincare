/**
 * Property-based tests for viewport state payload completeness.
 *
 * Feature: discovery-cockpit-frontend, Property 3: Viewport state payload completeness
 * Validates: Requirements 4.1, 8.4
 *
 * For any chat message sent, the viewport_state payload SHALL contain all required
 * fields. No required field SHALL be undefined (nullable fields may be null).
 */
import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import { buildViewportState, type BuildViewportStateInputs } from "../buildViewportState";

// --- Arbitraries ---

const arbPoincareColorMode = fc.constantFrom("cone_depth" as const, "uncertainty" as const);

const arbViewerColorMode = fc.constantFrom(
  "spectrum" as const, "cone_depth" as const, "epistemic" as const,
  "aleatoric" as const, "plasticity" as const, "allosteric" as const,
  "resistance" as const,
);

const arbActivePanel = fc.option(
  fc.constantFrom(
    "rcsb_search" as const, "graph_topology" as const, "hypothesis" as const,
    "data_tools" as const, "provenance" as const, "plot_generator" as const,
    "compare" as const, "data_inspector" as const,
  ),
  { nil: null },
);

const arbBuildViewportStateInputs: fc.Arbitrary<BuildViewportStateInputs> = fc.record({
  structureId: fc.option(fc.uuid(), { nil: null }),
  structureTitle: fc.option(fc.string({ minLength: 1, maxLength: 50 }), { nil: null }),
  poincareColorMode: arbPoincareColorMode,
  mobiusFocusEnabled: fc.boolean(),
  mobiusFocusResidue: fc.option(fc.string({ minLength: 3, maxLength: 10 }), { nil: null }),
  selectedResidue: fc.option(
    fc.record({
      residue_id: fc.string({ minLength: 3, maxLength: 10 }),
      residue_name: fc.option(fc.string({ maxLength: 5 }), { nil: null }),
      chain_label: fc.option(fc.constantFrom("A", "B", "C", "D"), { nil: null }),
      epistemic_uncertainty: fc.option(fc.float({ min: 0, max: 1, noNaN: true }), { nil: null }),
      cone_depth: fc.option(fc.float({ min: 0, max: 5, noNaN: true }), { nil: null }),
    }),
    { nil: null },
  ),
  brushSelectedIds: fc.array(fc.string({ minLength: 3, maxLength: 10 }), { maxLength: 10 }),
  viewerColorMode: arbViewerColorMode,
  riskThreshold: fc.float({ min: 0, max: 1, noNaN: true }),
  highlightedResidueIds: fc.array(fc.string({ minLength: 3, maxLength: 10 }), { maxLength: 10 }),
  residueCount: fc.nat({ max: 10000 }),
  sourceLeakCount: fc.nat({ max: 100 }),
  hypothesisCount: fc.nat({ max: 50 }),
  provenanceRunCount: fc.nat({ max: 100 }),
  annotationCount: fc.nat({ max: 200 }),
  activePanel: arbActivePanel,
  pipelineStatus: fc.constantFrom("never_run", "running", "complete", "failed"),
  pipelineCurrentStep: fc.option(fc.string({ maxLength: 20 }), { nil: null }),
  pipelineProgress: fc.option(fc.float({ min: 0, max: 100, noNaN: true }), { nil: null }),
});

// --- Property 3: Viewport state payload completeness ---

describe("Feature: discovery-cockpit-frontend, Property 3: Viewport state payload completeness", () => {
  it("For any inputs, the viewport_state payload contains all required top-level fields (never undefined)", () => {
    fc.assert(
      fc.property(arbBuildViewportStateInputs, (inputs) => {
        const state = buildViewportState(inputs);

        // Top-level required fields — may be null but never undefined
        expect(state.structure_id).not.toBe(undefined);
        expect(state.structure_title).not.toBe(undefined);
        expect(state.poincare).toBeDefined();
        expect(state.viewer_3d).toBeDefined();
        expect(state.data_summary).toBeDefined();
        expect(state.active_panel).not.toBe(undefined);
        expect(state.pipeline).toBeDefined();
      }),
      { numRuns: 200 },
    );
  });

  it("For any inputs, poincare sub-object contains all required fields", () => {
    fc.assert(
      fc.property(arbBuildViewportStateInputs, (inputs) => {
        const state = buildViewportState(inputs);

        expect(state.poincare.color_mode).not.toBe(undefined);
        expect(typeof state.poincare.color_mode).toBe("string");
        expect(state.poincare.mobius_focus_enabled).not.toBe(undefined);
        expect(typeof state.poincare.mobius_focus_enabled).toBe("boolean");
        // These may be null but not undefined
        expect("mobius_focus_residue" in state.poincare).toBe(true);
        expect("selected_residue" in state.poincare).toBe(true);
        expect(Array.isArray(state.poincare.brush_selected_ids)).toBe(true);
      }),
      { numRuns: 200 },
    );
  });

  it("For any inputs, viewer_3d sub-object contains all required fields", () => {
    fc.assert(
      fc.property(arbBuildViewportStateInputs, (inputs) => {
        const state = buildViewportState(inputs);

        expect(state.viewer_3d.color_mode).not.toBe(undefined);
        expect(typeof state.viewer_3d.color_mode).toBe("string");
        expect(typeof state.viewer_3d.risk_threshold).toBe("number");
        expect(Array.isArray(state.viewer_3d.highlighted_residue_ids)).toBe(true);
      }),
      { numRuns: 200 },
    );
  });

  it("For any inputs, data_summary contains all required fields", () => {
    fc.assert(
      fc.property(arbBuildViewportStateInputs, (inputs) => {
        const state = buildViewportState(inputs);

        expect(state.data_summary).not.toBeNull();
        const ds = state.data_summary!;
        expect(typeof ds.residue_count).toBe("number");
        expect(typeof ds.source_leak_count).toBe("number");
        expect(typeof ds.hypothesis_count).toBe("number");
        expect(typeof ds.provenance_run_count).toBe("number");
        expect(typeof ds.annotation_count).toBe("number");
        expect(Array.isArray(ds.top_uncertainty_residues)).toBe(true);
        expect("persistence_status" in ds).toBe(true);
        expect("resistance_summary" in ds).toBe(true);
      }),
      { numRuns: 200 },
    );
  });

  it("For any inputs, pipeline sub-object contains all required fields", () => {
    fc.assert(
      fc.property(arbBuildViewportStateInputs, (inputs) => {
        const state = buildViewportState(inputs);

        expect(typeof state.pipeline.status).toBe("string");
        expect("current_step" in state.pipeline).toBe(true);
        expect("progress" in state.pipeline).toBe(true);
      }),
      { numRuns: 200 },
    );
  });
});
