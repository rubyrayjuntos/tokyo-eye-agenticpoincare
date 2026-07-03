import { describe, expect, it } from "vitest";

import { proteinStateLabel, summarizeUncertainty } from "../briefingMetrics";
import { inferArtifactAvailability } from "../inferHydrateAvailability";
import type { HydrationResponse } from "../types";

describe("briefingMetrics", () => {
  it("detects localized spikes relative to structure distribution", () => {
    const residues = [
      { residue_id: "a:A:1", epistemic_uncertainty: 9 },
      { residue_id: "a:A:2", epistemic_uncertainty: 9.1 },
      { residue_id: "a:A:3", epistemic_uncertainty: 14 },
    ];
    const summary = summarizeUncertainty(residues);
    expect(summary.spikeCount).toBeGreaterThan(0);
    expect(proteinStateLabel(summary).title).toContain("Localized");
  });
});

describe("inferHydrateAvailability", () => {
  it("marks gnn_hyp present when hydrate has residues", () => {
    const hydration: HydrationResponse = {
      structure_id: "4uj1",
      embeddings: {
        structure_id: "4uj1",
        curvature: 1,
        residues: [{ residue_id: "4uj1:A:1" }],
      },
    };
    const availability = inferArtifactAvailability(hydration);
    expect(availability.gnn_hyp.present).toBe(true);
  });
});
