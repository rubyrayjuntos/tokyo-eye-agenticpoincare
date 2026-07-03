import { describe, expect, it } from "vitest";

import {
  formatArtifactReason,
  getArtifactSurfaceState,
  isArtifactPresent,
} from "../artifactAvailability";

describe("artifactAvailability", () => {
  const availability = {
    gnn_hyp: { present: true, tier: 1, reason: null },
    source_leaks: { present: false, tier: 1, reason: "absent" },
    motifs: { present: false, tier: 2, reason: "job_planned" },
  };

  it("detects present and missing tier-1 artifacts", () => {
    expect(isArtifactPresent(availability, "gnn_hyp")).toBe(true);
    expect(getArtifactSurfaceState(availability, "source_leaks")).toBe("tier1_empty");
    expect(getArtifactSurfaceState(availability, "motifs")).toBe("tier2_missing");
  });

  it("formats known availability reasons", () => {
    expect(formatArtifactReason("job_planned")).toBe("Pipeline job planned");
    expect(formatArtifactReason("not_in_hydrate_bundle")).toBe(
      "Available via dedicated endpoint",
    );
  });
});
