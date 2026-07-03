import { describe, expect, it } from "vitest";

import { TIER1_ARTIFACT_KEYS, isBindingScanComplete } from "../artifactContract";
import { inferArtifactAvailability } from "../inferHydrateAvailability";
import type { HydrationResponse } from "../types";

describe("artifactContract", () => {
  it("excludes allosteric_sites from tier-1 keys", () => {
    expect(TIER1_ARTIFACT_KEYS).not.toContain("allosteric_sites");
    expect(TIER1_ARTIFACT_KEYS).toContain("binding_scan");
  });

  it("requires complete binding_scan status before marking present", () => {
    const hydration = {
      structure_id: "9o0r",
      binding_scan: {
        status: "running",
        sites: [{ id: 1 }],
      },
    } as HydrationResponse;

    const availability = inferArtifactAvailability(hydration);
    expect(availability.binding_scan.present).toBe(false);
    expect(isBindingScanComplete("complete")).toBe(true);
    expect(isBindingScanComplete("running")).toBe(false);
  });
});