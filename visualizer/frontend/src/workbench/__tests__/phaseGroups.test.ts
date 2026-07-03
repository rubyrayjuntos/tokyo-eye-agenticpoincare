import { describe, expect, it } from "vitest";

import {
  PHASE_GROUP_CONFIG,
  discoveryPhaseToGroup,
  groupToDefaultDiscoveryPhase,
} from "../phaseGroups";

describe("phaseGroups", () => {
  it("maps discovery phases to three UX groups", () => {
    expect(discoveryPhaseToGroup("residue")).toBe("exploration");
    expect(discoveryPhaseToGroup("topology")).toBe("exploration");
    expect(discoveryPhaseToGroup("structure")).toBe("analysis");
    expect(discoveryPhaseToGroup("pocket")).toBe("analysis");
    expect(discoveryPhaseToGroup("screening")).toBe("review");
    expect(discoveryPhaseToGroup("report")).toBe("review");
  });

  it("returns default discovery phase per group", () => {
    expect(groupToDefaultDiscoveryPhase("exploration")).toBe("residue");
    expect(groupToDefaultDiscoveryPhase("analysis")).toBe("structure");
    expect(groupToDefaultDiscoveryPhase("review")).toBe("screening");
  });

  it("defines required dockview panels per group", () => {
    expect(PHASE_GROUP_CONFIG.exploration.requiredPanels).toContain("briefing-panel");
    expect(PHASE_GROUP_CONFIG.exploration.requiredPanels).toContain("triple-viewport-panel");
    expect(PHASE_GROUP_CONFIG.exploration.requiredPanels).toContain("findings-dock-panel");
    expect(PHASE_GROUP_CONFIG.analysis.requiredPanels).toContain("triple-viewport-panel");
    expect(PHASE_GROUP_CONFIG.analysis.requiredPanels).toContain("telemetry-panel");
    expect(PHASE_GROUP_CONFIG.review.requiredPanels).toContain("chat-panel");
  });
});
