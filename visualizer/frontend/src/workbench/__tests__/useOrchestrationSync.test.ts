import { describe, expect, it, vi } from "vitest";

import { WorkbenchBus } from "../WorkbenchBus";
import { applyBackendSnapshot } from "../useOrchestrationSync";
import type { StateSnapshotPayload } from "../../lib/useViewportSocket";

describe("applyBackendSnapshot", () => {
  it("ignores duplicate snapshots with the same fingerprint", () => {
    const bus = new WorkbenchBus();
    const onDiscoveryPhase = vi.fn();
    const snapshot: StateSnapshotPayload = {
      session_id: "session-1",
      discovery_phase: "residue",
      hypothesis_lifecycle: "emergent",
      structure_scope: {},
      selected_residue: null,
      policy: {
        allowed_tools: [],
        blocked_tools: [],
        preferred_tools: [],
        discovery_phase: "residue",
        hypothesis_state: "emergent",
        reasoning_mode: "explore",
        session_mode: "discovery",
      },
    };

    expect(applyBackendSnapshot(bus, snapshot, onDiscoveryPhase)).toBe(true);
    expect(onDiscoveryPhase).toHaveBeenCalledTimes(1);

    expect(applyBackendSnapshot(bus, snapshot, onDiscoveryPhase)).toBe(false);
    expect(onDiscoveryPhase).toHaveBeenCalledTimes(1);
  });
});
