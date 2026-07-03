import { describe, expect, it } from "vitest";

import { WorkbenchBus } from "../WorkbenchBus";

describe("WorkbenchBus", () => {
  it("blocks tool calls when orchestration snapshot marks tool as blocked", () => {
    const bus = new WorkbenchBus();
    bus.setOrchestrationSnapshot({
      discovery_phase: "residue",
      hypothesis_lifecycle: "emergent",
      structure_scope: {},
      selected_residue: null,
      policy: {
        allowed_tools: ["highlight_residues"],
        blocked_tools: ["screen_fragments"],
      },
    });

    let warningMessage = "";
    bus.subscribe("system:warning", (payload) => {
      warningMessage = payload.message;
    });

    const result = bus.publish("tool:call", { tool: "screen_fragments" });
    expect(result.accepted).toBe(false);
    expect(warningMessage).toContain("screen_fragments");
  });

  it("caches latest published state", () => {
    const bus = new WorkbenchBus();
    bus.publish("ui:selection_changed", { residueIds: ["rcsb_4obe:A:12"] });
    expect(bus.getLatestState("ui:selection_changed")?.residueIds).toEqual([
      "rcsb_4obe:A:12",
    ]);
  });
});
