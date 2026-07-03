import { describe, expect, it } from "vitest";
import { createActor } from "xstate";

import { hydrationMachine } from "../hydrationMachine";
import type { HydrationLoadResult } from "../hydrationLoader";

const sampleResult: HydrationLoadResult = {
  hydration: {
    structure_id: "9o0r",
    embeddings: {
      structure_id: "9o0r",
      curvature: -1,
      residues: [],
    },
  },
  artifactAvailability: {
    gnn_hyp: { present: true, tier: 1, reason: null },
  },
  hydrateMeta: {
    contract_version: "1.6",
    degraded: false,
    missing_tier1_count: 0,
    missing_keys: [],
  },
  signalsInferred: false,
};

describe("hydrationMachine", () => {
  it("transitions through loading to loaded", () => {
    const actor = createActor(hydrationMachine).start();

    actor.send({ type: "HYDRATION_REQUESTED", structureId: "9o0r" });
    expect(actor.getSnapshot().value).toBe("loading");

    actor.send({ type: "HYDRATION_SUCCEEDED", result: sampleResult });
    expect(actor.getSnapshot().value).toBe("loaded");
    expect(actor.getSnapshot().context.hydration?.structure_id).toBe("9o0r");
  });

  it("captures failures and clears on structure unload", () => {
    const actor = createActor(hydrationMachine).start();

    actor.send({ type: "HYDRATION_REQUESTED", structureId: "9o0r" });
    actor.send({ type: "HYDRATION_FAILED", error: "network" });
    expect(actor.getSnapshot().value).toBe("failed");
    expect(actor.getSnapshot().context.error).toBe("network");

    actor.send({ type: "HYDRATION_CLEARED" });
    expect(actor.getSnapshot().value).toBe("idle");
    expect(actor.getSnapshot().context.hydration).toBeNull();
  });
});