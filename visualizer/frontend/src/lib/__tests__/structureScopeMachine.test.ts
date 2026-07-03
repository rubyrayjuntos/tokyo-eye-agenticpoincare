import { describe, expect, it } from "vitest";
import { createActor } from "xstate";

import { structureScopeMachine } from "../structureScopeMachine";

describe("structureScopeMachine", () => {
  it("merges workspace scope and session mode", () => {
    const actor = createActor(structureScopeMachine).start();

    actor.send({
      type: "SYNC_FROM_WORKSPACE",
      scope: {
        scopeId: "scope-1",
        scopeType: "single_structure",
        structureIds: ["9o0r"],
        primaryStructureId: "9o0r",
        activeChainIds: ["A"],
        ingestionReady: true,
        inferenceReady: true,
        pipelineReady: false,
        provenanceLabel: "run-1",
        lastLoadedAt: 1,
        lastActor: "system",
        requiresReset: false,
        resetReason: null,
        sessionMode: "scientific_rigor",
      },
    });

    expect(actor.getSnapshot().context.primaryStructureId).toBe("9o0r");
    expect(actor.getSnapshot().context.pipelineReady).toBe(false);

    actor.send({ type: "SET_SESSION_MODE", mode: "free_flow" });
    expect(actor.getSnapshot().context.sessionMode).toBe("free_flow");
  });
});