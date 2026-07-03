import { describe, expect, it } from "vitest";
import { createActor } from "xstate";

import { viewportMachine } from "../viewportMachine";

describe("viewportMachine workbench state", () => {
  it("opens the selected tool panel and tracks sidebar visibility", () => {
    const actor = createActor(viewportMachine).start();

    actor.send({ type: "SET_ACTIVE_PANEL", panel: "graph_topology" });
    actor.send({ type: "SET_SIDEBAR_OPEN", open: true });

    expect(actor.getSnapshot().context.activePanel).toBe("graph_topology");
    expect(actor.getSnapshot().context.sidebarOpen).toBe(true);
  });

  it("tracks selection, brush, and mobius focus in machine context", () => {
    const actor = createActor(viewportMachine).start();

    actor.send({
      type: "SET_SELECTION",
      highlightedResidues: ["A:42"],
      brushSelectedIds: ["A:42"],
      selectedResidue: {
        residue_id: "A:42",
        residue_name: "GLU",
        chain_label: "A",
        epistemic_uncertainty: 0.9,
        cone_depth: 1.2,
      },
    });
    actor.send({ type: "SET_MOBIUS_FOCUS", enabled: true });

    const ctx = actor.getSnapshot().context;
    expect(ctx.highlightedResidues).toEqual(["A:42"]);
    expect(ctx.brushSelectedIds).toEqual(["A:42"]);
    expect(ctx.selectedResidue?.residue_id).toBe("A:42");
    expect(ctx.mobiusFocusEnabled).toBe(true);

    actor.send({ type: "CLEAR" });
    expect(actor.getSnapshot().context.highlightedResidues).toEqual([]);
    expect(actor.getSnapshot().context.brushSelectedIds).toEqual([]);
    expect(actor.getSnapshot().context.selectedResidue).toBeNull();
  });

  it("tracks editor and bottom panel selection independently", () => {
    const actor = createActor(viewportMachine).start();

    actor.send({ type: "SET_ACTIVE_EDITOR_TAB", tab: "agent" });
    actor.send({ type: "SET_ACTIVE_BOTTOM_PANEL", panel: "telemetry" });
    actor.send({ type: "SET_BOTTOM_PANEL_OPEN", open: false });

    expect(actor.getSnapshot().context.activeEditorTab).toBe("agent");
    expect(actor.getSnapshot().context.activeBottomPanel).toBe("telemetry");
    expect(actor.getSnapshot().context.bottomPanelOpen).toBe(false);
  });
});
