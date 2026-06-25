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
