import { describe, expect, it, vi } from "vitest";

import { createPanelBroker } from "../panelBroker";
import type { PanelManifest } from "../panelContract";
import type { ToolPanelId } from "../../../lib/types";

const manifest: PanelManifest<ToolPanelId> = {
  id: "data_inspector",
  title: "Data Inspector",
  region: "activity-sidebar",
  ports: [
    { name: "toggle", kind: "toggle", direction: "output" },
    { name: "selection", kind: "selection", direction: "output" },
    { name: "directive", kind: "directive", direction: "output" },
  ],
};

describe("panel broker", () => {
  it("routes toggle ports into workbench state events", () => {
    const sendViewport = vi.fn();
    const broker = createPanelBroker({
      sendViewport,
      emitDirective: vi.fn(),
      mergeLayoutModel: vi.fn(),
    });

    broker.emitPort(manifest, "toggle", { target: "sidebar", value: true });

    expect(sendViewport).toHaveBeenCalledWith({
      type: "SET_SIDEBAR_OPEN",
      open: true,
    });
  });

  it("routes selection ports into viewport selection events", () => {
    const sendViewport = vi.fn();
    const broker = createPanelBroker({
      sendViewport,
      emitDirective: vi.fn(),
      mergeLayoutModel: vi.fn(),
    });

    broker.emitPort(manifest, "selection", { residueIds: ["A:42"] });

    expect(sendViewport).toHaveBeenCalledWith({
      type: "USER_SELECT",
      residues: ["A:42"],
    });
  });

  it("rejects invalid payloads before dispatch", () => {
    const broker = createPanelBroker({
      sendViewport: vi.fn(),
      emitDirective: vi.fn(),
      mergeLayoutModel: vi.fn(),
    });

    expect(() =>
      broker.emitPort(manifest, "toggle", { target: "sidebar", value: "open" }),
    ).toThrow("Invalid payload");
  });
});
