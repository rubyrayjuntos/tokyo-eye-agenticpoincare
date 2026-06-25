import { describe, expect, it } from "vitest";

import { validatePanelPortPayload } from "../panelContract";

describe("panel contract payload validation", () => {
  it("accepts valid selection payloads", () => {
    expect(
      validatePanelPortPayload(
        { kind: "selection" },
        { residueIds: ["A:42", "B:17"] },
      ),
    ).toBe(true);
  });

  it("rejects malformed toggle payloads", () => {
    expect(
      validatePanelPortPayload(
        { kind: "toggle" },
        { target: "sidebar", value: "yes" },
      ),
    ).toBe(false);
  });

  it("accepts JSON layout payloads", () => {
    expect(
      validatePanelPortPayload(
        { kind: "layout" },
        { partialLayout: { workspace: [24, 76], nested: { open: true } } },
      ),
    ).toBe(true);
  });
});
