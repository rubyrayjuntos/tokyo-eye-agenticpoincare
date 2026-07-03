import { describe, expect, it, vi } from "vitest";

import {
  applyHypothesisLifecycleFromBackend,
  guardDiscoveryEvent,
  guardHypothesisEvent,
  isBackendAuthorizedDiscoveryEvent,
} from "../backendOrchestrationSync";

describe("backendOrchestrationSync", () => {
  it("allows backend-authorized discovery phase sync", () => {
    expect(
      isBackendAuthorizedDiscoveryEvent({
        type: "USER_SET_PHASE",
        phase: "topology",
        rationale: "backend_sync",
      }),
    ).toBe(true);
    expect(
      isBackendAuthorizedDiscoveryEvent({
        type: "USER_SET_PHASE",
        phase: "topology",
        rationale: "backend:orchestrator",
      }),
    ).toBe(true);
  });

  it("blocks user discovery phase changes in backend authority mode", () => {
    expect(
      guardDiscoveryEvent(
        {
          type: "USER_SET_PHASE",
          phase: "pocket",
          rationale: "manual_control_console",
        },
        "backend",
      ),
    ).toBeNull();
  });

  it("allows user discovery phase changes in local authority mode", () => {
    const event = {
      type: "USER_SET_PHASE" as const,
      phase: "pocket" as const,
      rationale: "manual_control_console",
    };
    expect(guardDiscoveryEvent(event, "local")).toEqual(event);
  });

  it("blocks hypothesis events in backend authority mode", () => {
    expect(
      guardHypothesisEvent({ type: "MARK_SUPPORTED" }, "backend"),
    ).toBeNull();
  });

  it("applies hypothesis lifecycle from backend snapshots", () => {
    const sendHypothesis = vi.fn();
    applyHypothesisLifecycleFromBackend(sendHypothesis, "emergent", "supported");
    expect(sendHypothesis).toHaveBeenCalledWith({ type: "MARK_SUPPORTED" });
  });

  it("skips hypothesis apply when lifecycle unchanged", () => {
    const sendHypothesis = vi.fn();
    applyHypothesisLifecycleFromBackend(sendHypothesis, "supported", "supported");
    expect(sendHypothesis).not.toHaveBeenCalled();
  });
});