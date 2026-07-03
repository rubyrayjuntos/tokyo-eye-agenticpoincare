import { assign, setup } from "xstate";

import type { HydrationLoadResult } from "./hydrationLoader";
import type {
  ArtifactAvailabilityEntry,
  HydrateMeta,
  HydrationResponse,
} from "./types";

export type HydrationMachineStatus = "idle" | "loading" | "loaded" | "failed";

export interface HydrationMachineContext {
  structureId: string | null;
  hydration: HydrationResponse | null;
  error: string | null;
  artifactAvailability: Record<string, ArtifactAvailabilityEntry> | null;
  hydrateMeta: HydrateMeta | null;
  signalsInferred: boolean;
}

export type HydrationMachineEvent =
  | { type: "HYDRATION_REQUESTED"; structureId: string }
  | { type: "HYDRATION_SUCCEEDED"; result: HydrationLoadResult }
  | { type: "HYDRATION_FAILED"; error: string }
  | { type: "HYDRATION_CLEARED" };

function applyLoadResult(result: HydrationLoadResult): Partial<HydrationMachineContext> {
  return {
    structureId: result.hydration.structure_id,
    hydration: result.hydration,
    error: null,
    artifactAvailability: result.artifactAvailability,
    hydrateMeta: result.hydrateMeta,
    signalsInferred: result.signalsInferred,
  };
}

const initialContext: HydrationMachineContext = {
  structureId: null,
  hydration: null,
  error: null,
  artifactAvailability: null,
  hydrateMeta: null,
  signalsInferred: false,
};

export const hydrationMachine = setup({
  types: {} as {
    context: HydrationMachineContext;
    events: HydrationMachineEvent;
  },
  actions: {
    assignRequestedStructure: assign({
      structureId: ({ event }) =>
        event.type === "HYDRATION_REQUESTED" ? event.structureId : null,
      error: () => null,
    }),
    assignLoadSuccess: assign(({ event }) => {
      if (event.type !== "HYDRATION_SUCCEEDED") return {};
      return applyLoadResult(event.result);
    }),
    assignLoadFailure: assign(({ event }) => {
      if (event.type !== "HYDRATION_FAILED") return {};
      return {
        error: event.error,
        hydration: null,
        artifactAvailability: null,
        hydrateMeta: null,
        signalsInferred: false,
      };
    }),
    clearHydration: assign(() => initialContext),
  },
}).createMachine({
  id: "hydration",
  initial: "idle",
  context: initialContext,
  states: {
    idle: {
      on: {
        HYDRATION_REQUESTED: {
          target: "loading",
          actions: "assignRequestedStructure",
        },
      },
    },
    loading: {
      on: {
        HYDRATION_SUCCEEDED: {
          target: "loaded",
          actions: "assignLoadSuccess",
        },
        HYDRATION_FAILED: {
          target: "failed",
          actions: "assignLoadFailure",
        },
        HYDRATION_CLEARED: {
          target: "idle",
          actions: "clearHydration",
        },
        HYDRATION_REQUESTED: {
          target: "loading",
          actions: "assignRequestedStructure",
        },
      },
    },
    loaded: {
      on: {
        HYDRATION_REQUESTED: {
          target: "loading",
          actions: "assignRequestedStructure",
        },
        HYDRATION_CLEARED: {
          target: "idle",
          actions: "clearHydration",
        },
      },
    },
    failed: {
      on: {
        HYDRATION_REQUESTED: {
          target: "loading",
          actions: "assignRequestedStructure",
        },
        HYDRATION_CLEARED: {
          target: "idle",
          actions: "clearHydration",
        },
      },
    },
  },
});

export function hydrationMachineStatus(
  stateValue: string | Record<string, unknown>,
): HydrationMachineStatus {
  if (typeof stateValue === "string") {
    return stateValue as HydrationMachineStatus;
  }
  return "idle";
}