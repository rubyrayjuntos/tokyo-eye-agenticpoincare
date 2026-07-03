import { assign, setup } from "xstate";

import {
  createDefaultStructureScope,
  type SessionMode,
  type StructureScopeContext,
} from "./plannerPolicySelector";

export interface StructureScopeMachineContext extends StructureScopeContext {}

export type StructureScopeEvent =
  | { type: "MERGE_SCOPE"; scope: Partial<StructureScopeContext> }
  | { type: "SYNC_FROM_WORKSPACE"; scope: StructureScopeContext }
  | { type: "SET_SESSION_MODE"; mode: SessionMode }
  | { type: "CLEAR_SCOPE" };

export const structureScopeMachine = setup({
  types: {} as {
    context: StructureScopeMachineContext;
    events: StructureScopeEvent;
  },
  actions: {
    mergeScope: assign(({ context, event }) => {
      if (event.type !== "MERGE_SCOPE") return context;
      return { ...context, ...event.scope };
    }),
    replaceScope: assign(({ event }) => {
      if (event.type !== "SYNC_FROM_WORKSPACE") {
        return createDefaultStructureScope();
      }
      return { ...event.scope };
    }),
    setSessionMode: assign({
      sessionMode: ({ context, event }) =>
        event.type === "SET_SESSION_MODE" ? event.mode : context.sessionMode,
    }),
    clearScope: assign(() => createDefaultStructureScope()),
  },
}).createMachine({
  id: "structureScope",
  initial: "active",
  context: createDefaultStructureScope(),
  states: {
    active: {
      on: {
        MERGE_SCOPE: { actions: "mergeScope" },
        SYNC_FROM_WORKSPACE: { actions: "replaceScope" },
        SET_SESSION_MODE: { actions: "setSessionMode" },
        CLEAR_SCOPE: { actions: "clearScope" },
      },
    },
  },
});