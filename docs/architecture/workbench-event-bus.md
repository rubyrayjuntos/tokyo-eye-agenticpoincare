# Workbench Event Bus Architecture

**Status:** Implemented (phases 0–6 complete)  
**Date:** 2026-06-25  
**Decisions locked:**


| Decision           | Choice                                               |
| ------------------ | ---------------------------------------------------- |
| Layout engine      | **Dockview** (prototype `LayoutEngineAdapter`)       |
| Phase UX           | **3 grouped tabs** (Exploration / Analysis / Review) |
| Layout persistence | **PostgreSQL** per user + session                    |


---

## Problem

The production frontend (`visualizer/frontend`) and the refactor prototype (`visualizer/frontend-refactor/dockable-panel-state-machine`) evolved parallel coordination layers:

- **Production:** XState machines (`discoveryPhaseMachine`, `hypothesisLifecycleMachine`, `viewportMachine`), a stub `panelBroker`, WebSocket viewport directives, and `react-resizable-panels` layout.
- **Prototype:** Typed `DataBroker` pub/sub, Dockview layout adapter, 3-phase state machine with middleware gating, client-side LLM stubs.
- **Backend:** `SessionOrchestrator` with phase-aware tool gating (now wired to `/api/agent/chat`).

Without consolidation, phase policy, selection sync, layout mutations, and agent tools can diverge across three surfaces. The `state-machine-audit.txt` (2026-06-24) documents concrete symptoms: hydration context not mirrored into machines, residue counts inconsistent, and backend snapshots not always driving UI state.

This document defines the **target architecture** and migration path.

---



## Design thesis

> One **typed event spine** on the frontend; one **authoritative orchestrator** on the backend; Dockview as the **layout runtime**.

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Workbench                          │
│  Dockview panels ──► WorkbenchBus ──► LayoutEngineAdapter       │
│       ▲                    │                    │               │
│       └────────────────────┴────────────────────┘               │
│                    subscribe / publish                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST + WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                   Agent Coordinator                             │
│  SessionOrchestrator (policy)  │  workspace_layout (DB)         │
│  /api/agent/chat (enforced)    │  /api/session/.../layout       │
└─────────────────────────────────────────────────────────────────┘
```

**Rules:**

1. **Backend enforces** agent tool policy (`SessionOrchestrator.can_invoke_tool`).
2. **Frontend middleware mirrors** policy for UX (block + warn before dispatch); never the sole security boundary.
3. **All cross-panel signals** go through `WorkbenchBus` topics (panel ports become thin adapters).
4. **Layout snapshots** persist to DB on debounced `layout:changed` events.
5. **No browser-side LLM** for production tools; `/api/agent/chat` remains the execution path.

---



## Layer 0 — Backend session state



### SessionOrchestrator (existing)

Canonical source for:

- `DiscoveryPhase` (6 internal values: `residue` → `report`)
- `HypothesisLifecycleState`
- `allowed_tools` / `blocked_tools`
- `PlannerPolicy` derivation

**Endpoints (new):**


| Method | Path                                            | Purpose                                           |
| ------ | ----------------------------------------------- | ------------------------------------------------- |
| `GET`  | `/api/session/{session_id}/orchestration`       | Full orchestrator snapshot                        |
| `POST` | `/api/session/{session_id}/orchestration/event` | Apply transition (`USER_SET_PHASE`, `ADVANCE`, …) |


Uses existing `agent/orchestration/session_store.py` per-session instances.

**WebSocket extension:** On `/ws/viewport` connect (after auth), push:

```json
{ "type": "state_snapshot", "snapshot": { ... SessionOrchestrator.get_state_snapshot() } }
```

Frontend `useOrchestrationSync` subscribes and publishes `system:orchestration_snapshot` on the bus.

### Workspace layout persistence (new)

Table: `agent_workspace_layout`


| Column                                                                           | Type        | Notes                                     |
| -------------------------------------------------------------------------------- | ----------- | ----------------------------------------- |
| `session_id`                                                                     | TEXT PK     | Ties to agent chat / viewport session     |
| `user_id`                                                                        | TEXT        | From JWT `sub`; scopes queries            |
| `workspace_id`                                                                   | TEXT        | Default `default`; future multi-workspace |
| `layout_json`                                                                    | JSONB       | Dockview `SerializedDockview`             |
| `active_phase_group`                                                             | TEXT        | `exploration` | `analysis` | `review`     |
| `updated_at`                                                                     | TIMESTAMPTZ | Audit                                     |


**Endpoints:**


| Method | Path                                         | Purpose                     |
| ------ | -------------------------------------------- | --------------------------- |
| `GET`  | `/api/session/{session_id}/workspace-layout` | Load layout + phase group   |
| `PUT`  | `/api/session/{session_id}/workspace-layout` | Upsert layout + phase group |


Auth: `Depends(get_current_user)`. Writes validate `user_id` matches JWT subject.

Operational tables (like agent memory) may bypass Normalizer; layout is session metadata, not scientific fact data.

---



## Layer 1 — WorkbenchBus (frontend)

Location: `visualizer/frontend/src/workbench/`

Port of prototype `DataBroker` with production topic registry.

### Core API

```typescript
class WorkbenchBus {
  subscribe<T extends WorkbenchTopic>(topic, callback): unsubscribe
  subscribeToAll(callback): unsubscribe
  publish<T extends WorkbenchTopic>(topic, payload): { accepted: boolean }
  getLatestState<T>(topic): EventRegistry[T] | undefined
  use(middleware): void
}
```



### Middleware chain (order matters)

1. **Phase gate** — reads latest `system:orchestration_snapshot`; blocks `tool:`* topics not in `allowed_tools` (UI feedback only).
2. **Schema guard** — TypeScript registry + runtime checks on publish.
3. **Dispatch** — update cache, notify subscribers, append `broker:log_added`.



### Topic namespaces


| Prefix     | Examples                                                | Source                  |
| ---------- | ------------------------------------------------------- | ----------------------- |
| `system:*` | `phase_transition`, `orchestration_snapshot`, `warning` | Backend WS, phase tabs  |
| `ui:*`     | `select_residue`, `interaction`                         | Panels, shell           |
| `data:*`   | `highlight_triggered`, `metric_changed`                 | Viewers                 |
| `tool:*`   | `call:highlight_residues`, agent results                | Chat / backend          |
| `layout:*` | `spawn_panel`, `sync_requirements`, `changed`           | LayoutEngineAdapter     |
| `broker:*` | `log_added`                                             | Telemetry / AgentLogHub |


Panel contract triggers map 1:1:


| `panelContract` trigger | Bus topic                |
| ----------------------- | ------------------------ |
| `selection.changed`     | `ui:selection_changed`   |
| `directive.emitted`     | `data:directive_emitted` |
| `layout.changed`        | `layout:changed`         |
| `panel.activated`       | `ui:panel_activated`     |


---



## Layer 2 — Phase UX (3 grouped tabs)

Internal model remains **6 discovery phases** (backend + DTIE). UI exposes **3 groups**:


| Tab (UX)                                       | `WorkbenchPhaseGroup` | Internal `DiscoveryPhase` values | Default on enter |
| ---------------------------------------------- | --------------------- | -------------------------------- | ---------------- |
| **Residue**                                    | `exploration`         | `residue`, `topology`            | `residue`        |
| **Topology** → label **Analysis** in prototype | `analysis`            | `structure`, `pocket`            | `structure`      |
| **Structure** → label **Review** in prototype  | `review`              | `screening`, `report`            | `screening`      |


Module: `workbench/phaseGroups.ts`

When user selects a tab:

1. `WorkbenchBus.publish("system:phase_transition", { group, discoveryPhase })`
2. `POST /api/session/{id}/orchestration/event` with `{ type: "USER_SET_PHASE", phase }`
3. `LayoutEngineAdapter.syncRequirements(PHASE_LAYOUT_PRESETS[group])`

Each group defines **required Dockview panel component types** (see below).

---



## Layer 3 — Dockview layout engine



### LayoutEngineAdapter

Binds `DockviewApi` ↔ `WorkbenchBus`:

- Listens: `layout:spawn_panel`, `layout:sync_requirements`
- Publishes: `layout:changed` with `{ workspaceId, layout: SerializedDockview }`
- Debounced persist → `PUT /api/session/{id}/workspace-layout`



### Default panel registry


| `componentType`   | Production component        |
| ----------------- | --------------------------- |
| `poincare-panel`  | `PoincarePanel`             |
| `molecular-panel` | `MolecularPanel`            |
| `chat-panel`      | `ChatRail` / agent chat     |
| `telemetry-panel` | Agent telemetry / KPI strip |


Phase presets:


| Group         | Default visible panels                         |
| ------------- | ---------------------------------------------- |
| `exploration` | poincare, molecular, chat                      |
| `analysis`    | + graph topology slot (future), data inspector |
| `review`      | + telemetry, export-oriented panels            |




### Hydration on session start

1. `GET /api/session/{id}/workspace-layout`
2. If `layout_json` present → `layoutEngineAdapter.loadLayoutFromSnapshot`
3. Else → apply group preset for `active_phase_group`

---



## Layer 4 — Slim frontend state machines


| Machine                      | Role after migration                                         |
| ---------------------------- | ------------------------------------------------------------ |
| `viewportMachine`            | Local viewer state (metric, highlights) — stays              |
| `discoveryPhaseMachine`      | **Mirror** backend snapshot; dispatches events upstream only |
| `hypothesisLifecycleMachine` | **Mirror** backend snapshot                                  |
| `useOrchestratorPolicy`      | Workbench mode uses `useWorkbenchPolicy` + backend snapshot for tool gating |


Do not delete XState until all AppCockpit paths read from bus/backend snapshots.

---



## Agent integration


| Concern               | Path                                                                    |
| --------------------- | ----------------------------------------------------------------------- |
| Tool execution        | `/api/agent/chat` with `SessionOrchestrator` gating                     |
| Spawn panel tool      | Backend tool emits viewport message → frontend bus `layout:spawn_panel` |
| Context block         | Orchestrator snapshot + bus `getLatestState` aggregation                |
| LLM persona per phase | Backend `reasoning_policy` + optional per-group prompt fragments        |


Prototype `LlmExecutionEngine` (browser Gemini) is **not** ported.

---



## Migration phases


| Phase | Deliverable                                                  | Status  |
| ----- | ------------------------------------------------------------ | ------- |
| **0** | This document                                                | ✅       |
| **1** | `WorkbenchBus`, `phaseGroups`, panelBroker adapter           | ✅       |
| **2** | Dockview canvas + `LayoutEngineAdapter` + feature flag shell | ✅       |
| **3** | Backend layout + orchestration REST; WS snapshot on connect  | ✅       |
| **4** | Persist layout on debounced `layout:changed`                 | ✅       |
| **5** | Collapse duplicate XState policy; audit fixes                | ✅       |
| **6** | Remove `react-resizable-panels` shell from default path      | ✅       |


Default shell: **Workbench** (Dockview + bus). Legacy `IdeShellLayout` remains available with `VITE_USE_LEGACY_SHELL=true`.

---



## Testing strategy


| Area                  | Tests                                            |
| --------------------- | ------------------------------------------------ |
| `phaseGroups`         | Group ↔ phase mapping, preset panel lists        |
| `WorkbenchBus`        | Middleware blocks gated tools; cache + subscribe |
| `LayoutEngineAdapter` | Mock Dockview API; spawn/sync events             |
| Workspace API         | Auth, user isolation, upsert idempotency         |
| Orchestration API     | Phase transition round-trip with session store   |


---



## References

- Prototype: `visualizer/frontend-refactor/dockable-panel-state-machine/`
- Production audit: `visualizer/frontend/state-machine-audit.txt`
- Backend orchestrator: `agent/orchestration/orchestrator.py`
- Panel contract: `visualizer/frontend/src/components/workbench/panelContract.ts`

