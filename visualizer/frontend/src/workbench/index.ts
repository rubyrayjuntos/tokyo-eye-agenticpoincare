export { WorkbenchBus } from "./WorkbenchBus";
export { WorkbenchProvider, useWorkbench, useWorkbenchBus, useLayoutEngineAdapter } from "./WorkbenchProvider";
export { LayoutEngineAdapter } from "./LayoutEngineAdapter";
export { WorkbenchCanvas } from "./WorkbenchCanvas";
export { WorkbenchCockpitLayout } from "./WorkbenchCockpitLayout";
export { PhaseGroupTabs, resolveActivePhaseGroup } from "./PhaseGroupTabs";
export { useOrchestrationSync, applyBackendSnapshot, snapshotFingerprint } from "./useOrchestrationSync";
export { useStructureScopeSync } from "./useStructureScopeSync";
export { useWorkbenchPolicy } from "./useWorkbenchPolicy";
export { deriveStructureScopeFromWorkspace } from "./deriveStructureScope";
export {
  PHASE_GROUP_CONFIG,
  PHASE_GROUP_ORDER,
  discoveryPhaseToGroup,
  groupToDefaultDiscoveryPhase,
  type WorkbenchPhaseGroup,
} from "./phaseGroups";
export {
  PINNED_ACTIVITY_PANELS,
  PINNED_PANEL_DOCK_SPAWN,
  type PinnedActivityPanel,
} from "./activityBarPolicy";
export type { WorkbenchTopic, EventRegistry } from "./eventRegistry";
