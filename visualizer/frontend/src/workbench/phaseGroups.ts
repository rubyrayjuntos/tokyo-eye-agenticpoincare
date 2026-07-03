import type { DiscoveryPhase } from "../lib/discoveryPhaseMachine";
import type { WorkbenchPanelComponentType } from "./eventRegistry";

export type WorkbenchPhaseGroup = "exploration" | "analysis" | "review";

export interface PhaseGroupConfig {
  id: WorkbenchPhaseGroup;
  label: string;
  description: string;
  discoveryPhases: DiscoveryPhase[];
  defaultDiscoveryPhase: DiscoveryPhase;
  requiredPanels: WorkbenchPanelComponentType[];
}

export const PHASE_GROUP_CONFIG: Record<WorkbenchPhaseGroup, PhaseGroupConfig> = {
  exploration: {
    id: "exploration",
    label: "Act 01 · Signal",
    description: "Wide-angle scanning, residue selection, and hot-spot detection.",
    discoveryPhases: ["residue", "topology"],
    defaultDiscoveryPhase: "residue",
    requiredPanels: [
      "briefing-panel",
      "triple-viewport-panel",
      "findings-dock-panel",
      "chat-panel",
    ],
  },
  analysis: {
    id: "analysis",
    label: "Acts 02–03 · Leak & Pocket",
    description: "Structure mapping, pocket evaluation, and topology analysis.",
    discoveryPhases: ["structure", "pocket"],
    defaultDiscoveryPhase: "structure",
    requiredPanels: [
      "briefing-panel",
      "triple-viewport-panel",
      "findings-dock-panel",
      "chat-panel",
      "telemetry-panel",
    ],
  },
  review: {
    id: "review",
    label: "Acts 04–05 · Fragment & Verdict",
    description: "Screening synthesis, export, and report-ready validation.",
    discoveryPhases: ["screening", "report"],
    defaultDiscoveryPhase: "screening",
    requiredPanels: [
      "briefing-panel",
      "triple-viewport-panel",
      "findings-dock-panel",
      "chat-panel",
      "telemetry-panel",
    ],
  },
};

export const PHASE_GROUP_ORDER: WorkbenchPhaseGroup[] = [
  "exploration",
  "analysis",
  "review",
];

export function discoveryPhaseToGroup(phase: DiscoveryPhase): WorkbenchPhaseGroup {
  for (const group of PHASE_GROUP_ORDER) {
    if (PHASE_GROUP_CONFIG[group].discoveryPhases.includes(phase)) {
      return group;
    }
  }
  return "exploration";
}

export function groupToDefaultDiscoveryPhase(group: WorkbenchPhaseGroup): DiscoveryPhase {
  return PHASE_GROUP_CONFIG[group].defaultDiscoveryPhase;
}
