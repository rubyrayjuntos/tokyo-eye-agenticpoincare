// Stub — full policy selector lives in the extended dashboard

export type SessionMode =
  | "scientific_rigor"
  | "exploratory"
  | "therapeutic_focus"
  | "comparison";

export interface StructureScopeContext {
  primaryStructureId: string | null;
  secondaryStructureId: string | null;
  focusedResidues: string[];
  activeModules: string[];
}

export interface PlannerPolicy {
  sessionMode: SessionMode;
  maxHypotheses: number;
  evidenceThreshold: number;
  autoEvaluate: boolean;
  preferredTools: string[];
}

export function defaultPlannerPolicy(mode: SessionMode): PlannerPolicy {
  switch (mode) {
    case "therapeutic_focus":
      return { sessionMode: mode, maxHypotheses: 5, evidenceThreshold: 0.7, autoEvaluate: true, preferredTools: ["allosteric_sites", "cryptic_pocket"] };
    case "exploratory":
      return { sessionMode: mode, maxHypotheses: 15, evidenceThreshold: 0.4, autoEvaluate: false, preferredTools: [] };
    case "comparison":
      return { sessionMode: mode, maxHypotheses: 8, evidenceThreshold: 0.6, autoEvaluate: true, preferredTools: ["compare_embeddings"] };
    default:
      return { sessionMode: mode, maxHypotheses: 10, evidenceThreshold: 0.6, autoEvaluate: true, preferredTools: [] };
  }
}
