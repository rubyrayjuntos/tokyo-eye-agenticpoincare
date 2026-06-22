// Stub — full XState machine lives in the extended dashboard

export type DiscoveryPhase =
  | "idle"
  | "structure_loading"
  | "embedding"
  | "clustering"
  | "hypothesis_generation"
  | "complete";

export interface DiscoveryPhaseContext {
  phase: DiscoveryPhase;
  structureId: string | null;
  error: string | null;
  progress: number;
}

export type DiscoveryPhaseEvent =
  | { type: "START"; structureId: string }
  | { type: "STRUCTURE_LOADED" }
  | { type: "EMBEDDING_COMPLETE" }
  | { type: "CLUSTERING_COMPLETE" }
  | { type: "HYPOTHESES_GENERATED" }
  | { type: "RESET" }
  | { type: "ERROR"; error: string };
