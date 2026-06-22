// Stub — full XState machine lives in the extended dashboard

export type HypothesisLifecycleState =
  | "draft"
  | "gathering_evidence"
  | "evaluating"
  | "supported"
  | "contradicted"
  | "inconclusive";

export interface HypothesisLifecycleContext {
  hypothesisId: string | null;
  state: HypothesisLifecycleState;
  confidence: number;
  evidenceCount: number;
}

export type HypothesisLifecycleEvent =
  | { type: "CREATE"; hypothesisId: string }
  | { type: "ADD_EVIDENCE"; supports: boolean }
  | { type: "EVALUATE" }
  | { type: "RESET" };
