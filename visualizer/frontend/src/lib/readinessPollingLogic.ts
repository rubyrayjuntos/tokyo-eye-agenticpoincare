/**
 * readinessPollingLogic — Pure logic for polling structure readiness during onboard.
 */
import type { Structure, StructureReadiness, PipelineJob } from "./types";
import { evaluatePipelineStatus } from "./autoIngestPipelineLogic";

export const READINESS_POLL_START_LABEL = "Starting discovery pipeline…";
export const MAX_READINESS_POLL_FAILURES = 3;

export type ReadinessCompletionAction =
  | { type: "complete"; structure: Structure; readiness: StructureReadiness }
  | { type: "failed"; error: string; readiness?: StructureReadiness }
  | { type: "continue" };

export type ReadinessPollCycleResult =
  | Exclude<ReadinessCompletionAction, { type: "continue" }>
  | { type: "continue"; readiness: StructureReadiness };

export function formatReadinessFailureMessage(readiness: StructureReadiness): string {
  const pipelineError = readiness.pipeline_job?.error?.trim();
  const gnnPresent = Boolean(
    readiness.tier1?.gnn_hyp ?? readiness.foundation?.gnn_hyp
  );

  if (
    pipelineError &&
    gnnPresent &&
    /no gnn nodes available/i.test(pipelineError)
  ) {
    const probeLine = readiness.degraded_reasons?.find((reason) =>
      reason.startsWith("Probe error:")
    );
    if (probeLine) {
      return probeLine;
    }
    const step = readiness.pipeline_job?.current_step ?? "downstream compute";
    return `Pipeline failed at ${step} (GNN embeddings are present — not an inference failure)`;
  }

  if (pipelineError) {
    return pipelineError;
  }
  if (!readiness.pipeline_job) {
    return "Discovery pipeline did not run — no compute job is queued for this structure.";
  }
  return (
    readiness.degraded_reasons?.[0] ??
    readiness.missing_artifacts?.join(", ") ??
    "Structure compute failed"
  );
}

/** Additional context lines after the primary pipeline failure message. */
export function supplementaryReadinessErrors(readiness: StructureReadiness): string[] {
  const pipelineError = readiness.pipeline_job?.error?.trim();
  const primary = formatReadinessFailureMessage(readiness);
  const reasons = readiness.degraded_reasons ?? [];
  return reasons.filter((reason) => reason !== primary && reason !== pipelineError);
}

export function evaluateReadinessStatus(
  readiness: StructureReadiness,
  structure: Structure
): ReadinessCompletionAction {
  if (readiness.readiness_status === "failed") {
    return {
      type: "failed",
      error: formatReadinessFailureMessage(readiness),
      readiness,
    };
  }

  if (readiness.readiness_status === "ready" || readiness.readiness_status === "degraded") {
    const hydrated: Structure = {
      ...structure,
      has_embeddings: Boolean(readiness.foundation?.gnn_hyp ?? readiness.tier1?.gnn_hyp),
      last_run_id: readiness.computation_run_id,
    };
    return { type: "complete", structure: hydrated, readiness };
  }

  return { type: "continue" };
}

export function actProgressLabel(readiness: StructureReadiness): string | null {
  const actId = actIdForNumber(readiness.current_act);
  if (!actId) return null;
  const act = readiness.acts?.[actId];
  if (!act) return null;
  return `Act ${act.number} — ${act.title}: ${act.question}`;
}

export function errorMessageFromUnknown(err: unknown): string {
  if (
    err &&
    typeof err === "object" &&
    "message" in err &&
    typeof (err as { message: string }).message === "string"
  ) {
    return (err as { message: string }).message;
  }
  return "Readiness check failed";
}

export async function runReadinessPollCycle(options: {
  structure: Structure;
  getReadiness: () => Promise<StructureReadiness>;
  getPipelineStatus?: (jobId: string) => Promise<PipelineJob>;
  pipelineJobId?: string | null;
}): Promise<ReadinessPollCycleResult> {
  try {
    const readiness = await options.getReadiness();
    const action = evaluateReadinessStatus(readiness, options.structure);
    if (action.type === "continue") {
      return { type: "continue", readiness };
    }
    return action;
  } catch (readinessErr) {
    if (options.getPipelineStatus && options.pipelineJobId) {
      try {
        const job = await options.getPipelineStatus(options.pipelineJobId);
        const pipelineAction = evaluatePipelineStatus(job, options.structure);
        if (pipelineAction.type === "failed") {
          return pipelineAction;
        }
        if (pipelineAction.type === "complete") {
          const readiness = await options.getReadiness();
          const action = evaluateReadinessStatus(readiness, options.structure);
          if (action.type === "continue") {
            return { type: "continue", readiness };
          }
          return action;
        }
      } catch {
        // Fall through to readiness error.
      }
    }
    throw readinessErr;
  }
}

function actIdForNumber(currentAct: number): string | null {
  const map: Record<number, string> = {
    1: "signal",
    2: "persistent_leak",
    3: "cryptic_pocket",
    4: "fragment",
    5: "verdict",
  };
  return map[currentAct] ?? null;
}
