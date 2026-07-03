/**
 * autoIngestPipelineLogic — Pure orchestration logic for ingest → pipeline flow.
 * Extracted for testability. The useAutoIngestPipeline hook wraps this with React state.
 *
 * Feature: structure-onboarding-panel
 * Requirements: 3.1, 4.1, 4.2, 4.3, 4.4
 */
import type { PipelineJob, IngestResponse, Structure, StructureReadiness } from "./types";

export interface PipelineApiAdapter {
  ingest: (pdbId: string) => Promise<IngestResponse>;
  getPipelineStatus: (jobId: string) => Promise<PipelineJob>;
  getStructureReadiness?: (structureId: string) => Promise<StructureReadiness>;
}

export interface IngestAndRunResult {
  structure: Structure;
  pipelineJob: PipelineJob | null;
  readinessUrl: string | null;
}

const INGEST_ONLY_MESSAGE =
  "Ingest did not queue a pipeline job. Compute is only triggered via POST /api/ingest.";

/**
 * Executes the ingest step and returns the coordinator-queued pipeline job.
 * Returns null pipelineJob for audit-only duplicate ingests.
 */
export async function ingestAndTriggerPipeline(
  pdbId: string,
  adapter: PipelineApiAdapter
): Promise<IngestAndRunResult> {
  const ingestResult = await adapter.ingest(pdbId.trim().toUpperCase());

  const structure: Structure = {
    structure_id: ingestResult.structure_id,
    pdb_id: ingestResult.pdb_id,
    title: ingestResult.title ?? ingestResult.pdb_id,
    resolution: null,
    method: "",
    source: "rcsb",
    chains: ingestResult.chains,
    residue_count: ingestResult.residue_count,
    has_embeddings: false,
    last_run_id: null,
    ingested_at: new Date().toISOString(),
  };

  const readinessUrl =
    ingestResult.readiness_url ??
    `/api/structures/${ingestResult.structure_id}/readiness`;

  let pipelineJob: PipelineJob | null = null;
  if (ingestResult.pipeline_job_id) {
    pipelineJob = {
      job_id: ingestResult.pipeline_job_id,
      structure_id: ingestResult.structure_id,
      status:
        ingestResult.pipeline_status && ingestResult.pipeline_status !== "skipped"
          ? ingestResult.pipeline_status
          : "queued",
      current_step: "gnn_inference",
      progress: 0,
      started_at: new Date().toISOString(),
      completed_at: null,
      error: null,
    };
  } else if (!ingestResult.audit_only) {
    throw new Error(INGEST_ONLY_MESSAGE);
  }

  return { structure, pipelineJob, readinessUrl };
}

/**
 * Evaluates what should happen when a pipeline status is received.
 */
export type PipelineCompletionAction =
  | { type: "complete"; structure: Structure }
  | { type: "failed"; error: string }
  | { type: "continue" };

export function evaluatePipelineStatus(
  job: PipelineJob,
  ingestedStructure: Structure
): PipelineCompletionAction {
  if (job.status === "complete") {
    return { type: "complete", structure: ingestedStructure };
  }
  if (job.status === "failed") {
    return { type: "failed", error: job.error || "Pipeline failed" };
  }
  return { type: "continue" };
}
