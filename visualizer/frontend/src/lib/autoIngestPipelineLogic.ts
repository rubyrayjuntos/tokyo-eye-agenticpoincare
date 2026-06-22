/**
 * autoIngestPipelineLogic — Pure orchestration logic for ingest → pipeline flow.
 * Extracted for testability. The useAutoIngestPipeline hook wraps this with React state.
 *
 * Feature: structure-onboarding-panel
 * Requirements: 3.1, 4.1, 4.2, 4.3, 4.4
 */
import type { PipelineJob, IngestResponse, Structure } from "./types";

export interface PipelineApiAdapter {
  ingest: (pdbId: string) => Promise<IngestResponse>;
  runPipeline: (req: { structure_id: string }) => Promise<PipelineJob>;
  getPipelineStatus: (jobId: string) => Promise<PipelineJob>;
}

export interface IngestAndRunResult {
  structure: Structure;
  pipelineJob: PipelineJob;
}

/**
 * Executes the ingest step and immediately triggers pipeline run.
 * Returns the created structure and initial pipeline job.
 *
 * Property 2: For any successful ingest response containing a valid structure_id,
 * this function SHALL call runPipeline with that structure_id without additional user interaction.
 */
export async function ingestAndTriggerPipeline(
  pdbId: string,
  adapter: PipelineApiAdapter
): Promise<IngestAndRunResult> {
  const ingestResult = await adapter.ingest(pdbId.trim().toUpperCase());

  const structure: Structure = {
    structure_id: ingestResult.structure_id,
    pdb_id: ingestResult.pdb_id,
    title: ingestResult.title,
    resolution: null,
    method: "",
    source: "rcsb",
    chains: ingestResult.chains,
    residue_count: ingestResult.residue_count,
    has_embeddings: false,
    last_run_id: null,
    ingested_at: new Date().toISOString(),
  };

  // Auto-trigger pipeline with all default modules (no modules = all enabled)
  const pipelineJob = await adapter.runPipeline({
    structure_id: ingestResult.structure_id,
  });

  return { structure, pipelineJob };
}

/**
 * Evaluates what should happen when a pipeline status is received.
 *
 * Property 3: For any pipeline job that reaches status === "complete",
 * this SHALL return an action to activate the structure and refresh the dashboard.
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
