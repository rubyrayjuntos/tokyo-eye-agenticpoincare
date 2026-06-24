/**
 * Property-based tests for autoIngestPipelineLogic.
 *
 * Feature: structure-onboarding-panel, Property 2: Auto-pipeline job exists after successful ingest
 * Feature: structure-onboarding-panel, Property 3: Pipeline completion activates structure and refreshes dashboard
 */
import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import {
  ingestAndTriggerPipeline,
  evaluatePipelineStatus,
  type PipelineApiAdapter,
} from "../autoIngestPipelineLogic";
import type { IngestResponse, PipelineJob, Structure } from "../types";

// --- Arbitraries ---

const arbPdbId = fc
  .array(fc.constantFrom("A", "B", "C", "D", "1", "2", "3", "4"), { minLength: 4, maxLength: 4 })
  .map((chars) => chars.join(""));

const arbStructureId = fc.uuid();

const arbChain = fc.constantFrom("A", "B", "C", "D");

const arbIngestResponse: fc.Arbitrary<IngestResponse> = fc.record({
  structure_id: arbStructureId,
  pdb_id: arbPdbId,
  title: fc.string({ minLength: 1, maxLength: 80 }),
  residue_count: fc.nat({ max: 10000 }),
  chains: fc.array(arbChain, { minLength: 1, maxLength: 4 }),
});

const arbIsoDate = fc.integer({ min: 1577836800000, max: 1893456000000 }).map((ts) => new Date(ts).toISOString());

const arbPipelineJob: fc.Arbitrary<PipelineJob> = fc.record({
  job_id: fc.uuid(),
  structure_id: arbStructureId,
  status: fc.constantFrom("queued" as const, "running" as const, "complete" as const, "failed" as const),
  current_step: fc.string({ minLength: 1, maxLength: 30 }),
  progress: fc.integer({ min: 0, max: 100 }),
  started_at: arbIsoDate,
  completed_at: fc.option(arbIsoDate, { nil: null }),
  error: fc.option(fc.string({ minLength: 1, maxLength: 100 }), { nil: null }),
});

const arbStructure: fc.Arbitrary<Structure> = fc.record({
  structure_id: arbStructureId,
  pdb_id: arbPdbId,
  title: fc.string({ minLength: 1, maxLength: 80 }),
  resolution: fc.option(fc.float({ min: 0.5, max: 5.0, noNaN: true }), { nil: null }),
  method: fc.string({ maxLength: 20 }),
  source: fc.constant("rcsb"),
  chains: fc.array(arbChain, { minLength: 1, maxLength: 4 }),
  residue_count: fc.nat({ max: 10000 }),
  has_embeddings: fc.boolean(),
  last_run_id: fc.option(fc.uuid(), { nil: null }),
  ingested_at: arbIsoDate,
});

// --- Property 2: Auto-pipeline job exists after successful ingest ---

describe("Feature: structure-onboarding-panel, Property 2: Auto-pipeline job exists after successful ingest", () => {
  it("For any ingest response without a queued job, runPipeline is called with that structure_id", async () => {
    await fc.assert(
      fc.asyncProperty(
        arbPdbId,
        arbIngestResponse.map((resp) => ({ ...resp, pipeline_job_id: undefined })),
        arbPipelineJob,
        async (pdbId, ingestResponse, pipelineJobTemplate) => {
          const runPipelineCalls: Array<{ structure_id: string }> = [];

          const adapter: PipelineApiAdapter = {
            ingest: async () => ingestResponse,
            runPipeline: async (req) => {
              runPipelineCalls.push(req);
              return { ...pipelineJobTemplate, structure_id: req.structure_id };
            },
            getPipelineStatus: async () => pipelineJobTemplate,
          };

          const result = await ingestAndTriggerPipeline(pdbId, adapter);

          // runPipeline MUST have been called exactly once
          expect(runPipelineCalls).toHaveLength(1);
          // runPipeline MUST have been called with the structure_id from ingest
          expect(runPipelineCalls[0].structure_id).toBe(ingestResponse.structure_id);
          // The returned structure must have the same structure_id
          expect(result.structure.structure_id).toBe(ingestResponse.structure_id);
        }
      ),
      { numRuns: 100 }
    );
  });

  it("For any ingest response with a queued job, runPipeline is not called again", async () => {
    await fc.assert(
      fc.asyncProperty(
        arbPdbId,
        arbIngestResponse,
        fc.uuid(),
        async (pdbId, ingestResponse, queuedJobId) => {
          const runPipelineCalls: Array<{ structure_id: string }> = [];

          const adapter: PipelineApiAdapter = {
            ingest: async () => ({
              ...ingestResponse,
              pipeline_job_id: queuedJobId,
              pipeline_status: "queued",
            }),
            runPipeline: async (req) => {
              runPipelineCalls.push(req);
              return {
                job_id: queuedJobId,
                structure_id: req.structure_id,
                status: "queued",
                current_step: "pipeline",
                progress: 0,
                started_at: new Date().toISOString(),
                completed_at: null,
                error: null,
              };
            },
            getPipelineStatus: async () => ({
              job_id: queuedJobId,
              structure_id: ingestResponse.structure_id,
              status: "queued",
              current_step: "pipeline",
              progress: 0,
              started_at: new Date().toISOString(),
              completed_at: null,
              error: null,
            }),
          };

          const result = await ingestAndTriggerPipeline(pdbId, adapter);

          expect(runPipelineCalls).toHaveLength(0);
          expect(result.pipelineJob.job_id).toBe(queuedJobId);
          expect(result.pipelineJob.structure_id).toBe(ingestResponse.structure_id);
        }
      ),
      { numRuns: 100 }
    );
  });
});

// --- Property 3: Pipeline completion activates structure and refreshes dashboard ---

describe("Feature: structure-onboarding-panel, Property 3: Pipeline completion activates structure and refreshes dashboard", () => {
  it("For any pipeline job with status 'complete', evaluatePipelineStatus returns a 'complete' action with the ingested structure", () => {
    fc.assert(
      fc.property(
        arbPipelineJob.map((j) => ({ ...j, status: "complete" as const })),
        arbStructure,
        (completedJob, structure) => {
          const action = evaluatePipelineStatus(completedJob, structure);

          expect(action.type).toBe("complete");
          if (action.type === "complete") {
            expect(action.structure).toBe(structure);
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  it("For any pipeline job with status 'failed', evaluatePipelineStatus returns a 'failed' action", () => {
    fc.assert(
      fc.property(
        arbPipelineJob.map((j) => ({ ...j, status: "failed" as const })),
        arbStructure,
        (failedJob, structure) => {
          const action = evaluatePipelineStatus(failedJob, structure);

          expect(action.type).toBe("failed");
          if (action.type === "failed") {
            expect(action.error).toBeTruthy();
          }
        }
      ),
      { numRuns: 100 }
    );
  });

  it("For any pipeline job with status 'running' or 'queued', evaluatePipelineStatus returns 'continue'", () => {
    fc.assert(
      fc.property(
        arbPipelineJob.chain((j) =>
          fc.constantFrom("running" as const, "queued" as const).map((s) => ({ ...j, status: s }))
        ),
        arbStructure,
        (activeJob, structure) => {
          const action = evaluatePipelineStatus(activeJob, structure);
          expect(action.type).toBe("continue");
        }
      ),
      { numRuns: 100 }
    );
  });
});
