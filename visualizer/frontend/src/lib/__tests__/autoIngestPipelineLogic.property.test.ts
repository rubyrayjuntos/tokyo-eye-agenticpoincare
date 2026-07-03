/**
 * Property-based tests for autoIngestPipelineLogic.
 */
import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import {
  ingestAndTriggerPipeline,
  evaluatePipelineStatus,
  type PipelineApiAdapter,
} from "../autoIngestPipelineLogic";
import type { IngestResponse, PipelineJob, Structure } from "../types";

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

describe("Feature: structure-onboarding-panel, Property 2: Pipeline job comes from ingest only", () => {
  it("throws when ingest succeeds without pipeline_job_id and not audit_only", async () => {
    await fc.assert(
      fc.asyncProperty(arbPdbId, arbIngestResponse, async (pdbId, ingestResponse) => {
        const adapter: PipelineApiAdapter = {
          ingest: async () => ({ ...ingestResponse, audit_only: false }),
          getPipelineStatus: async () => ({
            job_id: "unused",
            structure_id: ingestResponse.structure_id,
            status: "queued",
            current_step: "pipeline",
            progress: 0,
            started_at: new Date().toISOString(),
            completed_at: null,
            error: null,
          }),
        };

        await expect(ingestAndTriggerPipeline(pdbId, adapter)).rejects.toThrow(
          /did not queue a pipeline job/i
        );
      }),
      { numRuns: 100 }
    );
  });

  it("uses coordinator pipeline_job_id when present", async () => {
    await fc.assert(
      fc.asyncProperty(arbPdbId, arbIngestResponse, fc.uuid(), async (pdbId, ingestResponse, queuedJobId) => {
        const adapter: PipelineApiAdapter = {
          ingest: async () => ({
            ...ingestResponse,
            pipeline_job_id: queuedJobId,
            pipeline_status: "queued",
          }),
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

        expect(result.pipelineJob?.job_id).toBe(queuedJobId);
        expect(result.pipelineJob?.structure_id).toBe(ingestResponse.structure_id);
      }),
      { numRuns: 100 }
    );
  });

  it("returns pipeline job for audit-only ingest when coordinator re-queued compute", async () => {
    await fc.assert(
      fc.asyncProperty(arbPdbId, arbIngestResponse, fc.uuid(), async (pdbId, ingestResponse, queuedJobId) => {
        const adapter: PipelineApiAdapter = {
          ingest: async () => ({
            ...ingestResponse,
            audit_only: true,
            already_existed: true,
            pipeline_job_id: queuedJobId,
            pipeline_status: "queued",
          }),
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
        expect(result.pipelineJob?.job_id).toBe(queuedJobId);
      }),
      { numRuns: 100 }
    );
  });

  it("returns null pipeline job for audit-only duplicate ingests without re-queue", async () => {
    await fc.assert(
      fc.asyncProperty(arbPdbId, arbIngestResponse, async (pdbId, ingestResponse) => {
        const adapter: PipelineApiAdapter = {
          ingest: async () => ({ ...ingestResponse, audit_only: true }),
          getPipelineStatus: async () => ({
            job_id: "unused",
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
        expect(result.pipelineJob).toBeNull();
      }),
      { numRuns: 100 }
    );
  });
});

describe("Feature: structure-onboarding-panel, Property 3: Pipeline completion activates structure", () => {
  it("For any pipeline job with status 'complete', evaluatePipelineStatus returns a 'complete' action", () => {
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
