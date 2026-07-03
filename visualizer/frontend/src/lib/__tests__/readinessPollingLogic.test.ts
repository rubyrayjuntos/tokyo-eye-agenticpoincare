import { describe, it, expect } from "vitest";
import {
  actProgressLabel,
  evaluateReadinessStatus,
  formatReadinessFailureMessage,
  supplementaryReadinessErrors,
} from "../readinessPollingLogic";
import type { Structure, StructureReadiness } from "../types";

const baseStructure: Structure = {
  structure_id: "4obe",
  pdb_id: "4OBE",
  title: "KRAS",
  resolution: null,
  method: "",
  source: "rcsb",
  chains: ["A"],
  residue_count: 100,
  has_embeddings: false,
  last_run_id: null,
  ingested_at: "2026-06-25T00:00:00.000Z",
};

function readiness(overrides: Partial<StructureReadiness>): StructureReadiness {
  return {
    structure_id: "4obe",
    readiness_status: "running",
    pathway: "discovery_story",
    current_act: 1,
    foundation: { dims: true, scope: true, gnn_hyp: false, alignment: false },
    acts: {
      signal: {
        act_id: "signal",
        number: 1,
        title: "Signal",
        question: "Where is it strained?",
        color: "#5DDBC2",
        status: "running",
        jobs_complete: 0,
        jobs_total: 3,
        required_artifacts: { graph: false },
        optional_artifacts: {},
      },
    },
    artifacts: {},
    tier1: {},
    tier2: {},
    missing_artifacts: [],
    degraded_reasons: [],
    ...overrides,
  };
}

describe("readinessPollingLogic", () => {
  it("completes on ready status and hydrates embeddings flag", () => {
    const action = evaluateReadinessStatus(
      readiness({
        readiness_status: "ready",
        foundation: { dims: true, scope: true, gnn_hyp: true, alignment: true },
        tier1: { gnn_hyp: true },
      }),
      baseStructure
    );
    expect(action.type).toBe("complete");
    if (action.type === "complete") {
      expect(action.structure.has_embeddings).toBe(true);
    }
  });

  it("completes on degraded status", () => {
    const action = evaluateReadinessStatus(
      readiness({ readiness_status: "degraded" }),
      baseStructure
    );
    expect(action.type).toBe("complete");
  });

  it("fails on failed status", () => {
    const action = evaluateReadinessStatus(
      readiness({
        readiness_status: "failed",
        degraded_reasons: ["Missing tier-1 artifact: binding_scan"],
      }),
      baseStructure
    );
    expect(action.type).toBe("failed");
  });

  it("uses pipeline job error before degraded reasons", () => {
    const action = evaluateReadinessStatus(
      readiness({
        readiness_status: "failed",
        degraded_reasons: ["Missing tier-1 artifact: gnn_hyp — Hyperbolic GNN embeddings"],
        pipeline_job: {
          job_id: "job-1",
          structure_id: "4obe",
          status: "failed",
          error: "Preconditions not met for gnn_inference: missing dims",
        },
      }),
      baseStructure
    );
    expect(action.type).toBe("failed");
    if (action.type === "failed") {
      expect(action.error).toContain("missing dims");
    }
  });

  it("formatReadinessFailureMessage prefers pipeline error", () => {
    const message = formatReadinessFailureMessage(
      readiness({
        readiness_status: "failed",
        degraded_reasons: [
          "Pipeline job error: No GNN nodes available",
          "Missing tier-1 artifact: source_leaks — Source leaks",
        ],
        pipeline_job: {
          job_id: "job-1",
          structure_id: "4obe",
          status: "failed",
          error: "No GNN nodes available",
        },
      })
    );
    expect(message).toBe("No GNN nodes available");
    const extra = supplementaryReadinessErrors(
      readiness({
        readiness_status: "failed",
        degraded_reasons: [
          "Pipeline job error: No GNN nodes available",
          "Missing tier-1 artifact: source_leaks — Source leaks",
        ],
        pipeline_job: {
          job_id: "job-1",
          structure_id: "4obe",
          status: "failed",
          error: "No GNN nodes available",
        },
      })
    );
    expect(extra.some((line) => line.includes("source_leaks"))).toBe(true);
  });

  it("surfaces probe error when gnn_hyp present but pipeline shows no nodes", () => {
    const message = formatReadinessFailureMessage(
      readiness({
        readiness_status: "failed",
        tier1: { gnn_hyp: true },
        foundation: { dims: true, scope: true, gnn_hyp: true, alignment: false },
        degraded_reasons: [
          "Probe error: witness_embedding — ProgrammingError: bad placeholder",
          "Missing tier-1 artifact: source_leaks — Source leaks",
        ],
        pipeline_job: {
          job_id: "job-1",
          structure_id: "11qe",
          status: "failed",
          current_step: "source_leak_detection",
          error: "No GNN nodes available",
        },
      })
    );
    expect(message).toContain("Probe error: witness_embedding");
    expect(message).not.toBe("No GNN nodes available");
  });

  it("uses pipeline job error when readiness failed without degraded reasons", () => {
    const action = evaluateReadinessStatus(
      readiness({
        readiness_status: "failed",
        degraded_reasons: [],
        pipeline_job: {
          job_id: "job-1",
          structure_id: "4obe",
          status: "failed",
          error: "Preconditions not met for gnn_inference: missing dims",
        },
      }),
      baseStructure
    );
    expect(action.type).toBe("failed");
    if (action.type === "failed") {
      expect(action.error).toContain("missing dims");
    }
  });

  it("returns act progress label", () => {
    const label = actProgressLabel(
      readiness({
        current_act: 3,
        acts: {
          cryptic_pocket: {
            act_id: "cryptic_pocket",
            number: 3,
            title: "Cryptic Pocket",
            question: "Where is the backdoor?",
            color: "#5DDBC2",
            status: "running",
            jobs_complete: 1,
            jobs_total: 3,
            required_artifacts: { binding_scan: true },
            optional_artifacts: {},
          },
        },
      })
    );
    expect(label).toContain("Cryptic Pocket");
  });
});
