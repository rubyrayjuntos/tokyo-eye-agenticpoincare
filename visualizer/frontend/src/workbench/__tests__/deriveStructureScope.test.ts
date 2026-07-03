import { describe, expect, it } from "vitest";

import { deriveStructureScopeFromWorkspace } from "../deriveStructureScope";
import type { HydrationResponse, Structure } from "../../lib/types";

const STRUCTURE: Structure = {
  structure_id: "4uj1",
  pdb_id: "4uj1",
  title: "KRAS",
  resolution: 1.7,
  method: "X-RAY",
  source: "rcsb",
  chains: ["A", "B"],
  residue_count: 730,
  has_embeddings: true,
  last_run_id: null,
  ingested_at: "2026-06-24T20:04:48.560743+00:00",
};

describe("deriveStructureScopeFromWorkspace", () => {
  it("returns defaults when no structure is loaded", () => {
    const result = deriveStructureScopeFromWorkspace(null, null);
    expect(result.scope.primaryStructureId).toBeNull();
    expect(result.discoveryEvents).toEqual([]);
    expect(result.hydrationMeta).toBeNull();
  });

  it("maps loaded structure and hydration into scope and discovery events", () => {
    const hydration = {
      structure_id: "4uj1",
      embeddings: { structure_id: "4uj1", curvature: 1, residues: [{}, {}] as any },
      graph_metrics: { structure_id: "4uj1", metrics: [{}] } as any,
      allosteric_sites: { structure_id: "4uj1", sites: [{ site_id: "s1" }] } as any,
      drug_candidates: { structure_id: "4uj1", candidates: [{ candidate_id: "c1" }] } as any,
      persistence_status: { graph_persisted: true } as any,
      context_summary: {
        residue_count: 2,
        source_leak_count: 0,
        hypothesis_count: 0,
        top_uncertainty_residues: [],
        latest_run_ids_by_pipeline: { embeddings: "run-1" },
      },
    } as unknown as HydrationResponse;

    const result = deriveStructureScopeFromWorkspace(STRUCTURE, hydration);

    expect(result.scope.primaryStructureId).toBe("4uj1");
    expect(result.scope.structureIds).toEqual(["4uj1"]);
    expect(result.scope.activeChainIds).toEqual(["A", "B"]);
    expect(result.scope.ingestionReady).toBe(true);
    expect(result.scope.inferenceReady).toBe(true);
    expect(result.scope.pipelineReady).toBe(true);
    expect(result.discoveryEvents.map((event) => event.type)).toEqual([
      "STRUCTURE_MAPPED",
      "POCKET_EXTRACTED",
      "SCREENING_COMPLETED",
    ]);
    expect(result.hydrationMeta).toEqual({
      structureId: "4uj1",
      residueCount: 2,
      degraded: false,
    });
  });
});
