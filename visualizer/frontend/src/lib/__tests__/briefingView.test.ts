import { describe, expect, it } from "vitest";

import { buildBriefingView } from "../briefingView";
import type { HydrationResponse } from "../types";

const sampleHydration: HydrationResponse = {
  structure_id: "4uj1",
  artifact_availability: {
    gnn_hyp: { present: true, tier: 1, reason: null },
    graph: { present: true, tier: 1, reason: null },
    source_leaks: { present: true, tier: 1, reason: null },
    dims: { present: true, tier: 1, reason: null },
    scope: { present: true, tier: 1, reason: null },
    allosteric_sites: { present: true, tier: 1, reason: null },
    binding_scan: { present: true, tier: 1, reason: null },
  },
  binding_scan: {
    structure_id: "4uj1",
    status: "complete",
    sites: [
      {
        site_id: "site_1",
        site_type: "cryptic_wedge",
        druggability_score: 0.72,
        site_rank: 1,
      },
    ],
    count: 1,
  },
  hydrate_meta: {
    contract_version: "1.3",
    degraded: false,
    missing_tier1_count: 0,
    missing_keys: [],
  },
  embeddings: {
    structure_id: "4uj1",
    curvature: 1.2,
    residues: [
      {
        residue_id: "4uj1:A:10",
        residue_index: 10,
        chain_label: "A",
        x: 0.1,
        y: 0.2,
        cone_depth: 0.5,
        epistemic_uncertainty: 9.1,
        aleatoric_uncertainty: 0.2,
      },
    ],
  },
  graph_metrics: {
    structure_id: "4uj1",
    metrics: [
      {
        residue_id: "4uj1:A:10",
        degree: 8,
        betweenness: 0.05,
        clustering_coefficient: 0.1,
        is_bridge: false,
      },
    ],
  },
  source_leaks: {
    structure_id: "4uj1",
    leaks: [{ residue_id: "4uj1:A:10", leak_score: 0.8, source: "snapshot" }],
    count: 1,
  },
  persistence_status: {
    embeddings_persisted: true,
    graph_persisted: true,
    sites_persisted: true,
  },
  context_summary: {
    hypothesis_count: 0,
    latest_run_ids_by_pipeline: { embeddings: "run-abc123" },
  },
};

describe("buildBriefingView", () => {
  it("builds artifact-first sections from hydrate bundle", () => {
    const view = buildBriefingView(sampleHydration, "4uj1", false);
    expect(view).not.toBeNull();
    expect(view?.contractVersion).toBe("1.3");
    expect(view?.degraded).toBe(false);
    expect(view?.sections.find((s) => s.artifactKey === "binding_scan")?.present).toBe(true);
    expect(view?.sections.find((s) => s.artifactKey === "binding_scan")?.stats[0]?.value).toBe("1");
    expect(view?.sections.find((s) => s.artifactKey === "gnn_hyp")?.present).toBe(true);
    expect(view?.sections.find((s) => s.artifactKey === "gnn_hyp")?.rows[0]?.value).toBe(9.1);
  });
});
