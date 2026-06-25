import { describe, expect, it } from "vitest";

import { buildHydrationView } from "../hydrationView";
import type { HydrationResponse } from "../types";

const baseHydration: HydrationResponse = {
  structure_id: "rcsb:4obe",
  structure_snapshot: null,
  embeddings: {
    structure_id: "legacy-structure",
    curvature: 2,
    residues: [],
  },
  graph_metrics: {
    structure_id: "legacy-structure",
    metrics: [],
  },
  allosteric_sites: {
    structure_id: "legacy-structure",
    sites: [{ site_id: "legacy-site", confidence: 0.12, residue_ids: ["legacy:A:1"] }],
  },
  source_leaks: {
    structure_id: "legacy-structure",
    leaks: [{ residue_id: "legacy:A:2", leak_score: 0.3, source: "legacy" }],
    count: 1,
  },
  resistance_data: {
    residues: [],
    spectral: { lambda_2: 0.1, hinge_count: 0 },
  },
  hypotheses: [],
  provenance_runs: [],
  annotations: [],
  pharmacophore_pockets: {
    structure_id: "legacy-structure",
    pockets: [],
    count: 0,
  },
  drug_candidates: {
    structure_id: "legacy-structure",
    candidates: [],
    count: 0,
    admet_passed_count: 0,
    state_selective_count: 0,
  },
  phase4_resistance: null,
  buffering_atlas: null,
  persistence_status: {
    embeddings_persisted: false,
    graph_persisted: false,
    sites_persisted: false,
  },
};

describe("buildHydrationView", () => {
  it("prefers structure_snapshot when both snapshot and legacy fields exist", () => {
    const hydration: HydrationResponse = {
      ...baseHydration,
      structure_snapshot: {
        structure: {
          structure_id: "snapshot-structure",
          pdb_id: "4OBE",
          title: "Snapshot",
          method: "X-RAY",
          resolution: 2.1,
          source: "rcsb",
        },
        scope: {
          primary_chain_ids: ["A"],
          reference_chain: "A",
          exclude_chain_ids: [],
          normalization_protocol: "graph_default",
          scope_source: "auto",
          selection_reason: "test",
        },
        provenance: {
          latest_run_ids_by_pipeline: { embeddings: "run-1" },
          latest_model_versions: { embeddings: "v6" },
        },
        curvature: 1,
        residues: [
          {
            residue_id: "snapshot:A:10",
            residue_index: 10,
            chain_label: "A",
            x: 1,
            y: 2,
            cone_depth: 3,
            epistemic_uncertainty: 0.4,
            aleatoric_uncertainty: 0.1,
          },
        ],
        graph_metrics: {
          structure_id: "snapshot-structure",
          metrics: [{ residue_id: "snapshot:A:10", degree: 3, betweenness: 0.5, clustering_coefficient: 0.2, closeness: 0.4, eigenvector_centrality: 0.1, is_bridge: false, conductance: 0.3 }],
        },
        findings: {
          source_leaks: {
            structure_id: "snapshot-structure",
            source_leaks: [{ residue_id: "snapshot:A:10", leak_score: 0.9, source: "snapshot" }],
            count: 1,
          },
          allosteric_sites: {
            structure_id: "snapshot-structure",
            sites: [{ site_id: "site-1", confidence_score: 0.88, residue_ids: ["snapshot:A:10"] }],
            count: 1,
          },
          vulnerability_doorways: null,
          resistance: {
            residues: [{ residue_id: "snapshot:A:10", sensitivity_score: 0.7, coupling_count: 3, is_hinge: true, classification: "high_sensitivity" }],
            spectral: { lambda_2: 0.2, hinge_count: 1 },
          },
          pharmacophores: {
            structure_id: "snapshot-structure",
            pharmacophores: [{ pocket_index: 1, druggability_score: 0.8, residue_count: 3, volume_estimate: 12, allosteric_coupling: 4, center_x: 1, center_y: 2, center_z: 3, residue_ids: ["snapshot:A:10"] }],
            count: 1,
          },
          drug_candidates: {
            structure_id: "snapshot-structure",
            candidates: [{ pocket_index: 1, combined_druggability: 0.91, accessibility_score: 0.7, binding_potential: 0.8, admet_pass: true, selectivity_ratio: 1.2, is_state_selective: true }],
            count: 1,
            admet_passed_count: 1,
            state_selective_count: 1,
          },
        },
        status: {
          embeddings_persisted: true,
          graph_persisted: true,
          sites_persisted: true,
          phase4_persisted: true,
          phase5_persisted: true,
          phase6_persisted: true,
        },
      },
    };

    const view = buildHydrationView(hydration, "active-structure");

    expect(view.structureSnapshot?.structure.structure_id).toBe("snapshot-structure");
    expect(view.embeddings?.structure_id).toBe("snapshot-structure");
    expect(view.sourceLeaks?.structure_id).toBe("snapshot-structure");
    expect(view.sourceLeaks?.leaks[0]?.residue_id).toBe("snapshot:A:10");
    expect(view.allostericSites?.sites[0]?.confidence).toBe(0.88);
    expect(view.pharmacophorePockets?.pockets[0]?.pocket_index).toBe(1);
    expect(view.drugCandidates?.candidates[0]?.combined_druggability).toBe(0.91);
    expect(view.persistenceStatus?.resistance_data_available).toBe(true);
  });

  it("falls back to legacy hydration fields when snapshot is absent", () => {
    const view = buildHydrationView(baseHydration, "active-structure");

    expect(view.structureSnapshot).toBeNull();
    expect(view.embeddings?.structure_id).toBe("legacy-structure");
    expect(view.sourceLeaks?.leaks[0]?.source).toBe("legacy");
    expect(view.allostericSites?.sites[0]?.confidence).toBe(0.12);
    expect(view.persistenceStatus?.embeddings_persisted).toBe(false);
  });
});
