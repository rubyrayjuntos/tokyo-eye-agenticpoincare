/**
 * mergeResidueData — Pure function that merges per-residue data from all
 * computed DTIE pipeline phases into a unified table using residue_id as
 * the join key.
 *
 * Sources:
 *   - EmbeddingData.residues (Phase 1: cone_depth, uncertainty)
 *   - GraphMetricsData.metrics (graph: betweenness, degree, clustering, etc.)
 *   - SourceLeakData.leaks (source leaks: leak_score)
 *   - ResistanceData.residues (Phase 4: sensitivity, classification, hinge)
 *
 * Missing sources are handled gracefully — fields remain null.
 * Output is sorted by residue_index (ascending), with residues only present
 * in non-embedding sources sorted to the end by residue_id.
 */

import type {
  EmbeddingData,
  GraphMetricsData,
  SourceLeakData,
  ResistanceData,
} from "./types";

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

export interface MergedResidueRow {
  // Identity
  residue_id: string;
  chain_label: string | null;
  residue_index: number | null;

  // Embeddings (Phase 1)
  cone_depth: number | null;
  epistemic_uncertainty: number | null;
  aleatoric_uncertainty: number | null;

  // Graph metrics
  betweenness: number | null;
  degree: number | null;
  clustering_coefficient: number | null;
  closeness: number | null;
  eigenvector_centrality: number | null;
  is_bridge: boolean | null;

  // Source leaks
  leak_score: number | null;

  // Resistance (Phase 4)
  sensitivity_score: number | null;
  classification: "high_sensitivity" | "moderate" | "stable" | null;
  coupling_count: number | null;
  is_hinge: boolean | null;
}

// ---------------------------------------------------------------------------
// Merge function
// ---------------------------------------------------------------------------

export function mergeResidueData(
  embeddings: EmbeddingData | null,
  graphMetrics: GraphMetricsData | null,
  sourceLeaks: SourceLeakData | null,
  resistanceData: ResistanceData | null,
): MergedResidueRow[] {
  // Build a map keyed by residue_id
  const map = new Map<string, MergedResidueRow>();

  function getOrCreate(residueId: string): MergedResidueRow {
    let row = map.get(residueId);
    if (!row) {
      row = {
        residue_id: residueId,
        chain_label: null,
        residue_index: null,
        cone_depth: null,
        epistemic_uncertainty: null,
        aleatoric_uncertainty: null,
        betweenness: null,
        degree: null,
        clustering_coefficient: null,
        closeness: null,
        eigenvector_centrality: null,
        is_bridge: null,
        leak_score: null,
        sensitivity_score: null,
        classification: null,
        coupling_count: null,
        is_hinge: null,
      };
      map.set(residueId, row);
    }
    return row;
  }

  // --- Fill from embeddings ---
  if (embeddings?.residues) {
    for (const r of embeddings.residues) {
      const row = getOrCreate(r.residue_id);
      row.chain_label = r.chain_label;
      row.residue_index = r.residue_index;
      row.cone_depth = r.cone_depth;
      row.epistemic_uncertainty = r.epistemic_uncertainty;
      row.aleatoric_uncertainty = r.aleatoric_uncertainty;
    }
  }

  // --- Fill from graph metrics ---
  if (graphMetrics?.metrics) {
    for (const m of graphMetrics.metrics) {
      const row = getOrCreate(m.residue_id);
      row.betweenness = m.betweenness;
      row.degree = m.degree;
      row.clustering_coefficient = m.clustering_coefficient;
      row.closeness = m.closeness;
      row.eigenvector_centrality = m.eigenvector_centrality;
      row.is_bridge = m.is_bridge;
    }
  }

  // --- Fill from source leaks ---
  if (sourceLeaks) {
    // Backend may return as .leaks or .source_leaks
    const leaks = sourceLeaks.leaks ?? sourceLeaks.source_leaks ?? [];
    for (const l of leaks) {
      const row = getOrCreate(l.residue_id);
      row.leak_score = l.leak_score;
    }
  }

  // --- Fill from resistance data ---
  if (resistanceData?.residues) {
    for (const r of resistanceData.residues) {
      const row = getOrCreate(r.residue_id);
      row.sensitivity_score = r.sensitivity_score;
      row.classification = r.classification;
      row.coupling_count = r.coupling_count;
      row.is_hinge = r.is_hinge;
    }
  }

  // --- Sort: by residue_index ascending, nulls last, then by residue_id ---
  const rows = Array.from(map.values());
  rows.sort((a, b) => {
    if (a.residue_index !== null && b.residue_index !== null) {
      return a.residue_index - b.residue_index;
    }
    if (a.residue_index !== null) return -1;
    if (b.residue_index !== null) return 1;
    return a.residue_id.localeCompare(b.residue_id);
  });

  return rows;
}
