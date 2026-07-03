// AUTO-GENERATED from science/contracts/onboard_contract.yaml — do not edit by hand.
// Regenerate: make contract-sync

export const BINDING_SCAN_COMPLETE_STATUSES = new Set([
  "complete", "no_coordinates", "no_sites_found",
]);

export const TIER1_ARTIFACT_KEYS = [
  "dims",
  "scope",
  "gnn_hyp",
  "graph",
  "source_leaks",
  "binding_scan",
] as const;

export type Tier1ArtifactKey = (typeof TIER1_ARTIFACT_KEYS)[number];

export const ARTIFACT_TIERS: Record<string, number | null> = {
  "alignment": 2,
  "allele_selectivity": null,
  "allosteric_sites": null,
  "binding_scan": 1,
  "buffering_atlas": null,
  "dims": 1,
  "discovery_extended": 2,
  "drug_candidates": null,
  "fragment_hits": null,
  "gnn_euc": null,
  "gnn_hyp": 1,
  "graph": 1,
  "md_validation": 2,
  "motifs": 2,
  "pharmacophores": null,
  "pocket_pharmacophore": null,
  "resistance_pathway": null,
  "scope": 1,
  "source_leaks": 1,
  "strain_vulnerability": null,
  "witness_embedding": null,
};
