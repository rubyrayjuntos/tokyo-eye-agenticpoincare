import { buildHydrationView } from "./hydrationView";
import { ARTIFACT_TIERS, isBindingScanComplete } from "./artifactContract";
import type { ArtifactAvailabilityEntry, HydrateMeta, HydrationResponse } from "./types";

function hasResidues(hydration: HydrationResponse): boolean {
  const view = buildHydrationView(hydration);
  return (view.embeddings?.residues?.length ?? 0) > 0;
}

function isPresent(hydration: HydrationResponse, artifactKey: string): boolean {
  const view = buildHydrationView(hydration);
  const snap = hydration.structure_snapshot as
    | { structure?: unknown; scope?: { primary_chain_ids?: string[] }; residues?: unknown[] }
    | null
    | undefined;

  switch (artifactKey) {
    case "dims":
      return Boolean(snap?.structure);
    case "scope":
      return Boolean(snap?.scope?.primary_chain_ids?.length);
    case "gnn_hyp":
      return hasResidues(hydration);
    case "graph":
      return (view.graphMetrics?.metrics?.length ?? 0) > 0;
    case "source_leaks":
      return (view.sourceLeaks?.leaks?.length ?? 0) > 0;
    case "allosteric_sites":
      return (view.allostericSites?.sites?.length ?? 0) > 0;
    case "binding_scan": {
      const bindingScan = view.bindingScan;
      if (!bindingScan) return false;
      if (!isBindingScanComplete(bindingScan.status)) return false;
      return Boolean(bindingScan.sites?.length || (bindingScan as { count?: number }).count);
    }
    case "strain_vulnerability":
      return Boolean(hydration.phase2_vulnerability);
    case "resistance_pathway": {
      const p4 = hydration.phase4_resistance as
        | { pathways?: unknown[]; spectral?: unknown }
        | null
        | undefined;
      return Boolean(p4?.pathways?.length || p4?.spectral);
    }
    case "pharmacophores":
    case "pocket_pharmacophore":
      return (view.pharmacophorePockets?.pockets?.length ?? 0) > 0;
    case "drug_candidates":
      return (view.drugCandidates?.candidates?.length ?? 0) > 0;
    case "buffering_atlas":
      return Boolean(hydration.buffering_atlas);
    default:
      return false;
  }
}

export function inferArtifactAvailability(
  hydration: HydrationResponse,
): Record<string, ArtifactAvailabilityEntry> {
  const availability: Record<string, ArtifactAvailabilityEntry> = {};
  for (const [artifactKey, tier] of Object.entries(ARTIFACT_TIERS)) {
    const present = isPresent(hydration, artifactKey);
    availability[artifactKey] = {
      present,
      tier,
      reason: present ? null : "client_inferred",
    };
  }
  return availability;
}

export function inferHydrateMeta(
  availability: Record<string, ArtifactAvailabilityEntry>,
): HydrateMeta {
  const tier1Missing = Object.entries(availability)
    .filter(([, entry]) => entry.tier === 1 && !entry.present)
    .map(([key]) => key);
  return {
    contract_version: "inferred",
    degraded: tier1Missing.length > 0,
    missing_tier1_count: tier1Missing.length,
    missing_keys: tier1Missing.sort(),
  };
}

export interface ResolvedHydrateSignals {
  artifactAvailability: Record<string, ArtifactAvailabilityEntry>;
  hydrateMeta: HydrateMeta;
  signalsInferred: boolean;
}

export function resolveHydrateSignals(
  hydration: HydrationResponse | null,
): ResolvedHydrateSignals | null {
  if (!hydration) return null;

  if (hydration.artifact_availability && hydration.hydrate_meta) {
    return {
      artifactAvailability: hydration.artifact_availability,
      hydrateMeta: hydration.hydrate_meta,
      signalsInferred: false,
    };
  }

  const artifactAvailability = inferArtifactAvailability(hydration);
  return {
    artifactAvailability,
    hydrateMeta: inferHydrateMeta(artifactAvailability),
    signalsInferred: true,
  };
}