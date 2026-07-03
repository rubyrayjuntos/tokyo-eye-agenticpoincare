import { api } from "./api";
import { isArtifactPresent } from "./artifactAvailability";
import {
  inferArtifactAvailability,
  inferHydrateMeta,
  resolveHydrateSignals,
} from "./inferHydrateAvailability";
import type {
  ArtifactAvailabilityEntry,
  HydrateMeta,
  HydrationResponse,
} from "./types";

export interface HydrationLoadResult {
  hydration: HydrationResponse;
  artifactAvailability: Record<string, ArtifactAvailabilityEntry>;
  hydrateMeta: HydrateMeta;
  signalsInferred: boolean;
}

export async function loadHydrationBundle(
  structureId: string,
): Promise<HydrationLoadResult> {
  const data = await api.hydrate(structureId);
  const snapshot = data.structure_snapshot as { residues?: unknown[] } | null | undefined;
  const snapshotResidues = snapshot?.residues ?? [];
  const legacyResidues =
    (data.embeddings as { residues?: unknown[] } | null | undefined)?.residues ?? [];
  const gnnPresent =
    isArtifactPresent(data.artifact_availability, "gnn_hyp") ||
    snapshotResidues.length > 0 ||
    legacyResidues.length > 0;

  let merged: HydrationResponse = data;
  if (!gnnPresent) {
    try {
      const embeddings = await api.getEmbeddings(structureId);
      merged = { ...data, embeddings };
    } catch {
      merged = data;
    }
  }

  const signals = resolveHydrateSignals(merged);
  if (!signals || !data.artifact_availability || !gnnPresent) {
    const availability = inferArtifactAvailability(merged);
    merged = {
      ...merged,
      artifact_availability: availability,
      hydrate_meta: inferHydrateMeta(availability),
    };
    return {
      hydration: merged,
      artifactAvailability: availability,
      hydrateMeta: inferHydrateMeta(availability),
      signalsInferred: true,
    };
  }

  return {
    hydration: merged,
    artifactAvailability: signals.artifactAvailability,
    hydrateMeta: signals.hydrateMeta,
    signalsInferred: signals.signalsInferred,
  };
}