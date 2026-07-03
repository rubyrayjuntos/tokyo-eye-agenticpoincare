import type { ArtifactAvailabilityEntry } from "./types";

export type ArtifactSurfaceState =
  | "present"
  | "tier1_empty"
  | "tier2_missing"
  | "unknown";

export function getArtifactEntry(
  availability: Record<string, ArtifactAvailabilityEntry> | undefined | null,
  key: string,
): ArtifactAvailabilityEntry | null {
  return availability?.[key] ?? null;
}

export function getArtifactSurfaceState(
  availability: Record<string, ArtifactAvailabilityEntry> | undefined | null,
  key: string,
): ArtifactSurfaceState {
  const entry = getArtifactEntry(availability, key);
  if (!entry) return "unknown";
  if (entry.present) return "present";
  if (entry.tier === 1) return "tier1_empty";
  return "tier2_missing";
}

export function isArtifactPresent(
  availability: Record<string, ArtifactAvailabilityEntry> | undefined | null,
  key: string,
): boolean {
  return getArtifactEntry(availability, key)?.present === true;
}

export function formatArtifactReason(reason: string | null | undefined): string {
  switch (reason) {
    case "absent":
      return "Not computed yet";
    case "not_in_hydrate_bundle":
      return "Available via dedicated endpoint";
    case "job_planned":
      return "Pipeline job planned";
    case "job_partial":
      return "Pipeline job partially implemented";
    case "client_inferred":
      return "Inferred from hydrate payload (rebuild coordinator for server-side signals)";
    default:
      return reason ?? "Unavailable";
  }
}
