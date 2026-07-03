import {
  formatArtifactReason,
  getArtifactEntry,
  getArtifactSurfaceState,
  type ArtifactSurfaceState,
} from "../lib/artifactAvailability";
import type { ArtifactAvailabilityEntry } from "../lib/types";

interface ArtifactAvailabilityChipProps {
  artifactKey: string;
  availability: Record<string, ArtifactAvailabilityEntry> | undefined | null;
  label?: string;
}

function chipTone(state: ArtifactSurfaceState): string {
  switch (state) {
    case "present":
      return "border-success/40 text-success";
    case "tier1_empty":
      return "border-warning/50 text-warning";
    case "tier2_missing":
      return "border-slate text-text-secondary";
    default:
      return "border-slate text-text-secondary";
  }
}

function chipLabel(state: ArtifactSurfaceState): string {
  switch (state) {
    case "present":
      return "Ready";
    case "tier1_empty":
      return "Unavailable";
    case "tier2_missing":
      return "Coming soon";
    default:
      return "Unknown";
  }
}

export default function ArtifactAvailabilityChip({
  artifactKey,
  availability,
  label,
}: ArtifactAvailabilityChipProps) {
  const state = getArtifactSurfaceState(availability, artifactKey);
  const entry = getArtifactEntry(availability, artifactKey);
  const title = entry?.reason ? formatArtifactReason(entry.reason) : undefined;

  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] ${chipTone(state)}`}
    >
      {label ?? artifactKey}
      <span className="opacity-80">· {chipLabel(state)}</span>
    </span>
  );
}
