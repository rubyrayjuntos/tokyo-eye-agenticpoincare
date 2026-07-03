import type { DiscoveryPhase } from "./discoveryPhaseMachine";

/** Discovery Story act titles (pathway chapters). */
export const DISCOVERY_PHASE_ACT_LABELS: Record<DiscoveryPhase, string> = {
  residue: "Act 01 · Signal",
  topology: "Act 01 · Signal",
  structure: "Act 02 · Persistent Leak",
  pocket: "Act 03 · Cryptic Pocket",
  screening: "Act 04 · Fragment",
  report: "Act 05 · Verdict",
};

export const DISCOVERY_STORY_TAGLINE =
  "A loaded spring — signal, leak, pocket, fragment, verdict.";

export function discoveryPhaseActLabel(phase: DiscoveryPhase): string {
  return DISCOVERY_PHASE_ACT_LABELS[phase];
}
