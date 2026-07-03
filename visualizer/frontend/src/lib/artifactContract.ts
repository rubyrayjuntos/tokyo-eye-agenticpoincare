/**
 * Re-exports contract-driven artifact tiers from onboard_contract.yaml.
 * Hand-maintained helpers live alongside generated constants.
 */

import {
  ARTIFACT_TIERS,
  BINDING_SCAN_COMPLETE_STATUSES,
  TIER1_ARTIFACT_KEYS,
  type Tier1ArtifactKey,
} from "./generated/artifactContract.generated";

export { ARTIFACT_TIERS, BINDING_SCAN_COMPLETE_STATUSES, TIER1_ARTIFACT_KEYS, type Tier1ArtifactKey };

export function isBindingScanComplete(status: string | null | undefined): boolean {
  return BINDING_SCAN_COMPLETE_STATUSES.has(String(status ?? ""));
}