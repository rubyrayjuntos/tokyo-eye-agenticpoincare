import type { DiscoveryPhaseEvent } from "../lib/discoveryPhaseMachine";
import {
  createDefaultStructureScope,
  type StructureScopeContext,
} from "../lib/plannerPolicySelector";
import type { HydrationResponse, Structure } from "../lib/types";

export interface StructureScopeDerivation {
  scope: StructureScopeContext;
  discoveryEvents: DiscoveryPhaseEvent[];
  hydrationMeta: {
    structureId: string;
    residueCount: number;
    degraded: boolean;
  } | null;
}

export function deriveStructureScopeFromWorkspace(
  activeStructure: Structure | null,
  hydration: HydrationResponse | null,
): StructureScopeDerivation {
  const base = createDefaultStructureScope();
  if (!activeStructure) {
    return { scope: base, discoveryEvents: [], hydrationMeta: null };
  }

  const structureId = activeStructure.structure_id.toLowerCase();
  const snapshot = hydration?.structure_snapshot;
  const snapshotResidues = snapshot?.residues?.length ?? 0;
  const embeddingResidues = hydration?.embeddings?.residues?.length ?? 0;
  const residueCount = Math.max(
    snapshotResidues,
    embeddingResidues,
    hydration?.context_summary?.residue_count ?? 0,
  );

  const status = snapshot?.status ?? hydration?.persistence_status;
  const inferenceReady =
    activeStructure.has_embeddings || residueCount > 0 || Boolean(hydration?.embeddings);
  const pipelineReady = Boolean(
    inferenceReady &&
      (status?.graph_persisted || (hydration?.graph_metrics?.metrics?.length ?? 0) > 0),
  );

  const scope: StructureScopeContext = {
    ...base,
    primaryStructureId: structureId,
    structureIds: [structureId],
    activeChainIds: activeStructure.chains ?? [],
    ingestionReady: true,
    inferenceReady,
    pipelineReady,
    provenanceLabel:
      hydration?.context_summary?.latest_run_ids_by_pipeline?.embeddings ??
      activeStructure.last_run_id,
    lastLoadedAt: Date.now(),
    lastActor: "system",
  };

  const discoveryEvents: DiscoveryPhaseEvent[] = [
    { type: "STRUCTURE_MAPPED", structureId },
  ];

  if ((hydration?.allosteric_sites?.sites?.length ?? 0) > 0) {
    discoveryEvents.push({
      type: "POCKET_EXTRACTED",
      structureId,
      pocketIndex: 0,
    });
  }

  if ((hydration?.drug_candidates?.candidates?.length ?? 0) > 0) {
    discoveryEvents.push({
      type: "SCREENING_COMPLETED",
      runId:
        hydration?.context_summary?.latest_run_ids_by_pipeline?.screening ??
        "hydrate",
    });
  }

  const degraded =
    Boolean(hydration) &&
    residueCount === 0 &&
    Boolean(hydration?.context_summary?.latest_run_ids_by_pipeline?.embeddings);

  return {
    scope,
    discoveryEvents,
    hydrationMeta: hydration
      ? {
          structureId,
          residueCount,
          degraded,
        }
      : null,
  };
}
