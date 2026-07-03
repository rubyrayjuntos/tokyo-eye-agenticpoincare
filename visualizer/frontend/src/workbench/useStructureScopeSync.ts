import { useEffect, useRef } from "react";

import type { OrchestratorPolicyResult } from "../lib/useOrchestratorPolicy";
import type { HydrationResponse, Structure } from "../lib/types";
import { deriveStructureScopeFromWorkspace } from "./deriveStructureScope";
import type { WorkbenchBus } from "./WorkbenchBus";

interface UseStructureScopeSyncOptions {
  activeStructure: Structure | null;
  hydration: HydrationResponse | null;
  orchestrator: OrchestratorPolicyResult;
  workbenchBus?: WorkbenchBus;
  enabled?: boolean;
}

export function useStructureScopeSync({
  activeStructure,
  hydration,
  orchestrator,
  workbenchBus,
  enabled = true,
}: UseStructureScopeSyncOptions): void {
  const lastAppliedRef = useRef<string>("");
  const sendStructureScopeRef = useRef(orchestrator.sendStructureScope);
  const sendDiscoveryRef = useRef(orchestrator.sendDiscovery);
  const structureScopeRef = useRef(orchestrator.structureScope);
  sendStructureScopeRef.current = orchestrator.sendStructureScope;
  sendDiscoveryRef.current = orchestrator.sendDiscovery;
  structureScopeRef.current = orchestrator.structureScope;

  useEffect(() => {
    if (!enabled) return;

    const derivation = deriveStructureScopeFromWorkspace(activeStructure, hydration);
    const fingerprint = JSON.stringify({
      structureId: activeStructure?.structure_id ?? null,
      hydrationId: hydration?.structure_id ?? null,
      residueCount: derivation.hydrationMeta?.residueCount ?? 0,
      pipelineReady: derivation.scope.pipelineReady,
    });

    if (fingerprint === lastAppliedRef.current) {
      return;
    }
    lastAppliedRef.current = fingerprint;

    sendStructureScopeRef.current({
      type: "SYNC_FROM_WORKSPACE",
      scope: {
        ...structureScopeRef.current,
        ...derivation.scope,
      },
    });

    for (const event of derivation.discoveryEvents) {
      sendDiscoveryRef.current(event);
    }

    if (workbenchBus) {
      workbenchBus.publish("data:structure_scope_updated", {
        primaryStructureId: derivation.scope.primaryStructureId,
        structureIds: derivation.scope.structureIds,
        activeChainIds: derivation.scope.activeChainIds,
        ingestionReady: derivation.scope.ingestionReady,
        inferenceReady: derivation.scope.inferenceReady,
        pipelineReady: derivation.scope.pipelineReady,
      });

      if (derivation.hydrationMeta) {
        workbenchBus.publish("data:hydration_ready", derivation.hydrationMeta);
      }
    }
  }, [activeStructure, enabled, hydration, workbenchBus]);
}
