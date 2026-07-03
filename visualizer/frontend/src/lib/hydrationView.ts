import type {
  AllostericSitesData,
  BindingScanData,
  DrugCandidateData,
  EmbeddingData,
  GraphMetricsData,
  HydrationResponse,
  Hypothesis,
  PharmacophoreData,
  ProvenanceRun,
  ResistanceData,
  SourceLeakData,
  StructureAnalysisSnapshot,
  StructureSnapshotStatus,
} from "./types";

export interface HydrationView {
  structureSnapshot: StructureAnalysisSnapshot | null;
  embeddings: EmbeddingData | null;
  graphMetrics: GraphMetricsData | null;
  allostericSites: AllostericSitesData | null;
  sourceLeaks: SourceLeakData | null;
  resistanceData: ResistanceData | null;
  pharmacophorePockets: PharmacophoreData | null;
  drugCandidates: DrugCandidateData | null;
  bindingScan: BindingScanData | null;
  phase4Resistance: ResistanceData | null;
  persistenceStatus: StructureSnapshotStatus | null;
  hypotheses: Hypothesis[] | null;
  provenanceRuns: ProvenanceRun[] | null;
}

function normalizeSourceLeaks(
  sourceLeaks: SourceLeakData | null | undefined,
  structureId: string,
): SourceLeakData | null {
  if (!sourceLeaks) {
    return null;
  }

  const leaks = (sourceLeaks.source_leaks ?? sourceLeaks.leaks ?? []).map((leak) => ({
    ...leak,
    source: leak.source ?? "snapshot",
  }));

  return {
    structure_id: sourceLeaks.structure_id ?? structureId,
    leaks,
    source_leaks: leaks,
    count: sourceLeaks.count ?? leaks.length,
  };
}

function normalizeAllostericSites(
  allostericSites:
    | StructureAnalysisSnapshot["findings"]["allosteric_sites"]
    | AllostericSitesData
    | null
    | undefined,
  structureId: string,
): AllostericSitesData | null {
  if (!allostericSites) {
    return null;
  }

  return {
    structure_id: allostericSites.structure_id ?? structureId,
    sites: allostericSites.sites.map((site) => ({
      site_id: site.site_id,
      confidence: site.confidence ?? site.confidence_score ?? 0,
      residue_ids: site.residue_ids ?? [],
    })),
  };
}

function normalizePharmacophores(
  pharmacophores:
    | StructureAnalysisSnapshot["findings"]["pharmacophores"]
    | HydrationResponse["phase5_pharmacophore"]
    | PharmacophoreData
    | null
    | undefined,
  structureId: string,
): PharmacophoreData | null {
  if (!pharmacophores) {
    return null;
  }

  const pockets =
    ("pharmacophores" in pharmacophores ? pharmacophores.pharmacophores : undefined) ??
    pharmacophores.pockets ??
    [];

  return {
    structure_id: pharmacophores.structure_id ?? structureId,
    pockets,
    count: pharmacophores.count ?? pockets.length,
  };
}

function normalizeDrugCandidates(
  drugCandidates:
    | StructureAnalysisSnapshot["findings"]["drug_candidates"]
    | HydrationResponse["phase6_drug_candidates"]
    | DrugCandidateData
    | null
    | undefined,
  structureId: string,
): DrugCandidateData | null {
  if (!drugCandidates) {
    return null;
  }

  const candidates =
    drugCandidates.candidates ??
    ("drug_candidates" in drugCandidates ? drugCandidates.drug_candidates : undefined) ??
    [];

  return {
    structure_id: drugCandidates.structure_id ?? structureId,
    candidates,
    count: drugCandidates.count ?? candidates.length,
    admet_passed_count: drugCandidates.admet_passed_count ?? 0,
    state_selective_count: drugCandidates.state_selective_count ?? 0,
  };
}

function normalizeBindingScan(
  bindingScan:
    | StructureAnalysisSnapshot["findings"]["binding_scan"]
    | HydrationResponse["binding_scan"]
    | BindingScanData
    | null
    | undefined,
  structureId: string,
): BindingScanData | null {
  if (!bindingScan) {
    return null;
  }
  const sites = bindingScan.sites ?? [];
  return {
    structure_id: bindingScan.structure_id ?? structureId,
    run_id: bindingScan.run_id,
    status: bindingScan.status,
    sites_found: bindingScan.sites_found,
    heuristic_version: bindingScan.heuristic_version,
    model_version: bindingScan.model_version,
    sites,
    count: bindingScan.count ?? sites.length,
  };
}

/** Unwrap hydrate tool payloads (`{ hypotheses: [...] }`) or accept a bare array. */
export function normalizeHypotheses(
  hypotheses: HydrationResponse["hypotheses"] | Hypothesis[] | null | undefined,
): Hypothesis[] | null {
  if (hypotheses == null) {
    return null;
  }
  if (Array.isArray(hypotheses)) {
    return hypotheses;
  }
  if (typeof hypotheses === "object" && "hypotheses" in hypotheses) {
    const rows = (hypotheses as { hypotheses?: unknown }).hypotheses;
    return Array.isArray(rows) ? (rows as Hypothesis[]) : null;
  }
  return null;
}

/** Unwrap hydrate tool payloads (`{ runs: [...] }`) or accept a bare array. */
export function normalizeProvenanceRuns(
  provenanceRuns: HydrationResponse["provenance_runs"] | ProvenanceRun[] | null | undefined,
): ProvenanceRun[] | null {
  if (provenanceRuns == null) {
    return null;
  }
  if (Array.isArray(provenanceRuns)) {
    return provenanceRuns;
  }
  if (typeof provenanceRuns === "object" && "runs" in provenanceRuns) {
    const rows = (provenanceRuns as { runs?: unknown }).runs;
    return Array.isArray(rows) ? (rows as ProvenanceRun[]) : null;
  }
  return null;
}

export function buildHydrationView(
  hydration: HydrationResponse | null,
  activeStructureId?: string | null,
): HydrationView {
  const snapshot = hydration?.structure_snapshot ?? null;
  const snapshotResidues = snapshot?.residues ?? [];
  const fallbackEmbeddings = hydration?.embeddings ?? null;
  const structureId =
    snapshot?.structure.structure_id ??
    hydration?.structure_id ??
    activeStructureId ??
    "";

  return {
    structureSnapshot: snapshot,
    embeddings: snapshot && snapshotResidues.length > 0
      ? {
          structure_id: snapshot.structure.structure_id,
          curvature: snapshot.curvature,
          residues: snapshotResidues,
        }
      : fallbackEmbeddings,
    graphMetrics: snapshot?.graph_metrics ?? hydration?.graph_metrics ?? null,
    allostericSites: normalizeAllostericSites(
      snapshot?.findings.allosteric_sites ?? hydration?.allosteric_sites,
      structureId,
    ),
    sourceLeaks: normalizeSourceLeaks(
      snapshot?.findings.source_leaks ?? hydration?.source_leaks,
      structureId,
    ),
    resistanceData: snapshot?.findings.resistance ?? hydration?.resistance_data ?? null,
    pharmacophorePockets: normalizePharmacophores(
      snapshot?.findings.pharmacophores ?? hydration?.phase5_pharmacophore ?? hydration?.pharmacophore_pockets,
      structureId,
    ),
    drugCandidates: normalizeDrugCandidates(
      snapshot?.findings.drug_candidates ?? hydration?.phase6_drug_candidates ?? hydration?.drug_candidates,
      structureId,
    ),
    bindingScan: normalizeBindingScan(
      snapshot?.findings.binding_scan ?? hydration?.binding_scan,
      structureId,
    ),
    phase4Resistance:
      snapshot?.findings.resistance ??
      hydration?.phase4_resistance ??
      hydration?.resistance_data ??
      null,
    persistenceStatus: snapshot
      ? {
          ...snapshot.status,
          resistance_data_available: snapshot.findings.resistance != null,
        }
      : (hydration?.persistence_status ?? null),
    hypotheses: normalizeHypotheses(hydration?.hypotheses),
    provenanceRuns: normalizeProvenanceRuns(hydration?.provenance_runs),
  };
}
