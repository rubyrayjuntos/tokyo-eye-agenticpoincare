import { TIER1_ARTIFACT_KEYS } from "./artifactContract";
import { formatLambda2 } from "./briefingMetrics";
import { buildHydrationView } from "./hydrationView";
import { resolveHydrateSignals } from "./inferHydrateAvailability";
import { formatResidueLabel } from "./residueLabels";
import type {
  ArtifactAvailabilityEntry,
  HydrateMeta,
  HydrationResponse,
  ResidueEmbedding,
} from "./types";

export interface BriefingArtifactRow {
  key: string;
  label: string;
  present: boolean;
  tier: number | null;
  reason: string | null;
}

export interface BriefingResidueRow {
  residue_id: string;
  label: string;
  value: number;
  valueLabel: string;
}

export interface BriefingSection {
  artifactKey: string;
  title: string;
  present: boolean;
  reason: string | null;
  stats: Array<{ label: string; value: string }>;
  rows: BriefingResidueRow[];
  footnote: string | null;
}

export interface BriefingView {
  structureId: string;
  pdbId: string | null;
  title: string | null;
  contractVersion: string;
  degraded: boolean;
  missingTier1: string[];
  signalsInferred: boolean;
  isHydrating: boolean;
  artifacts: BriefingArtifactRow[];
  sections: BriefingSection[];
  persistence: Array<{ label: string; persisted: boolean }>;
  runIds: Record<string, string>;
  hypothesisCount: number;
}

const ARTIFACT_LABELS: Record<string, string> = {
  dims: "Structure dimensions",
  scope: "Computation scope",
  gnn_hyp: "Hyperbolic GNN",
  graph: "Graph topology",
  source_leaks: "Source leaks",
  allosteric_sites: "Allosteric sites",
  strain_vulnerability: "Strain vulnerability",
  resistance_pathway: "Resistance pathway",
  pharmacophores: "Pharmacophores",
  drug_candidates: "Drug candidates",
  buffering_atlas: "Buffering atlas",
  motifs: "Motifs",
  witness_embedding: "Witness embedding",
  binding_scan: "Binding scan",
};

function topResidueRows(
  residues: ResidueEmbedding[],
  valueKey: "epistemic_uncertainty" | "cone_depth",
  valueLabel: string,
  limit = 5,
): BriefingResidueRow[] {
  return [...residues]
    .sort((a, b) => b[valueKey] - a[valueKey])
    .slice(0, limit)
    .map((r) => ({
      residue_id: r.residue_id,
      label: formatResidueLabel(r.residue_id),
      value: r[valueKey],
      valueLabel,
    }));
}

function artifactRows(
  availability: Record<string, ArtifactAvailabilityEntry>,
): BriefingArtifactRow[] {
  return TIER1_ARTIFACT_KEYS.filter((key) => availability[key]).map((key) => {
    const entry = availability[key];
    return {
      key,
      label: ARTIFACT_LABELS[key] ?? key,
      present: entry.present,
      tier: entry.tier,
      reason: entry.reason,
    };
  });
}

export function buildBriefingView(
  hydration: HydrationResponse | null,
  activeStructureId: string | null | undefined,
  isHydrating: boolean,
): BriefingView | null {
  if (!hydration && !isHydrating) return null;

  const signals = resolveHydrateSignals(hydration);
  const availability = signals?.artifactAvailability ?? {};
  const meta: HydrateMeta = signals?.hydrateMeta ?? {
    contract_version: "—",
    degraded: false,
    missing_tier1_count: 0,
    missing_keys: [],
  };

  const view = buildHydrationView(hydration, activeStructureId);
  const snapshot = view.structureSnapshot;
  const structureId =
    hydration?.structure_id ?? activeStructureId ?? snapshot?.structure.structure_id ?? "—";
  const residues = view.embeddings?.residues ?? [];
  const graphRows = view.graphMetrics?.metrics ?? [];
  const leaks = view.sourceLeaks?.leaks ?? [];
  const sites = view.allostericSites?.sites ?? [];
  const bindingSites = view.bindingScan?.sites ?? [];
  const bridges = graphRows.filter((r) => r.is_bridge);
  const meanDegree =
    graphRows.length > 0
      ? graphRows.reduce((s, r) => s + r.degree, 0) / graphRows.length
      : 0;
  const lambda2 = formatLambda2(view.resistanceData?.spectral?.lambda_2);

  const contextSummary = hydration?.context_summary as
    | {
        hypothesis_count?: number;
        source_leak_count?: number;
        residue_count?: number;
        latest_run_ids_by_pipeline?: Record<string, string>;
        top_uncertainty_residues?: Array<{
          residue_id: string;
          epistemic_uncertainty?: number;
        }>;
      }
    | undefined;

  const sections: BriefingSection[] = [];

  if (availability.gnn_hyp?.present) {
    const contextTop =
      contextSummary?.top_uncertainty_residues?.map((r) => ({
        residue_id: r.residue_id,
        label: formatResidueLabel(r.residue_id),
        value: r.epistemic_uncertainty ?? 0,
        valueLabel: "e σ",
      })) ?? [];
    sections.push({
      artifactKey: "gnn_hyp",
      title: "Hyperbolic embeddings",
      present: true,
      reason: null,
      stats: [
        { label: "Residues", value: String(residues.length) },
        { label: "Curvature", value: String(view.embeddings?.curvature ?? "—") },
        {
          label: "Mean e σ",
          value:
            residues.length > 0
              ? (
                  residues.reduce((s, r) => s + r.epistemic_uncertainty, 0) /
                  residues.length
                ).toFixed(3)
              : "—",
        },
      ],
      rows: contextTop.length > 0 ? contextTop : topResidueRows(residues, "epistemic_uncertainty", "e σ"),
      footnote: "Values from governed fact_gnn_node_embedding / structure_snapshot.",
    });
  } else {
    sections.push({
      artifactKey: "gnn_hyp",
      title: "Hyperbolic embeddings",
      present: false,
      reason: availability.gnn_hyp?.reason ?? "absent",
      stats: [],
      rows: [],
      footnote: null,
    });
  }

  if (availability.graph?.present) {
    sections.push({
      artifactKey: "graph",
      title: "Graph topology",
      present: true,
      reason: null,
      stats: [
        { label: "Nodes", value: String(graphRows.length) },
        { label: "Mean degree", value: meanDegree.toFixed(1) },
        { label: "Bridges", value: String(bridges.length) },
      ],
      rows: [...graphRows]
        .sort((a, b) => b.betweenness - a.betweenness)
        .slice(0, 5)
        .map((r) => ({
          residue_id: r.residue_id,
          label: formatResidueLabel(r.residue_id),
          value: r.betweenness,
          valueLabel: "betweenness",
        })),
      footnote: null,
    });
  } else {
    sections.push({
      artifactKey: "graph",
      title: "Graph topology",
      present: false,
      reason: availability.graph?.reason ?? "absent",
      stats: [],
      rows: [],
      footnote: null,
    });
  }

  if (availability.source_leaks?.present) {
    sections.push({
      artifactKey: "source_leaks",
      title: "Source leaks",
      present: true,
      reason: null,
      stats: [
        {
          label: "Leak count",
          value: String(leaks.length || contextSummary?.source_leak_count || 0),
        },
      ],
      rows: [...leaks]
        .sort((a, b) => b.leak_score - a.leak_score)
        .slice(0, 5)
        .map((l) => ({
          residue_id: l.residue_id,
          label: formatResidueLabel(l.residue_id),
          value: l.leak_score,
          valueLabel: "leak score",
        })),
      footnote: null,
    });
  } else {
    sections.push({
      artifactKey: "source_leaks",
      title: "Source leaks",
      present: false,
      reason: availability.source_leaks?.reason ?? "absent",
      stats: [],
      rows: [],
      footnote: null,
    });
  }

  if (availability.resistance_pathway?.present || lambda2.display !== "n/a") {
    sections.push({
      artifactKey: "resistance_pathway",
      title: "Resistance spectral",
      present: true,
      reason: null,
      stats: [{ label: "λ₂", value: lambda2.display }],
      rows: [],
      footnote: lambda2.note,
    });
  }

  if (availability.allosteric_sites?.present) {
    sections.push({
      artifactKey: "allosteric_sites",
      title: "Allosteric sites",
      present: true,
      reason: null,
      stats: [{ label: "Sites", value: String(sites.length) }],
      rows: [],
      footnote: null,
    });
  }

  if (availability.binding_scan?.present) {
    sections.push({
      artifactKey: "binding_scan",
      title: "Binding scan",
      present: true,
      reason: null,
      stats: [
        { label: "Sites", value: String(bindingSites.length) },
        { label: "Status", value: view.bindingScan?.status ?? "—" },
      ],
      rows: [...bindingSites]
        .sort((a, b) => (b.druggability_score ?? 0) - (a.druggability_score ?? 0))
        .slice(0, 5)
        .map((site) => ({
          residue_id: site.site_id,
          label: site.site_type,
          value: site.druggability_score ?? 0,
          valueLabel: "druggability",
        })),
      footnote: "Ranked pockets from governed fact_cryptic_site / fact_binding_site_scan.",
    });
  } else {
    sections.push({
      artifactKey: "binding_scan",
      title: "Binding scan",
      present: false,
      reason: availability.binding_scan?.reason ?? "absent",
      stats: [],
      rows: [],
      footnote: null,
    });
  }

  const persistence = view.persistenceStatus;
  const persistenceRows = [
    { label: "Embeddings", persisted: Boolean(persistence?.embeddings_persisted) },
    { label: "Graph", persisted: Boolean(persistence?.graph_persisted) },
    { label: "Sites", persisted: Boolean(persistence?.sites_persisted) },
    { label: "Phase 4", persisted: Boolean(persistence?.phase4_persisted) },
    { label: "Phase 5", persisted: Boolean(persistence?.phase5_persisted) },
    { label: "Phase 6", persisted: Boolean(persistence?.phase6_persisted) },
  ];

  return {
    structureId,
    pdbId: snapshot?.structure.pdb_id ?? null,
    title: snapshot?.structure.title ?? null,
    contractVersion: meta.contract_version,
    degraded: meta.degraded,
    missingTier1: meta.missing_keys,
    signalsInferred: signals?.signalsInferred ?? false,
    isHydrating,
    artifacts: artifactRows(availability),
    sections,
    persistence: persistenceRows,
    runIds: contextSummary?.latest_run_ids_by_pipeline ?? {},
    hypothesisCount: contextSummary?.hypothesis_count ?? 0,
  };
}
