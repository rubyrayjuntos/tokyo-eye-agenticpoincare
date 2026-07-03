import { useMemo } from "react";

import { discoveryPhaseActLabel } from "../../lib/discoveryActLabels";
import { useHydration } from "../../context/HydrationProvider";
import { useDashboard } from "../../lib/context";
import {
  formatLambda2,
  proteinStateLabel,
  summarizeUncertainty,
} from "../../lib/briefingMetrics";
import { getArtifactSurfaceState } from "../../lib/artifactAvailability";
import type { ResidueEmbedding, ToolPanelId } from "../../lib/types";
import { formatResidueLabel, formatResiduePosition } from "../../lib/residueLabels";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";
import ArtifactAvailabilityChip from "../ArtifactAvailabilityChip";

interface PrototypeBriefingPanelProps {
  manifest: PanelManifest<ToolPanelId>;
}

function mean(values: number[]) {
  return values.length > 0
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

/** @deprecated Use StructureBriefingPanel — KRAS prototype narrative; kept for reference only. */
export default function PrototypeBriefingPanel({
  manifest,
}: PrototypeBriefingPanelProps) {
  const {
    embeddings,
    graphMetrics,
    resistanceData,
    sourceLeaks,
    persistenceStatus,
    hypotheses,
    hydration,
    artifactAvailability,
    hydrateMeta,
    hydrateSignalsInferred,
    isHydrating,
  } = useHydration();
  const { discoveryContext, activeStructure } = useDashboard();
  const panelBroker = usePanelBroker();

  const briefing = useMemo(() => {
    const residues = embeddings?.residues ?? [];
    const graphRows = graphMetrics?.metrics ?? [];
    const resistanceRows = resistanceData?.residues ?? [];
    const lambda2Info = formatLambda2(resistanceData?.spectral?.lambda_2);
    const hingeCount = resistanceRows.filter((residue) => residue.is_hinge).length;
    const bridgeRows = graphRows.filter((row) => row.is_bridge);
    const meanDegree = mean(graphRows.map((row) => row.degree));
    const uncertainty = summarizeUncertainty(residues);
    const proteinState = proteinStateLabel(uncertainty);
    const topSpikes = [...residues]
      .sort((a, b) => b.epistemic_uncertainty - a.epistemic_uncertainty)
      .slice(0, 5);
    const topLeak = [...(sourceLeaks?.leaks ?? [])]
      .sort((a, b) => b.leak_score - a.leak_score)[0];
    const topCentrality = [...graphRows]
      .sort((a, b) => b.betweenness - a.betweenness)
      .slice(0, 6);
    const hypothesisCount = hypotheses?.length ?? 0;
    const activePhase = discoveryContext?.phase ?? "structure";
    const actLabel = discoveryPhaseActLabel(activePhase);

    const stateDesc =
      uncertainty.spikeCount > 0
        ? `${actLabel}: top uncertainty at ${uncertainty.spikeResidueIds
            .slice(0, 3)
            .map((id) => formatResidueLabel(id))
            .join(", ")}.`
        : `${actLabel}: no standout uncertainty peaks for this structure.`;

    const coneClusterLabel = topLeak
      ? `Top leak ${formatResidueLabel(topLeak.residue_id)}`
      : "No source-leak signals";

    const contextSummary = hydration?.context_summary as
      | {
          latest_run_ids_by_pipeline?: Record<string, string>;
          residue_count?: number;
          source_leak_count?: number;
        }
      | undefined;

    return {
      proteinState,
      stateDesc,
      uncertainty,
      mechMetrics: [
        { label: "Residues", value: String(residues.length), color: "text-teal" },
        { label: "Hinges", value: String(hingeCount), color: "text-magenta" },
        { label: "Bridges", value: String(bridgeRows.length), color: "text-warning" },
        { label: "Mean conn.", value: meanDegree.toFixed(1), color: "text-teal" },
      ],
      uncertMetrics: [
        {
          label: "Mean e σ",
          value: uncertainty.mean.toFixed(3),
          color: "text-warning",
        },
        {
          label: "Median e σ",
          value: uncertainty.median.toFixed(3),
          color: "text-text-secondary",
        },
        { label: "Top leak", value: coneClusterLabel, color: "text-magenta" },
        {
          label: "Leak count",
          value: String(sourceLeaks?.leaks?.length ?? contextSummary?.source_leak_count ?? 0),
          color: "text-magenta",
        },
      ],
      topoMetrics: [
        {
          label: "Contacts",
          value: String(Math.round((meanDegree * residues.length) / 2)),
          color: "text-teal",
        },
        { label: "λ₂", value: lambda2Info.display, color: "text-success" },
      ],
      lambda2Note: lambda2Info.note,
      topSpikes,
      topCentrality,
      mechInterp:
        bridgeRows.length > 0
          ? `${bridgeRows.length} bridge residues in the Cα graph.`
          : "No bridge residues flagged in graph metrics.",
      uncertInterp: uncertainty.hasLocalizedSpikes
        ? `${uncertainty.spikeCount} residues exceed the structure 90th-percentile epistemic uncertainty (median ${uncertainty.median.toFixed(3)}).`
        : uncertainty.spikeCount > 0
          ? "Uncertainty is elevated broadly — not a tight localized cluster."
          : "Epistemic uncertainty is flat across residues in the current embedding run.",
      topoInterp:
        topCentrality.length > 0
          ? `${formatResidueLabel(topCentrality[0].residue_id)} has the highest betweenness in the Cα graph.`
          : "Topology hubs appear after graph metrics hydrate.",
      hypothesisCount,
      pipelineReady: Boolean(persistenceStatus?.embeddings_persisted),
      runIds: contextSummary?.latest_run_ids_by_pipeline ?? {},
    };
  }, [
    discoveryContext?.phase,
    embeddings?.residues,
    graphMetrics?.metrics,
    hypotheses?.length,
    hydration?.context_summary,
    persistenceStatus?.embeddings_persisted,
    resistanceData?.residues,
    resistanceData?.spectral?.lambda_2,
    sourceLeaks?.leaks,
  ]);

  const handleResidueSelect = (residueId: string) => {
    panelBroker.emitPort(manifest, "selection", { residueIds: [residueId] });
  };

  const gnnState = getArtifactSurfaceState(artifactAvailability, "gnn_hyp");
  const graphState = getArtifactSurfaceState(artifactAvailability, "graph");
  const sourceLeakState = getArtifactSurfaceState(artifactAvailability, "source_leaks");
  const hasGovernedData = gnnState === "present" && residuesCount(embeddings) > 0;
  const residues = embeddings?.residues ?? [];

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0b0d14] text-text-primary">
      <div className="border-b border-slate px-4 py-3">
        <div className="text-[9px] font-semibold uppercase tracking-[0.22em] text-text-secondary">
          Briefing · Discovery Story
        </div>
        <div className="mt-2 space-y-2">
          <div
            className={`rounded-md border px-2.5 py-2 text-[10px] leading-5 ${
              hasGovernedData
                ? "border-teal-dim/40 bg-teal-dim/5 text-teal"
                : "border-warning/40 bg-warning/5 text-warning"
            }`}
          >
            <span className="font-semibold uppercase tracking-[0.12em]">
              {hasGovernedData ? "Governed data" : "Incomplete bundle"}
            </span>
            {" · "}
            {activeStructure?.structure_id ?? "No structure"} · PDB{" "}
            {activeStructure?.pdb_id?.toUpperCase() ?? "—"}
            {hydrateSignalsInferred ? (
              <span className="text-text-secondary">
                {" "}
                · availability inferred client-side (rebuild coordinator for v1.3 signals)
              </span>
            ) : hydrateMeta?.contract_version ? (
              <span className="text-text-secondary">
                {" "}
                · contract {hydrateMeta.contract_version}
              </span>
            ) : null}
          </div>
          {hydrateMeta?.degraded ? (
            <div className="rounded-md border border-warning/40 bg-warning/5 px-2.5 py-2 text-[10px] leading-5 text-warning">
              Tier-1 gaps: {hydrateMeta.missing_keys.join(", ")}
            </div>
          ) : null}
          <p className="text-[10px] leading-5 text-text-secondary">
            Metrics below are computed from persisted pipeline outputs. Narrative labels are
            structure-relative heuristics — not fold-specific annotations (no Switch-I/II mapping).
          </p>
        </div>
      </div>
      <div className="ck-scroll flex-1 overflow-y-auto px-4 py-4">
        {isHydrating && gnnState !== "present" ? (
          <div className="mb-4 rounded-md border border-slate bg-bg-elevated/40 px-3 py-2 text-xs text-text-secondary">
            Hydrating structure bundle…
          </div>
        ) : null}
        {gnnState === "tier1_empty" ? (
          <section className="mb-5 rounded-md border border-warning/40 bg-warning/5 p-3">
            <div className="text-[9px] font-semibold uppercase tracking-[0.2em] text-warning">
              Embeddings unavailable
            </div>
            <p className="mt-2 text-xs leading-5 text-text-secondary">
              Tier-1 GNN embeddings are not in the hydrate bundle yet. Re-run ingest or wait for
              the onboard pathway to finish.
            </p>
            <div className="mt-3">
              <ArtifactAvailabilityChip
                artifactKey="gnn_hyp"
                availability={artifactAvailability}
                label="GNN"
              />
            </div>
          </section>
        ) : null}

        <section className="mb-5 rounded-md border border-magenta-dim/35 bg-gradient-to-b from-[#16101f] to-[#10131c] p-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <div className="text-[9px] font-semibold uppercase tracking-[0.2em] text-text-secondary">
              Protein State
            </div>
            <ArtifactAvailabilityChip artifactKey="gnn_hyp" availability={artifactAvailability} label="GNN" />
          </div>
          <div className="font-serif text-xl text-text-primary">{briefing.proteinState.title}</div>
          <p className="mt-2 text-xs leading-5 text-text-secondary">
            {briefing.proteinState.description}
          </p>
          <p className="mt-2 text-xs leading-5 text-text-secondary">{briefing.stateDesc}</p>
        </section>

        <section className="mb-5">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
              Mechanical Architecture
            </div>
            <ArtifactAvailabilityChip artifactKey="graph" availability={artifactAvailability} />
          </div>
          {graphState === "tier1_empty" ? (
            <p className="text-xs leading-5 text-text-secondary">
              Graph metrics are not available in the hydrate bundle.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2">
                {briefing.mechMetrics.map((metric) => (
                  <div key={metric.label} className="rounded-md border border-slate bg-bg-elevated/70 p-3">
                    <div className="text-[9px] font-semibold uppercase tracking-[0.1em] text-text-secondary">
                      {metric.label}
                    </div>
                    <div className={`mt-2 font-mono text-lg ${metric.color}`}>{metric.value}</div>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs leading-5 text-text-secondary">{briefing.mechInterp}</p>
            </>
          )}
        </section>

        <section className="mb-5">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
              Uncertainty &amp; Pocket Signals
            </div>
            <ArtifactAvailabilityChip artifactKey="gnn_hyp" availability={artifactAvailability} label="GNN" />
            <ArtifactAvailabilityChip artifactKey="source_leaks" availability={artifactAvailability} />
          </div>
          {sourceLeakState === "tier1_empty" ? (
            <p className="text-xs leading-5 text-text-secondary">
              Source-leak signals are not in the hydrate bundle yet.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2">
                {briefing.uncertMetrics.map((metric) => (
                  <div key={metric.label} className="rounded-md border border-slate bg-bg-elevated/70 p-3">
                    <div className="text-[9px] font-semibold uppercase tracking-[0.1em] text-text-secondary">
                      {metric.label}
                    </div>
                    <div className={`mt-2 font-mono text-sm ${metric.color}`}>{metric.value}</div>
                  </div>
                ))}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {briefing.topSpikes.map((residue) => (
                  <button
                    key={residue.residue_id}
                    onClick={() => handleResidueSelect(residue.residue_id)}
                    className="rounded-full border border-magenta-dim/40 px-2.5 py-1 font-mono text-[10px] text-magenta transition-colors hover:bg-magenta-dim/10"
                  >
                    {formatResidueLabel(residue.residue_id)} · {residue.epistemic_uncertainty.toFixed(3)}
                  </button>
                ))}
              </div>
              <p className="mt-3 text-xs leading-5 text-text-secondary">{briefing.uncertInterp}</p>
            </>
          )}
        </section>

        <section>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
              Allosteric Topology
            </div>
            <ArtifactAvailabilityChip artifactKey="graph" availability={artifactAvailability} label="Graph" />
          </div>
          <div className="mb-3 grid grid-cols-2 gap-2">
            {briefing.topoMetrics.map((metric) => (
              <div key={metric.label} className="rounded-md border border-slate bg-bg-elevated/70 p-3">
                <div className="text-[9px] font-semibold uppercase tracking-[0.1em] text-text-secondary">
                  {metric.label}
                </div>
                <div className={`mt-2 font-mono text-lg ${metric.color}`}>{metric.value}</div>
              </div>
            ))}
          </div>
          {briefing.lambda2Note ? (
            <p className="mb-3 text-[10px] leading-5 text-text-secondary">{briefing.lambda2Note}</p>
          ) : null}
          <div className="space-y-2">
            {briefing.topCentrality.map((row) => (
              <button
                key={row.residue_id}
                onClick={() => handleResidueSelect(row.residue_id)}
                className="flex w-full items-center justify-between rounded-md border border-slate bg-bg-elevated/40 px-3 py-2 text-left transition-colors hover:border-teal-dim/40 hover:bg-bg-elevated/70"
              >
                <div className="min-w-0">
                  <div className="font-mono text-xs text-teal">{row.residue_id}</div>
                  <div className="text-[10px] text-text-secondary">
                    {formatResiduePosition(
                      residues.find((residue) => residue.residue_id === row.residue_id) ??
                        ({
                          residue_id: row.residue_id,
                          residue_index: 0,
                          chain_label: "?",
                          x: 0,
                          y: 0,
                          cone_depth: 0,
                          epistemic_uncertainty: 0,
                          aleatoric_uncertainty: 0,
                        } satisfies ResidueEmbedding),
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {row.is_bridge ? (
                    <span className="rounded-full border border-warning/40 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-warning">
                      Bridge
                    </span>
                  ) : null}
                  <span className="font-mono text-xs text-text-primary">
                    {row.betweenness.toFixed(3)}
                  </span>
                </div>
              </button>
            ))}
          </div>
          <p className="mt-3 text-xs leading-5 text-text-secondary">{briefing.topoInterp}</p>
        </section>

        <div className="mt-5 rounded-md border border-slate bg-bg-elevated/40 p-3">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.16em] text-text-secondary">
            <span>Pipeline</span>
            <span>{briefing.pipelineReady ? "Embeddings persisted" : "Hydrating"}</span>
          </div>
          <div className="mt-2 space-y-1 text-xs text-text-secondary">
            <div>{briefing.hypothesisCount} hypotheses in workspace.</div>
            {Object.keys(briefing.runIds).length > 0 ? (
              <div className="font-mono text-[10px] text-text-muted">
                Runs:{" "}
                {Object.entries(briefing.runIds)
                  .map(([k, v]) => `${k}=${String(v).slice(0, 8)}`)
                  .join(" · ")}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function residuesCount(embeddings: { residues?: unknown[] } | null | undefined): number {
  return embeddings?.residues?.length ?? 0;
}
