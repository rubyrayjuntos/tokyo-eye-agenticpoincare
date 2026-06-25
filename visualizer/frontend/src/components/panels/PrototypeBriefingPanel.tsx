import { useMemo } from "react";

import { useHydration } from "../../context/HydrationProvider";
import { useDashboard } from "../../lib/context";
import type { ResidueEmbedding, ToolPanelId } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

interface PrototypeBriefingPanelProps {
  manifest: PanelManifest<ToolPanelId>;
}

function mean(values: number[]) {
  return values.length > 0
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function residueRegionLabel(residue: ResidueEmbedding) {
  if (residue.residue_index <= 25) return "P-loop";
  if (residue.residue_index <= 50) return "Switch-I";
  if (residue.residue_index <= 90) return "Core";
  if (residue.residue_index <= 120) return "Switch-II";
  return "Beta-sheet";
}

function formatResidueLabel(residueId: string) {
  const [, index] = residueId.split(":");
  return index ? residueId.replace(":", " · ") : residueId;
}

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
  } =
    useHydration();
  const { discoveryContext } = useDashboard();
  const panelBroker = usePanelBroker();

  const briefing = useMemo(() => {
    const residues = embeddings?.residues ?? [];
    const graphRows = graphMetrics?.metrics ?? [];
    const resistanceRows = resistanceData?.residues ?? [];
    const lambda2 = resistanceData?.spectral?.lambda_2;
    const hingeCount = resistanceRows.filter((residue) => residue.is_hinge).length;
    const bridgeRows = graphRows.filter((row) => row.is_bridge);
    const meanDegree = mean(graphRows.map((row) => row.degree));
    const meanEpistemic = mean(residues.map((residue) => residue.epistemic_uncertainty));
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

    const stateClass = meanEpistemic > 0.2
      ? "Adaptive / cryptic-pocket active"
      : "Stabilized / survey-ready";
    const stateDesc = topSpikes.length > 0
      ? `Phase ${activePhase} is anchored by localized uncertainty spikes around ${topSpikes
          .slice(0, 3)
          .map((residue) => residue.residue_id)
          .join(", ")}.`
      : "No localized uncertainty spikes have been detected yet.";

    const coneClusterLabel = topLeak
      ? `Leak cluster ${formatResidueLabel(topLeak.residue_id)}`
      : "No active leak cluster";

    return {
      stateClass,
      stateDesc,
      mechMetrics: [
        { label: "Residues", value: String(residues.length), color: "text-teal" },
        { label: "Hinges", value: String(hingeCount), color: "text-magenta" },
        { label: "Bridges", value: String(bridgeRows.length), color: "text-warning" },
        { label: "Mean conn.", value: meanDegree.toFixed(1), color: "text-teal" },
      ],
      uncertMetrics: [
        { label: "Mean e sigma", value: meanEpistemic.toFixed(3), color: "text-warning" },
        { label: "Cone cluster", value: coneClusterLabel, color: "text-magenta" },
      ],
      topoMetrics: [
        { label: "Contacts", value: String(Math.round((meanDegree * residues.length) / 2)), color: "text-teal" },
        { label: "lambda2", value: lambda2?.toFixed(3) ?? "n/a", color: "text-success" },
      ],
      topSpikes,
      topCentrality,
      mechInterp:
        bridgeRows.length > 0
          ? `${bridgeRows.length} bridge residues create likely allosteric choke points.`
          : "Graph bridge residues are not available yet.",
      uncertInterp:
        topSpikes.length > 0
          ? `Global uncertainty is low, but ${topSpikes.length} localized spikes remain candidate cryptic-pocket markers.`
          : "Uncertainty remains globally flat.",
      topoInterp:
        topCentrality.length > 0
          ? `${topCentrality[0].residue_id} currently anchors the strongest betweenness signal pathway.`
          : "Topology hubs will populate after graph hydration completes.",
      hypothesisCount,
      pipelineReady: Boolean(persistenceStatus?.embeddings_persisted),
    };
  }, [discoveryContext?.phase, embeddings?.residues, graphMetrics?.metrics, hypotheses?.length, persistenceStatus?.embeddings_persisted, resistanceData?.residues, resistanceData?.spectral?.lambda_2, sourceLeaks?.leaks]);

  const handleResidueSelect = (residueId: string) => {
    panelBroker.emitPort(manifest, "selection", { residueIds: [residueId] });
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0b0d14] text-text-primary">
      <div className="border-b border-slate px-4 py-3">
        <div className="text-[9px] font-semibold uppercase tracking-[0.22em] text-text-secondary">
          Briefing
        </div>
      </div>
      <div className="ck-scroll flex-1 overflow-y-auto px-4 py-4">
        <section className="mb-5 rounded-md border border-magenta-dim/35 bg-gradient-to-b from-[#16101f] to-[#10131c] p-3">
          <div className="mb-2 text-[9px] font-semibold uppercase tracking-[0.2em] text-text-secondary">
            Protein State
          </div>
          <div className="font-serif text-xl text-text-primary">{briefing.stateClass}</div>
          <p className="mt-2 text-xs leading-5 text-text-secondary">{briefing.stateDesc}</p>
        </section>

        <section className="mb-5">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Mechanical Architecture
          </div>
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
        </section>

        <section className="mb-5">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Uncertainty &amp; Pocket Signals
          </div>
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
        </section>

        <section>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Allosteric Topology
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
                    {residueRegionLabel(
                      (embeddings?.residues ?? []).find((residue) => residue.residue_id === row.residue_id) ??
                        {
                          residue_id: row.residue_id,
                          residue_index: 0,
                          chain_label: "A",
                          x: 0,
                          y: 0,
                          cone_depth: 0,
                          epistemic_uncertainty: 0,
                          aleatoric_uncertainty: 0,
                        },
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
            <span>{briefing.pipelineReady ? "Ready" : "Running"}</span>
          </div>
          <div className="mt-2 text-xs text-text-secondary">
            {briefing.hypothesisCount} hypotheses tracked in the current workspace.
          </div>
        </div>
      </div>
    </div>
  );
}
