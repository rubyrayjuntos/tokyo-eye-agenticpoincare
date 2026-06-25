import { useMemo } from "react";

import { useHydration } from "../../context/HydrationProvider";
import { useDashboard } from "../../lib/context";
import type { BottomPanelId } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

interface PrototypeSummaryPanelProps {
  manifest: PanelManifest<BottomPanelId>;
}

function average(values: number[]) {
  return values.length > 0
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

export default function PrototypeSummaryPanel({
  manifest,
}: PrototypeSummaryPanelProps) {
  const { activeStructure, discoveryContext, hypothesisContext } = useDashboard();
  const { embeddings, sourceLeaks, graphMetrics, persistenceStatus } = useHydration();
  const panelBroker = usePanelBroker();

  const summary = useMemo(() => {
    const residues = embeddings?.residues ?? [];
    const meanConeDepth = average(residues.map((residue) => residue.cone_depth));
    const meanEpistemic = average(residues.map((residue) => residue.epistemic_uncertainty));
    const graphRows = graphMetrics?.metrics ?? [];
    const bridgeCount = graphRows.filter((row) => row.is_bridge).length;

    return {
      residueCount: residues.length,
      sourceLeakCount: sourceLeaks?.leaks?.length ?? 0,
      bridgeCount,
      meanConeDepth,
      meanEpistemic,
    };
  }, [embeddings?.residues, graphMetrics?.metrics, sourceLeaks?.leaks?.length]);

  return (
    <div className="grid h-full min-h-0 gap-3 p-3 lg:grid-cols-[minmax(280px,360px)_1fr]">
      <section className="rounded-md border border-slate bg-bg-elevated/70 p-4">
        <div className="text-[9px] font-semibold uppercase tracking-[0.2em] text-text-secondary">
          Structure Identity
        </div>
        <div className="mt-3 font-serif text-2xl text-text-primary">
          {activeStructure?.pdb_id?.toUpperCase() ?? "No structure"}
        </div>
        <div className="mt-1 text-sm text-text-secondary">
          {activeStructure?.title ?? "Load a structure to hydrate the prototype panels."}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
          <div className="rounded-md border border-slate bg-bg px-3 py-2">
            <div className="text-[9px] uppercase tracking-[0.14em] text-text-secondary">Phase</div>
            <div className="mt-1 font-mono text-teal">{discoveryContext?.phase ?? "residue"}</div>
          </div>
          <div className="rounded-md border border-slate bg-bg px-3 py-2">
            <div className="text-[9px] uppercase tracking-[0.14em] text-text-secondary">Hypothesis</div>
            <div className="mt-1 font-mono text-magenta">{hypothesisContext?.stateLabel ?? "emergent"}</div>
          </div>
          <div className="rounded-md border border-slate bg-bg px-3 py-2">
            <div className="text-[9px] uppercase tracking-[0.14em] text-text-secondary">Method</div>
            <div className="mt-1 font-mono text-text-primary">{activeStructure?.method ?? "—"}</div>
          </div>
          <div className="rounded-md border border-slate bg-bg px-3 py-2">
            <div className="text-[9px] uppercase tracking-[0.14em] text-text-secondary">Resolution</div>
            <div className="mt-1 font-mono text-text-primary">
              {activeStructure?.resolution ? `${activeStructure.resolution.toFixed(2)} Å` : "—"}
            </div>
          </div>
        </div>
      </section>

      <section className="flex min-h-0 flex-col rounded-md border border-slate bg-bg-elevated/50 p-4">
        <div className="flex items-center justify-between">
          <div className="text-[9px] font-semibold uppercase tracking-[0.2em] text-text-secondary">
            Prototype Summary Tiles
          </div>
          <button
            onClick={() =>
              panelBroker.emitPort(manifest, "open-bottom-panel", {
                panelId: "results",
                open: true,
              })
            }
            className="rounded border border-slate px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-text-secondary transition-colors hover:border-teal-dim/40 hover:text-text-primary"
          >
            Open Ledger
          </button>
        </div>
        <div className="mt-4 grid flex-1 grid-cols-2 gap-3 xl:grid-cols-4">
          {[
            { label: "Residues", value: String(summary.residueCount), tone: "text-teal" },
            { label: "Leak signals", value: String(summary.sourceLeakCount), tone: "text-magenta" },
            { label: "Bridge hubs", value: String(summary.bridgeCount), tone: "text-warning" },
            { label: "Mean cone depth", value: summary.meanConeDepth.toFixed(3), tone: "text-success" },
            { label: "Mean epistemic", value: summary.meanEpistemic.toFixed(3), tone: "text-warning" },
            { label: "Pipeline", value: persistenceStatus?.embeddings_persisted ? "Complete" : "Hydrating", tone: "text-teal" },
            { label: "Chains", value: activeStructure?.chains?.join(", ") ?? "—", tone: "text-text-primary" },
            { label: "Last run", value: activeStructure?.last_run_id ? activeStructure.last_run_id.slice(0, 8) : "—", tone: "text-text-primary" },
          ].map((tile) => (
            <div key={tile.label} className="rounded-md border border-slate bg-bg px-3 py-3">
              <div className="text-[9px] uppercase tracking-[0.14em] text-text-secondary">{tile.label}</div>
              <div className={`mt-2 font-mono text-base ${tile.tone}`}>{tile.value}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
