import { useMemo } from "react";

import { useHydration } from "../../context/HydrationProvider";
import type { BottomPanelId } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

export default function PrototypeMotifsPanel({
  manifest,
}: {
  manifest: PanelManifest<BottomPanelId>;
}) {
  const { graphMetrics } = useHydration();
  const panelBroker = usePanelBroker();

  const motifs = useMemo(() => {
    const metrics = [...(graphMetrics?.metrics ?? [])].sort(
      (a, b) => b.betweenness - a.betweenness,
    );

    return metrics.slice(0, 6).map((row, index) => ({
      id: `M-${index + 1}`,
      residueId: row.residue_id,
      score: row.betweenness,
      meta: row.is_bridge ? "bridge hub" : "co-embedding hub",
      chips: [
        `deg ${row.degree}`,
        `clust ${row.clustering_coefficient.toFixed(2)}`,
        `cond ${row.conductance.toFixed(2)}`,
      ],
      tone: row.is_bridge ? "text-warning" : "text-magenta",
    }));
  }, [graphMetrics?.metrics]);

  if (motifs.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-text-muted">
        No graph motifs are available yet.
      </div>
    );
  }

  return (
    <div className="flex min-h-0 gap-3 overflow-x-auto pb-2">
      {motifs.map((motif) => (
        <article
          key={motif.id}
          className="flex min-h-[220px] min-w-[260px] max-w-[300px] flex-col rounded-md border border-slate bg-bg-elevated/60 p-3"
        >
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className={`font-mono text-[11px] ${motif.tone}`}>{motif.id}</span>
            <span className="font-mono text-[10px] text-text-secondary">
              sigma {motif.score.toFixed(3)}
            </span>
          </div>
          <h3 className="mb-2 text-sm font-semibold text-text-primary">
            {motif.residueId}
          </h3>
          <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-slate/40">
            <div
              className={`h-full rounded-full ${motif.tone === "text-warning" ? "bg-warning" : "bg-magenta"}`}
              style={{ width: `${Math.min(100, motif.score * 100)}%` }}
            />
          </div>
          <div className="mb-3 text-xs text-text-secondary">{motif.meta}</div>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {motif.chips.map((chip) => (
              <span
                key={chip}
                className="rounded-full border border-slate px-2 py-0.5 font-mono text-[9px] text-text-primary"
              >
                {chip}
              </span>
            ))}
          </div>
          <button
            onClick={() =>
              panelBroker.emitPort(manifest, "selection", {
                residueIds: [motif.residueId],
              })
            }
            className="mt-auto rounded border border-magenta-dim/35 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-magenta transition-colors hover:bg-magenta-dim/10"
          >
            Trace motif
          </button>
        </article>
      ))}
    </div>
  );
}
