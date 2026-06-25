import { useMemo } from "react";

import { useHydration } from "../../context/HydrationProvider";
import type { BottomPanelId } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

export default function PrototypePocketsPanel({
  manifest,
}: {
  manifest: PanelManifest<BottomPanelId>;
}) {
  const { pharmacophorePockets } = useHydration();
  const panelBroker = usePanelBroker();

  const pockets = useMemo(
    () =>
      [...(pharmacophorePockets?.pockets ?? [])].sort(
        (a, b) => b.druggability_score - a.druggability_score,
      ),
    [pharmacophorePockets?.pockets],
  );

  if (pockets.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-text-muted">
        No pockets have been extracted yet.
      </div>
    );
  }

  return (
    <div className="flex min-h-0 gap-3 overflow-x-auto pb-2">
      {pockets.map((pocket) => (
        <article
          key={pocket.pocket_index}
          className="flex min-h-[220px] min-w-[280px] max-w-[320px] flex-col rounded-md border border-slate bg-bg-elevated/60 p-3"
        >
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="font-mono text-[10px] text-text-secondary">
              P-{pocket.pocket_index}
            </span>
            <span className="rounded-full border border-magenta-dim/35 px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.12em] text-magenta">
              cryptic
            </span>
          </div>
          <h3 className="mb-3 text-sm font-semibold text-text-primary">
            Pocket {pocket.pocket_index} · {pocket.residue_count} residues
          </h3>
          <div className="mb-2 flex items-center justify-between text-[9px] uppercase tracking-[0.1em] text-text-secondary">
            <span>W_access</span>
            <span className="font-mono text-teal">
              {pocket.druggability_score.toFixed(2)}
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-slate/40">
            <div
              className="h-full rounded-full bg-teal"
              style={{
                width: `${Math.min(100, pocket.druggability_score * 100)}%`,
              }}
            />
          </div>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {pocket.residue_ids.slice(0, 6).map((residueId) => (
              <span
                key={residueId}
                className="rounded-full border border-slate px-2 py-0.5 font-mono text-[10px] text-text-primary"
              >
                {residueId}
              </span>
            ))}
          </div>
          <p className="mt-3 text-xs leading-5 text-text-secondary">
            Volume {pocket.volume_estimate.toFixed(1)} · coupling{" "}
            {pocket.allosteric_coupling.toFixed(1)}
          </p>
          <button
            onClick={() =>
              panelBroker.emitPort(manifest, "directive", {
                directive: {
                  action: "highlight",
                  highlight_groups: [
                    {
                      residue_ids: pocket.residue_ids,
                      color: "#4ade80",
                      style: "glow",
                      label: `Pocket ${pocket.pocket_index}`,
                    },
                  ],
                },
              })
            }
            className="mt-auto rounded border border-teal-dim/35 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-teal transition-colors hover:bg-teal-dim/10"
          >
            Trace pocket
          </button>
        </article>
      ))}
    </div>
  );
}
