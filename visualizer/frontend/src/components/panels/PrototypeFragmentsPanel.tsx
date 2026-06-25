import { useMemo } from "react";

import { useHydration } from "../../context/HydrationProvider";
import type { BottomPanelId } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

export default function PrototypeFragmentsPanel({
  manifest,
}: {
  manifest: PanelManifest<BottomPanelId>;
}) {
  const { drugCandidates, pharmacophorePockets } = useHydration();
  const panelBroker = usePanelBroker();

  const rows = useMemo(
    () =>
      [...(drugCandidates?.candidates ?? [])].sort(
        (a, b) => b.binding_potential - a.binding_potential,
      ),
    [drugCandidates?.candidates],
  );

  if (rows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-text-muted">
        No fragment candidates are available yet.
      </div>
    );
  }

  return (
    <div className="min-w-[720px]">
      <div className="grid grid-cols-[1.2fr_0.7fr_0.8fr_0.8fr_0.9fr_1fr] gap-3 border-b border-slate px-3 py-2">
        {["Fragment", "LE", "Bind", "Access", "Select", "Verdict"].map((label) => (
          <span
            key={label}
            className="text-[8px] font-semibold uppercase tracking-[0.12em] text-text-secondary"
          >
            {label}
          </span>
        ))}
      </div>
      <div className="space-y-1 pt-1">
        {rows.map((candidate) => {
          const pocket = (pharmacophorePockets?.pockets ?? []).find(
            (entry) => entry.pocket_index === candidate.pocket_index,
          );
          const verdict = candidate.admet_pass ? "advance" : "watch";
          const verdictTone = candidate.admet_pass ? "text-success" : "text-warning";

          return (
            <button
              key={`${candidate.pocket_index}-${candidate.combined_druggability}`}
              onClick={() =>
                panelBroker.emitPort(manifest, "directive", {
                  directive: {
                    action: "highlight",
                    highlight_groups: pocket
                      ? [
                          {
                            residue_ids: pocket.residue_ids,
                            color: candidate.admet_pass ? "#facc15" : "#f87171",
                            style: "glow",
                            label: `Candidate pocket ${candidate.pocket_index}`,
                          },
                        ]
                      : [],
                  },
                })
              }
              className="grid w-full grid-cols-[1.2fr_0.7fr_0.8fr_0.8fr_0.9fr_1fr] gap-3 rounded-md border border-transparent px-3 py-2 text-left transition-colors hover:border-slate hover:bg-bg-elevated/50"
            >
              <span className="font-mono text-xs text-teal">
                FRAG-{candidate.pocket_index}
              </span>
              <span className="font-mono text-xs text-text-primary">
                {candidate.combined_druggability.toFixed(2)}
              </span>
              <span className="font-mono text-xs text-text-primary">
                {candidate.binding_potential.toFixed(2)}
              </span>
              <span className="font-mono text-xs text-text-primary">
                {candidate.accessibility_score.toFixed(2)}
              </span>
              <span className="font-mono text-xs text-text-primary">
                {candidate.selectivity_ratio.toFixed(2)}
              </span>
              <span className={`text-[10px] font-semibold uppercase tracking-[0.1em] ${verdictTone}`}>
                {verdict}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
