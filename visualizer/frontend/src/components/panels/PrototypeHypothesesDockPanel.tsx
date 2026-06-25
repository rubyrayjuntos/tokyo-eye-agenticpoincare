import { useMemo } from "react";

import { useHydration } from "../../context/HydrationProvider";
import type { BottomPanelId, Hypothesis } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

function parseResidueRefs(text: string): string[] {
  const matches = text.match(/\b(\d{1,3})\b/g) ?? [];
  return matches.slice(0, 5).map((token) => `A:${token}`);
}

function confidenceBand(confidence: number) {
  if (confidence >= 0.75) return { label: "high", color: "text-success" };
  if (confidence >= 0.45) return { label: "watch", color: "text-warning" };
  return { label: "spec", color: "text-magenta" };
}

function statusTone(status: Hypothesis["status"]) {
  switch (status) {
    case "supported":
      return "text-success border-success/30";
    case "contradicted":
      return "text-error border-error/30";
    case "gathering":
      return "text-warning border-warning/30";
    default:
      return "text-text-secondary border-slate";
  }
}

export default function PrototypeHypothesesDockPanel({
  manifest,
}: {
  manifest: PanelManifest<BottomPanelId>;
}) {
  const { hypotheses } = useHydration();
  const panelBroker = usePanelBroker();

  const cards = useMemo(
    () =>
      (hypotheses ?? []).map((hypothesis) => ({
        ...hypothesis,
        residues: parseResidueRefs(
          [hypothesis.statement, hypothesis.mechanism ?? ""].join(" "),
        ),
        band: confidenceBand(hypothesis.confidence),
      })),
    [hypotheses],
  );

  if (cards.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-text-muted">
        No hypotheses are available yet for this structure.
      </div>
    );
  }

  return (
    <div className="flex min-h-0 gap-3 overflow-x-auto pb-2">
      {cards.map((hypothesis) => (
        <article
          key={hypothesis.hypothesis_id}
          className="flex min-h-[220px] min-w-[280px] max-w-[320px] flex-col rounded-md border border-slate bg-bg-elevated/60 p-3"
        >
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="font-mono text-[10px] text-text-secondary">
              {hypothesis.hypothesis_id}
            </span>
            <div className="flex items-center gap-2">
              <span
                className={`text-[8px] font-semibold uppercase tracking-[0.12em] ${hypothesis.band.color}`}
              >
                {hypothesis.band.label}
              </span>
              <span
                className={`rounded-full border px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.14em] ${statusTone(
                  hypothesis.status,
                )}`}
              >
                {hypothesis.status}
              </span>
            </div>
          </div>

          <p className="mb-3 text-sm leading-5 text-text-primary">
            {hypothesis.statement}
          </p>

          <div className="mb-3 flex flex-wrap gap-1.5">
            {hypothesis.residues.length > 0 ? (
              hypothesis.residues.map((residueId) => (
                <button
                  key={residueId}
                  onClick={() =>
                    panelBroker.emitPort(manifest, "selection", {
                      residueIds: [residueId],
                    })
                  }
                  className="rounded-full border border-teal-dim/35 px-2 py-0.5 font-mono text-[10px] text-teal transition-colors hover:bg-teal-dim/10"
                >
                  {residueId}
                </button>
              ))
            ) : (
              <span className="rounded-full border border-slate px-2 py-0.5 font-mono text-[10px] text-text-secondary">
                no residue refs
              </span>
            )}
          </div>

          <div className="mt-auto">
            <div className="mb-1 flex items-center justify-between text-[8px] uppercase tracking-[0.12em] text-text-secondary">
              <span>Confidence</span>
              <span className="font-mono text-text-primary">
                {Math.round(hypothesis.confidence * 100)}%
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate/40">
              <div
                className="h-full rounded-full bg-gradient-to-r from-magenta via-warning to-success"
                style={{ width: `${Math.max(4, hypothesis.confidence * 100)}%` }}
              />
            </div>
            <div className="mt-3 flex gap-3 font-mono text-[10px]">
              <span className="text-success">
                ▲ {hypothesis.evidence_supporting} support
              </span>
              <span className="text-magenta">
                ▼ {hypothesis.evidence_contradicting} counter
              </span>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
