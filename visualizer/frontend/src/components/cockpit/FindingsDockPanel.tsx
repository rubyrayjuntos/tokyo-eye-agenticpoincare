import { useMemo, useState } from "react";

import PrototypeFragmentsPanel from "../panels/PrototypeFragmentsPanel";
import PrototypeHypothesesDockPanel from "../panels/PrototypeHypothesesDockPanel";
import PrototypeMotifsPanel from "../panels/PrototypeMotifsPanel";
import PrototypePocketsPanel from "../panels/PrototypePocketsPanel";
import { useHydration } from "../../context/HydrationProvider";
import type { BottomPanelId } from "../../lib/types";
import type { PanelManifest } from "../workbench/panelContract";

type DockTab = "hypotheses" | "pockets" | "fragments" | "motifs";

const TAB_MANIFESTS: Record<DockTab, PanelManifest<BottomPanelId>> = {
  hypotheses: {
    id: "hypotheses",
    title: "Hypotheses",
    region: "bottom-panel",
    ports: [{ name: "selection", kind: "selection", direction: "output" }],
    emits: ["selection.changed"],
  },
  pockets: {
    id: "pockets",
    title: "Pockets",
    region: "bottom-panel",
    ports: [{ name: "directive", kind: "directive", direction: "output" }],
    emits: ["directive.emitted"],
  },
  fragments: {
    id: "fragments",
    title: "Fragments",
    region: "bottom-panel",
    ports: [{ name: "directive", kind: "directive", direction: "output" }],
    emits: ["directive.emitted"],
  },
  motifs: {
    id: "motifs",
    title: "Motifs",
    region: "bottom-panel",
    ports: [{ name: "selection", kind: "selection", direction: "output" }],
    emits: ["selection.changed"],
  },
};

const TABS: { id: DockTab; label: string }[] = [
  { id: "hypotheses", label: "Hypotheses" },
  { id: "pockets", label: "Pockets" },
  { id: "fragments", label: "Fragments" },
  { id: "motifs", label: "Motifs" },
];

export function FindingsDockPanel() {
  const [activeTab, setActiveTab] = useState<DockTab>("hypotheses");
  const { hypotheses, pharmacophorePockets } = useHydration();

  const counts = useMemo(
    () => ({
      hypotheses: hypotheses?.length ?? 0,
      pockets: pharmacophorePockets?.pockets?.length ?? 0,
      fragments: 0,
      motifs: 0,
    }),
    [hypotheses?.length, pharmacophorePockets?.pockets?.length],
  );

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0b0d14]">
      <div className="flex h-[38px] shrink-0 items-center gap-1 border-b border-slate/60 px-3">
        {TABS.map((tab) => {
          const active = activeTab === tab.id;
          const count = counts[tab.id];
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] transition-colors ${
                active
                  ? "bg-teal-dim/20 text-teal"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              {tab.label}
              {count > 0 ? (
                <span className="font-mono text-[9px] text-text-muted">{count}</span>
              ) : null}
            </button>
          );
        })}
        <span className="ml-auto font-mono text-[10px] text-text-muted">
          hypotheses · pockets · fragments · motifs
        </span>
      </div>
      <div className="ck-scroll min-h-0 flex-1 overflow-auto p-3">
        {activeTab === "hypotheses" ? (
          <PrototypeHypothesesDockPanel manifest={TAB_MANIFESTS.hypotheses} />
        ) : null}
        {activeTab === "pockets" ? (
          <PrototypePocketsPanel manifest={TAB_MANIFESTS.pockets} />
        ) : null}
        {activeTab === "fragments" ? (
          <PrototypeFragmentsPanel manifest={TAB_MANIFESTS.fragments} />
        ) : null}
        {activeTab === "motifs" ? (
          <PrototypeMotifsPanel manifest={TAB_MANIFESTS.motifs} />
        ) : null}
      </div>
    </div>
  );
}
