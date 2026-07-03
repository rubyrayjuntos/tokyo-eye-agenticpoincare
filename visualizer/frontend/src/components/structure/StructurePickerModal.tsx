import { useEffect, useState } from "react";
import { Dna, Search, X } from "lucide-react";

import { StructureOnboard } from "../onboard/StructureOnboard";
import RCSBSearchPanel from "../panels/RCSBSearchPanel";
import type { Structure } from "../../lib/types";

export type StructurePickerTab = "library" | "rcsb";

export interface StructurePickerModalProps {
  open: boolean;
  onClose: () => void;
  onStructureLoaded: (structure: Structure) => void;
  initialTab?: StructurePickerTab;
}

const TABS: { id: StructurePickerTab; label: string; icon: typeof Dna }[] = [
  { id: "library", label: "In database", icon: Dna },
  { id: "rcsb", label: "RCSB search", icon: Search },
];

export function StructurePickerModal({
  open,
  onClose,
  onStructureLoaded,
  initialTab = "rcsb",
}: StructurePickerModalProps) {
  const [tab, setTab] = useState<StructurePickerTab>(initialTab);

  useEffect(() => {
    if (open) {
      setTab(initialTab);
    }
  }, [open, initialTab]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const handleStructureLoaded = (structure: Structure) => {
    onStructureLoaded(structure);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-[120] flex items-center justify-center p-4">
      <button
        type="button"
        className="absolute inset-0 bg-[#05060a]/80 backdrop-blur-sm"
        aria-label="Close structure picker"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="structure-picker-title"
        className="relative z-10 flex h-[min(88vh,760px)] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-slate bg-bg shadow-2xl"
      >
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-slate px-4 py-3">
          <div>
            <h2
              id="structure-picker-title"
              className="text-sm font-display uppercase tracking-wider text-teal"
            >
              Load protein structure
            </h2>
            <p className="mt-0.5 text-[11px] text-text-muted">
              Pick an analyzed structure from the database or search RCSB and run the full discovery pipeline.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-[var(--radius-badge)] border border-slate-light p-1.5 text-text-muted hover:border-teal-dim hover:text-teal"
            aria-label="Close"
          >
            <X size={14} />
          </button>
        </header>

        <div className="flex shrink-0 gap-1 border-b border-slate px-4 py-2">
          {TABS.map((item) => {
            const Icon = item.icon;
            const active = tab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setTab(item.id)}
                className={`flex items-center gap-1.5 rounded-[var(--radius-badge)] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide transition-colors ${
                  active
                    ? "bg-teal-dim/20 text-teal border border-teal-dim/40"
                    : "text-text-muted hover:text-text-secondary border border-transparent"
                }`}
              >
                <Icon size={12} />
                {item.label}
              </button>
            );
          })}
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          {tab === "library" ? (
            <StructureOnboard onStructureLoaded={handleStructureLoaded} />
          ) : (
            <RCSBSearchPanel
              onStructureLoaded={handleStructureLoaded}
              className="h-full"
            />
          )}
        </div>
      </div>
    </div>
  );
}
