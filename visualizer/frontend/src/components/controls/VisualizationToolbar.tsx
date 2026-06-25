import React, { useEffect, useState } from "react";
import { Highlighter, Palette, Focus, XCircle } from "lucide-react";
import type { ViewportDirective } from "../../lib/types";
import { useDashboard } from "../../lib/context";

const COLOR_METRICS = [
  { value: "cone_depth", label: "Cone Depth" },
  { value: "epistemic", label: "Epistemic" },
  { value: "aleatoric", label: "Aleatoric" },
  { value: "total_uncertainty", label: "Total Uncertainty" },
] as const;

interface VisualizationToolbarProps {
  /** Currently selected residue IDs from any panel */
  selectedResidues: string[];
  /** Callback to emit viewport directives to the viewer */
  onDirective: (directive: ViewportDirective) => void;
}

/**
 * Toolbar for direct visualization controls: highlight, color metric, focus, clear.
 * Emits ViewportDirectives consumed by MolecularViewer and PoincareScatter.
 */
export default function VisualizationToolbar({
  selectedResidues,
  onDirective,
}: VisualizationToolbarProps) {
  const [metric, setMetric] = useState<string>("cone_depth");

  // Global radar state for persistence + active styling (hoisted in DashboardContext)
  const { isRadarActive, setIsRadarActive, currentDirective } = useDashboard();

  useEffect(() => {
    if (currentDirective?.action === "set_metric" && currentDirective.metric) {
      setMetric(currentDirective.metric);
    }
  }, [currentDirective]);

  const handleHighlight = () => {
    if (selectedResidues.length === 0) return;
    onDirective({
      action: "highlight",
      highlight_groups: [
        {
          residue_ids: selectedResidues,
          color: "#facc15",
          style: "glow",
        },
      ],
    });
  };

  const handleMetricChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newMetric = e.target.value;
    setMetric(newMetric);
    onDirective({
      action: "set_metric",
      metric: newMetric,
    });
  };

  const handleFocus = () => {
    if (selectedResidues.length === 0) return;
    onDirective({
      action: "focus",
      highlight_groups: [
        {
          residue_ids: selectedResidues,
          color: "#facc15",
          style: "glow",
        },
      ],
    });
  };

  const handleClear = () => {
    onDirective({ action: "clear" });
  };

  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-md"
      role="toolbar"
      aria-label="Visualization controls"
    >
      <button
        onClick={handleHighlight}
        disabled={selectedResidues.length === 0}
        className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Highlight selected residues"
        aria-label="Highlight selected residues"
      >
        <Highlighter className="w-3.5 h-3.5" aria-hidden="true" />
        Highlight
      </button>

      <div className="flex items-center gap-1">
        <Palette className="w-3.5 h-3.5 text-zinc-400" aria-hidden="true" />
        <select
          value={metric}
          onChange={handleMetricChange}
          className="text-xs bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
          aria-label="Color metric"
        >
          {COLOR_METRICS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={handleFocus}
        disabled={selectedResidues.length === 0}
        className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        title="Focus camera on selection"
        aria-label="Focus camera on selection"
      >
        <Focus className="w-3.5 h-3.5" aria-hidden="true" />
        Focus
      </button>

      <button
        onClick={() => {
          const next = !isRadarActive;
          setIsRadarActive(next);
          if (next) {
            // Emit the special directive so MolecularViewer applies the pseudo-bloom radar (ultramarine + amber locks + green targets)
            // The global flag ensures it persists across pocket selections / structure switches.
            onDirective({
              action: "highlight",
              highlight_groups: [{
                residue_ids: [],
                color: "#f59e0b",
                style: "glow",
                label: "Allosteric Locks — remote control via network"
              }],
            });
          } else {
            onDirective({ action: "clear" });
          }
        }}
        className={`flex items-center gap-1 px-2 py-1 text-xs rounded transition-all ${
          isRadarActive
            ? "bg-amber-600 text-amber-100 ring-1 ring-amber-400 shadow-inner"
            : "bg-amber-800 hover:bg-amber-700"
        }`}
        title={isRadarActive ? "Deactivate Allosteric Radar" : "Activate Allosteric Radar mode (dark ultramarine rest + amber glowing locks + green targets)"}
        aria-label="Toggle Allosteric Radar"
        aria-pressed={isRadarActive}
      >
        {isRadarActive ? "Radar: Active" : "Radar"}
      </button>

      <button
        onClick={handleClear}
        className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-zinc-800 hover:bg-zinc-700 transition-colors"
        title="Clear all highlights"
        aria-label="Clear all highlights"
      >
        <XCircle className="w-3.5 h-3.5" aria-hidden="true" />
        Clear
      </button>

      {selectedResidues.length > 0 && (
        <span className="text-xs text-zinc-500 ml-auto">
          {selectedResidues.length} residue{selectedResidues.length !== 1 ? "s" : ""} selected
        </span>
      )}
    </div>
  );
}
