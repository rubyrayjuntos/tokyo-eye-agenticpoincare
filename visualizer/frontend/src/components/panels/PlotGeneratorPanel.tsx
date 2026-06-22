/**
 * PlotGeneratorPanel — Generate publication-quality matplotlib figures.
 * Requirements: 6.1, 6.2, 6.3, 6.4
 */
import { useState, useCallback } from "react";
import { api } from "../../lib/api";
import { useDashboard } from "../../lib/context";
import type { PlotRequest, PlotResponse } from "../../lib/types";

type PlotType = PlotRequest["plot_type"];

const PLOT_TYPES: { value: PlotType; label: string }[] = [
  { value: "poincare_disc", label: "Poincaré Disc" },
  { value: "uncertainty_profile", label: "Uncertainty Profile" },
  { value: "cone_depth_histogram", label: "Cone Depth Histogram" },
  { value: "wt_vs_mutant", label: "WT vs Mutant" },
  { value: "persistence_barcode", label: "Persistence Barcode" },
  { value: "source_leak_map", label: "Source Leak Map" },
];

interface PlotParams {
  compare_structure_id?: string;
  chain?: string;
  color_by?: string;
  min_depth?: string;
  max_depth?: string;
}

export default function PlotGeneratorPanel() {
  const { activeStructure } = useDashboard();
  const structureId = activeStructure?.structure_id ?? null;

  const [plotType, setPlotType] = useState<PlotType>("poincare_disc");
  const [params, setParams] = useState<PlotParams>({});
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<PlotResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const updateParam = (key: keyof PlotParams, value: string) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const handleGenerate = useCallback(async () => {
    if (!structureId) return;
    setGenerating(true);
    setError(null);
    setResult(null);
    try {
      const parameters: Record<string, unknown> = {};
      if (params.compare_structure_id?.trim()) {
        parameters.compare_structure_id = params.compare_structure_id.trim();
      }
      if (params.chain?.trim()) {
        parameters.chain = params.chain.trim();
      }
      if (params.color_by?.trim()) {
        parameters.color_by = params.color_by.trim();
      }
      if (params.min_depth?.trim()) {
        parameters.min_depth = parseFloat(params.min_depth);
      }
      if (params.max_depth?.trim()) {
        parameters.max_depth = parseFloat(params.max_depth);
      }

      const req: PlotRequest = {
        structure_id: structureId,
        plot_type: plotType,
        parameters: Object.keys(parameters).length > 0 ? parameters : undefined,
      };

      const resp = await api.generatePlot(req);
      setResult(resp);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Plot generation failed";
      setError(msg);
    } finally {
      setGenerating(false);
    }
  }, [structureId, plotType, params]);

  if (!activeStructure) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-zinc-600 text-center">
          Select a structure to generate plots.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Error banner */}
      {error && (
        <div className="px-3 py-2 bg-red-900/20 border-b border-red-800/40">
          <p className="text-[11px] text-red-400">{error}</p>
        </div>
      )}

      <div className="flex-1 overflow-auto divide-y divide-zinc-800">
        {/* Plot Type Selection */}
        <div className="px-3 py-3 space-y-2">
          <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">
            Generate Plot
          </p>

          <select
            value={plotType}
            onChange={(e) => {
              setPlotType(e.target.value as PlotType);
              setParams({});
              setResult(null);
              setError(null);
            }}
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-cyan-600"
            aria-label="Plot type"
          >
            {PLOT_TYPES.map((pt) => (
              <option key={pt.value} value={pt.value}>
                {pt.label}
              </option>
            ))}
          </select>

          {/* Dynamic Parameters */}
          <div className="space-y-1.5">
            {plotType === "wt_vs_mutant" && (
              <input
                type="text"
                value={params.compare_structure_id ?? ""}
                onChange={(e) => updateParam("compare_structure_id", e.target.value)}
                placeholder="Compare structure ID (required)"
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
                aria-label="Compare structure ID"
              />
            )}

            {(plotType === "poincare_disc" ||
              plotType === "uncertainty_profile" ||
              plotType === "source_leak_map") && (
              <input
                type="text"
                value={params.chain ?? ""}
                onChange={(e) => updateParam("chain", e.target.value)}
                placeholder="Chain filter (optional, e.g. A)"
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
                aria-label="Chain filter"
              />
            )}

            {plotType === "poincare_disc" && (
              <select
                value={params.color_by ?? "cone_depth"}
                onChange={(e) => updateParam("color_by", e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-cyan-600"
                aria-label="Color by metric"
              >
                <option value="cone_depth">Cone Depth</option>
                <option value="epistemic">Epistemic Uncertainty</option>
                <option value="aleatoric">Aleatoric Uncertainty</option>
                <option value="total">Total Uncertainty</option>
              </select>
            )}

            {plotType === "cone_depth_histogram" && (
              <div className="grid grid-cols-2 gap-1.5">
                <input
                  type="number"
                  value={params.min_depth ?? ""}
                  onChange={(e) => updateParam("min_depth", e.target.value)}
                  placeholder="Min depth"
                  step="0.01"
                  className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
                  aria-label="Minimum depth"
                />
                <input
                  type="number"
                  value={params.max_depth ?? ""}
                  onChange={(e) => updateParam("max_depth", e.target.value)}
                  placeholder="Max depth"
                  step="0.01"
                  className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
                  aria-label="Maximum depth"
                />
              </div>
            )}
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={generating || (plotType === "wt_vs_mutant" && !params.compare_structure_id?.trim())}
            className="w-full px-2 py-1.5 bg-cyan-700/30 border border-cyan-600/40 rounded text-[11px] font-medium text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-40"
          >
            {generating ? "Generating..." : "Generate Plot"}
          </button>
        </div>

        {/* Result Display */}
        {result && (
          <div className="px-3 py-3 space-y-2">
            <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">
              Result
            </p>

            {result.success && result.download_url ? (
              <div className="space-y-2">
                <div className="border border-zinc-700 rounded overflow-hidden bg-zinc-900">
                  <img
                    src={result.download_url}
                    alt={`${result.plot_type} plot`}
                    className="w-full h-auto"
                  />
                </div>
                <a
                  href={result.download_url}
                  download
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 px-2 py-1 bg-emerald-700/30 border border-emerald-600/40 rounded text-[10px] text-emerald-300 hover:bg-emerald-700/50"
                >
                  ↓ Download PNG
                </a>
              </div>
            ) : (
              <p className="text-[11px] text-zinc-400">{result.message}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
