/**
 * ProvenanceTree — Run history with parent/child relationships, summaries, and comparison.
 * Requirements: 5.6, 5.7, 5.8
 */
import { useState, useCallback } from "react";
import { api } from "../../lib/api";
import { useDashboard } from "../../lib/context";
import { useHydration } from "../../context/HydrationProvider";
import type { RunSummary, RunCompareResult } from "../../lib/types";

export default function ProvenanceTree() {
  const { activeStructure } = useDashboard();
  const { provenanceRuns, loading } = useHydration();

  // Expanded run summaries
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [runSummary, setRunSummary] = useState<RunSummary | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(false);

  // Compare state
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [compareResult, setCompareResult] = useState<RunCompareResult | null>(null);
  const [comparing, setComparing] = useState(false);

  const [error, setError] = useState<string | null>(null);

  const handleExpandRun = useCallback(async (runId: string) => {
    if (expandedRun === runId) {
      setExpandedRun(null);
      setRunSummary(null);
      return;
    }
    setExpandedRun(runId);
    setLoadingSummary(true);
    setError(null);
    try {
      const summary = await api.getRunSummary(runId);
      setRunSummary(summary);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Failed to load run summary";
      setError(msg);
      setRunSummary(null);
    } finally {
      setLoadingSummary(false);
    }
  }, [expandedRun]);

  const toggleRunSelection = (runId: string) => {
    setSelectedRuns((prev) => {
      if (prev.includes(runId)) {
        return prev.filter((id) => id !== runId);
      }
      if (prev.length >= 2) {
        return [prev[1], runId];
      }
      return [...prev, runId];
    });
    setCompareResult(null);
  };

  const handleCompare = useCallback(async () => {
    if (selectedRuns.length !== 2) return;
    setComparing(true);
    setError(null);
    try {
      const result = await api.compareRuns(selectedRuns[0], selectedRuns[1]);
      setCompareResult(result);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Comparison failed";
      setError(msg);
    } finally {
      setComparing(false);
    }
  }, [selectedRuns]);

  if (!activeStructure) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-zinc-600 text-center">Select a structure to view provenance.</p>
      </div>
    );
  }

  if (loading.provenance_runs) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-xs text-zinc-500 animate-pulse">Loading provenance...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {error && (
        <div className="px-3 py-2 bg-red-900/20 border-b border-red-800/40">
          <p className="text-[11px] text-red-400">{error}</p>
        </div>
      )}

      {/* Header with compare button */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">
          Run History ({provenanceRuns?.length ?? 0})
        </p>
        <button
          onClick={handleCompare}
          disabled={selectedRuns.length !== 2 || comparing}
          className="px-2 py-1 bg-cyan-700/30 border border-cyan-600/40 rounded text-[10px] font-medium text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-40"
        >
          {comparing ? "Comparing..." : `Compare (${selectedRuns.length}/2)`}
        </button>
      </div>

      <div className="flex-1 overflow-auto">
        {!provenanceRuns || provenanceRuns.length === 0 ? (
          <div className="flex items-center justify-center h-full px-4">
            <p className="text-xs text-zinc-600 text-center">No pipeline runs recorded yet.</p>
          </div>
        ) : (
          <div className="divide-y divide-zinc-800">
            {provenanceRuns.map((run) => (
              <div key={run.run_id} className="px-3 py-2 space-y-1.5">
                <div className="flex items-center gap-2">
                  {/* Checkbox for compare selection */}
                  <input
                    type="checkbox"
                    checked={selectedRuns.includes(run.run_id)}
                    onChange={() => toggleRunSelection(run.run_id)}
                    className="w-3 h-3 rounded border-zinc-600 bg-zinc-800 accent-cyan-500"
                  />
                  {/* Run info */}
                  <button
                    onClick={() => handleExpandRun(run.run_id)}
                    className="flex-1 text-left"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-zinc-200 font-medium">
                        {run.pipeline_name}
                      </span>
                      <span className="text-[9px] text-zinc-500">
                        {new Date(run.started_at).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[9px] text-zinc-500">v{run.model_version}</span>
                      <span className="text-[9px] text-zinc-600">|</span>
                      <span className="text-[9px] text-zinc-500">{run.run_type}</span>
                      {run.parent_run_id && (
                        <>
                          <span className="text-[9px] text-zinc-600">|</span>
                          <span className="text-[9px] text-zinc-500">
                            parent: {run.parent_run_id.slice(0, 8)}…
                          </span>
                        </>
                      )}
                      {run.asset_count !== undefined && (
                        <>
                          <span className="text-[9px] text-zinc-600">|</span>
                          <span className="text-[9px] text-zinc-500">
                            {run.asset_count} assets
                          </span>
                        </>
                      )}
                    </div>
                  </button>
                </div>

                {/* Expanded run summary */}
                {expandedRun === run.run_id && (
                  <div className="ml-5 p-2 bg-zinc-800/50 rounded border border-zinc-700 space-y-1">
                    {loadingSummary ? (
                      <p className="text-[10px] text-zinc-500 animate-pulse">Loading summary...</p>
                    ) : runSummary ? (
                      <>
                        <div className="flex items-center gap-3 text-[10px]">
                          <span className="text-zinc-400">
                            Duration: {runSummary.duration_seconds.toFixed(1)}s
                          </span>
                          <span className="text-zinc-400">
                            Assets: {runSummary.assets_created}
                          </span>
                        </div>
                        <div className="text-[10px] text-zinc-400">
                          Phases: {runSummary.phases.join(" → ")}
                        </div>
                        {runSummary.errors.length > 0 && (
                          <div className="text-[10px] text-red-400">
                            Errors: {runSummary.errors.join(", ")}
                          </div>
                        )}
                      </>
                    ) : (
                      <p className="text-[10px] text-zinc-600">No summary available.</p>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Compare Results */}
        {compareResult && (
          <div className="px-3 py-3 border-t border-zinc-700 space-y-2">
            <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">
              Comparison: {compareResult.run_a.slice(0, 8)} vs {compareResult.run_b.slice(0, 8)}
            </p>
            {compareResult.residue_deltas.length === 0 ? (
              <p className="text-[10px] text-zinc-600">No differences found.</p>
            ) : (
              <div className="max-h-32 overflow-auto border border-zinc-700 rounded">
                <table className="w-full text-[10px]">
                  <thead className="bg-zinc-800 sticky top-0">
                    <tr>
                      <th className="px-1.5 py-1 text-left text-zinc-500">Residue</th>
                      <th className="px-1.5 py-1 text-right text-zinc-500">Δ Depth</th>
                      <th className="px-1.5 py-1 text-right text-zinc-500">Δ Uncertainty</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800">
                    {compareResult.residue_deltas.slice(0, 50).map((d) => (
                      <tr key={d.residue_id} className="hover:bg-zinc-800/50">
                        <td className="px-1.5 py-0.5 text-zinc-300">{d.residue_id}</td>
                        <td className={`px-1.5 py-0.5 text-right ${d.cone_depth_delta > 0 ? "text-emerald-400" : d.cone_depth_delta < 0 ? "text-red-400" : "text-zinc-500"}`}>
                          {d.cone_depth_delta > 0 ? "+" : ""}{d.cone_depth_delta.toFixed(4)}
                        </td>
                        <td className={`px-1.5 py-0.5 text-right ${d.uncertainty_delta > 0 ? "text-red-400" : d.uncertainty_delta < 0 ? "text-emerald-400" : "text-zinc-500"}`}>
                          {d.uncertainty_delta > 0 ? "+" : ""}{d.uncertainty_delta.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
