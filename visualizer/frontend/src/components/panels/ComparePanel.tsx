/**
 * ComparePanel — Structure comparison: displacements, graph diffs, summary.
 * Requirements: 2.1-2.4, 3.1-3.4, 5.1-5.4, 6.1-6.4
 */
import { useState, useMemo, useCallback } from "react";
import { useDashboard } from "../../lib/context";
import type { DisplacementRow, CompareGraphDiffResult } from "../../lib/types";

type Tab = "embeddings" | "graph" | "summary";
type DisplacementSortField = "residue_id" | "chain" | "index" | "primary_depth" | "secondary_depth" | "depth_delta" | "displacement";
type MetricDeltaSortField = "residue_id" | "chain" | "betweenness_delta" | "degree_delta" | "clustering_delta";
type SortDir = "asc" | "desc";

/** Export displacement data to CSV string */
export function exportDisplacementsToCSV(rows: DisplacementRow[]): string {
  const header = "residue_id,chain,index,primary_depth,secondary_depth,depth_delta,displacement";
  const lines = rows.map(
    (r) => `${r.residue_id},${r.chain},${r.index},${r.primary_depth},${r.secondary_depth},${r.depth_delta},${r.displacement}`
  );
  return [header, ...lines].join("\n");
}

export default function ComparePanel() {
  const { compareState, activeStructure, emitDirective } = useDashboard();
  const [activeTab, setActiveTab] = useState<Tab>("embeddings");

  // Embeddings tab sort state
  const [embSortField, setEmbSortField] = useState<DisplacementSortField>("displacement");
  const [embSortDir, setEmbSortDir] = useState<SortDir>("desc");

  // Graph tab sort state
  const [metricSortField, setMetricSortField] = useState<MetricDeltaSortField>("degree_delta");
  const [metricSortDir, setMetricSortDir] = useState<SortDir>("desc");

  // Summary threshold
  const [threshold, setThreshold] = useState<number>(
    compareState.displacements?.summary?.threshold ?? 0.5
  );

  // Not in compare mode — show instructions
  if (!compareState.active) {
    return (
      <div className="flex flex-col items-center justify-center h-full px-4 gap-3">
        <p className="text-xs text-zinc-500 text-center">
          No comparison active.
        </p>
        <p className="text-[11px] text-zinc-600 text-center leading-relaxed">
          To compare two structures, select a structure in the Discovery Ledger,
          then click the <span className="text-cyan-400 font-mono">⇔</span> compare button
          on a second structure.
        </p>
      </div>
    );
  }

  // Loading
  if (compareState.loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-xs text-zinc-500 animate-pulse">Loading comparison data...</p>
      </div>
    );
  }

  // Error
  if (compareState.error) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-red-400 text-center">{compareState.error}</p>
      </div>
    );
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: "embeddings", label: "Embeddings" },
    { key: "graph", label: "Graph" },
    { key: "summary", label: "Summary" },
  ];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-zinc-800">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex-1 px-2 py-1.5 text-[11px] font-medium transition-colors ${
              activeTab === tab.key
                ? "text-cyan-400 border-b-2 border-cyan-400"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto">
        {activeTab === "embeddings" && (
          <EmbeddingsTab
            displacements={compareState.displacements?.displacements ?? []}
            sortField={embSortField}
            sortDir={embSortDir}
            onSortFieldChange={setEmbSortField}
            onSortDirChange={setEmbSortDir}
            emitDirective={emitDirective}
            structureId={activeStructure?.structure_id ?? null}
          />
        )}
        {activeTab === "graph" && (
          <GraphTab
            graphDiff={compareState.graphDiff}
            sortField={metricSortField}
            sortDir={metricSortDir}
            onSortFieldChange={setMetricSortField}
            onSortDirChange={setMetricSortDir}
            emitDirective={emitDirective}
            structureId={activeStructure?.structure_id ?? null}
          />
        )}
        {activeTab === "summary" && (
          <SummaryTab
            displacements={compareState.displacements?.displacements ?? []}
            summary={compareState.displacements?.summary ?? null}
            threshold={threshold}
            onThresholdChange={setThreshold}
          />
        )}
      </div>
    </div>
  );
}


// --- Embeddings Tab ---

interface EmbeddingsTabProps {
  displacements: DisplacementRow[];
  sortField: DisplacementSortField;
  sortDir: SortDir;
  onSortFieldChange: (field: DisplacementSortField) => void;
  onSortDirChange: (dir: SortDir) => void;
  emitDirective: (d: any) => void;
  structureId: string | null;
}

const EMB_COLUMNS: { key: DisplacementSortField; label: string }[] = [
  { key: "residue_id", label: "Residue" },
  { key: "chain", label: "Chain" },
  { key: "index", label: "Idx" },
  { key: "primary_depth", label: "Pri Depth" },
  { key: "secondary_depth", label: "Sec Depth" },
  { key: "depth_delta", label: "Δ Depth" },
  { key: "displacement", label: "Displ." },
];

function EmbeddingsTab({
  displacements,
  sortField,
  sortDir,
  onSortFieldChange,
  onSortDirChange,
  emitDirective,
  structureId,
}: EmbeddingsTabProps) {
  const sorted = useMemo(() => {
    const data = [...displacements];
    data.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDir === "asc" ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
    });
    return data;
  }, [displacements, sortField, sortDir]);

  const handleSort = (field: DisplacementSortField) => {
    if (field === sortField) {
      onSortDirChange(sortDir === "asc" ? "desc" : "asc");
    } else {
      onSortFieldChange(field);
      onSortDirChange("desc");
    }
  };

  const highlightResidue = useCallback(
    (residueId: string) => {
      if (!structureId) return;
      emitDirective({
        action: "highlight",
        structure_id: structureId,
        highlight_groups: [
          {
            residue_ids: [residueId],
            color: "#06b6d4",
            style: "glow",
            label: residueId,
          },
        ],
      });
    },
    [structureId, emitDirective]
  );

  const highlightTopMovers = useCallback(() => {
    if (!structureId || displacements.length === 0) return;
    const top10 = [...displacements]
      .sort((a, b) => b.displacement - a.displacement)
      .slice(0, 10)
      .map((r) => r.residue_id);
    emitDirective({
      action: "highlight",
      structure_id: structureId,
      highlight_groups: [
        {
          residue_ids: top10,
          color: "#f97316",
          style: "glow",
          label: "Top 10 movers",
        },
      ],
    });
  }, [structureId, displacements, emitDirective]);

  if (displacements.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 px-4">
        <p className="text-xs text-zinc-600">No displacement data available.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Top Movers button */}
      <div className="px-3 py-2 border-b border-zinc-800">
        <button
          onClick={highlightTopMovers}
          className="px-2 py-1.5 bg-orange-700/30 border border-orange-600/40 rounded text-[11px] font-medium text-orange-300 hover:bg-orange-700/50"
        >
          Top Movers (10)
        </button>
      </div>

      {/* Displacement table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-zinc-900 z-10">
            <tr>
              {EMB_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="px-1.5 py-1.5 text-left text-zinc-500 font-medium cursor-pointer hover:text-zinc-300 select-none whitespace-nowrap"
                >
                  {col.label}
                  {sortField === col.key && (
                    <span className="ml-0.5 text-cyan-400">{sortDir === "asc" ? "↑" : "↓"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr
                key={row.residue_id}
                onClick={() => highlightResidue(row.residue_id)}
                className="border-t border-zinc-800/50 hover:bg-zinc-800/30 cursor-pointer"
              >
                <td className="px-1.5 py-1 text-zinc-300 font-mono">{row.residue_id}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.chain}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.index}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.primary_depth.toFixed(3)}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.secondary_depth.toFixed(3)}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.depth_delta.toFixed(3)}</td>
                <td className="px-1.5 py-1 text-cyan-400 font-medium">{row.displacement.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


// --- Graph Tab ---

interface GraphTabProps {
  graphDiff: CompareGraphDiffResult | null;
  sortField: MetricDeltaSortField;
  sortDir: SortDir;
  onSortFieldChange: (field: MetricDeltaSortField) => void;
  onSortDirChange: (dir: SortDir) => void;
  emitDirective: (d: any) => void;
  structureId: string | null;
}

const METRIC_COLUMNS: { key: MetricDeltaSortField; label: string }[] = [
  { key: "residue_id", label: "Residue" },
  { key: "chain", label: "Chain" },
  { key: "betweenness_delta", label: "Δ Betw." },
  { key: "degree_delta", label: "Δ Degree" },
  { key: "clustering_delta", label: "Δ Clust." },
];

function GraphTab({
  graphDiff,
  sortField,
  sortDir,
  onSortFieldChange,
  onSortDirChange,
  emitDirective,
  structureId,
}: GraphTabProps) {
  const sortedMetrics = useMemo(() => {
    if (!graphDiff?.metric_deltas) return [];
    const data = [...graphDiff.metric_deltas];
    data.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDir === "asc" ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
    });
    return data;
  }, [graphDiff, sortField, sortDir]);

  const handleSort = (field: MetricDeltaSortField) => {
    if (field === sortField) {
      onSortDirChange(sortDir === "asc" ? "desc" : "asc");
    } else {
      onSortFieldChange(field);
      onSortDirChange("desc");
    }
  };

  const highlightResidue = useCallback(
    (residueId: string) => {
      if (!structureId) return;
      emitDirective({
        action: "highlight",
        structure_id: structureId,
        highlight_groups: [
          {
            residue_ids: [residueId],
            color: "#06b6d4",
            style: "glow",
            label: residueId,
          },
        ],
      });
    },
    [structureId, emitDirective]
  );

  if (!graphDiff) {
    return (
      <div className="flex items-center justify-center h-32 px-4">
        <p className="text-xs text-zinc-600">No graph diff data available.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Edge diff summary card */}
      <div className="px-3 py-2 border-b border-zinc-800 space-y-1">
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">Edge Diff</p>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-sm font-bold text-emerald-400">+{graphDiff.edge_diff.gained_count}</p>
            <p className="text-[9px] text-zinc-500">Gained</p>
          </div>
          <div>
            <p className="text-sm font-bold text-red-400">−{graphDiff.edge_diff.lost_count}</p>
            <p className="text-[9px] text-zinc-500">Lost</p>
          </div>
          <div>
            <p className="text-sm font-bold text-yellow-400">{graphDiff.edge_diff.changed_count}</p>
            <p className="text-[9px] text-zinc-500">Changed</p>
          </div>
        </div>
        <div className="flex justify-center gap-4 pt-1">
          <span className="text-[10px] text-violet-400">H-bond +{graphDiff.hbond.gained}</span>
          <span className="text-[10px] text-violet-400">H-bond −{graphDiff.hbond.lost}</span>
        </div>
      </div>

      {/* Metric delta table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-zinc-900 z-10">
            <tr>
              {METRIC_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="px-1.5 py-1.5 text-left text-zinc-500 font-medium cursor-pointer hover:text-zinc-300 select-none whitespace-nowrap"
                >
                  {col.label}
                  {sortField === col.key && (
                    <span className="ml-0.5 text-cyan-400">{sortDir === "asc" ? "↑" : "↓"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedMetrics.map((row) => (
              <tr
                key={row.residue_id}
                onClick={() => highlightResidue(row.residue_id)}
                className="border-t border-zinc-800/50 hover:bg-zinc-800/30 cursor-pointer"
              >
                <td className="px-1.5 py-1 text-zinc-300 font-mono">{row.residue_id}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.chain}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.betweenness_delta.toFixed(4)}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.degree_delta}</td>
                <td className="px-1.5 py-1 text-zinc-400">{row.clustering_delta.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


// --- Summary Tab ---

interface SummaryTabProps {
  displacements: DisplacementRow[];
  summary: { mean_displacement: number; max_displacement: number; movers_above_threshold: number; threshold: number } | null;
  threshold: number;
  onThresholdChange: (t: number) => void;
}

function SummaryTab({ displacements, summary, threshold, onThresholdChange }: SummaryTabProps) {
  const computedSummary = useMemo(() => {
    if (displacements.length === 0) return null;
    const disps = displacements.map((r) => r.displacement);
    const mean = disps.reduce((a, b) => a + b, 0) / disps.length;
    const max = Math.max(...disps);
    const aboveThreshold = disps.filter((d) => d > threshold).length;
    return { mean, max, aboveThreshold };
  }, [displacements, threshold]);

  const topMovers = useMemo(() => {
    return [...displacements]
      .sort((a, b) => b.displacement - a.displacement)
      .slice(0, 10);
  }, [displacements]);

  const handleCopyCSV = useCallback(() => {
    const csv = exportDisplacementsToCSV(displacements);
    navigator.clipboard.writeText(csv).catch(() => {});
  }, [displacements]);

  if (!computedSummary) {
    return (
      <div className="flex items-center justify-center h-32 px-4">
        <p className="text-xs text-zinc-600">No displacement data to summarize.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-auto">
      {/* Summary stats card */}
      <div className="px-3 py-2 border-b border-zinc-800 space-y-2">
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">Summary Statistics</p>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-sm font-bold text-cyan-400">{computedSummary.mean.toFixed(3)}</p>
            <p className="text-[9px] text-zinc-500">Mean Displ.</p>
          </div>
          <div>
            <p className="text-sm font-bold text-orange-400">{computedSummary.max.toFixed(3)}</p>
            <p className="text-[9px] text-zinc-500">Max Displ.</p>
          </div>
          <div>
            <p className="text-sm font-bold text-emerald-400">{computedSummary.aboveThreshold}</p>
            <p className="text-[9px] text-zinc-500">Above Thresh.</p>
          </div>
        </div>

        {/* Threshold control */}
        <div className="flex items-center gap-2">
          <label className="text-[10px] text-zinc-500">Threshold:</label>
          <input
            type="number"
            step="0.1"
            min="0"
            value={threshold}
            onChange={(e) => onThresholdChange(parseFloat(e.target.value) || 0)}
            className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-xs text-zinc-200 focus:outline-none focus:border-cyan-600"
          />
        </div>
      </div>

      {/* Top movers list */}
      <div className="px-3 py-2 border-b border-zinc-800 space-y-1">
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">Top 10 Movers</p>
        <div className="space-y-0.5">
          {topMovers.map((row, i) => (
            <div key={row.residue_id} className="flex justify-between text-[11px]">
              <span className="text-zinc-400 font-mono">
                {i + 1}. {row.residue_id}
              </span>
              <span className="text-cyan-400">{row.displacement.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Copy to CSV */}
      <div className="px-3 py-2">
        <button
          onClick={handleCopyCSV}
          className="px-2 py-1.5 bg-zinc-700/50 border border-zinc-600/40 rounded text-[11px] font-medium text-zinc-300 hover:bg-zinc-700/80"
        >
          Copy to CSV
        </button>
      </div>
    </div>
  );
}
