/**
 * DataInspectorPanel — Surfaces all computed DTIE pipeline data in a browsable,
 * searchable, exportable interface. The capstone data access feature.
 *
 * Tabs: Overview | Residues | Pockets | Candidates
 * Requirements: 1.1–1.4, 2.1–2.5, 3.1–3.4, 4.1–4.4
 */
import { useState, useMemo, useCallback } from "react";
import { CheckCircle2, Circle, ArrowUpDown, Filter } from "lucide-react";
import { useDashboard } from "../../lib/context";
import { useHydration } from "../../context/HydrationProvider";
import { mergeResidueData, type MergedResidueRow } from "../../lib/mergeResidueData";
import { sortRows, filterRows, type SortConfig, type FilterCriteria } from "../../lib/tableUtils";
import type { PharmacophoreData, PharmacophoreRow, DrugCandidateData, DrugCandidateRow } from "../../lib/types";

type InspectorTab = "overview" | "residues" | "pockets" | "candidates";

// ---------------------------------------------------------------------------
// Phase status helpers
// ---------------------------------------------------------------------------

interface PhaseStatus {
  label: string;
  computed: boolean;
  count: number | null;
  unit: string;
}

function usePhaseStatuses(): PhaseStatus[] {
  const { embeddings, graphMetrics, sourceLeaks, resistanceData, persistenceStatus, pharmacophorePockets, drugCandidates } = useHydration();

  return useMemo(() => [
    {
      label: "Embeddings",
      computed: persistenceStatus?.embeddings_persisted ?? (embeddings?.residues?.length ?? 0) > 0,
      count: embeddings?.residues?.length ?? null,
      unit: "residues",
    },
    {
      label: "Graph",
      computed: persistenceStatus?.graph_persisted ?? (graphMetrics?.metrics?.length ?? 0) > 0,
      count: graphMetrics?.metrics?.length ?? null,
      unit: "nodes",
    },
    {
      label: "Sites",
      computed: persistenceStatus?.sites_persisted ?? false,
      count: null,
      unit: "sites",
    },
    {
      label: "Source Leaks",
      computed: (sourceLeaks?.leaks?.length ?? 0) > 0,
      count: sourceLeaks?.leaks?.length ?? null,
      unit: "leaks",
    },
    {
      label: "Resistance",
      computed: (resistanceData?.residues?.length ?? 0) > 0,
      count: resistanceData?.residues?.length ?? null,
      unit: "residues",
    },
    {
      label: "Phase 5",
      computed: persistenceStatus?.phase5_persisted ?? (pharmacophorePockets?.pockets?.length ?? 0) > 0,
      count: pharmacophorePockets?.pockets?.length ?? null,
      unit: "pockets",
    },
    {
      label: "Phase 6",
      computed: persistenceStatus?.phase6_persisted ?? (drugCandidates?.candidates?.length ?? 0) > 0,
      count: drugCandidates?.candidates?.length ?? null,
      unit: "candidates",
    },
  ], [embeddings, graphMetrics, sourceLeaks, resistanceData, persistenceStatus, pharmacophorePockets, drugCandidates]);
}

// ---------------------------------------------------------------------------
// Column definitions for residue table
// ---------------------------------------------------------------------------

interface ColumnDef {
  key: keyof MergedResidueRow;
  label: string;
  width: string;
  format: (val: unknown) => string;
}

const ALL_COLUMNS: ColumnDef[] = [
  { key: "residue_id", label: "ID", width: "w-28", format: (v) => String(v ?? "—") },
  { key: "chain_label", label: "Chain", width: "w-12", format: (v) => String(v ?? "—") },
  { key: "residue_index", label: "Idx", width: "w-12", format: (v) => v != null ? String(v) : "—" },
  { key: "cone_depth", label: "Depth", width: "w-14", format: (v) => typeof v === "number" ? v.toFixed(3) : "—" },
  { key: "epistemic_uncertainty", label: "Epist.", width: "w-14", format: (v) => typeof v === "number" ? v.toFixed(3) : "—" },
  { key: "aleatoric_uncertainty", label: "Aleat.", width: "w-14", format: (v) => typeof v === "number" ? v.toFixed(3) : "—" },
  { key: "betweenness", label: "Betw.", width: "w-14", format: (v) => typeof v === "number" ? v.toFixed(3) : "—" },
  { key: "degree", label: "Deg", width: "w-10", format: (v) => v != null ? String(v) : "—" },
  { key: "clustering_coefficient", label: "Clust.", width: "w-14", format: (v) => typeof v === "number" ? v.toFixed(3) : "—" },
  { key: "is_bridge", label: "Bridge", width: "w-14", format: (v) => v === true ? "Yes" : v === false ? "No" : "—" },
  { key: "leak_score", label: "Leak", width: "w-14", format: (v) => typeof v === "number" ? v.toFixed(3) : "—" },
  { key: "sensitivity_score", label: "Sens.", width: "w-14", format: (v) => typeof v === "number" ? v.toFixed(3) : "—" },
  { key: "classification", label: "Class", width: "w-20", format: (v) => v ? String(v) : "—" },
  { key: "is_hinge", label: "Hinge", width: "w-14", format: (v) => v === true ? "Yes" : v === false ? "No" : "—" },
];

// Default visible columns (subset for initial view)
const DEFAULT_VISIBLE: Set<keyof MergedResidueRow> = new Set([
  "residue_id", "chain_label", "residue_index", "cone_depth",
  "epistemic_uncertainty", "leak_score", "sensitivity_score", "classification",
]);

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DataInspectorPanel() {
  const { activeStructure, emitDirective, selectedPocketId, setSelectedPocketId } = useDashboard();
  const { embeddings, graphMetrics, sourceLeaks, resistanceData, isHydrating, pharmacophorePockets, drugCandidates, hydration, allostericSites } = useHydration();

  const [tab, setTab] = useState<InspectorTab>("overview");
  const [sortConfig, setSortConfig] = useState<SortConfig>({ column: "residue_index", direction: "asc" });
  const [filterCriteria, setFilterCriteria] = useState<FilterCriteria>({});
  const [visibleColumns, setVisibleColumns] = useState<Set<keyof MergedResidueRow>>(DEFAULT_VISIBLE);
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  const phaseStatuses = usePhaseStatuses();

  // Merge all residue data
  const mergedRows = useMemo(
    () => mergeResidueData(embeddings, graphMetrics, sourceLeaks, resistanceData),
    [embeddings, graphMetrics, sourceLeaks, resistanceData],
  );

  // Apply filter then sort
  const displayRows = useMemo(() => {
    const filtered = filterRows(mergedRows, filterCriteria);
    return sortRows(filtered, sortConfig);
  }, [mergedRows, filterCriteria, sortConfig]);

  // Column header click toggles sort
  const handleSort = useCallback((column: keyof MergedResidueRow) => {
    setSortConfig((prev) => ({
      column,
      direction: prev.column === column && prev.direction === "asc" ? "desc" : "asc",
    }));
  }, []);

  // Row click highlights residue in viewers
  const handleRowClick = useCallback((row: MergedResidueRow) => {
    emitDirective({
      action: "highlight",
      highlight_groups: [{
        residue_ids: [row.residue_id],
        color: "#22d3ee",
        style: "glow",
        label: `${row.chain_label ?? ""}:${row.residue_index ?? ""}`,
      }],
    });
  }, [emitDirective]);

  // Pocket click highlights pocket residues + its *specific* connected allosteric locks
  // (precise remote control sub-graph for this pocket's pathways)
  // Also sets global selectedPocketId so GraphTopologyPanel renders the dynamic sub-graph
  const handlePocketClick = useCallback((pocket: PharmacophoreRow) => {
    const lockIds = pocket.connected_allosteric_locks || allostericSites?.sites?.[0]?.residue_ids || [];
    setSelectedPocketId(pocket.pocket_index);
    emitDirective({
      action: "highlight",
      highlight_groups: [
        {
          residue_ids: pocket.residue_ids,
          color: "#4ade80",
          style: "glow",
          label: `Pocket ${pocket.pocket_index} (coupling: ${pocket.allosteric_coupling.toFixed(0)})`,
        },
        {
          residue_ids: lockIds,
          color: "#c026ff",  // purple for precision locks
          style: "glow",
          label: `Allosteric Locks (${lockIds.length}) — remote via pathways`,
        },
      ],
    });
  }, [emitDirective, allostericSites, setSelectedPocketId]);

  // Candidate click highlights binding site residues + its *specific* connected locks
  const handleCandidateClick = useCallback((candidate: DrugCandidateRow) => {
    const pocket = pharmacophorePockets?.pockets.find((p) => p.pocket_index === candidate.pocket_index);
    const lockIds = pocket?.connected_allosteric_locks || allostericSites?.sites?.[0]?.residue_ids || [];
    if (pocket) {
      setSelectedPocketId(pocket.pocket_index);
      emitDirective({
        action: "highlight",
        highlight_groups: [
          {
            residue_ids: pocket.residue_ids,
            color: candidate.admet_pass ? "#facc15" : "#f87171",
            style: "glow",
            label: `Candidate P${candidate.pocket_index} (coupling: ${pocket.allosteric_coupling?.toFixed(0) || 'N/A'})`,
          },
          {
            residue_ids: lockIds,
            color: "#c026ff",
            style: "glow",
            label: `Allosteric Locks (${lockIds.length}) — remote via pathways`,
          },
        ],
      });
    }
  }, [emitDirective, pharmacophorePockets, allostericSites, setSelectedPocketId]);

  // Toggle column visibility
  const toggleColumn = useCallback((key: keyof MergedResidueRow) => {
    setVisibleColumns((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  if (!activeStructure) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-zinc-600 text-center">Select a structure to inspect data.</p>
      </div>
    );
  }

  if (isHydrating) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-zinc-500 animate-pulse">Loading data...</p>
      </div>
    );
  }

  const columns = ALL_COLUMNS.filter((c) => visibleColumns.has(c.key));

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-zinc-800 px-2 shrink-0">
        {(["overview", "residues", "pockets", "candidates"] as InspectorTab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-2.5 py-1.5 text-[10px] font-medium uppercase tracking-wide border-b-2 transition-colors ${
              tab === t
                ? "border-cyan-500 text-cyan-300"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto">
        {tab === "overview" && <OverviewTab phases={phaseStatuses} totalResidues={mergedRows.length} />}
        {tab === "residues" && (
          <ResiduesTab
            rows={displayRows}
            columns={columns}
            allColumns={ALL_COLUMNS}
            visibleColumns={visibleColumns}
            sortConfig={sortConfig}
            filterCriteria={filterCriteria}
            showColumnPicker={showColumnPicker}
            showFilters={showFilters}
            onSort={handleSort}
            onRowClick={handleRowClick}
            onToggleColumn={toggleColumn}
            onToggleColumnPicker={() => setShowColumnPicker((p) => !p)}
            onToggleFilters={() => setShowFilters((p) => !p)}
            onFilterChange={setFilterCriteria}
            totalMerged={mergedRows.length}
          />
        )}
        {tab === "pockets" && <PocketsTab data={pharmacophorePockets} onSelectPocket={handlePocketClick} />}
        {tab === "candidates" && <CandidatesTab data={drugCandidates} onSelectCandidate={handleCandidateClick} />}
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Overview Tab
// ---------------------------------------------------------------------------

function OverviewTab({ phases, totalResidues }: { phases: PhaseStatus[]; totalResidues: number }) {
  return (
    <div className="px-3 py-3 space-y-3">
      {/* Persistence status grid */}
      <div>
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide mb-2">
          Pipeline Phase Status
        </p>
        <div className="grid grid-cols-2 gap-1.5">
          {phases.map((p) => (
            <div
              key={p.label}
              className={`flex items-center gap-2 px-2 py-1.5 rounded border ${
                p.computed
                  ? "border-emerald-700/40 bg-emerald-900/10"
                  : "border-zinc-700/40 bg-zinc-800/30"
              }`}
            >
              {p.computed ? (
                <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
              ) : (
                <Circle className="w-3 h-3 text-zinc-600 shrink-0" />
              )}
              <div className="min-w-0">
                <p className={`text-[10px] font-medium truncate ${p.computed ? "text-emerald-300" : "text-zinc-500"}`}>
                  {p.label}
                </p>
                <p className="text-[9px] text-zinc-500">
                  {p.computed && p.count != null ? `${p.count} ${p.unit}` : p.computed ? "Computed" : "Not computed"}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Aggregate statistics */}
      <div>
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide mb-2">
          Aggregate Statistics
        </p>
        <div className="grid grid-cols-2 gap-1.5">
          <StatCard label="Total residues (merged)" value={totalResidues} />
          <StatCard label="Phases computed" value={phases.filter((p) => p.computed).length} />
          <StatCard
            label="Source leaks"
            value={phases.find((p) => p.label === "Source Leaks")?.count ?? 0}
          />
          <StatCard
            label="Graph nodes"
            value={phases.find((p) => p.label === "Graph")?.count ?? 0}
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="px-2 py-1.5 bg-zinc-800/40 rounded border border-zinc-700/40">
      <p className="text-[9px] text-zinc-500">{label}</p>
      <p className="text-sm font-medium text-zinc-200">{value}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Residues Tab
// ---------------------------------------------------------------------------

interface ResiduesTabProps {
  rows: MergedResidueRow[];
  columns: ColumnDef[];
  allColumns: ColumnDef[];
  visibleColumns: Set<keyof MergedResidueRow>;
  sortConfig: SortConfig;
  filterCriteria: FilterCriteria;
  showColumnPicker: boolean;
  showFilters: boolean;
  onSort: (col: keyof MergedResidueRow) => void;
  onRowClick: (row: MergedResidueRow) => void;
  onToggleColumn: (key: keyof MergedResidueRow) => void;
  onToggleColumnPicker: () => void;
  onToggleFilters: () => void;
  onFilterChange: (criteria: FilterCriteria) => void;
  totalMerged: number;
}

function ResiduesTab({
  rows,
  columns,
  allColumns,
  visibleColumns,
  sortConfig,
  filterCriteria,
  showColumnPicker,
  showFilters,
  onSort,
  onRowClick,
  onToggleColumn,
  onToggleColumnPicker,
  onToggleFilters,
  onFilterChange,
  totalMerged,
}: ResiduesTabProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-zinc-800 shrink-0">
        <span className="text-[10px] text-zinc-500">
          {rows.length === totalMerged ? `${rows.length} residues` : `${rows.length} / ${totalMerged} residues`}
        </span>
        <div className="flex gap-1.5">
          <button
            onClick={onToggleFilters}
            className={`px-1.5 py-0.5 rounded text-[10px] border ${
              showFilters ? "border-cyan-600/50 text-cyan-300 bg-cyan-900/20" : "border-zinc-700 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Filter className="w-3 h-3 inline-block mr-0.5" />
            Filter
          </button>
          <button
            onClick={onToggleColumnPicker}
            className={`px-1.5 py-0.5 rounded text-[10px] border ${
              showColumnPicker ? "border-cyan-600/50 text-cyan-300 bg-cyan-900/20" : "border-zinc-700 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            Columns
          </button>
        </div>
      </div>

      {/* Filter bar */}
      {showFilters && (
        <FilterBar criteria={filterCriteria} onChange={onFilterChange} />
      )}

      {/* Column picker */}
      {showColumnPicker && (
        <div className="px-3 py-2 border-b border-zinc-800 bg-zinc-900/80">
          <div className="flex flex-wrap gap-1">
            {allColumns.map((col) => (
              <button
                key={col.key}
                onClick={() => onToggleColumn(col.key)}
                className={`px-1.5 py-0.5 rounded text-[9px] border ${
                  visibleColumns.has(col.key)
                    ? "border-cyan-600/50 text-cyan-300 bg-cyan-900/20"
                    : "border-zinc-700 text-zinc-600"
                }`}
              >
                {col.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[10px]">
          <thead className="bg-zinc-800/80 sticky top-0 z-10">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => onSort(col.key)}
                  className={`px-1.5 py-1 text-left text-zinc-500 cursor-pointer hover:text-zinc-300 select-none ${col.width}`}
                >
                  <span className="inline-flex items-center gap-0.5">
                    {col.label}
                    {sortConfig.column === col.key && (
                      <ArrowUpDown className="w-2.5 h-2.5 text-cyan-400" />
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {rows.map((row, i) => (
              <tr
                key={row.residue_id + "-" + i}
                onClick={() => onRowClick(row)}
                className="hover:bg-cyan-900/10 cursor-pointer transition-colors"
              >
                {columns.map((col) => (
                  <td key={col.key} className={`px-1.5 py-0.5 text-zinc-300 ${col.width}`}>
                    {col.format(row[col.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <div className="flex items-center justify-center py-8">
            <p className="text-[10px] text-zinc-600">No residue data available.</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Filter Bar
// ---------------------------------------------------------------------------

function FilterBar({
  criteria,
  onChange,
}: {
  criteria: FilterCriteria;
  onChange: (c: FilterCriteria) => void;
}) {
  return (
    <div className="px-3 py-2 border-b border-zinc-800 bg-zinc-900/60 space-y-1.5">
      <div className="grid grid-cols-3 gap-1.5">
        <input
          type="text"
          value={criteria.chain ?? ""}
          onChange={(e) => onChange({ ...criteria, chain: e.target.value || null })}
          placeholder="Chain"
          className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
        />
        <input
          type="number"
          value={criteria.minUncertainty ?? ""}
          onChange={(e) => onChange({ ...criteria, minUncertainty: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Min uncert."
          step="0.01"
          className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
        />
        <input
          type="number"
          value={criteria.maxUncertainty ?? ""}
          onChange={(e) => onChange({ ...criteria, maxUncertainty: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max uncert."
          step="0.01"
          className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
        />
        <input
          type="number"
          value={criteria.minDepth ?? ""}
          onChange={(e) => onChange({ ...criteria, minDepth: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Min depth"
          step="0.01"
          className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
        />
        <input
          type="number"
          value={criteria.maxDepth ?? ""}
          onChange={(e) => onChange({ ...criteria, maxDepth: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max depth"
          step="0.01"
          className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
        />
        <select
          value={criteria.classification ?? ""}
          onChange={(e) => onChange({ ...criteria, classification: (e.target.value || null) as FilterCriteria["classification"] })}
          className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-zinc-200 focus:outline-none focus:border-cyan-600"
        >
          <option value="">All classes</option>
          <option value="high_sensitivity">High sensitivity</option>
          <option value="moderate">Moderate</option>
          <option value="stable">Stable</option>
        </select>
      </div>
      <button
        onClick={() => onChange({})}
        className="text-[9px] text-zinc-500 hover:text-zinc-300"
      >
        Clear filters
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pockets Tab (Phase 5)
// ---------------------------------------------------------------------------

interface PocketsTabProps {
  data: PharmacophoreData | null;
  onSelectPocket: (pocket: PharmacophoreRow) => void;
}

function PocketsTab({ data, onSelectPocket }: PocketsTabProps) {
  if (!data || data.pockets.length === 0) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <div className="text-center space-y-1">
          <p className="text-xs text-zinc-500">Phase 5 — Pharmacophore Pockets</p>
          <p className="text-[10px] text-zinc-600">
            Not computed — run pipeline with Phase 5 enabled.
          </p>
        </div>
      </div>
    );
  }

  // Sort by druggability_score descending
  const sorted = [...data.pockets].sort((a, b) => b.druggability_score - a.druggability_score);

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-1.5 border-b border-zinc-800 shrink-0">
        <span className="text-[10px] text-zinc-500">{sorted.length} pockets found</span>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[10px]">
          <thead className="bg-zinc-800/80 sticky top-0 z-10">
            <tr>
              <th className="px-1.5 py-1 text-left text-zinc-500">#</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Druggability</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Residues</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Volume</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Coupling</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Locks</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {sorted.map((pocket) => (
              <tr
                key={pocket.pocket_index}
                onClick={() => onSelectPocket(pocket)}
                className="hover:bg-emerald-900/10 cursor-pointer transition-colors"
              >
                <td className="px-1.5 py-1 text-zinc-300 font-medium">{pocket.pocket_index}</td>
                <td className="px-1.5 py-1 text-right">
                  <span className={`font-medium ${
                    pocket.druggability_score >= 0.7 ? "text-emerald-400" :
                    pocket.druggability_score >= 0.4 ? "text-amber-400" : "text-zinc-400"
                  }`}>
                    {pocket.druggability_score.toFixed(3)}
                  </span>
                </td>
                <td className="px-1.5 py-1 text-right text-zinc-400">{pocket.residue_count}</td>
                <td className="px-1.5 py-1 text-right text-zinc-400">{pocket.volume_estimate.toFixed(1)}</td>
                <td className="px-1.5 py-1 text-right text-zinc-400">{pocket.allosteric_coupling.toFixed(3)}</td>
                <td className="px-1.5 py-1 text-right text-zinc-400">
                  {pocket.connected_allosteric_locks ? pocket.connected_allosteric_locks.length : 0}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Candidates Tab (Phase 6)
// ---------------------------------------------------------------------------

interface CandidatesTabProps {
  data: DrugCandidateData | null;
  onSelectCandidate: (candidate: DrugCandidateRow) => void;
}

function CandidatesTab({ data, onSelectCandidate }: CandidatesTabProps) {
  const [filterAdmet, setFilterAdmet] = useState<boolean | null>(null);
  const [filterSelective, setFilterSelective] = useState<boolean | null>(null);

  if (!data || data.candidates.length === 0) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <div className="text-center space-y-1">
          <p className="text-xs text-zinc-500">Phase 6 — Drug Candidates</p>
          <p className="text-[10px] text-zinc-600">
            Not computed — run pipeline with Phase 6 enabled.
          </p>
        </div>
      </div>
    );
  }

  const filtered = data.candidates.filter((c) => {
    if (filterAdmet !== null && c.admet_pass !== filterAdmet) return false;
    if (filterSelective !== null && c.is_state_selective !== filterSelective) return false;
    return true;
  });

  const totalCandidates = data.candidates.length;
  const admetPassedCount = data.candidates.filter((c) => c.admet_pass).length;
  const stateSelectiveCount = data.candidates.filter((c) => c.is_state_selective).length;

  return (
    <div className="flex flex-col h-full">
      {/* Summary counts */}
      <div className="px-3 py-2 border-b border-zinc-800 shrink-0 space-y-1.5">
        <div className="grid grid-cols-3 gap-1.5">
          <div className="px-2 py-1 bg-zinc-800/40 rounded border border-zinc-700/40 text-center">
            <p className="text-[9px] text-zinc-500">Total</p>
            <p className="text-xs font-medium text-zinc-200">{totalCandidates}</p>
          </div>
          <div className="px-2 py-1 bg-zinc-800/40 rounded border border-zinc-700/40 text-center">
            <p className="text-[9px] text-zinc-500">ADMET Pass</p>
            <p className="text-xs font-medium text-emerald-400">{admetPassedCount}</p>
          </div>
          <div className="px-2 py-1 bg-zinc-800/40 rounded border border-zinc-700/40 text-center">
            <p className="text-[9px] text-zinc-500">Selective</p>
            <p className="text-xs font-medium text-amber-400">{stateSelectiveCount}</p>
          </div>
        </div>

        {/* Filter toggles */}
        <div className="flex gap-1.5">
          <button
            onClick={() => setFilterAdmet(filterAdmet === true ? null : true)}
            className={`px-1.5 py-0.5 rounded text-[9px] border ${
              filterAdmet === true ? "border-emerald-600/50 text-emerald-300 bg-emerald-900/20" : "border-zinc-700 text-zinc-500"
            }`}
          >
            ADMET Pass
          </button>
          <button
            onClick={() => setFilterAdmet(filterAdmet === false ? null : false)}
            className={`px-1.5 py-0.5 rounded text-[9px] border ${
              filterAdmet === false ? "border-red-600/50 text-red-300 bg-red-900/20" : "border-zinc-700 text-zinc-500"
            }`}
          >
            ADMET Fail
          </button>
          <button
            onClick={() => setFilterSelective(filterSelective === true ? null : true)}
            className={`px-1.5 py-0.5 rounded text-[9px] border ${
              filterSelective === true ? "border-amber-600/50 text-amber-300 bg-amber-900/20" : "border-zinc-700 text-zinc-500"
            }`}
          >
            Selective
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[10px]">
          <thead className="bg-zinc-800/80 sticky top-0 z-10">
            <tr>
              <th className="px-1.5 py-1 text-left text-zinc-500">Pocket</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Druggability</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Access.</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Binding</th>
              <th className="px-1.5 py-1 text-center text-zinc-500">ADMET</th>
              <th className="px-1.5 py-1 text-right text-zinc-500">Select.</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {filtered.map((c, i) => (
              <tr
                key={`${c.pocket_index}-${i}`}
                onClick={() => onSelectCandidate(c)}
                className="hover:bg-amber-900/10 cursor-pointer transition-colors"
              >
                <td className="px-1.5 py-1 text-zinc-300 font-medium">{c.pocket_index}</td>
                <td className="px-1.5 py-1 text-right text-zinc-300">{c.combined_druggability.toFixed(3)}</td>
                <td className="px-1.5 py-1 text-right text-zinc-400">{c.accessibility_score.toFixed(3)}</td>
                <td className="px-1.5 py-1 text-right text-zinc-400">{c.binding_potential.toFixed(3)}</td>
                <td className="px-1.5 py-1 text-center">
                  <span className={`px-1 py-0.5 rounded text-[9px] font-medium ${
                    c.admet_pass ? "bg-emerald-900/30 text-emerald-400" : "bg-red-900/30 text-red-400"
                  }`}>
                    {c.admet_pass ? "Pass" : "Fail"}
                  </span>
                </td>
                <td className="px-1.5 py-1 text-right text-zinc-400">
                  {c.selectivity_ratio.toFixed(2)}
                  {c.is_state_selective && <span className="ml-0.5 text-amber-400">★</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="flex items-center justify-center py-8">
            <p className="text-[10px] text-zinc-600">No candidates match current filters.</p>
          </div>
        )}
      </div>
    </div>
  );
}
