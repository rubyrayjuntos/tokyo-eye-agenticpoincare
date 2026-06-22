/**
 * GraphTopologyPanel — Graph topology analysis: metrics table, bridges, H-bonds, paths, compare.
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
 */
import { useState, useMemo, useCallback } from "react";
import { api } from "../../lib/api";
import { useDashboard } from "../../lib/context";
import { useHydration } from "../../context/HydrationProvider";
import type { GraphCompareResult, ShortestPathResult } from "../../lib/types";

type SortField =
  | "residue_id"
  | "degree"
  | "betweenness"
  | "clustering_coefficient"
  | "closeness"
  | "eigenvector_centrality"
  | "conductance";

type SortDir = "asc" | "desc";

const COLUMNS: { key: SortField; label: string }[] = [
  { key: "residue_id", label: "Residue" },
  { key: "degree", label: "Degree" },
  { key: "betweenness", label: "Betweenness" },
  { key: "clustering_coefficient", label: "Clustering" },
  { key: "closeness", label: "Closeness" },
  { key: "eigenvector_centrality", label: "Eigenvector" },
  { key: "conductance", label: "Conductance" },
];

export default function GraphTopologyPanel() {
  const { activeStructure, setHighlightedResidues, emitDirective, selectedPocketId, setSelectedPocketId } = useDashboard();
  const { graphMetrics, loading, pharmacophorePockets, allostericSites, phase4_resistance } = useHydration();

  // Sort state
  const [sortField, setSortField] = useState<SortField>("betweenness");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // Bridges / H-bonds state
  const [bridgeResidues, setBridgeResidues] = useState<string[] | null>(null);
  const [hbondEdges, setHbondEdges] = useState<Array<{ source: string; target: string }> | null>(null);
  const [bridgeLoading, setBridgeLoading] = useState(false);
  const [hbondLoading, setHbondLoading] = useState(false);

  // Shortest path state
  const [pathSource, setPathSource] = useState("");
  const [pathTarget, setPathTarget] = useState("");
  const [pathResult, setPathResult] = useState<ShortestPathResult | null>(null);
  const [pathLoading, setPathLoading] = useState(false);

  // Compare state
  const [compareId, setCompareId] = useState("");
  const [compareResult, setCompareResult] = useState<GraphCompareResult | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  // Error
  const [error, setError] = useState<string | null>(null);

  // Hover state for per-pathway isolation in the sub-graph SVG and table
  const [hoveredPathKey, setHoveredPathKey] = useState<string | null>(null);

  const structureId = activeStructure?.structure_id ?? null;

  // Sorted metrics
  const sortedMetrics = useMemo(() => {
    if (!graphMetrics?.metrics) return [];
    const data = [...graphMetrics.metrics];
    data.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      const aNum = Number(aVal);
      const bNum = Number(bVal);
      return sortDir === "asc" ? aNum - bNum : bNum - aNum;
    });
    return data;
  }, [graphMetrics, sortField, sortDir]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const handleFindBridges = useCallback(async () => {
    if (!structureId) return;
    setBridgeLoading(true);
    setError(null);
    try {
      const res = await api.getGraphBridges(structureId);
      setBridgeResidues(res.residues);
      setHighlightedResidues(res.residues);
      emitDirective({
        action: "highlight",
        structure_id: structureId,
        highlight_groups: [
          {
            residue_ids: res.residues,
            color: "#f59e0b",
            style: "glow",
            label: "Bridge residues",
          },
        ],
      });
    } catch (err: unknown) {
      setError(err && typeof err === "object" && "message" in err ? (err as { message: string }).message : "Failed to fetch bridges");
    } finally {
      setBridgeLoading(false);
    }
  }, [structureId, setHighlightedResidues, emitDirective]);

  const handleHbondNetwork = useCallback(async () => {
    if (!structureId) return;
    setHbondLoading(true);
    setError(null);
    try {
      const res = await api.getGraphHbonds(structureId);
      setHbondEdges(res.edges);
      // Highlight all residues participating in H-bonds
      const residueSet = new Set<string>();
      res.edges.forEach((e) => {
        residueSet.add(e.source);
        residueSet.add(e.target);
      });
      const residueList = Array.from(residueSet);
      setHighlightedResidues(residueList);
      emitDirective({
        action: "highlight",
        structure_id: structureId,
        highlight_groups: [
          {
            residue_ids: residueList,
            color: "#8b5cf6",
            style: "glow",
            label: "H-bond network",
          },
        ],
      });
    } catch (err: unknown) {
      setError(err && typeof err === "object" && "message" in err ? (err as { message: string }).message : "Failed to fetch H-bond network");
    } finally {
      setHbondLoading(false);
    }
  }, [structureId, setHighlightedResidues, emitDirective]);

  const handleFindPath = useCallback(async () => {
    if (!structureId || !pathSource.trim() || !pathTarget.trim()) return;
    setPathLoading(true);
    setError(null);
    try {
      const res = await api.getShortestPath(structureId, pathSource.trim(), pathTarget.trim());
      setPathResult(res);
      if (!res.disconnected && res.path.length > 0) {
        setHighlightedResidues(res.path);
        emitDirective({
          action: "highlight",
          structure_id: structureId,
          highlight_groups: [
            {
              residue_ids: [res.path[0]],
              color: "#22c55e",
              style: "glow",
              label: "Path start",
            },
            {
              residue_ids: res.path.slice(1, -1),
              color: "#facc15",
              style: "color",
              label: "Path",
            },
            {
              residue_ids: [res.path[res.path.length - 1]],
              color: "#ef4444",
              style: "glow",
              label: "Path end",
            },
          ],
        });
      }
    } catch (err: unknown) {
      setError(err && typeof err === "object" && "message" in err ? (err as { message: string }).message : "Failed to find path");
    } finally {
      setPathLoading(false);
    }
  }, [structureId, pathSource, pathTarget, setHighlightedResidues, emitDirective]);

  const handleCompare = useCallback(async () => {
    if (!structureId || !compareId.trim()) return;
    setCompareLoading(true);
    setError(null);
    try {
      const res = await api.compareGraphs(structureId, compareId.trim());
      setCompareResult(res);
    } catch (err: unknown) {
      setError(err && typeof err === "object" && "message" in err ? (err as { message: string }).message : "Compare failed");
    } finally {
      setCompareLoading(false);
    }
  }, [structureId, compareId]);

  // No structure selected
  if (!activeStructure) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-zinc-600 text-center">Select a structure to view graph topology.</p>
      </div>
    );
  }

  // Loading state
  if (loading.graph_metrics) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-xs text-zinc-500 animate-pulse">Loading graph metrics...</p>
      </div>
    );
  }

  // No graph data
  if (!graphMetrics || !graphMetrics.metrics || graphMetrics.metrics.length === 0) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-zinc-600 text-center">No graph data available. Run the pipeline first.</p>
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

      {/* Action buttons */}
      <div className="flex gap-2 px-3 py-2 border-b border-zinc-800">
        <button
          onClick={handleFindBridges}
          disabled={bridgeLoading}
          className="px-2 py-1.5 bg-amber-700/30 border border-amber-600/40 rounded text-[11px] font-medium text-amber-300 hover:bg-amber-700/50 disabled:opacity-40"
        >
          {bridgeLoading ? "..." : "Find Bridges"}
        </button>
        <button
          onClick={handleHbondNetwork}
          disabled={hbondLoading}
          className="px-2 py-1.5 bg-violet-700/30 border border-violet-600/40 rounded text-[11px] font-medium text-violet-300 hover:bg-violet-700/50 disabled:opacity-40"
        >
          {hbondLoading ? "..." : "H-Bond Network"}
        </button>
      </div>

      {/* Bridge results */}
      {bridgeResidues && (
        <div className="px-3 py-1.5 border-b border-zinc-800 bg-amber-900/10">
          <p className="text-[10px] text-amber-400">
            {bridgeResidues.length} bridge residue{bridgeResidues.length !== 1 ? "s" : ""} found
            {bridgeResidues.length > 0 && `: ${bridgeResidues.slice(0, 8).join(", ")}${bridgeResidues.length > 8 ? "..." : ""}`}
          </p>
        </div>
      )}

      {/* H-bond results */}
      {hbondEdges && (
        <div className="px-3 py-1.5 border-b border-zinc-800 bg-violet-900/10">
          <p className="text-[10px] text-violet-400">
            {hbondEdges.length} H-bond edge{hbondEdges.length !== 1 ? "s" : ""} highlighted
          </p>
        </div>
      )}

      {/* Shortest path section */}
      <div className="px-3 py-2 border-b border-zinc-800 space-y-1.5">
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">Shortest Path</p>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Source residue"
            value={pathSource}
            onChange={(e) => setPathSource(e.target.value)}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
          />
          <input
            type="text"
            placeholder="Target residue"
            value={pathTarget}
            onChange={(e) => setPathTarget(e.target.value)}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
          />
          <button
            onClick={handleFindPath}
            disabled={pathLoading || !pathSource.trim() || !pathTarget.trim()}
            className="px-2 py-1 bg-cyan-700/30 border border-cyan-600/40 rounded text-[11px] text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-40"
          >
            {pathLoading ? "..." : "Find"}
          </button>
        </div>
        {pathResult && (
          <div className="text-[10px] text-zinc-400">
            {pathResult.disconnected ? (
              <span className="text-red-400">Residues are disconnected — no path exists.</span>
            ) : (
              <span>
                Path length: {pathResult.path_length} | Distance: {pathResult.total_distance.toFixed(2)} |{" "}
                {pathResult.path.join(" → ")}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Compare section */}
      <div className="px-3 py-2 border-b border-zinc-800 space-y-1.5">
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">Compare Graphs</p>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Second structure ID"
            value={compareId}
            onChange={(e) => setCompareId(e.target.value)}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
          />
          <button
            onClick={handleCompare}
            disabled={compareLoading || !compareId.trim()}
            className="px-2 py-1 bg-emerald-700/30 border border-emerald-600/40 rounded text-[11px] text-emerald-300 hover:bg-emerald-700/50 disabled:opacity-40"
          >
            {compareLoading ? "..." : "Compare"}
          </button>
        </div>
        {compareResult && (
          <div className="text-[10px] space-y-0.5">
            <p className="text-emerald-400">+{compareResult.edge_diff.gained.length} gained edges</p>
            <p className="text-red-400">−{compareResult.edge_diff.lost.length} lost edges</p>
            <p className="text-zinc-400">{compareResult.edge_diff.changed.length} changed edges</p>
          </div>
        )}
      </div>

      {/* Metrics table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-zinc-900 z-10">
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="px-2 py-1.5 text-left text-zinc-500 font-medium cursor-pointer hover:text-zinc-300 select-none whitespace-nowrap"
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
                className={`border-t border-zinc-800/50 hover:bg-zinc-800/30 ${
                  row.is_bridge ? "bg-amber-900/10" : ""
                }`}
              >
                <td className="px-2 py-1 text-zinc-300 font-mono">{row.residue_id}</td>
                <td className="px-2 py-1 text-zinc-400">{row.degree}</td>
                <td className="px-2 py-1 text-zinc-400">{row.betweenness.toFixed(4)}</td>
                <td className="px-2 py-1 text-zinc-400">{row.clustering_coefficient.toFixed(3)}</td>
                <td className="px-2 py-1 text-zinc-400">{row.closeness.toFixed(4)}</td>
                <td className="px-2 py-1 text-zinc-400">{row.eigenvector_centrality.toFixed(4)}</td>
                <td className="px-2 py-1 text-zinc-400">{row.conductance.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Allosteric Control Sub-graph (dynamic, data-driven from hydration) */}
      <div className="px-3 py-2 border-t border-zinc-800 bg-zinc-950/50 text-[10px] space-y-1.5">
        <div className="flex items-center justify-between">
          <p className="text-[10px] text-amber-400 font-medium uppercase tracking-wide">
            Allosteric Control Sub-graph
          </p>
          {selectedPocketId ? (
            <button
              onClick={() => setSelectedPocketId(null)}
              className="text-[9px] text-zinc-400 hover:text-zinc-200"
            >
              Clear selection (Macro View)
            </button>
          ) : (
            <span className="text-[9px] text-zinc-500">Select a pocket in Inspector for focused view</span>
          )}
        </div>

        {selectedPocketId ? (
          // Focused sub-graph for selected pocket
          (() => {
            const pocket = pharmacophorePockets?.pockets?.find((p) => p.pocket_index === selectedPocketId);
            const allLocks = allostericSites?.sites?.[0]?.residue_ids || [];
            const connectedLocks = pocket?.connected_allosteric_locks || [];
            const displayLocks = connectedLocks.length > 0 ? connectedLocks : allLocks.slice(0, 8);

            // Compute (or use backend-enriched) per-pathway data for the receipt table and precise SVG.
            // Backend now provides connecting_pathways (exact subset with normalized full IDs + coupling)
            // for the 20 pockets after B-chain-aware mapping and source-leak lock intersection.
            const pocketResSet = new Set(pocket?.residue_ids || []);
            const lockSet = new Set(displayLocks);
            let connectingPathways: any[] = [];
            if (pocket?.connecting_pathways && pocket.connecting_pathways.length > 0) {
              // Prefer the exact pre-computed subset from hydration (governed, includes the real high-M couplings for KRAS)
              connectingPathways = pocket.connecting_pathways.map((pw: any, idx: number) => {
                const src = String(pw.source_residue || pw.src || '');
                const tgt = String(pw.target_residue || pw.tgt || '');
                return {
                  ...pw,
                  _key: `${src}-${tgt}-${idx}`,
                  src,
                  tgt,
                  coupling: parseFloat(pw.coupling_strength || pw.coupling || 0),
                };
              });
            } else {
              const allPathways = phase4_resistance?.pathways || [];
              connectingPathways = allPathways
                .map((pw: any, idx: number) => {
                  const src = String(pw.source_residue || '');
                  const tgt = String(pw.target_residue || '');
                  return {
                    ...pw,
                    _key: `${src}-${tgt}-${idx}`,
                    src,
                    tgt,
                    coupling: parseFloat(pw.coupling_strength || 0),
                  };
                })
                .filter((pw: any) =>
                  (lockSet.has(pw.src) && pocketResSet.has(pw.tgt)) ||
                  (lockSet.has(pw.tgt) && pocketResSet.has(pw.src))
                )
                .sort((a: any, b: any) => b.coupling - a.coupling);
            }

            const maxCoupling = Math.max(1, ...connectingPathways.map(p => p.coupling), 1);

            return (
              <div className="space-y-1.5">
                <div className="text-zinc-400">
                  Target: <span className="font-mono text-emerald-300">Pocket {selectedPocketId}</span> (coupling {pocket?.allosteric_coupling?.toFixed(0) || 'N/A'})
                  <br />
                  Connected Precision Locks ({displayLocks.length}) • {connectingPathways.length} pathways
                </div>
                {/* Bipartite visual with inline SVG edges: Locks (left, amber) → Pocket (right, green).
                   Dark ultramarine/indigo bg for "resting state". Amber curves for "conductive channels" glowing with energy.
                   Stroke scaled with log for 60k+ outliers: thick/high-opacity for strong signals (e.g. P3 60k), thin/ghostly for weak.
                   Hover on table rows isolates the specific curve.
                */}
                <div className="flex items-stretch h-20 bg-[#1a2744] rounded border border-zinc-700 overflow-hidden text-[8px]">
                  {/* Left: Locks column (amber nodes) */}
                  <div className="w-2/5 p-1 overflow-y-auto flex flex-col gap-0.5 border-r border-zinc-600/50 bg-black/20 text-[7px]">
                    {displayLocks.slice(0, 8).map((lid, i) => (
                      <div key={i} 
                        className="px-1 py-0.5 bg-amber-900/40 text-amber-200 rounded font-mono border border-amber-700/60 cursor-pointer hover:bg-amber-800/60 truncate"
                        onClick={() => {
                          setHighlightedResidues([lid]);
                          emitDirective({ action: "highlight", highlight_groups: [{ residue_ids: [lid], color: "#f59e0b", style: "glow", label: `Lock ${lid}` }] });
                        }}>
                        {lid}
                      </div>
                    ))}
                    {displayLocks.length > 8 && <div className="text-amber-300/60">+{displayLocks.length-8} more locks</div>}
                  </div>

                  {/* Center: SVG for flowing bezier "energy" edges, weighted by coupling. Hover dims non-hovered. */}
                  <div className="w-1/5 relative flex items-center justify-center bg-black/10">
                    <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0">
                      {connectingPathways.map((pw, i) => {
                        const isHovered = hoveredPathKey === pw._key;
                        const sourceLock = lockSet.has(pw.src) ? pw.src : pw.tgt;
                        const y = 8 + (i * 84 / Math.max(1, connectingPathways.length - 1));
                        
                        // Log scaling per edge (handles 60k outliers)
                        const c = pw.coupling;
                        const logScale = Math.log10(c + 1) / Math.log10(maxCoupling + 1);
                        const sw = isHovered ? 1.2 + logScale * 6 : 0.5 + logScale * 3.5;
                        const op = isHovered ? 0.95 : 0.2 + logScale * 0.55;
                        const strokeCol = isHovered ? "#fbbf24" : "#f59e0b";

                        return (
                          <path 
                            key={pw._key} 
                            d={`M5,${y} Q32,${52 + (i % 5 - 2) * 5} 95,50`} 
                            stroke={strokeCol} 
                            strokeWidth={sw} 
                            strokeOpacity={op} 
                            fill="none" 
                            strokeLinecap="round"
                            style={{ transition: 'all 0.12s ease' }}
                          />
                        );
                      })}
                    </svg>
                    <div className="relative text-[6px] text-amber-300/60 font-mono z-10">paths</div>
                  </div>

                  {/* Right: Target pocket (green) */}
                  <div className="w-2/5 flex items-center justify-center bg-black/10">
                    <div 
                      className="px-2 py-1 bg-emerald-900/40 text-emerald-200 rounded font-mono text-[8px] border border-emerald-600/60 cursor-pointer hover:bg-emerald-800/50 text-center leading-tight"
                      onClick={() => {
                        const ids = pocket?.residue_ids || [];
                        setHighlightedResidues(ids);
                        emitDirective({ action: "highlight", highlight_groups: [{ residue_ids: ids, color: "#4ade80", style: "glow", label: `P${selectedPocketId} target` }] });
                      }}>
                      P{selectedPocketId} TARGET<br/>
                      <span className="text-[6px] opacity-70">{connectingPathways.length} paths • {(pocket?.connecting_coupling_sum || 0).toFixed(0)}</span>
                    </div>
                  </div>
                </div>
                {/* Per-pathway receipt table - sortable by coupling, hover isolates specific curve in SVG + 3D highlight for that lock→pocket edge */}
                {connectingPathways.length > 0 && (
                  <div className="mt-1 max-h-20 overflow-auto border border-zinc-700 rounded bg-zinc-950/70">
                    <table className="w-full text-[7px]">
                      <thead className="bg-zinc-900 sticky top-0">
                        <tr>
                          <th className="px-1 py-0 text-left text-zinc-400">Lock (Source)</th>
                          <th className="px-1 py-0 text-left text-zinc-400">Pocket Residue</th>
                          <th className="px-1 py-0 text-right text-zinc-400">Hops</th>
                          <th className="px-1 py-0 text-right text-zinc-400">Coupling</th>
                        </tr>
                      </thead>
                      <tbody>
                        {connectingPathways.map((pw) => {
                          const srcLock = lockSet.has(pw.src) ? pw.src : pw.tgt;
                          const tgtRes = lockSet.has(pw.src) ? pw.tgt : pw.src;
                          const isHovered = hoveredPathKey === pw._key;
                          return (
                            <tr 
                              key={pw._key}
                              className={`border-t border-zinc-800/40 ${isHovered ? 'bg-amber-900/30' : 'hover:bg-zinc-800/40'}`}
                              onMouseEnter={() => setHoveredPathKey(pw._key)}
                              onMouseLeave={() => setHoveredPathKey(null)}
                              onClick={() => {
                                setHighlightedResidues([srcLock, tgtRes]);
                                emitDirective({
                                  action: "highlight",
                                  highlight_groups: [
                                    { residue_ids: [srcLock], color: "#f59e0b", style: "glow", label: `Lock ${srcLock}` },
                                    { residue_ids: [tgtRes], color: "#4ade80", style: "glow", label: `Pocket res ${tgtRes}` },
                                  ],
                                });
                              }}
                            >
                              <td className="px-1 py-0 font-mono text-amber-200">{srcLock}</td>
                              <td className="px-1 py-0 font-mono text-emerald-200">{tgtRes}</td>
                              <td className="px-1 py-0 text-right text-zinc-400">1</td>
                              <td className="px-1 py-0 text-right font-medium text-amber-300">{pw.coupling.toFixed(0)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                <div className="text-[9px] text-zinc-400">
                  {connectingPathways.length} connecting pathways • total coupling {(pocket?.connecting_coupling_sum || 0).toFixed(0) || 'N/A'}
                </div>
                <button
                  onClick={() => {
                    const ids = [...displayLocks, ... (pocket?.residue_ids || [])];
                    setHighlightedResidues(ids);
                    emitDirective({
                      action: "highlight",
                      highlight_groups: [
                        { residue_ids: displayLocks, color: "#f59e0b", style: "glow", label: "Precision Locks (amber network)" },
                        { residue_ids: pocket?.residue_ids || [], color: "#4ade80", style: "glow", label: `Pocket ${selectedPocketId} (target)` },
                      ],
                    });
                  }}
                  className="px-2 py-0.5 bg-amber-700/30 border border-amber-600/40 rounded text-[9px] text-amber-300 hover:bg-amber-700/50"
                >
                  Highlight Locks + Pocket in 3D (glowing channels)
                </button>
              </div>
            );
          })()
        ) : (
          // Macro / Fallback view
          <div className="text-zinc-400">
            <div>Precision Locks (the 23 allosteric network): NMP domain (A:35/36/43), Hinges (A:102/107), LID (A:129/130/157/171/187 + B counterparts).</div>
            <div className="mt-1 text-[9px]">Select a pocket (e.g. in Inspector) to see its pocket-specific conductive sub-graph (locks + connecting resistance pathways scaled by coupling).</div>
            <div className="mt-1 text-emerald-400 text-[9px]">Top coupled pockets typically show strong remote control from the locks (see P3 with 60k coupling).</div>
          </div>
        )}
        <p className="text-[8px] text-zinc-500">Data-driven from hydrated phase5 (connected_locks) + phase4 pathways. Amber = active network; click to inspect/3D highlight.</p>
      </div>
    </div>
  );
}
