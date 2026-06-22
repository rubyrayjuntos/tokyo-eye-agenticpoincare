/**
 * ResultsTable — discovery ledger showing ingested structures.
 * Requirements: 2.4, 3.2, 3.5
 */
import { useEffect, useState, useCallback } from "react";
import { api } from "../lib/api";
import { useDashboard } from "../lib/context";
import type { Structure } from "../lib/types";

export default function ResultsTable() {
  const { activeStructure, setActiveStructure, refreshKey, enterCompareMode } = useDashboard();
  const [structures, setStructures] = useState<Structure[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchStructures = useCallback(() => {
    api
      .getStructures()
      .then((data) => {
        setStructures(data);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  // Initial load + refresh on refreshKey change
  useEffect(() => {
    fetchStructures();
  }, [fetchStructures, refreshKey]);

  const handleRowClick = (structure: Structure) => {
    setActiveStructure(structure);
  };

  const getStatusBadge = (structure: Structure) => {
    if (structure.has_embeddings) {
      return (
        <span className="px-1.5 py-0.5 text-[10px] rounded bg-emerald-900/30 text-emerald-400">
          Complete
        </span>
      );
    }
    if (structure.last_run_id) {
      return (
        <span className="px-1.5 py-0.5 text-[10px] rounded bg-cyan-900/30 text-cyan-400">
          Processing
        </span>
      );
    }
    return (
      <span className="px-1.5 py-0.5 text-[10px] rounded bg-zinc-800 text-zinc-500">
        Ingested
      </span>
    );
  };

  return (
    <div className="border border-zinc-800 rounded-lg bg-zinc-900/40 overflow-hidden">
      <div className="px-4 py-2 border-b border-zinc-800 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Discovery Ledger
        </h3>
        <div className="flex items-center gap-2">
          {activeStructure && (
            <span className="text-[10px] text-cyan-400">
              Active: {activeStructure.pdb_id}
            </span>
          )}
          <span className="text-[10px] text-zinc-600">
            {structures.length} structure{structures.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800 text-zinc-500 text-left">
              <th className="px-4 py-2 font-medium">PDB ID</th>
              <th className="px-4 py-2 font-medium">Title</th>
              <th className="px-4 py-2 font-medium">Residues</th>
              <th className="px-4 py-2 font-medium">Status</th>
              <th className="px-4 py-2 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={5} className="px-4 py-6 text-center text-zinc-600">
                  Loading structures...
                </td>
              </tr>
            )}

            {!loading && structures.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-6 text-center text-zinc-600">
                  No structures ingested yet. Use Pipeline Controls to ingest a
                  PDB.
                </td>
              </tr>
            )}

            {structures.map((s) => {
              const isActive =
                activeStructure?.structure_id === s.structure_id;
              return (
                <tr
                  key={s.structure_id}
                  onClick={() => handleRowClick(s)}
                  className={`border-b border-zinc-800/50 cursor-pointer transition-colors ${
                    isActive
                      ? "bg-cyan-900/10 border-l-2 border-l-cyan-500"
                      : "hover:bg-zinc-800/30"
                  }`}
                >
                  <td className="px-4 py-2 font-mono text-cyan-300">
                    {s.pdb_id}
                  </td>
                  <td className="px-4 py-2 text-zinc-300 max-w-[200px] truncate">
                    {s.title || "—"}
                  </td>
                  <td className="px-4 py-2 tabular-nums text-zinc-400">
                    {s.residue_count}
                  </td>
                  <td className="px-4 py-2">{getStatusBadge(s)}</td>
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-1.5">
                      {s.has_embeddings && (
                        <>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRowClick(s);
                            }}
                            title="View 2D embeddings"
                            className="px-1.5 py-0.5 text-[10px] rounded bg-zinc-800 text-zinc-400 hover:text-cyan-300 hover:bg-zinc-700"
                          >
                            2D
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRowClick(s);
                            }}
                            title="View 3D structure"
                            className="px-1.5 py-0.5 text-[10px] rounded bg-zinc-800 text-zinc-400 hover:text-cyan-300 hover:bg-zinc-700"
                          >
                            3D
                          </button>
                        </>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          enterCompareMode(s);
                        }}
                        disabled={isActive || !s.has_embeddings || !activeStructure?.has_embeddings}
                        title={
                          !activeStructure
                            ? "Select a structure first"
                            : isActive
                            ? "Cannot compare with itself"
                            : !s.has_embeddings
                            ? "No embeddings available"
                            : !activeStructure.has_embeddings
                            ? "Active structure has no embeddings"
                            : `Compare with ${activeStructure.pdb_id}`
                        }
                        className="px-1.5 py-0.5 text-[10px] rounded bg-zinc-800 text-zinc-400 hover:text-amber-300 hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:text-zinc-400 disabled:hover:bg-zinc-800"
                      >
                        ⇔
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
