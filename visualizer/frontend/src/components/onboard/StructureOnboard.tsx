import { useState, useEffect, useCallback } from "react";
import { Search, Loader2, AlertCircle, Check } from "lucide-react";
import { api } from "../../lib/api";
import type { Structure, IngestResponse } from "../../lib/types";

/**
 * StructureOnboard — PDB ID input, ingestion progress, quick-select
 *
 * Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
 */

export interface StructureOnboardProps {
  onStructureLoaded: (structure: Structure) => void;
}

type IngestionState =
  | { status: "idle" }
  | { status: "validating" }
  | { status: "ingesting"; pdbId: string }
  | { status: "complete"; response: IngestResponse }
  | { status: "error"; message: string };

/** Validates PDB ID format: 4-character alphanumeric */
function isValidPdbId(value: string): boolean {
  return /^[A-Za-z0-9]{4}$/.test(value);
}

export function StructureOnboard({ onStructureLoaded }: StructureOnboardProps) {
  const [pdbInput, setPdbInput] = useState("");
  const [ingestionState, setIngestionState] = useState<IngestionState>({ status: "idle" });
  const [previousStructures, setPreviousStructures] = useState<Structure[]>([]);
  const [loadingList, setLoadingList] = useState(true);

  // Load previously ingested structures
  useEffect(() => {
    api.getStructures()
      .then((structures) => {
        setPreviousStructures(structures);
        setLoadingList(false);
      })
      .catch(() => setLoadingList(false));
  }, []);

  const handleSubmit = useCallback(async () => {
    const trimmed = pdbInput.trim().toUpperCase();

    if (!isValidPdbId(trimmed)) {
      setIngestionState({ status: "error", message: "PDB ID must be exactly 4 alphanumeric characters" });
      return;
    }

    setIngestionState({ status: "ingesting", pdbId: trimmed });

    try {
      const response = await api.ingest(trimmed);
      setIngestionState({ status: "complete", response });

      // Build a Structure object from the ingest response and load it
      const structure: Structure = {
        structure_id: response.structure_id,
        pdb_id: response.pdb_id,
        title: response.title,
        resolution: null,
        method: "",
        source: "rcsb",
        chains: response.chains,
        residue_count: response.residue_count,
        has_embeddings: false,
        last_run_id: null,
        ingested_at: new Date().toISOString(),
      };

      onStructureLoaded(structure);

      // Add to the local list
      setPreviousStructures((prev) => [structure, ...prev.filter((s) => s.structure_id !== structure.structure_id)]);
      setPdbInput("");

      // Reset state after brief delay
      setTimeout(() => setIngestionState({ status: "idle" }), 2000);
    } catch (e: any) {
      const message = e?.message?.includes("404")
        ? `PDB ID "${trimmed}" not found in RCSB`
        : e?.message || "Ingestion failed";
      setIngestionState({ status: "error", message });
    }
  }, [pdbInput, onStructureLoaded]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleQuickSelect = useCallback(
    (structure: Structure) => {
      onStructureLoaded(structure);
    },
    [onStructureLoaded],
  );

  const isIngesting = ingestionState.status === "ingesting";

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
      {/* PDB ID input */}
      <div>
        <label className="text-[10px] uppercase tracking-wider text-text-muted mb-1 block">
          Load Structure by PDB ID
        </label>
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              type="text"
              value={pdbInput}
              onChange={(e) => {
                setPdbInput(e.target.value);
                if (ingestionState.status === "error") setIngestionState({ status: "idle" });
              }}
              onKeyDown={handleKeyDown}
              placeholder="e.g. 4OBE"
              maxLength={4}
              disabled={isIngesting}
              className="w-full bg-bg-elevated border border-slate-light rounded-[var(--radius-button)] pl-7 pr-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-teal disabled:opacity-50 font-mono uppercase"
            />
          </div>
          <button
            onClick={handleSubmit}
            disabled={isIngesting || !pdbInput.trim()}
            className="px-3 py-1.5 rounded-[var(--radius-button)] bg-teal-dim/30 text-teal text-xs hover:bg-teal-dim/50 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            {isIngesting ? <Loader2 size={12} className="animate-spin" /> : "Load"}
          </button>
        </div>
      </div>

      {/* Progress / status */}
      {ingestionState.status === "ingesting" && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-card)] bg-teal-dim/10 border border-teal-dim/30">
          <Loader2 size={14} className="text-teal animate-spin" />
          <span className="text-xs text-teal">
            Ingesting {ingestionState.pdbId}…
          </span>
        </div>
      )}

      {ingestionState.status === "complete" && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-card)] bg-success/10 border border-success/30">
          <Check size={14} className="text-success" />
          <span className="text-xs text-success">
            {ingestionState.response.pdb_id.toUpperCase()} loaded — {ingestionState.response.residue_count} residues
          </span>
        </div>
      )}

      {ingestionState.status === "error" && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-card)] bg-error/10 border border-error/30">
          <AlertCircle size={14} className="text-error" />
          <span className="text-xs text-error">{ingestionState.message}</span>
        </div>
      )}

      {/* Previously ingested structures */}
      <div className="mt-2">
        <label className="text-[10px] uppercase tracking-wider text-text-muted mb-2 block">
          Previously Loaded
        </label>

        {loadingList ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 size={14} className="text-text-muted animate-spin" />
          </div>
        ) : previousStructures.length === 0 ? (
          <div className="text-xs text-text-muted text-center py-4">
            No structures loaded yet
          </div>
        ) : (
          <div className="space-y-1">
            {previousStructures.map((s) => (
              <button
                key={s.structure_id}
                onClick={() => handleQuickSelect(s)}
                className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-[var(--radius-button)] text-left hover:bg-slate/50 border border-transparent hover:border-slate-light transition-colors group"
              >
                <span className="text-xs font-mono text-teal-bright group-hover:text-teal">
                  {s.pdb_id.toUpperCase()}
                </span>
                <span className="text-[10px] text-text-muted truncate flex-1">
                  {s.title || s.structure_id}
                </span>
                <span className="text-[9px] text-text-muted">
                  {s.residue_count}r
                </span>
                {s.has_embeddings && (
                  <span className="w-1.5 h-1.5 rounded-full bg-success shrink-0" title="Has embeddings" />
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
