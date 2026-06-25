/**
 * RCSBSearchPanel — 4-mode RCSB search with results cards.
 * Requirements: 2.1, 2.5, 2.6, 2.7
 */
import { useState, useCallback } from "react";
import { api } from "../../lib/api";
import { useDashboard } from "../../lib/context";
import type { RCSBSearchRequest, RCSBSearchResult, Structure } from "../../lib/types";

type SearchMode = "text" | "sequence" | "structure" | "functional";

const MODES: { id: SearchMode; label: string }[] = [
  { id: "text", label: "Text" },
  { id: "sequence", label: "Sequence" },
  { id: "structure", label: "Structure" },
  { id: "functional", label: "Functional" },
];

export default function RCSBSearchPanel() {
  const { setActiveStructure } = useDashboard();

  const [mode, setMode] = useState<SearchMode>("text");
  const [query, setQuery] = useState("");
  const [organism, setOrganism] = useState("");
  const [maxResolution, setMaxResolution] = useState("");
  const [minIdentity, setMinIdentity] = useState("");
  const [evalCutoff, setEvalCutoff] = useState("");
  const [maxResults, setMaxResults] = useState("10");

  const [results, setResults] = useState<RCSBSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingestingId, setIngestingId] = useState<string | null>(null);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResults([]);

    const req: RCSBSearchRequest = {
      mode,
      query: query.trim(),
      max_results: maxResults ? parseInt(maxResults, 10) : undefined,
    };

    if (mode === "text" || mode === "functional") {
      if (organism.trim()) req.organism = organism.trim();
      if (maxResolution) req.max_resolution = parseFloat(maxResolution);
    }

    if (mode === "sequence") {
      if (minIdentity) req.min_identity = parseFloat(minIdentity);
      if (evalCutoff) req.evalue_cutoff = parseFloat(evalCutoff);
    }

    try {
      let data: RCSBSearchResult[];
      if (mode === "sequence") {
        data = await api.rcsbSequenceSearch(req);
      } else if (mode === "structure") {
        data = await api.rcsbStructureSearch(req);
      } else {
        data = await api.rcsbSearch(req);
      }
      setResults(data);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Search failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [mode, query, organism, maxResolution, minIdentity, evalCutoff, maxResults]);

  const handleIngest = useCallback(
    async (pdbId: string) => {
      setIngestingId(pdbId);
      try {
        const result = await api.ingest(pdbId);
        const newStructure: Structure = {
          structure_id: result.structure_id,
          pdb_id: result.pdb_id,
          title: result.title,
          resolution: null,
          method: "",
          source: "rcsb",
          chains: result.chains,
          residue_count: result.residue_count,
          has_embeddings: false,
          last_run_id: null,
          ingested_at: new Date().toISOString(),
        };
        setActiveStructure(newStructure);
      } catch {
        // Ingest error is non-critical for the panel
      } finally {
        setIngestingId(null);
      }
    },
    [setActiveStructure]
  );

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex border-b border-zinc-800">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`flex-1 px-2 py-2 text-[11px] font-medium transition-colors ${
              mode === m.id
                ? "text-cyan-300 border-b-2 border-cyan-500 bg-zinc-800/40"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Search form */}
      <div className="px-3 py-3 space-y-2 border-b border-zinc-800">
        {/* Query input */}
        <input
          type="text"
          placeholder={
            mode === "text"
              ? "Keyword (e.g. KRAS, kinase)"
              : mode === "sequence"
              ? "FASTA sequence or PDB chain (e.g. 4OBE_A)"
              : mode === "structure"
              ? "PDB ID (e.g. 4OBE)"
              : "GO term, EC number, or Pfam domain"
          }
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSearch();
          }}
          disabled={loading}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
        />

        {/* Mode-specific filters */}
        {(mode === "text" || mode === "functional") && (
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Organism"
              value={organism}
              onChange={(e) => setOrganism(e.target.value)}
              disabled={loading}
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
            />
            <input
              type="number"
              placeholder="Max Å"
              value={maxResolution}
              onChange={(e) => setMaxResolution(e.target.value)}
              disabled={loading}
              step="0.1"
              className="w-16 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
            />
          </div>
        )}

        {mode === "sequence" && (
          <div className="flex gap-2">
            <input
              type="number"
              placeholder="Min identity %"
              value={minIdentity}
              onChange={(e) => setMinIdentity(e.target.value)}
              disabled={loading}
              step="1"
              min="0"
              max="100"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
            />
            <input
              type="number"
              placeholder="E-value"
              value={evalCutoff}
              onChange={(e) => setEvalCutoff(e.target.value)}
              disabled={loading}
              step="0.001"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
            />
          </div>
        )}

        {/* Max results + search button */}
        <div className="flex gap-2">
          <input
            type="number"
            placeholder="Max results"
            value={maxResults}
            onChange={(e) => setMaxResults(e.target.value)}
            disabled={loading}
            min="1"
            max="100"
            className="w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
          />
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="flex-1 py-1.5 bg-cyan-700/30 border border-cyan-600/40 rounded text-xs font-medium text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-40"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="px-3 py-2 bg-red-900/20 border-b border-red-800/40">
          <p className="text-[11px] text-red-400">{error}</p>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-xs text-zinc-500 animate-pulse">
            Searching RCSB...
          </div>
        </div>
      )}

      {/* Results */}
      {!loading && results.length > 0 && (
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
          {results.map((r) => (
            <div
              key={r.pdb_id}
              className="bg-zinc-800/60 border border-zinc-700/50 rounded p-2.5 space-y-1"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-zinc-200 truncate">
                    {r.pdb_id} — {r.title}
                  </p>
                  <div className="flex gap-3 mt-0.5">
                    {r.resolution != null && (
                      <span className="text-[10px] text-zinc-500">
                        {r.resolution.toFixed(1)} Å
                      </span>
                    )}
                    {r.method && (
                      <span className="text-[10px] text-zinc-500">
                        {r.method}
                      </span>
                    )}
                    {r.organism && (
                      <span className="text-[10px] text-zinc-500">
                        {r.organism}
                      </span>
                    )}
                    {r.similarity_score != null && (
                      <span className="text-[10px] text-cyan-400">
                        {mode === "sequence"
                          ? `${r.similarity_score.toFixed(1)}% identity`
                          : `Score: ${r.similarity_score.toFixed(3)}`}
                      </span>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => handleIngest(r.pdb_id)}
                  disabled={ingestingId === r.pdb_id}
                  className="shrink-0 px-2 py-1 bg-emerald-700/30 border border-emerald-600/40 rounded text-[10px] text-emerald-300 hover:bg-emerald-700/50 disabled:opacity-40"
                >
                  {ingestingId === r.pdb_id ? "..." : "Ingest"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && results.length === 0 && query.trim() && (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-xs text-zinc-600">No results yet. Hit Search.</p>
        </div>
      )}

      {/* Initial state */}
      {!loading && !error && results.length === 0 && !query.trim() && (
        <div className="flex-1 flex items-center justify-center px-4">
          <p className="text-xs text-zinc-600 text-center">
            Search RCSB by keyword, sequence, structure similarity, or functional annotation.
          </p>
        </div>
      )}
    </div>
  );
}
