/**
 * RCSBSearchPanel — 4-mode RCSB search with results cards.
 * Requirements: 2.1, 2.5, 2.6, 2.7
 */
import { useState, useCallback } from "react";
import { Loader2 } from "lucide-react";
import { api } from "../../lib/api";
import { useDashboard } from "../../lib/context";
import { useAutoIngestPipeline } from "../../lib/useAutoIngestPipeline";
import { supplementaryReadinessErrors } from "../../lib/readinessPollingLogic";
import { DiscoveryStoryActRail } from "../onboard/DiscoveryStoryActRail";
import type { RCSBSearchRequest, RCSBSearchResult, Structure } from "../../lib/types";

type SearchMode = "text" | "sequence" | "structure" | "functional";

const MODES: { id: SearchMode; label: string }[] = [
  { id: "text", label: "Text" },
  { id: "sequence", label: "Sequence" },
  { id: "structure", label: "Structure" },
  { id: "functional", label: "Functional" },
];

export interface RCSBSearchPanelProps {
  onStructureLoaded?: (structure: Structure) => void;
  className?: string;
}

export default function RCSBSearchPanel({
  onStructureLoaded,
  className = "",
}: RCSBSearchPanelProps = {}) {
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

  const applyStructure = useCallback(
    (structure: Structure) => {
      if (onStructureLoaded) {
        onStructureLoaded(structure);
      } else {
        setActiveStructure(structure);
      }
    },
    [onStructureLoaded, setActiveStructure],
  );

  const {
    ingestAndRun,
    isIngesting,
    isRunningPipeline,
    progressLabel,
    readiness,
    error: ingestError,
  } = useAutoIngestPipeline({
    onComplete: (structure) => {
      applyStructure(structure);
      setIngestingId(null);
    },
    onError: (message) => {
      setError(message);
      setIngestingId(null);
    },
  });

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
      setError(null);
      await ingestAndRun(pdbId);
    },
    [ingestAndRun],
  );

  const pipelineBusy = isIngesting || isRunningPipeline;

  return (
    <div className={`flex h-full flex-col ${className}`}>
      <div className="flex border-b border-slate">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setMode(m.id)}
            className={`flex-1 px-2 py-2 text-[11px] font-medium transition-colors ${
              mode === m.id
                ? "border-b-2 border-teal bg-teal-dim/10 text-teal"
                : "text-text-muted hover:text-text-secondary"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="space-y-2 border-b border-slate px-3 py-3">
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
          disabled={loading || pipelineBusy}
          className="w-full rounded-[var(--radius-button)] border border-slate-light bg-bg-elevated px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-teal focus:outline-none disabled:opacity-50"
        />

        {(mode === "text" || mode === "functional") && (
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Organism"
              value={organism}
              onChange={(e) => setOrganism(e.target.value)}
              disabled={loading || pipelineBusy}
              className="flex-1 rounded-[var(--radius-button)] border border-slate-light bg-bg-elevated px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-teal focus:outline-none disabled:opacity-50"
            />
            <input
              type="number"
              placeholder="Max Å"
              value={maxResolution}
              onChange={(e) => setMaxResolution(e.target.value)}
              disabled={loading || pipelineBusy}
              step="0.1"
              className="w-16 rounded-[var(--radius-button)] border border-slate-light bg-bg-elevated px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-teal focus:outline-none disabled:opacity-50"
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
              disabled={loading || pipelineBusy}
              step="1"
              min="0"
              max="100"
              className="flex-1 rounded-[var(--radius-button)] border border-slate-light bg-bg-elevated px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-teal focus:outline-none disabled:opacity-50"
            />
            <input
              type="number"
              placeholder="E-value"
              value={evalCutoff}
              onChange={(e) => setEvalCutoff(e.target.value)}
              disabled={loading || pipelineBusy}
              step="0.001"
              className="flex-1 rounded-[var(--radius-button)] border border-slate-light bg-bg-elevated px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-teal focus:outline-none disabled:opacity-50"
            />
          </div>
        )}

        <div className="flex gap-2">
          <input
            type="number"
            placeholder="Max results"
            value={maxResults}
            onChange={(e) => setMaxResults(e.target.value)}
            disabled={loading || pipelineBusy}
            min="1"
            max="100"
            className="w-20 rounded-[var(--radius-button)] border border-slate-light bg-bg-elevated px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-teal focus:outline-none disabled:opacity-50"
          />
          <button
            type="button"
            onClick={handleSearch}
            disabled={loading || pipelineBusy || !query.trim()}
            className="flex-1 rounded-[var(--radius-button)] border border-teal-dim/40 bg-teal-dim/20 py-1.5 text-xs font-medium text-teal hover:bg-teal-dim/35 disabled:opacity-40"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {(error || ingestError) && (
        <div className="border-b border-error/30 bg-error/10 px-3 py-2 space-y-1">
          <p className="text-[11px] text-error">{error ?? ingestError}</p>
          {ingestError && readiness
            ? supplementaryReadinessErrors(readiness).map((line) => (
                <p key={line} className="text-[10px] text-error/80 pl-2 border-l border-error/30">
                  {line}
                </p>
              ))
            : null}
        </div>
      )}

      {pipelineBusy && (
        <div className="flex flex-col gap-2 border-b border-teal-dim/30 bg-teal-dim/10 px-3 py-2">
          <div className="flex items-center gap-2">
            <Loader2 size={14} className="animate-spin text-teal" />
            <span className="text-xs text-teal">
              {isIngesting
                ? `Ingesting ${ingestingId?.toUpperCase() ?? "structure"}…`
                : `Running discovery pipeline for ${ingestingId?.toUpperCase() ?? "structure"}…`}
            </span>
          </div>
          {progressLabel ? (
            <span className="pl-6 text-[10px] text-text-muted">{progressLabel}</span>
          ) : null}
          {readiness?.pipeline_job?.error ? (
            <p className="pl-6 text-[10px] text-error">{readiness.pipeline_job.error}</p>
          ) : null}
          <DiscoveryStoryActRail
            acts={readiness?.acts}
            currentAct={readiness?.current_act}
          />
        </div>
      )}

      {loading && (
        <div className="flex flex-1 items-center justify-center">
          <div className="animate-pulse text-xs text-text-muted">Searching RCSB…</div>
        </div>
      )}

      {!loading && results.length > 0 && (
        <div className="flex-1 space-y-2 overflow-y-auto px-3 py-2">
          {results.map((r) => (
            <div
              key={r.pdb_id}
              className="space-y-1 rounded-[var(--radius-card)] border border-slate-light bg-bg-elevated/60 p-2.5"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium text-text-primary">
                    {r.pdb_id} — {r.title}
                  </p>
                  <div className="mt-0.5 flex gap-3">
                    {r.resolution != null && (
                      <span className="text-[10px] text-text-muted">
                        {r.resolution.toFixed(1)} Å
                      </span>
                    )}
                    {r.method && (
                      <span className="text-[10px] text-text-muted">{r.method}</span>
                    )}
                    {r.organism && (
                      <span className="text-[10px] text-text-muted">{r.organism}</span>
                    )}
                    {r.similarity_score != null && (
                      <span className="text-[10px] text-teal">
                        {mode === "sequence"
                          ? `${r.similarity_score.toFixed(1)}% identity`
                          : `Score: ${r.similarity_score.toFixed(3)}`}
                      </span>
                    )}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => handleIngest(r.pdb_id)}
                  disabled={pipelineBusy}
                  className="shrink-0 rounded-[var(--radius-badge)] border border-teal-dim/40 bg-teal-dim/20 px-2 py-1 text-[10px] text-teal hover:bg-teal-dim/35 disabled:opacity-40"
                >
                  {ingestingId === r.pdb_id && pipelineBusy ? "…" : "Load & run"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && !error && results.length === 0 && query.trim() && (
        <div className="flex flex-1 items-center justify-center">
          <p className="text-xs text-text-muted">No results yet. Hit Search.</p>
        </div>
      )}

      {!loading && !error && results.length === 0 && !query.trim() && (
        <div className="flex flex-1 items-center justify-center px-4">
          <p className="text-center text-xs text-text-muted">
            Search RCSB by keyword, sequence, structure similarity, or functional annotation.
            Ingest runs the full downstream discovery pipeline automatically.
          </p>
        </div>
      )}
    </div>
  );
}
