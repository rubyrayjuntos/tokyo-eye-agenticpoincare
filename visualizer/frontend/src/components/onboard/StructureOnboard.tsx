import { useState, useEffect, useCallback, useRef } from "react";
import { Search, Loader2, AlertCircle, Check } from "lucide-react";
import { api } from "../../lib/api";
import { DiscoveryStoryActRail } from "./DiscoveryStoryActRail";
import type { Structure, IngestResponse, StructureReadiness } from "../../lib/types";
import {
  actProgressLabel,
  errorMessageFromUnknown,
  MAX_READINESS_POLL_FAILURES,
  READINESS_POLL_START_LABEL,
  runReadinessPollCycle,
  supplementaryReadinessErrors,
} from "../../lib/readinessPollingLogic";

/**
 * StructureOnboard — PDB ID input, ingestion + discovery compute progress.
 */

export interface StructureOnboardProps {
  onStructureLoaded: (structure: Structure) => void;
}

type IngestionState =
  | { status: "idle" }
  | { status: "validating" }
  | { status: "ingesting"; pdbId: string }
  | {
      status: "computing";
      pdbId: string;
      structureId: string;
      label: string;
      readiness?: StructureReadiness;
    }
  | { status: "complete"; response: IngestResponse }
  | { status: "error"; message: string; details?: string[] };

function isValidPdbId(value: string): boolean {
  return /^[A-Za-z0-9]{4}$/.test(value);
}

const POLL_INTERVAL_MS = 2000;

export function StructureOnboard({ onStructureLoaded }: StructureOnboardProps) {
  const [pdbInput, setPdbInput] = useState("");
  const [ingestionState, setIngestionState] = useState<IngestionState>({ status: "idle" });
  const [previousStructures, setPreviousStructures] = useState<Structure[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const structureRef = useRef<Structure | null>(null);
  const pipelineJobIdRef = useRef<string | null>(null);
  const pollFailureCountRef = useRef(0);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  useEffect(() => {
    api.getStructures()
      .then((structures) => {
        setPreviousStructures(structures);
        setLoadingList(false);
      })
      .catch(() => setLoadingList(false));
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startReadinessPolling = useCallback(
    (structure: Structure, pdbId: string, pipelineJobId?: string | null) => {
      structureRef.current = structure;
      pipelineJobIdRef.current = pipelineJobId ?? null;
      pollFailureCountRef.current = 0;
      setIngestionState({
        status: "computing",
        pdbId,
        structureId: structure.structure_id,
        label: READINESS_POLL_START_LABEL,
      });

      const poll = async () => {
        const structureSnapshot = structureRef.current;
        if (!structureSnapshot) return;

        try {
          const result = await runReadinessPollCycle({
            structure: structureSnapshot,
            getReadiness: () => api.getStructureReadiness(structureSnapshot.structure_id),
            getPipelineStatus: (jobId) => api.getPipelineStatus(jobId),
            pipelineJobId: pipelineJobIdRef.current,
          });

          pollFailureCountRef.current = 0;

          if (result.type === "continue") {
            const label = actProgressLabel(result.readiness) ?? "Discovery compute running…";
            setIngestionState((prev) =>
              prev.status === "computing"
                ? { ...prev, label, readiness: result.readiness }
                : prev
            );
            return;
          }

          if (result.type === "complete") {
            stopPolling();
            onStructureLoaded(result.structure);
            setPreviousStructures((prev) => [
              result.structure,
              ...prev.filter((s) => s.structure_id !== result.structure.structure_id),
            ]);
            setIngestionState({
              status: "complete",
              response: {
                structure_id: result.structure.structure_id,
                pdb_id: result.structure.pdb_id,
                residue_count: result.structure.residue_count,
                chains: result.structure.chains,
              },
            });
            setPdbInput("");
            setTimeout(() => setIngestionState({ status: "idle" }), 2500);
            return;
          }

          stopPolling();
          setIngestionState({
            status: "error",
            message: result.error,
            details: result.readiness
              ? supplementaryReadinessErrors(result.readiness)
              : undefined,
          });
        } catch (err: unknown) {
          pollFailureCountRef.current += 1;
          if (pollFailureCountRef.current >= MAX_READINESS_POLL_FAILURES) {
            stopPolling();
            setIngestionState({
              status: "error",
              message: errorMessageFromUnknown(err),
            });
          }
        }
      };

      pollRef.current = setInterval(() => {
        void poll();
      }, POLL_INTERVAL_MS);
      void poll();
    },
    [onStructureLoaded, stopPolling]
  );

  const handleSubmit = useCallback(async () => {
    const trimmed = pdbInput.trim().toUpperCase();

    if (!isValidPdbId(trimmed)) {
      setIngestionState({
        status: "error",
        message: "PDB ID must be exactly 4 alphanumeric characters",
      });
      return;
    }

    stopPolling();
    setIngestionState({ status: "ingesting", pdbId: trimmed });

    try {
      const response = await api.ingest(trimmed);

      const structure: Structure = {
        structure_id: response.structure_id,
        pdb_id: response.pdb_id,
        title: response.title ?? response.pdb_id,
        resolution: null,
        method: "",
        source: "rcsb",
        chains: response.chains,
        residue_count: response.residue_count,
        has_embeddings: false,
        last_run_id: response.pipeline_job_id,
        ingested_at: new Date().toISOString(),
      };

      if (response.pipeline_status === "skipped" && !response.pipeline_job_id) {
        onStructureLoaded(structure);
        setPreviousStructures((prev) => [
          structure,
          ...prev.filter((s) => s.structure_id !== structure.structure_id),
        ]);
        setIngestionState({ status: "complete", response });
        setPdbInput("");
        setTimeout(() => setIngestionState({ status: "idle" }), 2000);
        return;
      }

      startReadinessPolling(structure, trimmed, response.pipeline_job_id);
    } catch (e: any) {
      const message = e?.message?.includes("404")
        ? `PDB ID "${trimmed}" not found in RCSB`
        : e?.message || "Ingestion failed";
      setIngestionState({ status: "error", message });
    }
  }, [pdbInput, onStructureLoaded, startReadinessPolling, stopPolling]);

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
    [onStructureLoaded]
  );

  const isBusy =
    ingestionState.status === "ingesting" || ingestionState.status === "computing";

  return (
    <div className="flex flex-col gap-4 p-4 h-full overflow-y-auto">
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
              disabled={isBusy}
              className="w-full bg-bg-elevated border border-slate-light rounded-[var(--radius-button)] pl-7 pr-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-teal disabled:opacity-50 font-mono uppercase"
            />
          </div>
          <button
            onClick={handleSubmit}
            disabled={isBusy || !pdbInput.trim()}
            className="px-3 py-1.5 rounded-[var(--radius-button)] bg-teal-dim/30 text-teal text-xs hover:bg-teal-dim/50 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            {isBusy ? <Loader2 size={12} className="animate-spin" /> : "Load"}
          </button>
        </div>
      </div>

      {ingestionState.status === "ingesting" && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-card)] bg-teal-dim/10 border border-teal-dim/30">
          <Loader2 size={14} className="text-teal animate-spin" />
          <span className="text-xs text-teal">Ingesting {ingestionState.pdbId}…</span>
        </div>
      )}

      {ingestionState.status === "computing" && (
        <div className="flex flex-col gap-2 px-3 py-2 rounded-[var(--radius-card)] bg-teal-dim/10 border border-teal-dim/30">
          <div className="flex items-center gap-2">
            <Loader2 size={14} className="text-teal animate-spin" />
            <span className="text-xs text-teal">
              Computing discovery path for {ingestionState.pdbId}…
            </span>
          </div>
          <span className="text-[10px] text-text-muted pl-6">{ingestionState.label}</span>
          {ingestionState.readiness?.pipeline_job?.error ? (
            <p className="pl-6 text-[10px] text-error">{ingestionState.readiness.pipeline_job.error}</p>
          ) : null}
          <DiscoveryStoryActRail
            acts={ingestionState.readiness?.acts}
            currentAct={ingestionState.readiness?.current_act}
          />
        </div>
      )}

      {ingestionState.status === "complete" && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-card)] bg-success/10 border border-success/30">
          <Check size={14} className="text-success" />
          <span className="text-xs text-success">
            {ingestionState.response.pdb_id.toUpperCase()} ready —{" "}
            {ingestionState.response.residue_count} residues
          </span>
        </div>
      )}

      {ingestionState.status === "error" && (
        <div className="flex flex-col gap-1 px-3 py-2 rounded-[var(--radius-card)] bg-error/10 border border-error/30">
          <div className="flex items-center gap-2">
            <AlertCircle size={14} className="text-error shrink-0" />
            <span className="text-xs text-error">{ingestionState.message}</span>
          </div>
          {ingestionState.details?.map((line) => (
            <p key={line} className="text-[10px] text-error/80 pl-6 border-l border-error/30 ml-1">
              {line}
            </p>
          ))}
        </div>
      )}

      <div className="mt-2">
        <label className="text-[10px] uppercase tracking-wider text-text-muted mb-2 block">
          Previously Loaded
        </label>

        {loadingList ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 size={14} className="text-text-muted animate-spin" />
          </div>
        ) : previousStructures.length === 0 ? (
          <div className="text-xs text-text-muted text-center py-4">No structures loaded yet</div>
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
                <span className="text-[9px] text-text-muted">{s.residue_count}r</span>
                {s.has_embeddings && (
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-success shrink-0"
                    title="Has embeddings"
                  />
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
