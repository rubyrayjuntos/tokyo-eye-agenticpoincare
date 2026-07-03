/**
 * PipelineControls — left sidebar with ingest (canonical onboard trigger).
 */
import { useState, useCallback, useEffect, useRef } from "react";
import { api } from "../lib/api";
import { useDashboard } from "../lib/context";
import type { Structure, PipelineJob } from "../lib/types";

const PIPELINE_STEPS = [
  "ingestion",
  "graph_build",
  "gnn_forward_pass",
  "discovery_story",
  "complete",
];

export default function PipelineControls() {
  const { activeStructure, setActiveStructure, triggerRefresh } = useDashboard();

  const [pdbId, setPdbId] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const [structures, setStructures] = useState<Structure[]>([]);
  const [selectedStructureId, setSelectedStructureId] = useState<string>("");

  const [pipelineJob, setPipelineJob] = useState<PipelineJob | null>(null);
  const [executing, setExecuting] = useState(false);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadStructures = useCallback(() => {
    api.getStructures().then(setStructures).catch(() => {});
  }, []);

  useEffect(() => {
    loadStructures();
  }, [loadStructures]);

  useEffect(() => {
    if (activeStructure) {
      setSelectedStructureId(activeStructure.structure_id);
    }
  }, [activeStructure]);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startPipelinePolling = useCallback(
    (jobId: string) => {
      if (pollRef.current) clearInterval(pollRef.current);
      setExecuting(true);
      setPipelineError(null);

      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getPipelineStatus(jobId);
          setPipelineJob(status);

          if (status.status === "complete" || status.status === "failed") {
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }
            setExecuting(false);

            if (status.status === "failed") {
              setPipelineError(status.error || "Pipeline failed");
            } else {
              loadStructures();
              triggerRefresh();
            }
          }
        } catch {
          // Keep polling on transient errors.
        }
      }, 2000);
    },
    [loadStructures, triggerRefresh]
  );

  const handleIngest = useCallback(async () => {
    const trimmed = pdbId.trim().toUpperCase();
    if (!trimmed) return;

    setIngesting(true);
    setIngestError(null);
    setPipelineError(null);
    setPipelineJob(null);

    try {
      const result = await api.ingest(trimmed);
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
      setStructures((prev) => [newStructure, ...prev]);
      setSelectedStructureId(result.structure_id);
      setActiveStructure(newStructure);
      setPdbId("");

      if (result.pipeline_job_id) {
        try {
          const status = await api.getPipelineStatus(result.pipeline_job_id);
          setPipelineJob(status);
        } catch {
          setPipelineJob({
            job_id: result.pipeline_job_id,
            structure_id: result.structure_id,
            status: result.pipeline_status === "skipped" ? "queued" : result.pipeline_status,
            current_step: "gnn_inference",
            progress: 0,
            started_at: new Date().toISOString(),
            completed_at: null,
            error: null,
          });
        }
        startPipelinePolling(result.pipeline_job_id);
      } else if (!result.audit_only) {
        setPipelineError(
          "Ingest did not queue compute. Re-ingest or check coordinator logs."
        );
      }
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Ingestion failed";
      setIngestError(msg);
    } finally {
      setIngesting(false);
    }
  }, [pdbId, setActiveStructure, startPipelinePolling]);

  const handleTargetChange = (structureId: string) => {
    setSelectedStructureId(structureId);
    const found = structures.find((s) => s.structure_id === structureId);
    if (found) setActiveStructure(found);
  };

  const getStepIndex = (step: string) => {
    const idx = PIPELINE_STEPS.indexOf(step);
    return idx >= 0 ? idx : 0;
  };

  const isJobActive =
    pipelineJob &&
    (pipelineJob.status === "queued" || pipelineJob.status === "running");

  return (
    <aside className="w-64 shrink-0 border-r border-zinc-800 bg-zinc-900/60 flex flex-col overflow-y-auto">
      <div className="px-4 py-3 border-b border-zinc-800">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Pipeline Controls
        </h2>
      </div>

      <div className="px-4 py-3 border-b border-zinc-800 space-y-2">
        <label className="text-[10px] uppercase tracking-wider text-zinc-500">
          PDB ID
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="e.g. 4OBE"
            value={pdbId}
            onChange={(e) => setPdbId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleIngest();
            }}
            disabled={ingesting || executing}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
          />
          <button
            onClick={handleIngest}
            disabled={ingesting || executing || !pdbId.trim()}
            className="px-3 py-1.5 bg-cyan-700/30 border border-cyan-600/40 rounded text-xs text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-40"
          >
            {ingesting ? "..." : "Ingest"}
          </button>
        </div>
        <p className="text-[10px] text-zinc-500">
          Ingest queues governed structure load and the full discovery pathway.
        </p>
        {ingestError && (
          <p className="text-[10px] text-red-400">{ingestError}</p>
        )}
      </div>

      <div className="px-4 py-3 border-b border-zinc-800 space-y-2">
        <label className="text-[10px] uppercase tracking-wider text-zinc-500">
          Target
        </label>
        <select
          value={selectedStructureId}
          onChange={(e) => handleTargetChange(e.target.value)}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-cyan-600"
        >
          <option value="">Select structure...</option>
          {structures.map((s) => (
            <option key={s.structure_id} value={s.structure_id}>
              {s.pdb_id} — {s.title || s.structure_id}
            </option>
          ))}
        </select>
      </div>

      <div className="px-4 py-3 flex-1">
        <label className="text-[10px] uppercase tracking-wider text-zinc-500">
          Status
        </label>

        {!pipelineJob && !pipelineError && !executing && (
          <p className="text-xs text-zinc-600 mt-1">Idle</p>
        )}

        {executing && !pipelineJob && (
          <p className="text-xs text-cyan-400 mt-1">Queuing discovery pathway…</p>
        )}

        {pipelineError && (
          <p className="text-xs text-red-400 mt-1">{pipelineError}</p>
        )}

        {pipelineJob && (
          <div className="mt-2 space-y-1.5">
            <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  pipelineJob.status === "failed"
                    ? "bg-red-500"
                    : pipelineJob.status === "complete"
                    ? "bg-emerald-500"
                    : "bg-cyan-500"
                }`}
                style={{ width: `${pipelineJob.progress}%` }}
              />
            </div>

            <div className="space-y-1">
              {PIPELINE_STEPS.map((step, idx) => {
                const currentIdx = getStepIndex(
                  pipelineJob.current_step || ""
                );
                const isDone = idx < currentIdx || pipelineJob.status === "complete";
                const isCurrent =
                  idx === currentIdx && pipelineJob.status === "running";

                return (
                  <div
                    key={step}
                    className={`flex items-center gap-1.5 text-[10px] ${
                      isDone
                        ? "text-emerald-400"
                        : isCurrent
                        ? "text-cyan-400"
                        : "text-zinc-600"
                    }`}
                  >
                    <span className="w-3 text-center">
                      {isDone ? "✓" : isCurrent ? "●" : "○"}
                    </span>
                    <span>{step.replace(/_/g, " ")}</span>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center justify-between text-[10px]">
              <span
                className={
                  pipelineJob.status === "failed"
                    ? "text-red-400"
                    : pipelineJob.status === "complete"
                    ? "text-emerald-400"
                    : "text-cyan-400"
                }
              >
                {pipelineJob.status}
              </span>
              {isJobActive && (
                <span className="text-zinc-500">{pipelineJob.progress}%</span>
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
